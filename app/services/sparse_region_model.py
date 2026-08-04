"""Category-agnostic sparse region model for very small annotation sets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors

from app.services.pu_query import normalize_feature_map
from app.services.s2_ml import load_s2_features


CHECKPOINT_FORMAT = "sparse_region_model_v1"
RANDOM_SEED = 20260802


def load_optical_features(
    region_id: str,
    patch_id: str,
    month: str,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Load RGB reflectance as HWC and align it to an embedding grid."""
    features, valid, _ = load_s2_features(region_id, patch_id, month)
    rgb = np.moveaxis(features[[2, 1, 0]], 0, -1)
    rgb[~valid] = 0.0
    if rgb.shape[:2] != shape:
        zoom = (shape[0] / rgb.shape[0], shape[1] / rgb.shape[1], 1.0)
        rgb = ndimage.zoom(rgb, zoom, order=1)
        rgb = rgb[: shape[0], : shape[1]]
    finite = rgb[np.isfinite(rgb)]
    scale = max(float(np.quantile(finite, 0.99)), 1e-4) if finite.size else 1.0
    return np.clip(rgb / scale, 0.0, 1.0).astype(np.float32)


def local_descriptors(embedding: np.ndarray, optical: np.ndarray) -> np.ndarray:
    """Combine semantic, color, texture and boundary cues without class priors."""
    color_sum = np.maximum(optical.sum(axis=-1, keepdims=True), 1e-4)
    chroma = optical / color_sum
    descriptors = [embedding, chroma]
    for size in (3, 7):
        mean = ndimage.uniform_filter(
            embedding, size=(size, size, 1), mode="reflect"
        )
        mean_square = ndimage.uniform_filter(
            embedding**2, size=(size, size, 1), mode="reflect"
        )
        local_std = np.sqrt(np.maximum(mean_square - mean**2, 0.0)).mean(
            axis=-1, keepdims=True
        )
        color_mean = ndimage.uniform_filter(
            chroma, size=(size, size, 1), mode="reflect"
        )
        color_square = ndimage.uniform_filter(
            chroma**2, size=(size, size, 1), mode="reflect"
        )
        color_std = np.sqrt(np.maximum(color_square - color_mean**2, 0.0))
        descriptors.extend([local_std, color_mean, color_std])

    gradient = np.zeros(embedding.shape[:2], dtype=np.float32)
    for axis in (0, 1):
        shifted = np.roll(embedding, 1, axis=axis)
        difference = np.linalg.norm(embedding - shifted, axis=-1)
        if axis == 0:
            difference[0] = 0
        else:
            difference[:, 0] = 0
        gradient += difference
    gradient /= max(float(np.quantile(gradient, 0.98)), 1e-6)
    descriptors.append(np.clip(gradient, 0.0, 2.0)[..., None])
    return np.concatenate(descriptors, axis=-1).astype(np.float32)


def _guided_refine(
    probability: np.ndarray,
    embedding: np.ndarray,
    optical: np.ndarray,
) -> np.ndarray:
    edge_feature = np.concatenate([embedding * 0.45, optical * 1.8], axis=-1)
    distances = []
    for dy, dx in ((1, 0), (0, 1)):
        shifted = np.roll(edge_feature, (dy, dx), axis=(0, 1))
        distance = np.linalg.norm(edge_feature - shifted, axis=-1)
        if dy:
            distance[0] = np.nan
        if dx:
            distance[:, 0] = np.nan
        distances.append(distance)
    sigma = max(
        float(np.nanmedian(np.concatenate([item.ravel() for item in distances]))),
        0.05,
    )
    current = probability.astype(np.float32)
    for _ in range(5):
        numerator = np.zeros_like(current)
        denominator = np.zeros_like(current)
        for dy, dx in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            neighbor_feature = np.roll(edge_feature, (dy, dx), axis=(0, 1))
            neighbor_value = np.roll(current, (dy, dx), axis=(0, 1))
            weight = np.exp(
                -np.sum((edge_feature - neighbor_feature) ** 2, axis=-1)
                / (2.0 * sigma**2)
            )
            if dy < 0:
                weight[dy:] = 0
            elif dy > 0:
                weight[:dy] = 0
            if dx < 0:
                weight[:, dx:] = 0
            elif dx > 0:
                weight[:, :dx] = 0
            numerator += weight * neighbor_value
            denominator += weight
        current = 0.72 * probability + 0.28 * (
            numerator / np.maximum(denominator, 1e-6)
        )
    return current


def _remove_tiny_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    result = np.zeros_like(mask, dtype=bool)
    for identifier in range(1, count + 1):
        component = labels == identifier
        if int(component.sum()) >= minimum:
            result[component] = True
    return result


def train_sparse_region_model(
    polygon_samples: List[Tuple[str, np.ndarray, np.ndarray]],
    optical_by_support: Dict[str, np.ndarray],
    *,
    n_estimators: int = 240,
) -> Dict[str, Any]:
    """Train an ExtraTrees PU model from positives and reliable unlabeled pixels."""
    if not polygon_samples:
        raise ValueError("No valid polygon samples after filtering")
    features: Dict[str, np.ndarray] = {}
    masks_by_support: Dict[str, List[np.ndarray]] = defaultdict(list)
    for key, feature, mask in polygon_samples:
        features[key] = feature
        masks_by_support[key].append(mask.astype(bool))
    missing = sorted(set(features) - set(optical_by_support))
    if missing:
        raise ValueError(f"Optical features are missing for: {', '.join(missing)}")

    channels = features[next(iter(features))].shape[0]
    pixels = np.concatenate([item.reshape(channels, -1).T for item in features.values()])
    mean = pixels.mean(axis=0).astype(np.float32)
    std = np.maximum(pixels.std(axis=0), 1e-5).astype(np.float32)
    normalized = {
        key: normalize_feature_map(feature, mean, std)
        for key, feature in features.items()
    }
    descriptors = {
        key: local_descriptors(normalized[key], optical_by_support[key])
        for key in features
    }

    positives = np.concatenate(
        [descriptors[key][mask] for key, _, mask in polygon_samples]
    )
    positive_center = np.concatenate(
        [normalized[key][mask] for key, _, mask in polygon_samples]
    ).mean(axis=0)
    positive_center /= max(float(np.linalg.norm(positive_center)), 1e-8)

    negative_values = []
    negative_similarity = []
    for key, masks in masks_by_support.items():
        positive_mask = np.logical_or.reduce(masks)
        outside = ~ndimage.binary_dilation(positive_mask, iterations=3)
        negative_values.append(descriptors[key][outside])
        negative_similarity.append(normalized[key][outside] @ positive_center)
    unlabeled = np.concatenate(negative_values)
    similarities = np.concatenate(negative_similarity)
    cutoff = float(np.quantile(similarities, 0.25))
    reliable = unlabeled[similarities <= cutoff]
    rng = np.random.default_rng(RANDOM_SEED)
    count = min(len(reliable), max(3000, len(positives) * 5))
    if count == 0:
        raise ValueError("No reliable unlabeled pixels are available")
    reliable = reliable[rng.choice(len(reliable), count, replace=False)]

    classifier = ExtraTreesClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        max_features=0.65,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=min(4, n_estimators),
    )
    classifier.fit(
        np.concatenate([positives, reliable]),
        np.concatenate([
            np.ones(len(positives), dtype=np.uint8),
            np.zeros(len(reliable), dtype=np.uint8),
        ]),
    )
    reference = np.concatenate([positives, reliable])
    center = reference.mean(axis=0).astype(np.float32)
    scale = np.maximum(reference.std(axis=0), 1e-3).astype(np.float32)
    scaled_positives = (positives - center) / scale
    neighbor_count = min(8, len(scaled_positives))
    density_model = NearestNeighbors(
        n_neighbors=neighbor_count, metric="euclidean", n_jobs=1
    ).fit(scaled_positives)
    support_distance = density_model.kneighbors(
        scaled_positives, return_distance=True
    )[0][:, -1]
    density_radius = max(float(np.quantile(support_distance, 0.98)), 1e-3)

    model = {
        "__format__": CHECKPOINT_FORMAT,
        "classifier": classifier,
        "feature_mean": mean,
        "feature_std": std,
        "descriptor_center": center,
        "descriptor_scale": scale,
        "density_model": density_model,
        "density_radius": density_radius,
        "density_alpha": 0.30,
        "minimum_component_pixels": max(
            3, int(np.median([int(mask.sum()) for _, _, mask in polygon_samples]) * 0.025)
        ),
        "reliable_negative_cutoff": cutoff,
    }
    positive_probability = []
    for key, _, mask in polygon_samples:
        probability = score_sparse_region_model(
            features[key], optical_by_support[key], model, refine=False
        )
        positive_probability.append(probability[mask])
    model["threshold"] = float(
        np.clip(np.quantile(np.concatenate(positive_probability), 0.08), 0.42, 0.68)
    )
    model["training_positive_recall"] = float(
        np.mean(np.concatenate(positive_probability) >= model["threshold"])
    )
    return model


def score_sparse_region_model(
    feature: np.ndarray,
    optical: np.ndarray,
    model: Dict[str, Any],
    *,
    refine: bool = True,
) -> np.ndarray:
    normalized = normalize_feature_map(
        feature,
        np.asarray(model["feature_mean"], dtype=np.float32),
        np.asarray(model["feature_std"], dtype=np.float32),
    )
    descriptor = local_descriptors(normalized, optical)
    flat = descriptor.reshape(-1, descriptor.shape[-1])
    probability = model["classifier"].predict_proba(flat)[:, 1]
    scaled = (
        flat - np.asarray(model["descriptor_center"], dtype=np.float32)
    ) / np.asarray(model["descriptor_scale"], dtype=np.float32)
    distance = model["density_model"].kneighbors(
        scaled, return_distance=True
    )[0][:, -1]
    density = np.exp(-0.5 * (distance / float(model["density_radius"])) ** 2)
    probability *= np.power(density, float(model.get("density_alpha", 0.30)))
    probability = probability.reshape(descriptor.shape[:2]).astype(np.float32)
    if refine:
        probability = _guided_refine(probability, normalized, optical)
    return probability


def predict_sparse_region_model(score: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    mask = np.asarray(score) >= float(model["threshold"])
    return _remove_tiny_components(
        mask, int(model.get("minimum_component_pixels", 3))
    )


def select_sparse_strategy(
    polygon_samples: List[Tuple[str, np.ndarray, np.ndarray]],
    optical_by_support: Dict[str, np.ndarray],
) -> Tuple[str, Dict[str, Any]]:
    """Select the new model only when sparse-label validation supports it."""
    from app.services.pu_query import predict_pu_query, score_pu_query, train_pu_query

    baseline = train_pu_query(polygon_samples)
    thickness = [
        float(ndimage.distance_transform_edt(mask.astype(bool)).max())
        for _, _, mask in polygon_samples
    ]
    if float(np.median(thickness)) <= 2.0:
        baseline["selection"] = {
            "selected": "pu_query_retrieval",
            "reason": "thin_annotation_guard",
            "median_annotation_half_width": float(np.median(thickness)),
        }
        return "pu_query_retrieval", baseline

    adaptive = train_sparse_region_model(polygon_samples, optical_by_support)
    if len(polygon_samples) == 1:
        adaptive["selection"] = {
            "selected": "sparse_region_model",
            "reason": "single_polygon_no_holdout",
        }
        return "sparse_region_model", adaptive

    indices = np.linspace(
        0, len(polygon_samples) - 1, min(3, len(polygon_samples)), dtype=int
    )
    baseline_quality = []
    adaptive_quality = []
    baseline_recall = []
    adaptive_recall = []
    for held_index in sorted(set(indices.tolist())):
        held_key, held_feature, held_mask = polygon_samples[held_index]
        train_samples = [
            sample for index, sample in enumerate(polygon_samples)
            if index != held_index
        ]
        if not train_samples:
            continue
        try:
            fold_baseline = train_pu_query(train_samples)
            fold_adaptive = train_sparse_region_model(
                train_samples,
                {
                    key: value for key, value in optical_by_support.items()
                    if key in {sample[0] for sample in train_samples}
                },
                n_estimators=64,
            )
        except ValueError:
            continue
        baseline_score, _ = score_pu_query(held_feature, fold_baseline)
        baseline_mask = predict_pu_query(baseline_score, fold_baseline)
        adaptive_score = score_sparse_region_model(
            held_feature, optical_by_support[held_key], fold_adaptive
        )
        adaptive_mask = predict_sparse_region_model(adaptive_score, fold_adaptive)
        target_area = max(1, int(held_mask.sum()))
        for prediction, recalls, qualities in (
            (baseline_mask, baseline_recall, baseline_quality),
            (adaptive_mask, adaptive_recall, adaptive_quality),
        ):
            recall = float(np.logical_and(prediction, held_mask).sum() / target_area)
            expansion = float(prediction.sum() / target_area)
            penalty = 0.08 * max(0.0, np.log2(max(expansion, 1e-6) / 4.0))
            recalls.append(recall)
            qualities.append(recall - penalty)

    if not adaptive_quality:
        adaptive["selection"] = {
            "selected": "sparse_region_model",
            "reason": "holdout_unavailable",
        }
        return "sparse_region_model", adaptive

    old_quality = float(np.mean(baseline_quality))
    new_quality = float(np.mean(adaptive_quality))
    old_recall = float(np.mean(baseline_recall))
    new_recall = float(np.mean(adaptive_recall))
    selection = {
        "baseline_quality": old_quality,
        "adaptive_quality": new_quality,
        "baseline_positive_recall": old_recall,
        "adaptive_positive_recall": new_recall,
        "fold_count": len(adaptive_quality),
    }
    if new_quality >= old_quality + 0.01 and new_recall >= old_recall * 0.85:
        selection.update({"selected": "sparse_region_model", "reason": "holdout_win"})
        adaptive["selection"] = selection
        return "sparse_region_model", adaptive
    selection.update({"selected": "pu_query_retrieval", "reason": "baseline_retained"})
    baseline["selection"] = selection
    return "pu_query_retrieval", baseline
