"""Positive-unlabeled prototype retrieval for very sparse custom training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage


CHECKPOINT_FORMAT = "pu_query_retrieval_v1"
BACKGROUND_WEIGHT = 0.65
BACKGROUND_QUANTILE = 0.30
BACKGROUND_EXCLUSION_PIXELS = 3
MAX_BACKGROUND_PER_SUPPORT = 2048
QUERY_BLEND = 0.12
QUERY_QUANTILE = 0.997
QUERY_MIN_PIXELS = 4
QUERY_MAX_PIXELS = 128
QUERY_MIN_MARGIN = 0.05
QUERY_MAX_GROWTH = 1.35
QUERY_MIN_AREA_CAP = 64
POSTPROCESS_SEED_QUANTILE = 0.99
POSTPROCESS_RECALL_RATIO = 0.85
_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def normalize_feature_map(
    feature: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Return a globally standardized, per-pixel L2-normalized HWC map."""
    pixels = np.moveaxis(feature.astype(np.float32, copy=False), 0, -1)
    pixels = (pixels - mean) / np.maximum(std, 1e-5)
    return _l2_normalize(pixels)


def estimate_feature_stats(features: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate channel statistics from all pixels in the labelled support patches."""
    if not features:
        raise ValueError("At least one support embedding is required")
    total = np.zeros(features[0].shape[0], dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = 0
    for feature in features:
        pixels = feature.reshape(feature.shape[0], -1).T.astype(np.float64, copy=False)
        total += pixels.sum(axis=0)
        total_sq += np.square(pixels).sum(axis=0)
        count += len(pixels)
    mean = total / max(1, count)
    variance = np.maximum(total_sq / max(1, count) - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(
            [padded[y : y + result.shape[0], x : x + result.shape[1]]
             for y in range(3) for x in range(3)]
        )
    return result


def _gaussian_smooth(score: np.ndarray, sigma: float = 0.55) -> np.ndarray:
    radius = 2
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(score.astype(np.float32), ((radius, radius), (0, 0)), mode="reflect")
    vertical = sum(kernel[i] * padded[i : i + score.shape[0]] for i in range(len(kernel)))
    padded = np.pad(vertical, ((0, 0), (radius, radius)), mode="reflect")
    return sum(kernel[i] * padded[:, i : i + score.shape[1]] for i in range(len(kernel)))


def _f_beta(labels: np.ndarray, predicted: np.ndarray, beta: float = 0.5) -> float:
    tp = int(np.logical_and(predicted, labels == 1).sum())
    fp = int(np.logical_and(predicted, labels == 0).sum())
    fn = int(np.logical_and(~predicted, labels == 1).sum())
    beta_sq = beta * beta
    return float((1.0 + beta_sq) * tp / max(1.0, (1.0 + beta_sq) * tp + beta_sq * fn + fp))


def tune_threshold(positive_scores: np.ndarray, negative_scores: np.ndarray) -> Tuple[float, float]:
    max_negatives = max(4096, len(positive_scores) * 4)
    if len(negative_scores) > max_negatives:
        indices = np.linspace(0, len(negative_scores) - 1, max_negatives, dtype=int)
        negative_scores = negative_scores[indices]
    scores = np.concatenate([positive_scores, negative_scores]).astype(np.float32)
    labels = np.concatenate([
        np.ones(len(positive_scores), dtype=np.uint8),
        np.zeros(len(negative_scores), dtype=np.uint8),
    ])
    best_threshold = float(scores.max())
    best_f05 = -1.0
    for threshold in np.linspace(float(scores.min()), float(scores.max()), 180):
        metric = _f_beta(labels, scores >= threshold)
        if metric > best_f05:
            best_threshold, best_f05 = float(threshold), metric
    return best_threshold, best_f05


def _area_guard_prediction(
    score: np.ndarray,
    *,
    low: float,
    seed_quantile: float,
    min_pixels: int,
    max_component_pixels: int,
    max_total_ratio: float,
) -> np.ndarray:
    score_array = np.asarray(score, dtype=np.float32)
    high = max(float(low), float(np.quantile(score_array, seed_quantile)))
    candidates = score_array >= float(low)
    seeds = score_array >= high
    components, count = ndimage.label(candidates, structure=_EIGHT_CONNECTED)
    accepted = []
    for component_id in range(1, count + 1):
        component = components == component_id
        area = int(component.sum())
        if (
            area < int(min_pixels)
            or area > int(max_component_pixels)
            or not seeds[component].any()
        ):
            continue
        accepted.append(
            (float(score_array[np.logical_and(component, seeds)].max()), component)
        )

    pixel_cap = max(1, int(np.floor(score_array.size * max_total_ratio)))
    result = np.zeros(score_array.shape, dtype=bool)
    retained = 0
    for _, component in sorted(accepted, key=lambda item: item[0], reverse=True):
        area = int(component.sum())
        if retained + area > pixel_cap:
            continue
        result[component] = True
        retained += area
    return result


def _calibrate_postprocess(
    scores: List[np.ndarray],
    masks: List[np.ndarray],
    positive_scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Fit generic spatial guards without treating unlabeled pixels as negatives."""
    baseline = np.concatenate(
        [(score >= threshold).reshape(-1) for score in scores]
    )
    expected = np.concatenate([mask.reshape(-1) for mask in masks])
    positive_count = max(1, int(expected.sum()))
    baseline_recall = float(np.logical_and(baseline, expected).sum() / positive_count)
    recall_floor = max(
        0.55, min(0.90, baseline_recall * POSTPROCESS_RECALL_RATIO)
    )
    component_areas = np.asarray(
        [int(component.sum()) for mask in masks for component in [mask] if component.any()],
        dtype=np.float32,
    )
    min_area = max(1, int(component_areas.min()))
    max_area = max(1, int(component_areas.max()))
    support_ratios = [float(mask.mean()) for mask in masks]
    base_ratio = max(support_ratios)
    lows = sorted(
        {
            float(threshold),
            *[
                max(float(threshold), float(value))
                for value in np.quantile(positive_scores, [0.01, 0.05, 0.10])
            ],
        }
    )
    min_values = sorted({1, max(2, int(min_area * 0.05)), max(4, int(min_area * 0.10))})
    max_values = sorted({max_area * 2, max_area * 4, max_area * 8})
    ratio_values = sorted(
        {
            min(1.0, max(0.02, base_ratio * scale))
            for scale in (2.0, 3.0, 5.0)
        }
    )

    best_parameters = None
    best_rank = None
    best_recall = 0.0
    for seed_quantile in (0.95, 0.98, POSTPROCESS_SEED_QUANTILE):
        for low in lows:
            for min_pixels in min_values:
                for max_component_pixels in max_values:
                    for max_total_ratio in ratio_values:
                        predictions = [
                            _area_guard_prediction(
                                score,
                                low=low,
                                seed_quantile=seed_quantile,
                                min_pixels=min_pixels,
                                max_component_pixels=max_component_pixels,
                                max_total_ratio=max_total_ratio,
                            )
                            for score in scores
                        ]
                        predicted = np.concatenate(
                            [prediction.reshape(-1) for prediction in predictions]
                        )
                        recall = float(
                            np.logical_and(predicted, expected).sum()
                            / positive_count
                        )
                        if recall < recall_floor:
                            continue
                        rank = (
                            -max_total_ratio,
                            float(low),
                            min_pixels,
                            -max_component_pixels,
                            seed_quantile,
                            recall,
                        )
                        if best_rank is None or rank > best_rank:
                            best_rank = rank
                            best_recall = recall
                            best_parameters = {
                                "method": "relative_seed_area_guard",
                                "low": float(low),
                                "seed_quantile": float(seed_quantile),
                                "min_pixels": int(min_pixels),
                                "max_component_pixels": int(max_component_pixels),
                                "max_total_ratio": float(max_total_ratio),
                            }
    if best_parameters is None:
        return {"method": "fixed_threshold", "threshold": float(threshold)}
    best_parameters["calibration_positive_recall"] = float(best_recall)
    best_parameters["baseline_positive_recall"] = float(baseline_recall)
    best_parameters["recall_floor"] = float(recall_floor)
    return best_parameters


def predict_pu_query(score: np.ndarray, model_data: Dict[str, Any]) -> np.ndarray:
    """Convert a PU score map into a mask, preserving old checkpoints."""
    parameters = model_data.get("postprocess")
    if (
        not isinstance(parameters, dict)
        or not parameters.get("enabled", False)
        or parameters.get("method") != "relative_seed_area_guard"
    ):
        return np.asarray(score) >= float(model_data["threshold"])
    return _area_guard_prediction(
        score,
        low=float(parameters["low"]),
        seed_quantile=float(parameters["seed_quantile"]),
        min_pixels=int(parameters["min_pixels"]),
        max_component_pixels=int(parameters["max_component_pixels"]),
        max_total_ratio=float(parameters["max_total_ratio"]),
    )


def train_pu_query(
    polygon_samples: List[Tuple[str, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Fit foreground/background prototypes from individual polygon masks.

    ``polygon_samples`` contains ``(support_key, feature, polygon_mask)``. The
    same support feature may appear more than once when a patch has multiple
    labelled polygons.
    """
    if not polygon_samples:
        raise ValueError("No valid polygon samples after filtering")

    unique_features: Dict[str, np.ndarray] = {}
    masks_by_support: Dict[str, List[np.ndarray]] = defaultdict(list)
    for support_key, feature, mask in polygon_samples:
        unique_features[support_key] = feature
        masks_by_support[support_key].append(mask.astype(bool))

    mean, std = estimate_feature_stats(list(unique_features.values()))
    normalized = {
        key: normalize_feature_map(feature, mean, std)
        for key, feature in unique_features.items()
    }

    polygon_centers = []
    for support_key, _, mask in polygon_samples:
        pixels = normalized[support_key][mask.astype(bool)]
        if len(pixels):
            polygon_centers.append(_l2_normalize(pixels.mean(axis=0, keepdims=True))[0])
    if not polygon_centers:
        raise ValueError("Training polygons do not cover any embedding pixels")
    foreground = _l2_normalize(np.mean(polygon_centers, axis=0, keepdims=True))[0]

    positive_pixels, negative_pixels = [], []
    for support_key, masks in masks_by_support.items():
        pixels = normalized[support_key]
        positive_mask = np.logical_or.reduce(masks)
        positive_pixels.append(pixels[positive_mask])
        available = ~_binary_dilation(positive_mask, BACKGROUND_EXCLUSION_PIXELS)
        if not np.any(available):
            continue
        foreground_similarity = pixels @ foreground
        cutoff = float(np.quantile(foreground_similarity[available], BACKGROUND_QUANTILE))
        candidates = pixels[np.logical_and(available, foreground_similarity <= cutoff)]
        if len(candidates) > MAX_BACKGROUND_PER_SUPPORT:
            indices = np.linspace(0, len(candidates) - 1, MAX_BACKGROUND_PER_SUPPORT, dtype=int)
            candidates = candidates[indices]
        if len(candidates):
            negative_pixels.append(candidates)

    if not negative_pixels:
        raise ValueError("No reliable unlabeled background pixels are available")
    positives = np.concatenate(positive_pixels)
    negatives = np.concatenate(negative_pixels)
    background = _l2_normalize(negatives.mean(axis=0, keepdims=True))[0]
    positive_scores = positives @ foreground - BACKGROUND_WEIGHT * (positives @ background)
    negative_scores = negatives @ foreground - BACKGROUND_WEIGHT * (negatives @ background)
    threshold, f05 = tune_threshold(positive_scores, negative_scores)
    result = {
        "feature_mean": mean,
        "feature_std": std,
        "foreground_center": foreground,
        "background_center": background,
        "threshold": threshold,
        "training_f05": f05,
    }
    support_scores = [
        score_pu_query(feature, result)[0]
        for feature in unique_features.values()
    ]
    support_masks = [
        np.logical_or.reduce(masks_by_support[key])
        for key in unique_features
    ]
    result["postprocess"] = _calibrate_postprocess(
        support_scores,
        support_masks,
        positive_scores,
        threshold,
    )
    result["postprocess"]["enabled"] = False
    return result


def score_pu_query(feature: np.ndarray, model_data: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    """Score one query patch and apply one guarded query-adaptation step."""
    pixels = normalize_feature_map(
        feature,
        np.asarray(model_data["feature_mean"], dtype=np.float32),
        np.asarray(model_data["feature_std"], dtype=np.float32),
    )
    foreground = np.asarray(model_data["foreground_center"], dtype=np.float32)
    background = np.asarray(model_data["background_center"], dtype=np.float32)
    threshold = float(model_data["threshold"])
    base = _gaussian_smooth(pixels @ foreground - BACKGROUND_WEIGHT * (pixels @ background))

    cutoff = max(float(np.quantile(base, QUERY_QUANTILE)), threshold + QUERY_MIN_MARGIN)
    confident = base >= cutoff
    count = int(confident.sum())
    if not QUERY_MIN_PIXELS <= count <= QUERY_MAX_PIXELS:
        return base, False

    query = _l2_normalize(pixels[confident].mean(axis=0, keepdims=True))[0]
    query_score = _gaussian_smooth(pixels @ query - BACKGROUND_WEIGHT * (pixels @ background))
    candidate = (1.0 - QUERY_BLEND) * base + QUERY_BLEND * query_score
    base_area = int((base >= threshold).sum())
    candidate_area = int((candidate >= threshold).sum())
    if candidate_area > max(QUERY_MIN_AREA_CAP, int(base_area * QUERY_MAX_GROWTH)):
        return base, False
    return candidate, True
