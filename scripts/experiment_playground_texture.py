"""Test optical texture-boundary constraints for ``playground_xuannv``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_playground_xuannv import (
    EMBEDDING_ROOT,
    MODEL_ID,
    MODEL_NAME,
    OPTICAL_ROOT,
    REGISTRY_PATH,
    REFERENCE_LIMITATION,
    _combined_metrics,
    distribution,
    load_registered_model,
    predict_variants,
    score_with_and_without_query,
    write_strict_json,
)
from scripts.playground_pu_postprocess import (
    component_statistics,
    hysteresis_prediction,
)


TRAINING_PATCH_IDS = ("patch_000059", "patch_000060", "patch_000064")
INDEPENDENT_OSM_PATCH_IDS = ("patch_000076",)
GLOBAL_HIGH_FALSE_PATCH_IDS = (
    "patch_000232",
    "patch_000249",
    "patch_000154",
)
_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    finite = values[np.isfinite(values)]
    nonzero = finite[finite != 0]
    sample = nonzero if len(nonzero) >= max(32, int(finite.size * 0.02)) else finite
    if not len(sample):
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.quantile(sample, [0.02, 0.98])
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _read_stretched_optical(path: Path) -> np.ndarray:
    if not Path(path).is_file():
        raise FileNotFoundError(f"High-resolution optical image not found: {path}")
    with rasterio.open(path) as dataset:
        count = min(3, dataset.count)
        values = dataset.read(list(range(1, count + 1)))
    if count == 1:
        values = np.repeat(values, 3, axis=0)
    elif count == 2:
        values = np.concatenate([values, values[-1:]], axis=0)
    return np.stack([_stretch_channel(values[index]) for index in range(3)], axis=-1)


def _robust_unit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return np.zeros(array.shape, dtype=np.float32)
    low, high = np.quantile(finite, [0.05, 0.995])
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def compute_highres_texture_boundary(
    optical_path: Path,
    output_shape: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Derive a semantic-free edge and local-texture boundary map."""
    optical = _read_stretched_optical(optical_path)
    luminance = (
        0.299 * optical[..., 0]
        + 0.587 * optical[..., 1]
        + 0.114 * optical[..., 2]
    )
    gradient_x = ndimage.sobel(luminance, axis=1, mode="reflect")
    gradient_y = ndimage.sobel(luminance, axis=0, mode="reflect")
    gradient = np.hypot(gradient_x, gradient_y)
    local_mean = ndimage.uniform_filter(luminance, size=11, mode="reflect")
    local_sq_mean = ndimage.uniform_filter(
        np.square(luminance),
        size=11,
        mode="reflect",
    )
    local_std = np.sqrt(np.maximum(local_sq_mean - np.square(local_mean), 0.0))
    color_gradient = np.zeros(luminance.shape, dtype=np.float32)
    for channel in range(3):
        gx = ndimage.sobel(optical[..., channel], axis=1, mode="reflect")
        gy = ndimage.sobel(optical[..., channel], axis=0, mode="reflect")
        color_gradient = np.maximum(color_gradient, np.hypot(gx, gy))
    boundary = (
        0.55 * _robust_unit(gradient)
        + 0.30 * _robust_unit(color_gradient)
        + 0.15 * _robust_unit(local_std)
    )
    boundary = ndimage.maximum_filter(boundary, size=3, mode="reflect")
    resized = Image.fromarray(boundary.astype(np.float32)).resize(
        (output_shape[1], output_shape[0]),
        resample=Image.Resampling.BILINEAR,
    )
    return _robust_unit(np.asarray(resized, dtype=np.float32))


def relative_seed_threshold(score: np.ndarray, seed_quantile: float) -> float:
    if not 0.0 < seed_quantile < 1.0:
        raise ValueError("seed_quantile must be between 0 and 1")
    values = np.asarray(score, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("score must contain only finite values")
    return float(np.quantile(values, seed_quantile))


def area_guard_prediction(
    score: np.ndarray,
    *,
    seed_quantile: float,
    low: float,
    min_pixels: int,
    max_component_pixels: int,
    max_total_ratio: float,
) -> np.ndarray:
    """Use patch-relative seeds with score connectivity and area guards only."""
    high = max(float(low), relative_seed_threshold(score, seed_quantile))
    return hysteresis_prediction(
        score,
        high=high,
        low=float(low),
        min_pixels=min_pixels,
        max_component_pixels=max_component_pixels,
        max_total_ratio=max_total_ratio,
    )


def texture_boundary_area_prediction(
    score: np.ndarray,
    texture_boundary: np.ndarray,
    *,
    seed_quantile: float,
    low: float,
    boundary_quantile: float,
    min_pixels: int,
    max_component_pixels: int,
    max_total_ratio: float,
) -> np.ndarray:
    """Prevent score expansion from crossing strong optical texture boundaries."""
    score_array = np.asarray(score, dtype=np.float32)
    boundary = np.asarray(texture_boundary, dtype=np.float32)
    if score_array.shape != boundary.shape:
        raise ValueError("score and texture_boundary must have the same shape")
    if not np.isfinite(boundary).all():
        raise ValueError("texture_boundary must contain only finite values")
    if not 0.0 < boundary_quantile < 1.0:
        raise ValueError("boundary_quantile must be between 0 and 1")
    high = max(float(low), relative_seed_threshold(score_array, seed_quantile))
    seeds = score_array >= high
    boundary_limit = float(np.quantile(boundary, boundary_quantile))
    barriers = boundary > boundary_limit
    filtered = score_array.copy()
    filtered[np.logical_and(barriers, ~seeds)] = np.nextafter(
        np.float32(low),
        np.float32(-np.inf),
    )
    return hysteresis_prediction(
        filtered,
        high=high,
        low=float(low),
        min_pixels=min_pixels,
        max_component_pixels=max_component_pixels,
        max_total_ratio=max_total_ratio,
    )


def _candidate_area_parameters(
    pixel_count: int,
    production_threshold: float,
) -> Iterable[Dict[str, Any]]:
    total_ratios = (1.0,) if pixel_count < 100 else (0.02, 0.03, 0.05)
    max_components = (64,) if pixel_count < 100 else (128, 256, 512)
    min_components = (1, 2, 4) if pixel_count < 100 else (4, 8, 16)
    for seed_quantile in (0.90, 0.95, 0.98, 0.99):
        for low in (production_threshold, 0.35, 0.45, 0.55):
            for min_pixels in min_components:
                for max_component_pixels in max_components:
                    for max_total_ratio in total_ratios:
                        yield {
                            "seed_quantile": seed_quantile,
                            "low": float(low),
                            "min_pixels": min_pixels,
                            "max_component_pixels": max_component_pixels,
                            "max_total_ratio": max_total_ratio,
                        }


def calibrate_texture_experiment(
    scores_by_patch: Mapping[str, np.ndarray],
    boundaries_by_patch: Mapping[str, np.ndarray],
    references_by_patch: Mapping[str, np.ndarray],
    calibration_patch_ids: Sequence[str],
    excluded_patch_ids: Sequence[str],
    production_threshold: float,
) -> Dict[str, Any]:
    """Freeze area-only and texture parameters on support polygons only."""
    calibration_ids = list(calibration_patch_ids)
    excluded_ids = list(excluded_patch_ids)
    overlap = sorted(set(calibration_ids) & set(excluded_ids))
    if overlap:
        raise ValueError(
            "Independent holdout patches must not enter calibration: "
            + ", ".join(overlap)
        )
    if not calibration_ids:
        raise ValueError("At least one calibration patch is required")
    for patch_id in calibration_ids:
        if (
            patch_id not in scores_by_patch
            or patch_id not in boundaries_by_patch
            or patch_id not in references_by_patch
        ):
            raise ValueError(f"Missing calibration input for {patch_id}")
    scores = [scores_by_patch[patch_id] for patch_id in calibration_ids]
    boundaries = [boundaries_by_patch[patch_id] for patch_id in calibration_ids]
    references = [references_by_patch[patch_id] for patch_id in calibration_ids]
    baseline = _combined_metrics(
        [score >= production_threshold for score in scores],
        references,
    )
    recall_floor = max(0.55, min(0.85, baseline["recall"] * 0.85))
    best_area = None
    best_area_metrics = None
    best_area_rank = None
    fallback = None
    fallback_metrics = None
    fallback_rank = None
    for parameters in _candidate_area_parameters(
        scores[0].size,
        production_threshold,
    ):
        predictions = [
            area_guard_prediction(score, **parameters) for score in scores
        ]
        metrics = _combined_metrics(predictions, references)
        fallback_candidate = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            -metrics["positive_ratio"],
        )
        if fallback_rank is None or fallback_candidate > fallback_rank:
            fallback_rank = fallback_candidate
            fallback = parameters
            fallback_metrics = metrics
        if metrics["recall"] < recall_floor:
            continue
        rank = (
            metrics["precision"],
            metrics["f1"],
            -metrics["positive_ratio"],
            metrics["recall"],
        )
        if best_area_rank is None or rank > best_area_rank:
            best_area_rank = rank
            best_area = parameters
            best_area_metrics = metrics
    if best_area is None:
        best_area = fallback
        best_area_metrics = fallback_metrics
    if best_area is None or best_area_metrics is None:
        raise RuntimeError("Unable to calibrate area-only parameters")

    texture_recall_floor = max(0.50, best_area_metrics["recall"] * 0.90)
    best_boundary_quantile = None
    best_texture_metrics = None
    best_texture_rank = None
    for boundary_quantile in (0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        parameters = dict(best_area, boundary_quantile=boundary_quantile)
        predictions = [
            texture_boundary_area_prediction(score, boundary, **parameters)
            for score, boundary in zip(scores, boundaries)
        ]
        metrics = _combined_metrics(predictions, references)
        if metrics["recall"] < texture_recall_floor:
            continue
        rank = (
            metrics["precision"],
            metrics["f1"],
            -metrics["positive_ratio"],
            metrics["recall"],
        )
        if best_texture_rank is None or rank > best_texture_rank:
            best_texture_rank = rank
            best_boundary_quantile = boundary_quantile
            best_texture_metrics = metrics
    if best_boundary_quantile is None:
        best_boundary_quantile = 0.95
        parameters = dict(best_area, boundary_quantile=best_boundary_quantile)
        best_texture_metrics = _combined_metrics(
            [
                texture_boundary_area_prediction(score, boundary, **parameters)
                for score, boundary in zip(scores, boundaries)
            ],
            references,
        )
    texture_parameters = dict(
        best_area,
        boundary_quantile=float(best_boundary_quantile),
    )
    return {
        "calibration_patch_ids": calibration_ids,
        "excluded_patch_ids": excluded_ids,
        "production_threshold": float(production_threshold),
        "area_guard": best_area,
        "texture_boundary_area_guard": texture_parameters,
        "search_evidence": {
            "reference_limitation": REFERENCE_LIMITATION,
            "baseline_reference_metrics": baseline,
            "area_guard_reference_metrics": best_area_metrics,
            "texture_reference_metrics": best_texture_metrics,
            "area_recall_floor": float(recall_floor),
            "texture_recall_floor": float(texture_recall_floor),
            "selection_rule": (
                "仅在训练 Polygon 上先满足召回下限，再最大化参考相对"
                " Precision/F1 并压低预测面积；独立 OSM 不参与。"
            ),
        },
    }


def _patch_center(patch: Mapping[str, Any]) -> Tuple[float, float]:
    bounds = patch["bounds_wgs84"]
    return (
        float((bounds[0] + bounds[2]) / 2.0),
        float((bounds[1] + bounds[3]) / 2.0),
    )


def select_typical_patches(
    metadata: Mapping[str, Any],
    baseline_ratios: Mapping[str, float],
) -> Dict[str, Any]:
    """Select fixed positives, extreme errors, and one spatially distinct patch."""
    patches = metadata.get("patches")
    if not isinstance(patches, list):
        raise ValueError("Patch metadata must contain a patches list")
    by_id = {patch["patch_id"]: patch for patch in patches}
    centers = {patch_id: _patch_center(patch) for patch_id, patch in by_id.items()}
    longitude_median = float(np.median([center[0] for center in centers.values()]))
    latitude_median = float(np.median([center[1] for center in centers.values()]))
    reserved = set(TRAINING_PATCH_IDS + INDEPENDENT_OSM_PATCH_IDS + GLOBAL_HIGH_FALSE_PATCH_IDS)
    positive_centers = [
        centers[patch_id]
        for patch_id in TRAINING_PATCH_IDS + INDEPENDENT_OSM_PATCH_IDS
        if patch_id in centers
    ]

    def distinct_from_positives(patch_id: str) -> bool:
        if not positive_centers:
            return True
        return min(
            math.dist(centers[patch_id], positive_center)
            for positive_center in positive_centers
        ) >= 0.04

    candidates = [
        patch_id
        for patch_id, center in centers.items()
        if patch_id not in reserved
        and patch_id in baseline_ratios
        and center[0] >= longitude_median
        and center[1] <= latitude_median
        and distinct_from_positives(patch_id)
    ]
    if not candidates:
        candidates = [
            patch_id
            for patch_id in centers
            if patch_id not in reserved and patch_id in baseline_ratios
        ]
    spatial_patch = max(
        candidates,
        key=lambda patch_id: (baseline_ratios[patch_id], patch_id),
    )
    groups = {
        "training": list(TRAINING_PATCH_IDS),
        "independent_osm": list(INDEPENDENT_OSM_PATCH_IDS),
        "global_high_false_positive": list(GLOBAL_HIGH_FALSE_PATCH_IDS),
        "spatial_high_score": [spatial_patch],
    }
    reasons = {
        **{patch_id: "原模型训练 Polygon 所在 Patch" for patch_id in TRAINING_PATCH_IDS},
        "patch_000076": "未参与调参的独立 OSM 操场",
        "patch_000232": "全域原始阈值预测面积最高的极端误检候选",
        "patch_000249": "全域原始阈值预测面积第二高的极端误检候选",
        "patch_000154": "接近区域中部且原始阈值预测面积较高的误检候选",
        spatial_patch: (
            "位于区域中心线以东、中心线以南，且与训练/OSM Patch 保持"
            "空间距离的高分代表"
        ),
    }
    evidence = []
    for group_patch_ids in groups.values():
        for patch_id in group_patch_ids:
            if patch_id not in centers:
                raise ValueError(f"Selected patch missing from metadata: {patch_id}")
            evidence.append(
                {
                    "patch_id": patch_id,
                    "center_wgs84": list(centers[patch_id]),
                    "baseline_positive_ratio": float(baseline_ratios[patch_id]),
                    "reason": reasons[patch_id],
                }
            )
    groups["selection_evidence"] = evidence
    groups["spatial_rule"] = {
        "longitude_median": longitude_median,
        "latitude_median": latitude_median,
        "minimum_distance_degrees_from_known_positive": 0.04,
    }
    return groups


def _reference_metrics(
    predictions_by_patch: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    return _combined_metrics(
        [predictions_by_patch[patch_id] for patch_id in references],
        [references[patch_id] for patch_id in references],
    )


def _variant_summary(
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    ratios = [float(prediction.mean()) for prediction in predictions.values()]
    component_areas = []
    component_count = 0
    for patch_id, prediction in predictions.items():
        components = component_statistics(prediction, scores[patch_id])
        component_count += len(components)
        component_areas.extend(item["area"] for item in components)
    return {
        "patch_count": len(predictions),
        "nonempty_patch_count": int(sum(value.any() for value in predictions.values())),
        "mean_positive_ratio": float(np.mean(ratios)),
        "median_positive_ratio": float(np.median(ratios)),
        "component_count": component_count,
        "component_area_pixels": distribution(np.asarray(component_areas)),
    }


def assess_texture_improvement(
    *,
    area_osm: Mapping[str, float],
    texture_osm: Mapping[str, float],
    area_high_false_ratio: float,
    texture_high_false_ratio: float,
    area_training: Mapping[str, float],
    texture_training: Mapping[str, float],
    legacy_osm: Mapping[str, float],
) -> Dict[str, Any]:
    """Require a material, not merely numerical, gain over the fair control."""
    osm_f1_delta = float(texture_osm["f1"] - area_osm["f1"])
    osm_recall_delta = float(texture_osm["recall"] - area_osm["recall"])
    high_false_delta = float(texture_high_false_ratio - area_high_false_ratio)
    training_recall_delta = float(
        texture_training["recall"] - area_training["recall"]
    )
    recall_preserved = osm_recall_delta >= -0.02
    training_preserved = training_recall_delta >= -0.05
    material_signal = (
        osm_f1_delta >= 0.02
        or (
            high_false_delta <= -0.002
            and osm_recall_delta >= 0.0
        )
    )
    materially_improved = (
        recall_preserved and training_preserved and material_signal
    )
    improved_over_legacy = (
        texture_osm["recall"] - legacy_osm["recall"] >= 0.20
    )
    return {
        "materially_improved_over_area_guard": bool(materially_improved),
        "improved_over_legacy_guarded": bool(improved_over_legacy),
        "independent_osm_f1_delta": osm_f1_delta,
        "independent_osm_recall_delta": osm_recall_delta,
        "high_false_mean_area_ratio_delta": high_false_delta,
        "training_recall_delta": training_recall_delta,
        "minimum_material_change": {
            "independent_osm_f1_delta": 0.02,
            "high_false_mean_area_ratio_reduction": 0.002,
            "maximum_osm_recall_drop": 0.02,
            "maximum_training_recall_drop": 0.05,
        },
        "verdict": (
            "纹理边界有实质增益"
            if materially_improved
            else "纹理边界未证明有实质增益"
        ),
        "interpretation": (
            "与现有全局阈值面积保护相比，新流程恢复独立 OSM 操场主要来自"
            " Patch 内相对种子阈值；与使用同一相对种子和面积限制的公平对照"
            "相比，纹理边界本身只有达到最小实质变化才算改善。"
        ),
    }


def run_texture_experiment(
    task3_input: Path,
    output: Path,
    repo_root: Path = ROOT,
) -> Dict[str, Any]:
    task3_input = Path(task3_input).resolve()
    output = Path(output).resolve()
    repo_root = Path(repo_root).resolve()
    task3_manifest = json.loads(
        (task3_input / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    task3_metrics = json.loads(
        (task3_input / "metrics.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (repo_root / "data/haidian/patches_meta_v2.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_ratios = {
        patch_id: float(values["query"]["baseline"]["positive_ratio"])
        for patch_id, values in task3_metrics["per_patch"].items()
    }
    selection = select_typical_patches(metadata, baseline_ratios)
    selected_patch_ids = [
        patch_id
        for key in (
            "training",
            "independent_osm",
            "global_high_false_positive",
            "spatial_high_score",
        )
        for patch_id in selection[key]
    ]

    task3_artifacts = task3_manifest["artifacts"]["arrays"]
    missing_artifacts = [
        patch_id for patch_id in selected_patch_ids if patch_id not in task3_artifacts
    ]
    missing_model_data = None
    if missing_artifacts:
        _, missing_model_data = load_registered_model(
            REGISTRY_PATH,
            MODEL_NAME,
            MODEL_ID,
        )
    scores = {}
    references = {}
    boundaries = {}
    optical_paths = {}
    task3_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for patch_id in selected_patch_ids:
        if patch_id in task3_artifacts:
            artifact = task3_artifacts[patch_id]
            arrays = artifact["arrays"]
            scores[patch_id] = np.load(
                task3_input / arrays["score_query"],
                allow_pickle=False,
            )
            references[patch_id] = np.load(
                task3_input / arrays["reference"],
                allow_pickle=False,
            ).astype(bool)
            task3_predictions[patch_id] = {
                name: np.load(
                    task3_input / arrays[name],
                    allow_pickle=False,
                ).astype(bool)
                for name in ("baseline", "strict", "guarded")
            }
            optical_path = repo_root / artifact["optical_path"]
        else:
            if missing_model_data is None:
                raise RuntimeError("Missing checkpoint for selected-patch scoring")
            feature = np.load(
                EMBEDDING_ROOT / "202604" / f"{patch_id}.npy",
                allow_pickle=False,
            )
            score, _, _ = score_with_and_without_query(feature, missing_model_data)
            scores[patch_id] = score
            references[patch_id] = np.zeros(score.shape, dtype=bool)
            task3_predictions[patch_id] = predict_variants(
                score,
                float(task3_manifest["model"]["production_threshold"]),
                task3_metrics["parameters"],
            )
            optical_matches = sorted(
                OPTICAL_ROOT.glob(
                    f"highres_optical_20260401_{patch_id}.tif"
                )
            )
            if not optical_matches:
                raise FileNotFoundError(
                    f"No 202604 optical image for selected patch {patch_id}"
                )
            optical_path = optical_matches[0]
        optical_paths[patch_id] = optical_path
        boundaries[patch_id] = compute_highres_texture_boundary(optical_path)

    training_references = {
        patch_id: references[patch_id] for patch_id in TRAINING_PATCH_IDS
    }
    production_threshold = float(task3_manifest["model"]["production_threshold"])
    parameters = calibrate_texture_experiment(
        scores,
        boundaries,
        training_references,
        calibration_patch_ids=TRAINING_PATCH_IDS,
        excluded_patch_ids=INDEPENDENT_OSM_PATCH_IDS,
        production_threshold=production_threshold,
    )
    area_predictions = {
        patch_id: area_guard_prediction(
            score,
            **parameters["area_guard"],
        )
        for patch_id, score in scores.items()
    }
    texture_predictions = {
        patch_id: texture_boundary_area_prediction(
            score,
            boundaries[patch_id],
            **parameters["texture_boundary_area_guard"],
        )
        for patch_id, score in scores.items()
    }
    variants = {
        "baseline": {
            patch_id: values["baseline"]
            for patch_id, values in task3_predictions.items()
        },
        "strict": {
            patch_id: values["strict"]
            for patch_id, values in task3_predictions.items()
        },
        "guarded": {
            patch_id: values["guarded"]
            for patch_id, values in task3_predictions.items()
        },
        "area_guard": area_predictions,
        "texture_boundary_area_guard": texture_predictions,
    }
    independent_references = {
        patch_id: references[patch_id] for patch_id in INDEPENDENT_OSM_PATCH_IDS
    }
    high_false_ids = list(
        selection["global_high_false_positive"]
        + selection["spatial_high_score"]
    )
    metrics = {
        "reference_policy": {
            "limitation": REFERENCE_LIMITATION,
            "unlabeled_pixels_are_reliable_negatives": False,
            "texture_prior": (
                "仅使用真实高分辨率光学 RGB 的梯度和局部纹理边界；"
                "未使用建筑、道路、操场或其他人工类别掩膜。"
            ),
        },
        "parameters": parameters,
        "selection": selection,
        "selected_patch_summary": {
            name: _variant_summary(predictions, scores)
            for name, predictions in variants.items()
        },
        "reference_relative_metrics": {
            "training_polygons": {
                name: _reference_metrics(predictions, training_references)
                for name, predictions in variants.items()
            },
            "independent_osm_polygon": {
                name: _reference_metrics(predictions, independent_references)
                for name, predictions in variants.items()
            },
        },
        "high_false_candidate_mean_positive_ratio": {
            name: float(
                np.mean(
                    [
                        predictions[patch_id].mean()
                        for patch_id in high_false_ids
                    ]
                )
            )
            for name, predictions in variants.items()
        },
        "per_patch": {
            patch_id: {
                "role": next(
                    key
                    for key in (
                        "training",
                        "independent_osm",
                        "global_high_false_positive",
                        "spatial_high_score",
                    )
                    if patch_id in selection[key]
                ),
                "variants": {
                    name: {
                        "positive_pixels": int(predictions[patch_id].sum()),
                        "positive_ratio": float(predictions[patch_id].mean()),
                        "component_count": len(
                            component_statistics(
                                predictions[patch_id],
                                scores[patch_id],
                            )
                        ),
                    }
                    for name, predictions in variants.items()
                },
            }
            for patch_id in selected_patch_ids
        },
    }
    osm_metrics = metrics["reference_relative_metrics"]["independent_osm_polygon"]
    high_false = metrics["high_false_candidate_mean_positive_ratio"]
    training_metrics = metrics["reference_relative_metrics"]["training_polygons"]
    metrics["texture_assessment"] = assess_texture_improvement(
        area_osm=osm_metrics["area_guard"],
        texture_osm=osm_metrics["texture_boundary_area_guard"],
        area_high_false_ratio=high_false["area_guard"],
        texture_high_false_ratio=high_false["texture_boundary_area_guard"],
        area_training=training_metrics["area_guard"],
        texture_training=training_metrics["texture_boundary_area_guard"],
        legacy_osm=osm_metrics["guarded"],
    )

    arrays_dir = output / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for patch_id in selected_patch_ids:
        values = {
            "score_query": scores[patch_id].astype(np.float32),
            "reference": references[patch_id].astype(np.uint8),
            "texture_boundary": boundaries[patch_id].astype(np.float32),
            **{
                name: predictions[patch_id].astype(np.uint8)
                for name, predictions in variants.items()
            },
        }
        array_paths = {}
        for name, value in values.items():
            relative = Path("arrays") / f"{patch_id}_{name}.npy"
            np.save(output / relative, value, allow_pickle=False)
            array_paths[name] = relative.as_posix()
        artifacts[patch_id] = {
            "role": metrics["per_patch"][patch_id]["role"],
            "optical_path": str(optical_paths[patch_id].relative_to(repo_root)),
            "arrays": array_paths,
        }
    manifest = {
        "experiment": {
            "name": "playground_xuannv optical texture boundary experiment",
            "month": "202604",
            "scope": "selected_typical_patches_only",
            "selected_patch_count": len(selected_patch_ids),
            "full_domain_statistics_reused_from_task3": True,
            "checkpoint_retrained": False,
            "production_api_modified": False,
        },
        "model": task3_manifest["model"],
        "reference_policy": metrics["reference_policy"],
        "selection": selection,
        "artifacts": {
            "metrics": "metrics.json",
            "arrays": artifacts,
        },
    }
    write_strict_json(output / "experiment_manifest.json", manifest)
    write_strict_json(output / "metrics.json", metrics)
    return {"manifest": manifest, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "Tmp/playground_pu_query_20260731",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Tmp/playground_texture_20260731",
    )
    args = parser.parse_args()
    result = run_texture_experiment(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selection": result["manifest"]["selection"],
                "texture_assessment": result["metrics"]["texture_assessment"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
