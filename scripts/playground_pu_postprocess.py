"""Offline postprocessing utilities for Haidian PU + Query score maps."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import ndimage


PRODUCTION_THRESHOLD = 0.247057
_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def _validated_pairs(
    scores: Sequence[np.ndarray], labels: Sequence[np.ndarray]
) -> List[tuple[np.ndarray, np.ndarray]]:
    if not scores or not labels:
        raise ValueError("scores and labels must be non-empty")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    pairs = []
    for score, label in zip(scores, labels):
        score_array = np.asarray(score)
        label_array = np.asarray(label, dtype=bool)
        if score_array.shape != label_array.shape:
            raise ValueError("each score array must match its label array")
        if not np.isfinite(score_array).all():
            raise ValueError("score arrays must contain only finite values")
        pairs.append((score_array, label_array))
    return pairs


def strict_threshold(scores: List[np.ndarray], labels: List[np.ndarray]) -> float:
    """Select a strict offline threshold from calibration patches only."""
    pairs = _validated_pairs(scores, labels)
    all_scores = np.concatenate([score.reshape(-1) for score, _ in pairs])
    all_labels = np.concatenate([label.reshape(-1) for _, label in pairs])
    upper = float(np.quantile(all_scores, 0.999))
    lower = PRODUCTION_THRESHOLD
    if upper < lower:
        upper = lower

    best_threshold = lower
    best_metrics = None
    for threshold in np.linspace(lower, upper, 180):
        metrics = binary_metrics(
            (all_scores >= threshold).reshape(1, -1), all_labels.reshape(1, -1)
        )
        rank = (
            metrics["f1"],
            metrics["precision"],
            -metrics["positive_ratio"],
        )
        if best_metrics is None or rank > best_metrics:
            best_metrics = rank
            best_threshold = float(threshold)
    return best_threshold


def hysteresis_prediction(
    score: np.ndarray,
    high: float,
    low: float,
    min_pixels: int,
    max_component_pixels: Optional[int] = None,
    max_total_ratio: Optional[float] = None,
) -> np.ndarray:
    """Keep seeded low-threshold components subject to optional area guards."""
    score_array = np.asarray(score)
    if score_array.ndim != 2:
        raise ValueError("score must be a two-dimensional array")
    if not np.isfinite(score_array).all():
        raise ValueError("score must contain only finite values")
    if not np.isfinite([high, low]).all():
        raise ValueError("thresholds must be finite")
    if high < low:
        raise ValueError("high must be greater than or equal to low")
    if min_pixels < 1:
        raise ValueError("min_pixels must be at least 1")
    if max_component_pixels is not None and max_component_pixels < min_pixels:
        raise ValueError("max_component_pixels must be at least min_pixels")
    if max_total_ratio is not None and not 0.0 <= max_total_ratio <= 1.0:
        raise ValueError("max_total_ratio must be between 0 and 1")

    candidates = score_array >= low
    seeds = score_array >= high
    components, count = ndimage.label(candidates, structure=_EIGHT_CONNECTED)
    accepted = []
    for component_id in range(1, count + 1):
        component = components == component_id
        area = int(component.sum())
        if area < min_pixels or not seeds[component].any():
            continue
        if max_component_pixels is not None and area > max_component_pixels:
            continue
        seed_confidence = float(score_array[np.logical_and(component, seeds)].max())
        accepted.append((seed_confidence, component_id, area, component))

    if max_total_ratio is None:
        retained = accepted
    else:
        pixel_cap = int(np.floor(score_array.size * max_total_ratio))
        retained = []
        retained_area = 0
        for candidate in sorted(accepted, key=lambda item: (-item[0], item[1])):
            if retained_area + candidate[2] <= pixel_cap:
                retained.append(candidate)
                retained_area += candidate[2]

    result = np.zeros_like(candidates, dtype=bool)
    for _, _, _, component in retained:
        result[component] = True
    return result


def binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Return pixel metrics and 8-connected component count."""
    predicted = np.asarray(prediction, dtype=bool)
    expected = np.asarray(reference, dtype=bool)
    if predicted.shape != expected.shape:
        raise ValueError("prediction and reference must have the same shape")
    if predicted.ndim != 2:
        raise ValueError("prediction and reference must be two-dimensional")

    true_positive = int(np.logical_and(predicted, expected).sum())
    false_positive = int(np.logical_and(predicted, ~expected).sum())
    false_negative = int(np.logical_and(~predicted, expected).sum())
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    iou = _safe_ratio(true_positive, true_positive + false_positive + false_negative)
    _, component_count = ndimage.label(predicted, structure=_EIGHT_CONNECTED)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "positive_ratio": float(predicted.mean()),
        "component_count": int(component_count),
    }


def component_statistics(prediction: np.ndarray, score: np.ndarray) -> List[Dict[str, object]]:
    """Describe 8-connected predicted components in raster row-major order."""
    predicted = np.asarray(prediction, dtype=bool)
    score_array = np.asarray(score)
    if predicted.shape != score_array.shape:
        raise ValueError("prediction and score must have the same shape")
    if predicted.ndim != 2:
        raise ValueError("prediction and score must be two-dimensional")
    if not np.isfinite(score_array).all():
        raise ValueError("score must contain only finite values")

    components, count = ndimage.label(predicted, structure=_EIGHT_CONNECTED)
    statistics = []
    for component_id in range(1, count + 1):
        component = components == component_id
        rows, columns = np.nonzero(component)
        values = score_array[component]
        statistics.append({
            "area": int(component.sum()),
            "mean_score": float(values.mean()),
            "max_score": float(values.max()),
            "bbox": [
                int(rows.min()),
                int(columns.min()),
                int(rows.max()) + 1,
                int(columns.max()) + 1,
            ],
        })
    return statistics


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
