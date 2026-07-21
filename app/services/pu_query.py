"""Positive-unlabeled prototype retrieval for very sparse custom training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


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
    return {
        "feature_mean": mean,
        "feature_std": std,
        "foreground_center": foreground,
        "background_center": background,
        "threshold": threshold,
        "training_f05": f05,
    }


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
