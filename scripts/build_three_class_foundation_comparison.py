#!/usr/bin/env python3
"""Build same-patch sparse-label comparisons across four feature methods."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import experiment_no_positive_memory as experiment
import experiment_prototype_similarity as base
from experiment_pu_query_label_counts import sparse_polygons_from_one_patch
from experiment_sparse_retrieval_vs_conv import ROOT

from app.services.external_embeddings import load_aef_embedding, load_dino_embedding
from app.services.s2_ml import load_s2_features


OUTPUT = ROOT / "Tmp/pu_query_aef_traditional_single_patch_20260726"
MONTH = "202604"
CASES = {
    "building_extraction": {"name": "建筑", "patch_id": "patch_000205", "count": 3},
    "road_extraction": {"name": "道路", "patch_id": "patch_000264", "count": 2},
    "water": {"name": "水体", "patch_id": "patch_000106", "count": 1},
}


def resize_feature(feature: np.ndarray, size: int = 128) -> np.ndarray:
    if feature.shape[-2:] == (size, size):
        return feature.astype(np.float32, copy=False)
    zoom = (1, size / feature.shape[-2], size / feature.shape[-1])
    return ndimage.zoom(feature, zoom, order=1).astype(np.float32)


def sparse_mask(polygons) -> np.ndarray:
    value = np.zeros((128, 128), dtype=bool)
    for _, mask, _ in polygons:
        value |= mask
    return value


def standardized_pixels(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.nan_to_num(feature.reshape(feature.shape[0], -1).T)
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), 1e-5)
    return (flat - mean) / std, np.isfinite(feature).all(axis=0).reshape(-1)


def pseudo_training(feature: np.ndarray, positive_mask: np.ndarray, seed: int):
    x, valid = standardized_pixels(feature)
    positive_index = np.flatnonzero(positive_mask.reshape(-1) & valid)
    if not len(positive_index):
        raise ValueError("Sparse annotation contains no valid feature pixels")
    positive = x[positive_index]
    center = positive.mean(axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-6)
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
    score = normalized @ center
    excluded = ndimage.binary_dilation(positive_mask, iterations=6).reshape(-1)
    candidates = np.flatnonzero(valid & ~excluded)
    negative_count = min(max(len(positive_index) * 4, 256), 3000, len(candidates))
    negative_index = candidates[np.argsort(score[candidates])[:negative_count]]
    rng = np.random.default_rng(seed)
    positive_index = rng.choice(
        positive_index, size=min(max(len(positive_index), 64), 1200), replace=len(positive_index) < 64
    )
    train_index = np.concatenate([positive_index, negative_index])
    labels = np.concatenate([
        np.ones(len(positive_index), dtype=np.uint8),
        np.zeros(len(negative_index), dtype=np.uint8),
    ])
    order = rng.permutation(len(train_index))
    return x, train_index[order], labels[order], valid


def train_predict(feature: np.ndarray, positive_mask: np.ndarray, method: str, seed: int):
    x, train_index, labels, valid = pseudo_training(feature, positive_mask, seed)
    if method == "traditional":
        model = RandomForestClassifier(
            n_estimators=180, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", alpha=1e-3,
            max_iter=220, early_stopping=True, random_state=seed,
        )
    model.fit(x[train_index], labels)
    probability = np.zeros(len(x), dtype=np.float32)
    probability[valid] = model.predict_proba(x[valid])[:, 1]
    return probability.reshape(128, 128) >= 0.5


def metrics(prediction: np.ndarray, label: np.ndarray) -> dict:
    tp = int(np.logical_and(prediction, label).sum())
    fp = int(np.logical_and(prediction, ~label).sum())
    fn = int(np.logical_and(~prediction, label).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {"f1": f1, "iou": iou}


def save_rgb(name: str, value: np.ndarray) -> None:
    Image.fromarray(value.astype(np.uint8)).save(OUTPUT / name)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    value = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    value[mask] = (235, 45, 50)
    return value


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels = OUTPUT / "_same_patch_evaluation_labels"
    experiment.prepare_all_labels(labels)
    base.TASK_ROOT = labels
    global_mean, global_std = experiment.global_stats()
    results = {}

    for task_index, (task, case) in enumerate(CASES.items()):
        patch_id = case["patch_id"]
        polygons = sparse_polygons_from_one_patch(task, [patch_id], count=case["count"])
        visible, _ = experiment.model_visible_label_view(polygons)
        positive_mask = sparse_mask(polygons)
        embedding, label, pca = base.load_pair(task, patch_id)
        positive = experiment.polygon_center(polygons, task, global_mean, global_std)
        negative, positive_pixels, negative_pixels = experiment.reliable_background_center(
            polygons, task, positive, global_mean, global_std
        )
        positive_scores = positive_pixels @ positive - 0.65 * (positive_pixels @ negative)
        negative_scores = negative_pixels @ positive - 0.65 * (negative_pixels @ negative)
        threshold = experiment.tune_threshold(positive_scores, negative_scores)
        xuannv_score = experiment.contrast_score(
            embedding, positive, negative, global_mean, global_std, True, threshold
        )
        predictions = {
            "xuannv": xuannv_score >= threshold,
            "aef": train_predict(
                resize_feature(load_aef_embedding("haidian", patch_id, MONTH)),
                positive_mask, "mlp", 100 + task_index,
            ),
            "traditional": train_predict(
                resize_feature(load_s2_features("haidian", patch_id, MONTH)[0]),
                positive_mask, "traditional", 200 + task_index,
            ),
            "dino": train_predict(
                resize_feature(load_dino_embedding("haidian", patch_id, MONTH)),
                positive_mask, "mlp", 300 + task_index,
            ),
        }
        optical = experiment.true_color_composite(patch_id)
        save_rgb(f"{task}_optical.png", optical)
        save_rgb(f"{task}_pca.png", pca)
        save_rgb(f"{task}_visible.png", visible)
        item = {
            "name": case["name"],
            "patch_id": patch_id,
            "polygon_count": case["count"],
            "methods": {},
        }
        for method, prediction in predictions.items():
            save_rgb(f"{task}_{method}.png", mask_rgb(prediction))
            overlay = optical.copy()
            overlay[prediction] = (
                0.45 * overlay[prediction] + 0.55 * np.array([235, 45, 50])
            ).astype(np.uint8)
            save_rgb(f"{task}_{method}_overlay.png", overlay)
            item["methods"][method] = metrics(prediction, label)
        results[task] = item
        print(task, json.dumps(item["methods"], ensure_ascii=False), flush=True)

    (OUTPUT / "same_patch_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
