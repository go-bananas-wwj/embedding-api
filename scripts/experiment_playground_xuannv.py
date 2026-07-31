"""Reproduce and diagnose the existing Haidian ``playground_xuannv`` head."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.geojson_adapter import rasterize_patch_geometry
from app.services.pu_query import score_pu_query
from scripts.playground_pu_postprocess import (
    binary_metrics,
    component_statistics,
    hysteresis_prediction,
    strict_threshold,
)

MODEL_NAME = "playground_xuannv"
MODEL_ID = "model_756ed870"
REGISTRY_PATH = ROOT / "users/default/models_index.json"
AUDIT_PATH = ROOT / "logs/request_audit.jsonl"
PATCH_METADATA_PATH = ROOT / "data/haidian/patches_meta_v2.json"
OSM_ROOT = ROOT / "data/haidian/labels/osm_playgrounds"
EMBEDDING_ROOT = ROOT / "data/haidian/embeddings/v1"
OPTICAL_ROOT = (
    ROOT
    / "data/haidian/archive/processed_training_data/extracted/patches/highres_optical"
)
TRAINING_PATCH_IDS = ("patch_000059", "patch_000060", "patch_000064")
INDEPENDENT_OSM_PATCH_IDS = ("patch_000076",)
REQUIRED_HIGH_AREA_PATCH_IDS = ("patch_000232", "patch_000249", "patch_000154")
REFERENCE_LIMITATION = (
    "训练 Polygon 和独立 OSM Polygon 都是不完整正类参考；参考范围外像素仅为"
    "未标注像素，不能视为可靠负样本。Precision、F1 和 IoU 仅表示相对于不完整"
    "参考标签的数值，不代表真实全域精度。"
)
_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def load_registered_model(
    registry_path: Path,
    model_name: str,
    model_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve one registry record and load its PyTorch checkpoint."""
    records = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Model registry must be a JSON list")
    matches = [item for item in records if item.get("name") == model_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one model named {model_name!r}, found {len(matches)}"
        )
    record = matches[0]
    if model_id is not None and record.get("id") != model_id:
        raise ValueError(
            f"Model {model_name!r} resolved to {record.get('id')!r}, not {model_id!r}"
        )

    checkpoint_path = Path(record["model_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model_data = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(model_data, dict):
        raise ValueError("Model checkpoint must contain a dictionary")
    return record, model_data


def extract_training_request(audit_path: Path, model_name: str) -> Dict[str, Any]:
    """Return the latest exact POST /models body for ``model_name``."""
    candidates = []
    with Path(audit_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("method") != "POST" or entry.get("path") != "/models":
                continue
            body = entry.get("body", {}).get("content")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    continue
            if isinstance(body, dict) and body.get("name") == model_name:
                candidates.append((str(entry.get("ts", "")), line_number, body))
    if not candidates:
        raise ValueError(f"No POST /models audit request found for {model_name!r}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def load_patch_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    patches = payload.get("patches") if isinstance(payload, dict) else payload
    if not isinstance(patches, list):
        raise ValueError("Patch metadata must contain a patch list")
    return {patch["patch_id"]: patch for patch in patches}


def _spatial_ref(patch: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "bounds_wgs84": patch.get("bounds_wgs84"),
        "bounds": patch.get("bounds"),
        "crs": patch.get("crs"),
    }


def rasterize_training_masks(
    request: Mapping[str, Any],
    patch_metadata_path: Path,
) -> Dict[str, np.ndarray]:
    """Rasterize and merge the audited training polygons per support patch."""
    annotations = request.get("annotations", {})
    features = annotations.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("Training request contains no annotation features")
    patches = load_patch_metadata(patch_metadata_path)
    masks: Dict[str, np.ndarray] = {}
    for feature in features:
        properties = feature.get("properties", {})
        patch_id = properties.get("patch_id")
        if patch_id not in patches:
            raise ValueError(f"Unknown training patch: {patch_id!r}")
        if properties.get("region_id") != request.get("region_id"):
            raise ValueError(f"Training feature region mismatch for {patch_id}")
        mask = rasterize_patch_geometry(
            feature["geometry"],
            _spatial_ref(patches[patch_id]),
            size=(128, 128),
        ).astype(bool)
        masks[patch_id] = np.logical_or(
            masks.get(patch_id, np.zeros((128, 128), dtype=bool)),
            mask,
        )
    if not all(mask.any() for mask in masks.values()):
        raise ValueError("At least one training polygon does not cover any pixels")
    return masks


def load_osm_reference_masks(
    osm_root: Path,
    patch_metadata_path: Path,
) -> Dict[str, np.ndarray]:
    """Rasterize the independent OSM polygons onto their listed patch grids."""
    manifest = json.loads((Path(osm_root) / "manifest.json").read_text(encoding="utf-8"))
    geojson = json.loads(
        (Path(osm_root) / "playgrounds.geojson").read_text(encoding="utf-8")
    )
    feature_by_id = {
        int(feature["properties"]["osm_id"]): feature
        for feature in geojson.get("features", [])
    }
    patches = load_patch_metadata(patch_metadata_path)
    masks: Dict[str, np.ndarray] = {}
    for playground in manifest.get("playgrounds", []):
        feature = feature_by_id[int(playground["osm_id"])]
        for match in playground.get("patches", []):
            patch_id = match["patch_id"]
            mask = rasterize_patch_geometry(
                feature["geometry"],
                _spatial_ref(patches[patch_id]),
                size=(128, 128),
            ).astype(bool)
            masks[patch_id] = np.logical_or(
                masks.get(patch_id, np.zeros((128, 128), dtype=bool)),
                mask,
            )
    if not masks:
        raise ValueError("Independent OSM reference contains no rasterized polygons")
    return masks


def score_with_and_without_query(
    feature: np.ndarray,
    model_data: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Call production scoring for its adapted and unadapted score maps."""
    query_score, query_used = score_pu_query(feature, dict(model_data))
    no_query_model = dict(model_data)
    no_query_model["threshold"] = float(np.finfo(np.float32).max / 4.0)
    no_query_score, no_query_used = score_pu_query(feature, no_query_model)
    if no_query_used:
        raise RuntimeError("No-query scoring unexpectedly applied query adaptation")
    return (
        np.asarray(query_score, dtype=np.float32),
        np.asarray(no_query_score, dtype=np.float32),
        bool(query_used),
    )


def _combined_metrics(
    predictions: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
) -> Dict[str, float]:
    prediction = np.concatenate([item.reshape(-1) for item in predictions])
    reference = np.concatenate([item.reshape(-1) for item in references])
    metrics = binary_metrics(prediction.reshape(1, -1), reference.reshape(1, -1))
    metrics["component_count"] = int(
        sum(
            ndimage.label(np.asarray(item, dtype=bool), structure=_EIGHT_CONNECTED)[1]
            for item in predictions
        )
    )
    return metrics


def _guard_grid(
    scores: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    production_threshold: float,
    strict: float,
) -> Iterable[Dict[str, Any]]:
    positive_values = np.concatenate(
        [score[reference] for score, reference in zip(scores, references)]
    )
    high_values = {
        float(strict),
        *[
            float(np.quantile(positive_values, quantile))
            for quantile in (0.45, 0.60, 0.75, 0.85)
        ],
    }
    for high in sorted(value for value in high_values if np.isfinite(value)):
        high = max(float(production_threshold), high)
        low_values = {
            float(production_threshold),
            high,
            float(production_threshold + 0.35 * (high - production_threshold)),
            float(production_threshold + 0.65 * (high - production_threshold)),
        }
        for low in sorted(value for value in low_values if value <= high):
            for min_pixels in (4, 8, 16):
                for max_component_pixels in (128, 256, 512):
                    for max_total_ratio in (0.03, 0.05, 0.08, 0.12):
                        yield {
                            "high": high,
                            "low": low,
                            "min_pixels": min_pixels,
                            "max_component_pixels": max_component_pixels,
                            "max_total_ratio": max_total_ratio,
                        }


def calibrate_postprocessing(
    scores_by_patch: Mapping[str, np.ndarray],
    references_by_patch: Mapping[str, np.ndarray],
    calibration_patch_ids: Sequence[str],
    excluded_patch_ids: Sequence[str],
    production_threshold: float,
) -> Dict[str, Any]:
    """Freeze strict and guarded parameters using calibration patches only."""
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
        if patch_id not in scores_by_patch or patch_id not in references_by_patch:
            raise ValueError(f"Missing calibration score/reference for {patch_id}")

    scores = [np.asarray(scores_by_patch[patch_id]) for patch_id in calibration_ids]
    references = [
        np.asarray(references_by_patch[patch_id], dtype=bool)
        for patch_id in calibration_ids
    ]
    strict = strict_threshold(scores, references)
    baseline_metrics = _combined_metrics(
        [score >= production_threshold for score in scores],
        references,
    )
    recall_floor = max(0.55, min(0.85, baseline_metrics["recall"] * 0.85))
    best_parameters = None
    best_metrics = None
    best_rank = None
    fallback_rank = None
    fallback_parameters = None
    fallback_metrics = None
    for parameters in _guard_grid(
        scores,
        references,
        production_threshold,
        strict,
    ):
        predictions = [
            hysteresis_prediction(score, **parameters) for score in scores
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
            fallback_parameters = parameters
            fallback_metrics = metrics
        if metrics["recall"] < recall_floor:
            continue
        rank = (
            metrics["precision"],
            metrics["f1"],
            -metrics["positive_ratio"],
            metrics["recall"],
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_parameters = parameters
            best_metrics = metrics
    if best_parameters is None:
        best_parameters = fallback_parameters
        best_metrics = fallback_metrics
    if best_parameters is None or best_metrics is None:
        raise RuntimeError("Unable to calibrate guarded postprocessing")

    return {
        "calibration_patch_ids": calibration_ids,
        "excluded_patch_ids": excluded_ids,
        "production_threshold": float(production_threshold),
        "strict_threshold": float(strict),
        "guarded": best_parameters,
        "selection": {
            "recall_floor": float(recall_floor),
            "objective": (
                "在训练 Polygon 不完整参考下，先满足召回下限，再依次最大化"
                "参考相对 Precision、F1，并压低预测面积。"
            ),
            "reference_limitation": REFERENCE_LIMITATION,
        },
        "calibration_metrics": {
            "baseline": baseline_metrics,
            "strict": _combined_metrics(
                [score >= strict for score in scores],
                references,
            ),
            "guarded": best_metrics,
        },
    }


def predict_variants(
    score: np.ndarray,
    production_threshold: float,
    parameters: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Return baseline, strict, and area-guarded boolean predictions."""
    score_array = np.asarray(score)
    return {
        "baseline": np.asarray(
            score_array >= float(production_threshold), dtype=bool
        ),
        "strict": np.asarray(
            score_array >= float(parameters["strict_threshold"]), dtype=bool
        ),
        "guarded": hysteresis_prediction(
            score_array,
            **dict(parameters["guarded"]),
        ),
    }


def distribution(values: np.ndarray) -> Dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    quantiles = np.quantile(finite, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(finite.max()),
    }


def _aggregate_domain(
    score_by_patch: Mapping[str, np.ndarray],
    production_threshold: float,
    parameters: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    per_patch: Dict[str, Dict[str, Any]] = {}
    ratios: Dict[str, List[float]] = {
        "baseline": [],
        "strict": [],
        "guarded": [],
    }
    component_areas: Dict[str, List[int]] = {
        "baseline": [],
        "strict": [],
        "guarded": [],
    }
    component_counts: Dict[str, int] = {
        "baseline": 0,
        "strict": 0,
        "guarded": 0,
    }
    nonempty: Dict[str, int] = {
        "baseline": 0,
        "strict": 0,
        "guarded": 0,
    }
    for patch_id, score in score_by_patch.items():
        variants = predict_variants(score, production_threshold, parameters)
        patch_metrics: Dict[str, Any] = {}
        for name, prediction in variants.items():
            components = component_statistics(prediction, score)
            ratio = float(prediction.mean())
            ratios[name].append(ratio)
            component_areas[name].extend(item["area"] for item in components)
            component_counts[name] += len(components)
            nonempty[name] += int(prediction.any())
            patch_metrics[name] = {
                "positive_pixels": int(prediction.sum()),
                "positive_ratio": ratio,
                "component_count": len(components),
                "largest_component_pixels": max(
                    [item["area"] for item in components],
                    default=0,
                ),
            }
        per_patch[patch_id] = patch_metrics

    aggregate = {}
    for name, values in ratios.items():
        ratios_array = np.asarray(values, dtype=np.float64)
        aggregate[name] = {
            "nonempty_patch_count": int(nonempty[name]),
            "patch_count": int(len(values)),
            "mean_positive_ratio": float(ratios_array.mean()),
            "median_positive_ratio": float(np.median(ratios_array)),
            "p95_positive_ratio": float(np.quantile(ratios_array, 0.95)),
            "component_count": int(component_counts[name]),
            "component_area_pixels": distribution(
                np.asarray(component_areas[name], dtype=np.float64)
            ),
            "top_area_patches": sorted(
                [
                    {
                        "patch_id": patch_id,
                        "positive_ratio": metrics[name]["positive_ratio"],
                    }
                    for patch_id, metrics in per_patch.items()
                ],
                key=lambda item: (-item["positive_ratio"], item["patch_id"]),
            )[:20],
        }
    return aggregate, per_patch


def _reference_metrics(
    score_by_patch: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
    production_threshold: float,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    scores = [score_by_patch[patch_id] for patch_id in references]
    labels = [references[patch_id] for patch_id in references]
    variants = [
        predict_variants(score, production_threshold, parameters)
        for score in scores
    ]
    return {
        name: _combined_metrics(
            [item[name] for item in variants],
            labels,
        )
        for name in ("baseline", "strict", "guarded")
    }


def _edge_ring(mask: np.ndarray) -> np.ndarray:
    expanded = ndimage.binary_dilation(mask, structure=_EIGHT_CONNECTED, iterations=2)
    contracted = ndimage.binary_erosion(mask, structure=_EIGHT_CONNECTED, iterations=1)
    return np.logical_and(expanded, ~contracted)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == int(np.argmax(sizes) + 1)


def _score_groups(
    query_scores: Mapping[str, np.ndarray],
    no_query_scores: Mapping[str, np.ndarray],
    training_masks: Mapping[str, np.ndarray],
    osm_masks: Mapping[str, np.ndarray],
    production_threshold: float,
    top_unlabeled_patch_ids: Sequence[str],
    random_patch_ids: Sequence[str],
) -> Dict[str, Any]:
    rng = np.random.default_rng(20260731)

    def collect(
        score_maps: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        return np.concatenate(
            [score_maps[patch_id][mask] for patch_id, mask in masks.items()]
        )

    training_ring = {
        patch_id: _edge_ring(mask) for patch_id, mask in training_masks.items()
    }
    osm_ring = {patch_id: _edge_ring(mask) for patch_id, mask in osm_masks.items()}
    high_area_values = []
    for patch_id in top_unlabeled_patch_ids:
        baseline = query_scores[patch_id] >= production_threshold
        component = _largest_component(baseline)
        if component.any():
            high_area_values.append(query_scores[patch_id][component])
    random_values = []
    for patch_id in random_patch_ids:
        flat = query_scores[patch_id].reshape(-1)
        sample_size = min(256, len(flat))
        indices = rng.choice(len(flat), size=sample_size, replace=False)
        random_values.append(flat[indices])

    groups = {
        "training_polygon": distribution(collect(query_scores, training_masks)),
        "independent_osm_polygon": distribution(collect(query_scores, osm_masks)),
        "training_polygon_edge_ring": distribution(
            collect(query_scores, training_ring)
        ),
        "independent_osm_edge_ring": distribution(
            collect(query_scores, osm_ring)
        ),
        "baseline_high_area_unlabeled_components": distribution(
            np.concatenate(high_area_values)
            if high_area_values
            else np.asarray([], dtype=np.float32)
        ),
        "fixed_random_unlabeled_pixels": distribution(
            np.concatenate(random_values)
            if random_values
            else np.asarray([], dtype=np.float32)
        ),
        "query_minus_no_query_all_pixels": distribution(
            np.concatenate(
                [
                    (query_scores[patch_id] - no_query_scores[patch_id]).reshape(-1)
                    for patch_id in sorted(query_scores)
                ]
            )
        ),
    }
    return {
        "reference_policy": {
            "limitation": REFERENCE_LIMITATION,
            "unlabeled_groups_are_reliable_negatives": False,
            "note": (
                "名称中的 unlabeled 只表示没有参考标签；高面积连通域是误检候选，"
                "固定随机像素是分布对照，二者都不是负类真值。"
            ),
        },
        "groups": groups,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optical_path(patch_id: str, month: str) -> Optional[Path]:
    candidates = sorted(OPTICAL_ROOT.glob(f"highres_optical_{month}01_{patch_id}.tif"))
    return candidates[0] if candidates else None


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write UTF-8 JSON while rejecting NaN and Infinity."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text + "\n", encoding="utf-8")


def run_experiment(month: str, output: Path) -> Dict[str, Any]:
    """Run the frozen-head experiment over every embedding in ``month``."""
    record, model_data = load_registered_model(REGISTRY_PATH, MODEL_NAME, MODEL_ID)
    request = extract_training_request(AUDIT_PATH, MODEL_NAME)
    training_masks = rasterize_training_masks(request, PATCH_METADATA_PATH)
    osm_masks = load_osm_reference_masks(OSM_ROOT, PATCH_METADATA_PATH)
    if sorted(training_masks) != sorted(TRAINING_PATCH_IDS):
        raise ValueError("Audited training patches differ from the frozen experiment")
    if set(osm_masks) & set(training_masks):
        raise ValueError("Independent OSM reference overlaps the calibration patches")

    embedding_dir = EMBEDDING_ROOT / month
    embedding_paths = sorted(embedding_dir.glob("patch_*.npy"))
    if len(embedding_paths) != 320:
        raise ValueError(
            f"Expected 320 Haidian embeddings for {month}, found {len(embedding_paths)}"
        )

    query_scores: Dict[str, np.ndarray] = {}
    no_query_scores: Dict[str, np.ndarray] = {}
    query_used: Dict[str, bool] = {}
    for embedding_path in embedding_paths:
        feature = np.load(embedding_path, allow_pickle=False)
        query, no_query, used = score_with_and_without_query(feature, model_data)
        if query.shape != (128, 128) or no_query.shape != (128, 128):
            raise ValueError(f"Unexpected score shape for {embedding_path.name}")
        patch_id = embedding_path.stem
        query_scores[patch_id] = query
        no_query_scores[patch_id] = no_query
        query_used[patch_id] = used

    production_threshold = float(model_data["threshold"])
    parameters = calibrate_postprocessing(
        query_scores,
        training_masks,
        calibration_patch_ids=TRAINING_PATCH_IDS,
        excluded_patch_ids=tuple(osm_masks),
        production_threshold=production_threshold,
    )
    query_aggregate, query_per_patch = _aggregate_domain(
        query_scores,
        production_threshold,
        parameters,
    )
    no_query_aggregate, no_query_per_patch = _aggregate_domain(
        no_query_scores,
        production_threshold,
        parameters,
    )

    excluded = set(training_masks) | set(osm_masks)
    top_unlabeled = [
        item["patch_id"]
        for item in query_aggregate["baseline"]["top_area_patches"]
        if item["patch_id"] not in excluded
    ][:8]
    random_population = sorted(
        set(query_scores)
        - excluded
        - set(REQUIRED_HIGH_AREA_PATCH_IDS)
        - set(top_unlabeled)
    )
    random_patch_ids = sorted(
        random.Random(20260731).sample(random_population, 8)
    )
    score_groups = _score_groups(
        query_scores,
        no_query_scores,
        training_masks,
        osm_masks,
        production_threshold,
        top_unlabeled,
        random_patch_ids,
    )

    reference_policy = {
        "limitation": REFERENCE_LIMITATION,
        "unlabeled_pixels_are_reliable_negatives": False,
        "metric_interpretation": (
            "所有 Precision、F1、IoU 都是参考相对指标；Recall 也只衡量已知"
            "Polygon 的覆盖。独立 OSM 只用于冻结参数后的验证。"
        ),
    }
    metrics = {
        "reference_policy": reference_policy,
        "parameters": parameters,
        "domain": {
            "query_adapted_scores": query_aggregate,
            "no_query_scores": no_query_aggregate,
        },
        "reference_relative_metrics": {
            "calibration_training_polygons": _reference_metrics(
                query_scores,
                training_masks,
                production_threshold,
                parameters,
            ),
            "independent_osm_polygon": _reference_metrics(
                query_scores,
                osm_masks,
                production_threshold,
                parameters,
            ),
        },
        "query_adaptation": {
            "used_patch_count": int(sum(query_used.values())),
            "patch_count": len(query_used),
            "used_patch_ids": [
                patch_id for patch_id in sorted(query_used) if query_used[patch_id]
            ],
            "all_pixel_delta": score_groups["groups"][
                "query_minus_no_query_all_pixels"
            ],
            "baseline_positive_ratio_delta": distribution(
                np.asarray(
                    [
                        query_per_patch[patch_id]["baseline"]["positive_ratio"]
                        - no_query_per_patch[patch_id]["baseline"]["positive_ratio"]
                        for patch_id in sorted(query_scores)
                    ],
                    dtype=np.float64,
                )
            ),
        },
        "per_patch": {
            patch_id: {
                "query_used": query_used[patch_id],
                "query": query_per_patch[patch_id],
                "no_query": no_query_per_patch[patch_id],
            }
            for patch_id in sorted(query_scores)
        },
    }

    selected_patch_ids = sorted(
        set(TRAINING_PATCH_IDS)
        | set(osm_masks)
        | set(REQUIRED_HIGH_AREA_PATCH_IDS)
        | set(top_unlabeled[:5])
        | set(random_patch_ids)
    )
    arrays_dir = Path(output) / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    array_artifacts: Dict[str, Dict[str, Any]] = {}
    for patch_id in selected_patch_ids:
        variants = predict_variants(
            query_scores[patch_id],
            production_threshold,
            parameters,
        )
        reference = np.logical_or(
            training_masks.get(patch_id, np.zeros((128, 128), dtype=bool)),
            osm_masks.get(patch_id, np.zeros((128, 128), dtype=bool)),
        )
        values = {
            "score_query": query_scores[patch_id].astype(np.float32),
            "score_no_query": no_query_scores[patch_id].astype(np.float32),
            "baseline": variants["baseline"].astype(np.uint8),
            "strict": variants["strict"].astype(np.uint8),
            "guarded": variants["guarded"].astype(np.uint8),
            "reference": reference.astype(np.uint8),
        }
        artifact_paths = {}
        for name, value in values.items():
            relative_path = Path("arrays") / f"{patch_id}_{name}.npy"
            np.save(Path(output) / relative_path, value, allow_pickle=False)
            artifact_paths[name] = relative_path.as_posix()
        optical = _optical_path(patch_id, month)
        array_artifacts[patch_id] = {
            "role": (
                "training"
                if patch_id in training_masks
                else "independent_osm"
                if patch_id in osm_masks
                else "high_area_unlabeled_candidate"
                if patch_id
                in set(REQUIRED_HIGH_AREA_PATCH_IDS) | set(top_unlabeled)
                else "fixed_random_unlabeled"
            ),
            "arrays": artifact_paths,
            "optical_path": str(optical.relative_to(ROOT)) if optical else None,
        }

    training_geojson = request["annotations"]
    write_strict_json(Path(output) / "training_annotations.geojson", training_geojson)
    checkpoint_path = ROOT / record["model_path"]
    manifest = {
        "experiment": {
            "name": "playground_xuannv Haidian false-positive diagnosis",
            "month": month,
            "patch_count": len(embedding_paths),
            "production_api_modified": False,
            "checkpoint_retrained": False,
        },
        "model": {
            "id": record["id"],
            "name": record["name"],
            "head_type": record.get("head_type"),
            "checkpoint_format": record.get("checkpoint_format"),
            "foundation_model_id": record.get("foundation_model_id"),
            "foundation_model_version": record.get("foundation_model_version"),
            "feature_dimension": record.get("feature_dimension"),
            "checkpoint_path": record["model_path"],
            "checkpoint_sha256": _sha256(checkpoint_path),
            "production_threshold": production_threshold,
        },
        "reference_policy": reference_policy,
        "calibration": {
            "patch_ids": list(TRAINING_PATCH_IDS),
            "source": "logs/request_audit.jsonl exact POST /models body",
            "annotations": "training_annotations.geojson",
        },
        "independent_evaluation": {
            "patch_ids": sorted(osm_masks),
            "source": "OpenStreetMap athletics playground reference",
            "used_for_parameter_selection": False,
            "manifest": str((OSM_ROOT / "manifest.json").relative_to(ROOT)),
        },
        "selection": {
            "required_high_area_patch_ids": list(REQUIRED_HIGH_AREA_PATCH_IDS),
            "top_baseline_area_unlabeled_patch_ids": top_unlabeled,
            "fixed_random_unlabeled_patch_ids": random_patch_ids,
        },
        "artifacts": {
            "metrics": "metrics.json",
            "score_groups": "score_groups.json",
            "arrays": array_artifacts,
        },
    }
    write_strict_json(Path(output) / "experiment_manifest.json", manifest)
    write_strict_json(Path(output) / "metrics.json", metrics)
    write_strict_json(Path(output) / "score_groups.json", score_groups)
    return {
        "manifest": manifest,
        "metrics": metrics,
        "score_groups": score_groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="202604")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Tmp/playground_pu_query_20260731",
    )
    args = parser.parse_args()
    result = run_experiment(args.month, args.output)
    domain = result["metrics"]["domain"]["query_adapted_scores"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "patch_count": result["manifest"]["experiment"]["patch_count"],
                "query_used_patch_count": result["metrics"]["query_adaptation"][
                    "used_patch_count"
                ],
                "baseline": domain["baseline"],
                "strict": domain["strict"],
                "guarded": domain["guarded"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
