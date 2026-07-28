"""Sentinel-2 feature loading and Random Forest training helpers."""

import re
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from app.config import get_config
from app.services.model_registry import get_model_registry
from app.services.model_binding import build_model_binding

CHECKPOINT_FORMAT = "s2_random_forest_v1"
CANONICAL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def _date_prefix(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) < 6:
        raise ValueError("Sentinel-2 month must use YYYY-MM, YYYYMM, or YYYYMMDD")
    return digits[:8] if len(digits) >= 8 else digits[:6]


def resolve_s2_path(region_id: str, patch_id: str, month: str) -> Path:
    region = get_config().get_region(region_id) or {}
    root_value = region.get("s2_dir")
    if not root_value:
        raise FileNotFoundError(f"Sentinel-2 source is not configured for region '{region_id}'")
    root = Path(root_value)
    prefix = _date_prefix(month)
    candidates: List[Path] = []
    for directory in (root / patch_id, root / patch_id / "s2", root):
        if not directory.is_dir():
            continue
        candidates.extend(directory.glob(f"{prefix}*.tif"))
        candidates.extend(directory.glob(f"s2_{prefix}*_{patch_id}.tif"))
    candidates = [p for p in candidates if not p.stem.endswith("_mask")]
    if not candidates:
        raise FileNotFoundError(
            f"No Sentinel-2 image found for {region_id}/{patch_id}/{month}"
        )
    # Product rule: when a month has multiple scenes, use the latest scene.
    return max(candidates, key=lambda p: (re.findall(r"\d{8}", p.stem) or [p.stem], p.name))


def _band_indexes(descriptions: Tuple[Optional[str], ...], count: int) -> Dict[str, int]:
    by_name = {str(name).upper(): i for i, name in enumerate(descriptions) if name}
    if all(name in by_name for name in CANONICAL_BANDS):
        return {name: by_name[name] for name in CANONICAL_BANDS}
    if count == 6:
        return {name: i for i, name in enumerate(CANONICAL_BANDS)}
    raise ValueError(
        "Sentinel-2 band metadata is incomplete; expected named B02/B03/B04/B08/B11/B12 or the documented 6-band layout"
    )


def load_s2_features(region_id: str, patch_id: str, month: str) -> Tuple[np.ndarray, np.ndarray, str]:
    path = resolve_s2_path(region_id, patch_id, month)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32, copy=False)
        indexes = _band_indexes(src.descriptions, src.count)
        bands = np.stack([data[indexes[name]] for name in CANONICAL_BANDS])
        valid = np.all(np.isfinite(bands), axis=0)
        if src.nodata is not None:
            valid &= np.all(bands != float(src.nodata), axis=0)
        valid &= np.any(bands != 0, axis=0)

    # Harmonize integer reflectance (0..10000) and already-scaled reflectance.
    finite = bands[:, valid]
    if finite.size and float(np.nanpercentile(np.abs(finite), 99)) > 2.0:
        bands = bands / 10000.0
    b02, b03, b04, b08, b11, b12 = bands
    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    mndwi = (b03 - b11) / (b03 + b11 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)
    features = np.concatenate([bands, np.stack([ndvi, ndwi, mndwi, ndbi])], axis=0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32, copy=False), valid, str(path)


def train_random_forest(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[RandomForestClassifier, float, Dict[str, float]]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    rng = np.random.default_rng(42)
    for feature, positive_mask, valid_mask in samples:
        flat = feature.reshape(feature.shape[0], -1).T
        pos = (positive_mask & valid_mask).reshape(-1)
        unlabeled = ((~positive_mask) & valid_mask).reshape(-1)
        pos_idx = np.flatnonzero(pos)
        unlabeled_idx = np.flatnonzero(unlabeled)
        if not len(pos_idx) or not len(unlabeled_idx):
            continue
        # Outside polygons is unlabeled, not ground-truth background. Select only
        # spectrally distant pixels as conservative weak negatives.
        proto = np.median(flat[pos_idx], axis=0)
        scale = np.median(np.abs(flat[pos_idx] - proto), axis=0) + 1e-4
        distance = np.mean(np.abs((flat[unlabeled_idx] - proto) / scale), axis=1)
        negative_pool = unlabeled_idx[np.argsort(distance)[::-1][: max(64, len(pos_idx) * 3)]]
        per_class = min(4000, len(pos_idx), len(negative_pool))
        if per_class < 4:
            continue
        p = rng.choice(pos_idx, per_class, replace=False)
        n = rng.choice(negative_pool, per_class, replace=False)
        xs.extend([flat[p], flat[n]])
        ys.extend([np.ones(per_class, dtype=np.uint8), np.zeros(per_class, dtype=np.uint8)])
    if not xs:
        raise ValueError("No valid Sentinel-2 positive/weak-negative samples were found")
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=min(4, os.cpu_count() or 1),
        random_state=42,
        oob_score=True,
    )
    model.fit(x, y)
    probability = model.predict_proba(x)[:, 1]
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in np.linspace(0.2, 0.8, 13):
        score = f1_score(y, probability >= threshold)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return model, best_threshold, {
        "training_f1": best_f1,
        "oob_score": float(model.oob_score_),
        "labeled_pixel_count": float(len(y)),
    }


def save_random_forest_checkpoint(
    user_id: str,
    model_id: str,
    model: RandomForestClassifier,
    metadata: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> Path:
    record = get_model_registry(user_id).get_model(model_id)
    if record is None:
        raise ValueError(f"Model {model_id} not found in registry")
    path = checkpoint_path or Path(record["model_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    try:
        checkpoint = {"__format__": CHECKPOINT_FORMAT, "model": model, **metadata}
        checkpoint.update(build_model_binding(checkpoint))
        joblib.dump(checkpoint, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
