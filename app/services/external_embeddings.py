"""Frozen AEF and DINOv3-SAT493M feature loading."""

import os
import re
import threading
import uuid
import fcntl
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from app.services.s2_ml import resolve_s2_path

DINO_MODEL_DIR = Path(os.environ.get("DINOV3_SAT493M_MODEL_DIR", "models/dinov3_sat493m"))
DINO_CACHE_DIR = Path(os.environ.get("DINOV3_SAT493M_CACHE_DIR", "data/feature_cache/dinov3_sat493m"))
AEF_EMBEDDING_DIR = Path(os.environ.get("AEF_EMBEDDING_DIR", "data/external_embeddings/aef"))
EXTERNAL_MLP_CHECKPOINT_FORMAT = "external_embedding_mlp_v1"

_dino_model = None
_dino_lock = threading.Lock()


def aef_assets_available() -> bool:
    return AEF_EMBEDDING_DIR.is_dir() and any(AEF_EMBEDDING_DIR.rglob("*.npy"))


def aef_assets_available_for_region(region_id: str) -> bool:
    root = AEF_EMBEDDING_DIR / region_id
    return root.is_dir() and any(root.rglob("*.npy"))


def dino_assets_available() -> bool:
    return (DINO_MODEL_DIR / "model.safetensors").is_file() and (
        DINO_MODEL_DIR / "config.json"
    ).is_file()


def _period_keys(month: str):
    digits = re.sub(r"\D", "", month or "")
    return [month, digits, digits[:6]]


def load_aef_embedding(region_id: str, patch_id: str, month: str) -> np.ndarray:
    keys = _period_keys(month)
    digits = re.sub(r"\D", "", month or "")
    if len(digits) >= 4:
        keys.append(digits[:4])
    for key in dict.fromkeys(keys):
        candidates = (
            AEF_EMBEDDING_DIR / region_id / key / f"{patch_id}.npy",
            AEF_EMBEDDING_DIR / region_id / f"{patch_id}_{key}.npy",
        )
        for path in candidates:
            if path.is_file():
                value = np.load(path)
                _validate_embedding(value, "AEF")
                return value.astype(np.float32, copy=False)
    raise FileNotFoundError(
        f"No AEF embedding found for {region_id}/{patch_id}/{month}; configure AEF_EMBEDDING_DIR"
    )


def _validate_embedding(value: np.ndarray, source: str, expected_channels: Optional[int] = None) -> None:
    if value.ndim != 3 or any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{source} embedding must be non-empty [C,H,W], got {value.shape}")
    if expected_channels is not None and value.shape[0] != expected_channels:
        raise ValueError(
            f"{source} embedding channel mismatch: expected {expected_channels}, got {value.shape[0]}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{source} embedding contains NaN or infinite values")


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _dino_cache_version() -> str:
    config = DINO_MODEL_DIR / "config.json"
    weights = DINO_MODEL_DIR / "model.safetensors"
    material = f"token14-v2:{config.stat().st_size}:{config.stat().st_mtime_ns}:{weights.stat().st_size}:{weights.stat().st_mtime_ns}"
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def _load_dino_model():
    global _dino_model
    if _dino_model is not None:
        return _dino_model
    with _dino_lock:
        if _dino_model is not None:
            return _dino_model
        if not dino_assets_available():
            raise FileNotFoundError(
                "DINOv3-SAT493M weights are not installed; configure DINOV3_SAT493M_MODEL_DIR"
            )
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(str(DINO_MODEL_DIR), local_files_only=True)
        model = AutoModel.from_config(config)
        state = load_file(str(DINO_MODEL_DIR / "model.safetensors"))
        state = {key.removeprefix("model."): value for key, value in state.items()}
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "DINOv3 checkpoint does not match configured architecture: "
                f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dino_model = model.eval().to(device)
        return _dino_model


def _read_s2_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32, copy=False)
        names = {str(name).upper(): i for i, name in enumerate(src.descriptions) if name}
        if all(name in names for name in ("B02", "B03", "B04")):
            rgb = np.stack([data[names["B04"]], data[names["B03"]], data[names["B02"]]])
        elif src.count >= 3:
            # Documented Harbin 6-band layout: B02,B03,B04,B08,B11,B12.
            rgb = data[[2, 1, 0]]
        else:
            raise ValueError("DINOv3 requires a Sentinel-2 image with RGB bands")
    finite = rgb[np.isfinite(rgb)]
    if finite.size and float(np.percentile(np.abs(finite), 99)) > 2.0:
        rgb = rgb / 10000.0
    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def load_dino_embedding(region_id: str, patch_id: str, month: str) -> np.ndarray:
    if not dino_assets_available():
        raise FileNotFoundError("DINOv3-SAT493M weights are not installed")
    cache_path = DINO_CACHE_DIR / _dino_cache_version() / region_id / re.sub(r"\D", "", month) / f"{patch_id}.npy"
    lock_path = cache_path.with_suffix(".lock")
    with _file_lock(lock_path):
        if cache_path.is_file():
            try:
                value = np.load(cache_path).astype(np.float32, copy=False)
                _validate_embedding(value, "DINOv3", expected_channels=1024)
                return value
            except (OSError, ValueError):
                cache_path.unlink(missing_ok=True)
        source = resolve_s2_path(region_id, patch_id, month)
        rgb = _read_s2_rgb(source)
        image = torch.from_numpy(rgb).unsqueeze(0)
        image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.43, 0.411, 0.296]).view(1, 3, 1, 1)
        std = torch.tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)
        with _file_lock(DINO_CACHE_DIR / ".gpu.lock"):
            model = _load_dino_model()
            device = next(model.parameters()).device
            with torch.inference_mode():
                output = model(pixel_values=((image - mean) / std).to(device))
                register_count = int(getattr(model.config, "num_register_tokens", 0))
                tokens = output.last_hidden_state[:, 1 + register_count :]
                side = int(tokens.shape[1] ** 0.5)
                if side * side != tokens.shape[1]:
                    raise RuntimeError(f"Unexpected DINO patch-token count: {tokens.shape[1]}")
                value = tokens.reshape(1, side, side, -1).permute(0, 3, 1, 2).squeeze(0).float().cpu().numpy()
        _validate_embedding(value, "DINOv3", expected_channels=1024)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(f".tmp.{uuid.uuid4().hex}.npy")
        try:
            np.save(tmp, value)
            tmp.replace(cache_path)
        finally:
            tmp.unlink(missing_ok=True)
        return value


def load_external_embedding(
    method: str, region_id: str, patch_id: str, month: str
) -> np.ndarray:
    if method == "aef":
        return load_aef_embedding(region_id, patch_id, month)
    if method == "dinov3_sat493m":
        return load_dino_embedding(region_id, patch_id, month)
    raise ValueError(f"Unsupported external embedding method: {method}")
