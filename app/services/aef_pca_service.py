"""AEF 2025 global-PCA visualization for Haidian patches."""

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


AEF_YEAR = "2025"
PCA_VERSION = "aef-haidian-2025-global-v1"
PATCH_PATTERN = re.compile(r"^patch_\d{6}$")
AEF_ROOT = Path(
    os.environ.get("AEF_EMBEDDING_DIR", "data/external_embeddings/aef")
) / "haidian" / AEF_YEAR
MODEL_ROOT = Path("models/haidian/aef/2025")
PCA_PATH = MODEL_ROOT / "pca_global_v1.npz"
PCA_META_PATH = MODEL_ROOT / "pca_global_v1.json"
CACHE_ROOT = Path(
    os.environ.get(
        "AEF_PCA_CACHE_DIR",
        "data/visualization_cache/aef/haidian/2025/pca_global_v1",
    )
)
MAX_ELEMENTS = 64 * 512 * 512
_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_params = None
_params_lock = threading.Lock()


class AefPcaError(Exception):
    """Base exception for AEF PCA failures."""


class InvalidPatchId(AefPcaError):
    """Patch ID does not match the public contract."""


class AefEmbeddingNotFound(AefPcaError):
    """AEF source file is unavailable."""


def validate_patch_id(patch_id: str) -> None:
    if not PATCH_PATTERN.fullmatch(patch_id):
        raise InvalidPatchId("Invalid patch_id. Use format patch_000000")


def embedding_path(patch_id: str) -> Path:
    validate_patch_id(patch_id)
    path = (AEF_ROOT / f"{patch_id}.npy").resolve()
    root = AEF_ROOT.resolve()
    if root not in path.parents:
        raise InvalidPatchId("Invalid patch_id. Use format patch_000000")
    if not path.is_file():
        raise AefEmbeddingNotFound(
            f"AEF 2025 embedding not found for Haidian patch '{patch_id}'"
        )
    return path


def load_embedding(path: Path) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.ndim != 3 or value.shape[0] <= 3:
        raise AefPcaError(f"AEF embedding must use [C,H,W], got {value.shape}")
    if value.size > MAX_ELEMENTS:
        raise AefPcaError("AEF embedding exceeds the visualization safety limit")
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise AefPcaError("AEF embedding contains NaN or infinite values")
    return array


def fit_global_pca(
    samples_per_patch: int = 512,
    random_seed: int = 2025,
) -> dict:
    """Fit one stable PCA/color range for every Haidian AEF patch."""
    paths = sorted(AEF_ROOT.glob("patch_*.npy"))
    if not paths:
        raise AefPcaError(f"No AEF embeddings found under {AEF_ROOT}")
    rng = np.random.default_rng(random_seed)
    samples = []
    expected_channels = None
    for path in paths:
        embedding = load_embedding(path)
        channels = embedding.shape[0]
        if expected_channels is None:
            expected_channels = channels
        elif channels != expected_channels:
            raise AefPcaError(
                f"Inconsistent AEF channels: {path.name} has {channels}, "
                f"expected {expected_channels}"
            )
        pixels = embedding.reshape(channels, -1).T
        count = min(samples_per_patch, len(pixels))
        indices = rng.choice(len(pixels), size=count, replace=False)
        samples.append(pixels[indices])
    sample = np.concatenate(samples, axis=0).astype(np.float32, copy=False)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=random_seed)
    projected = pca.fit_transform(sample)
    display_low = np.percentile(projected, 2, axis=0).astype(np.float32)
    display_high = np.percentile(projected, 98, axis=0).astype(np.float32)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(
        PCA_PATH,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        display_low=display_low,
        display_high=display_high,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        sample_count=np.int64(len(sample)),
        random_seed=np.int64(random_seed),
        source_patch_count=np.int64(len(paths)),
    )
    metadata = {
        "version": PCA_VERSION,
        "region_id": "haidian",
        "source": "AEF",
        "year": AEF_YEAR,
        "source_patch_count": len(paths),
        "sample_count": len(sample),
        "samples_per_patch": samples_per_patch,
        "random_seed": random_seed,
        "feature_channels": expected_channels,
        "display_percentiles": [2, 98],
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    PCA_META_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    global _params
    _params = None
    return metadata


def load_pca_params():
    global _params
    if _params is not None:
        return _params
    with _params_lock:
        if _params is None:
            if not PCA_PATH.is_file():
                raise AefPcaError(
                    f"AEF global PCA model is missing: {PCA_PATH}"
                )
            data = np.load(PCA_PATH, allow_pickle=False)
            _params = {
                key: np.asarray(data[key])
                for key in ("mean", "components", "display_low", "display_high")
            }
    return _params


def render_pca(embedding: np.ndarray) -> np.ndarray:
    params = load_pca_params()
    channels, height, width = embedding.shape
    if channels != len(params["mean"]):
        raise AefPcaError(
            f"AEF channel mismatch: expected {len(params['mean'])}, got {channels}"
        )
    pixels = embedding.reshape(channels, -1).T
    projected = (pixels - params["mean"]) @ params["components"].T
    low = params["display_low"]
    high = params["display_high"]
    rgb = np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)
    return np.rint(rgb.reshape(height, width, 3) * 255).astype(np.uint8)


def _patch_lock(patch_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(patch_id, threading.Lock())


def get_or_create_pca_png(patch_id: str) -> Path:
    source = embedding_path(patch_id)
    output = CACHE_ROOT / f"{patch_id}.png"
    if output.is_file() and output.stat().st_mtime_ns >= max(
        source.stat().st_mtime_ns, PCA_PATH.stat().st_mtime_ns
    ):
        return output
    with _patch_lock(patch_id):
        if output.is_file() and output.stat().st_mtime_ns >= max(
            source.stat().st_mtime_ns, PCA_PATH.stat().st_mtime_ns
        ):
            return output
        rgb = render_pca(load_embedding(source))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f".tmp.{os.getpid()}.png")
        try:
            Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def response_etag(path: Path) -> str:
    material = f"{PCA_VERSION}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
    return '"' + hashlib.sha256(material.encode()).hexdigest()[:24] + '"'
