"""Build region-wide mosaic PNG/GeoTIFF from per-patch raw satellite TIFFs."""

import io
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge as rio_merge

from app.config import get_config
from app.services.data_service import DataNotFoundError, DataValidationError

logger = logging.getLogger(__name__)

RAW_ROOT = "/workspace/raw"

# Visualization defaults per sensor.
# Each entry maps sensor_type -> (red_band, green_band, blue_band).
# Band indices are 0-based into the raw TIFF band order.
_SENSOR_RGB = {
    # Sentinel-2: common 6-band order B2,B3,B4,B8,B11,B12 -> R=B4(idx2), G=B3(idx1), B=B2(idx0)
    "s2": (2, 1, 0),
    # Landsat: common 6-band order B2,B3,B4,B5,B6,B7 -> R=B4(idx2), G=B3(idx1), B=B2(idx0)
    "landsat": (2, 1, 0),
    # Sentinel-1: 2 bands VV(idx0), VH(idx1) -> R=VV, G=VH, B=VH/VV ratio
    "s1": (0, 1, None),
}


def _resolve_period(date: str) -> str:
    """Map YYYY-MM to quarterly filename, or return exact filename stem."""
    if "Q" in date:
        return date
    if "-" in date:
        year, month = date.split("-", 1)
        try:
            quarter = (int(month) - 1) // 3 + 1
            return f"{year}Q{quarter}"
        except ValueError:
            pass
    return date


def _get_raw_tiff_path(
    region_id: str,
    patch_id: str,
    sensor_type: str,
    period: str,
) -> Optional[str]:
    """Resolve a per-patch raw TIFF path.

    Layout: /workspace/raw/{region_id}/{sensor_type}/{patch_id}/{period}.tif
    """
    path = os.path.join(
        RAW_ROOT, region_id, sensor_type, patch_id, f"{period}.tif"
    )
    if os.path.exists(path) and os.path.isfile(path):
        return path
    return None


def build_mosaic(
    region_id: str,
    date: str,
    sensor_type: str = "s2",
    version: Optional[str] = None,
    fmt: str = "png",
    patch_ids: Optional[List[str]] = None,
    cache_dir: str = "users/default/mosaic",
) -> Tuple[bytes, str]:
    """Build a mosaic image for the given region, date and sensor.

    Reads raw per-patch TIFFs from /workspace/raw/{region_id}/{sensor_type}.
    The `version` parameter is kept for API compatibility but is ignored for
    raw satellite sensors.

    Returns:
        (image_bytes, mime_type)
    """
    del version  # raw sensors do not use embedding versions

    config = get_config()
    if region_id not in config.list_regions():
        raise DataValidationError(f"Region '{region_id}' does not exist")

    sensor_type = sensor_type.lower()
    if sensor_type not in _SENSOR_RGB:
        raise DataValidationError(
            f"sensor_type '{sensor_type}' is not supported; "
            f"use one of {list(_SENSOR_RGB.keys())}"
        )

    fmt = fmt.lower()
    if fmt not in ("png", "tif", "tiff"):
        raise DataValidationError("format must be 'png' or 'tif'")

    period = _resolve_period(date)
    allowed_ids = set(patch_ids) if patch_ids else None
    cache_suffix = ""
    if allowed_ids:
        cache_suffix = "_" + "_".join(sorted(allowed_ids))[:64]
    ext = "tif" if fmt in ("tif", "tiff") else "png"
    cache_path = Path(cache_dir) / f"{region_id}_{sensor_type}_{period}{cache_suffix}.{ext}"
    if cache_path.exists():
        return cache_path.read_bytes(), f"image/{ext}"

    patches = config.get_patches(region_id)
    if not patches:
        raise DataNotFoundError(f"No patches found for region '{region_id}'")

    paths = []
    for patch in patches:
        patch_id = patch.get("patch_id")
        if not patch_id:
            continue
        if allowed_ids is not None and patch_id not in allowed_ids:
            continue
        path = _get_raw_tiff_path(region_id, patch_id, sensor_type, period)
        if path:
            paths.append(path)

    if not paths:
        raise DataNotFoundError(
            f"No raw {sensor_type} images found for {region_id}/{period}; "
            "check date/sensor_type. Date 'YYYY-MM' is mapped to quarterly files."
        )

    with rasterio.Env():
        datasets = [rasterio.open(p) for p in paths]
        try:
            merged, transform = rio_merge(datasets)
        finally:
            for ds in datasets:
                ds.close()

    crs = datasets[0].crs if datasets else None

    if fmt in ("tif", "tiff"):
        return _write_raw_geotiff(merged, transform, crs, cache_path)

    rgb = _to_rgb(merged, sensor_type)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data = buf.getvalue()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data, "image/png"


def _to_rgb(arr: np.ndarray, sensor_type: str) -> np.ndarray:
    """Convert a multi-band float array to an 8-bit RGBA image."""
    count, height, width = arr.shape
    red_i, green_i, blue_i = _SENSOR_RGB[sensor_type]

    red = arr[red_i]
    green = arr[green_i]
    if blue_i is None:
        # Sentinel-1 synthetic blue channel: VH / VV ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            blue = np.where(green != 0, green / red, 0)
    else:
        blue = arr[blue_i]

    rgb = np.stack([red, green, blue], axis=0)
    rgb = _stretch_percentile(rgb)

    # Alpha: mark pixels where all bands are zero/NaN as transparent.
    valid = np.isfinite(rgb).all(axis=0) & ~(np.all(rgb == 0, axis=0))
    alpha = (valid * 255).astype(np.uint8)

    rgba = np.concatenate([rgb, alpha[None, ...]], axis=0)
    return np.transpose(rgba, (1, 2, 0))


def _stretch_percentile(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Linearly stretch each band to 0-255 using percentile clipping."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for i in range(arr.shape[0]):
        band = arr[i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [low, high])
        if hi == lo:
            norm = np.zeros_like(band)
        else:
            norm = (band - lo) / (hi - lo)
        norm = np.clip(norm, 0, 1)
        out[i] = (norm * 255).astype(np.uint8)
    return out


def _write_raw_geotiff(
    arr: np.ndarray,
    transform,
    crs,
    cache_path: Path,
) -> Tuple[bytes, str]:
    """Write the merged multi-band float array as a GeoTIFF."""
    count, height, width = arr.shape
    dtype = arr.dtype

    crs_kwargs = {}
    if crs is not None:
        try:
            crs_kwargs["crs"] = crs
        except Exception as e:
            logger.warning(f"Could not set CRS, writing plain TIFF: {e}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(cache_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        transform=transform,
        **crs_kwargs,
    ) as dst:
        for i in range(count):
            dst.write(arr[i], i + 1)

    return cache_path.read_bytes(), "image/tiff"
