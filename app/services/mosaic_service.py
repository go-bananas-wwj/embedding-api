"""Build region-wide mosaic PNG/GeoTIFF from per-patch raw satellite TIFFs."""

import io
import logging
import os
import re
from datetime import date as date_cls
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge as rio_merge

from app.config import get_config
from app.services.data_service import DataNotFoundError, DataValidationError
from app.services.time_utils import normalize_quarter_date

logger = logging.getLogger(__name__)

RAW_ROOT = "/workspace/raw"
_YYYYMMDD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_YYYYMM_RE = re.compile(r"^(\d{4})(\d{2})$")
_YYYY_HYPHEN_MM_RE = re.compile(r"^(\d{4})-(\d{2})$")
_YYYY_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")

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


def _candidate_period_prefixes(periods: List[str]) -> List[str]:
    """Return month/quarter prefixes to use for fuzzy filename matching.

    Some raw scene archives store daily files (YYYYMMDD). When an exact
    quarterly/monthly filename is missing, we fall back to files inside the
    requested month or quarter. Exact day requests are intentionally strict.
    We also intentionally do not fall back to a bare year prefix; that could
    select an unrelated month when several scenes exist in the same year.
    """
    if not periods:
        return []

    first = periods[0]
    prefixes = []

    def add(value: str) -> None:
        if value not in prefixes:
            prefixes.append(value)

    if _YYYYMMDD_RE.match(first):
        add(first)
        return prefixes

    m_hyphen = _YYYY_HYPHEN_MM_RE.match(first)
    if m_hyphen:
        year, month = m_hyphen.groups()
        add(first)
        add(f"{year}{month}")
        return prefixes

    m_month = _YYYYMM_RE.match(first)
    if m_month:
        add(first)
        return prefixes

    m_quarter = _YYYY_QUARTER_RE.match(first)
    if m_quarter:
        year, q_str = m_quarter.groups()
        q = int(q_str)
        add(first)
        for month in range((q - 1) * 3 + 1, q * 3 + 1):
            add(f"{year}{month:02d}")
        return prefixes

    for p in periods:
        add(p)
    return prefixes


def _target_scene_date(prefixes: List[str]) -> Optional[date_cls]:
    """Return the representative target date for choosing among daily scenes."""
    if not prefixes:
        return None
    first = prefixes[0]
    m_exact = _YYYYMMDD_RE.match(first)
    if m_exact:
        year, month, day = m_exact.groups()
        return date_cls(int(year), int(month), int(day))
    m_month = _YYYYMM_RE.match(first)
    if m_month:
        year, month = m_month.groups()
        return date_cls(int(year), int(month), 15)
    m_hyphen = _YYYY_HYPHEN_MM_RE.match(first)
    if m_hyphen:
        year, month = m_hyphen.groups()
        return date_cls(int(year), int(month), 15)
    m_quarter = _YYYY_QUARTER_RE.match(first)
    if m_quarter:
        year, q_str = m_quarter.groups()
        middle_month = (int(q_str) - 1) * 3 + 2
        return date_cls(int(year), middle_month, 15)
    return None


def _daily_scene_sort_key(path: Path, target: Optional[date_cls]) -> Tuple[int, str]:
    """Prefer the daily scene closest to the requested month/quarter midpoint."""
    match = _YYYYMMDD_RE.match(path.stem)
    if not match or target is None:
        return (10**9, path.name)
    year, month, day = match.groups()
    scene_date = date_cls(int(year), int(month), int(day))
    return (abs((scene_date - target).days), path.name)


def _select_daily_candidate(candidates: List[Path], prefixes: List[str]) -> Optional[Path]:
    """Select a deterministic daily scene inside the requested month/quarter."""
    matching = []
    for path in candidates:
        stem = path.stem
        if any(stem.startswith(prefix) for prefix in prefixes):
            matching.append(path)
    if not matching:
        return None
    target = _target_scene_date(prefixes)
    return sorted(matching, key=lambda p: _daily_scene_sort_key(p, target))[0]


def _get_raw_tiff_path(
    region_id: str,
    patch_id: str,
    sensor_type: str,
    periods: List[str],
    roots: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve a per-patch raw TIFF path across multiple storage layouts.

    Tried layouts (in order):
      1. {root}/{region_id}/{sensor_type}/{patch_id}/{period}.tif  (Harbin quarterly)
      2. {root}/{patch_id}/{sensor_type}/{period}.tif              (Haidian/OLMO daily)

    ``periods`` already contains YYYY-MM, YYYYMM, YYYYQn forms from
    ``normalize_quarter_date``. Exact day/month files are preferred first. If
    the request is monthly or quarterly and no exact file exists, daily files
    inside that period are accepted and selected deterministically. Quarterly
    compatibility filenames are used only after the monthly/day candidates.
    """
    roots = roots or [RAW_ROOT]
    # Derive request-scoped prefixes once for exact and daily-scene matching.
    fuzzy_prefixes = _candidate_period_prefixes(periods)
    fallback_exact_periods = [p for p in periods if p not in fuzzy_prefixes]

    for root in roots:
        if not root:
            continue
        layouts = [
            Path(root) / region_id / sensor_type / patch_id,
            Path(root) / patch_id / sensor_type,
        ]
        for layout_dir in layouts:
            # 1) Exact day/month/quarter match for the user-requested period.
            for period in fuzzy_prefixes:
                path = layout_dir / f"{period}.tif"
                if path.exists() and path.is_file():
                    return str(path)
            # 2) Daily scene fallback inside the requested month/quarter.
            if layout_dir.exists():
                try:
                    candidates = sorted(layout_dir.glob("*.tif"))
                except OSError:
                    candidates = []
                selected = _select_daily_candidate(candidates, fuzzy_prefixes)
                if selected:
                    logger.info(
                        "Selected raw scene %s for %s/%s/%s from candidates=%s",
                        selected.name,
                        region_id,
                        patch_id,
                        periods[0] if periods else "",
                        [p.name for p in candidates if p.stem[:8].isdigit()],
                    )
                    return str(selected)
            # 3) Backward-compatible fallback for legacy quarterly archives
            # when the caller supplied a month but only a quarter file exists.
            for period in fallback_exact_periods:
                path = layout_dir / f"{period}.tif"
                if path.exists() and path.is_file():
                    return str(path)
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

    # Allow per-region raw scene directory (e.g. Haidian scenes live under
    # /workspace/olmo/data/haidian/scenes, not /workspace/raw/haidian/s2).
    region_cfg = config.get_region(region_id) or {}
    s2_dir = region_cfg.get("s2_dir")
    roots = [RAW_ROOT]
    if s2_dir:
        roots.append(s2_dir)

    sensor_type = sensor_type.lower()
    if sensor_type not in _SENSOR_RGB:
        raise DataValidationError(
            f"sensor_type '{sensor_type}' is not supported; "
            f"use one of {list(_SENSOR_RGB.keys())}"
        )

    fmt = fmt.lower()
    if fmt not in ("png", "tif", "tiff"):
        raise DataValidationError("format must be 'png' or 'tif'")

    periods = normalize_quarter_date(date)
    if not periods:
        raise DataValidationError(f"Invalid date format: '{date}'")
    # Use the quarterly form for cache naming when available, otherwise the
    # first candidate, so cache keys remain deterministic.
    cache_period = next((p for p in periods if "Q" in p), periods[0])

    allowed_ids = set(patch_ids) if patch_ids else None
    cache_suffix = ""
    if allowed_ids:
        cache_suffix = "_" + "_".join(sorted(allowed_ids))[:64]
    ext = "tif" if fmt in ("tif", "tiff") else "png"
    cache_path = Path(cache_dir) / f"{region_id}_{sensor_type}_{cache_period}{cache_suffix}.{ext}"
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
        path = _get_raw_tiff_path(region_id, patch_id, sensor_type, periods, roots=roots)
        if path:
            paths.append(path)

    if not paths:
        raise DataNotFoundError(
            f"No raw {sensor_type} images found for {region_id}/{date}; "
            "check date/sensor_type. Supported formats: YYYY-MM, YYYYMM, YYYYMMDD."
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
