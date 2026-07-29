"""Build region-wide mosaic PNG/GeoTIFF from per-patch raw satellite TIFFs."""

import io
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge as rio_merge

from app.config import get_config
from app.services.data_service import DataNotFoundError, DataService, DataValidationError
from app.services.time_utils import normalize_quarter_date

logger = logging.getLogger(__name__)

RAW_ROOT = "/workspace/data/raw"
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
    # Generic high-resolution optical GeoTIFF: RGB bands in display order.
    # Files are discovered from a ``highres`` sensor directory, matching the
    # same per-region/per-patch layouts used by the other sensors.
    "highres": (0, 1, 2),
    "highres_sar": (0, 0, 0),
    "s1_hr": (0, 0, 0),
    "s2_hr": (0, 1, 2),
}

# Public API values follow the frontend database. Storage keys may differ.
_SENSOR_STORAGE_KEYS = {
    "s1": "s1",
    "s2": "s2",
    "landsat": "landsat",
    "highres": "highres_optical",
    "highres_sar": "highres_sar",
    "s1_hr": "s1_hr",
    "s2_hr": "s2_hr",
}

# Fixed source-to-PNG display ranges. Unlike percentile stretching, these do
# not change according to the pixels included in a request.
_SENSOR_DISPLAY_RANGES = {
    "s1": (0.0, 1.0),
    "s2": (0.0, 10000.0),
    "landsat": (0.0, 1.0),
    "highres": (0.0, 4095.0),
    "highres_sar": (0.0, 1.0),
    "s1_hr": (-3.0, 3.0),
    "s2_hr": (0.0, 255.0),
}

_MOSAIC_CACHE_VERSION = "raw-v3"


def _sensor_storage_key(sensor_type: str) -> str:
    """Map a public frontend sensor value to its on-disk filename prefix."""
    return _SENSOR_STORAGE_KEYS.get(sensor_type, sensor_type)


def _available_sensor_months(
    roots: List[str], region_id: str, sensor_type: str
) -> List[str]:
    """Discover available YYYYMM values for a configured sensor root."""
    storage_key = _sensor_storage_key(sensor_type)
    months = set()
    for root_value in reversed(roots):
        root = Path(root_value)
        candidates = (
            root,
            root / storage_key,
            root / region_id / storage_key,
        )
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                paths = candidate.rglob("*.tif")
                for path in paths:
                    match = re.search(r"(?<!\d)(\d{6})\d{2}(?!\d)", path.stem)
                    if match:
                        months.add(match.group(1))
            except OSError:
                continue
        if months:
            break
    return sorted(months)


def _configured_sensor_roots(
    region_cfg: dict,
    sensor_type: str,
    *,
    raw_root: str = RAW_ROOT,
) -> List[str]:
    """Return storage roots for the requested sensor, not just Sentinel-2."""
    roots = [raw_root]
    configured = region_cfg.get(f"{sensor_type}_dir")
    if configured and configured not in roots:
        roots.append(configured)
    return roots


def _candidate_period_prefixes(periods: List[str]) -> List[str]:
    """Return month prefixes to use for fuzzy filename matching.

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

    for p in periods:
        if not _YYYY_QUARTER_RE.match(p):
            add(p)
    return prefixes


def _select_daily_candidate(candidates: List[Path], prefixes: List[str]) -> Optional[Path]:
    """Select the latest daily scene inside the requested month/quarter."""
    matching = []
    for path in candidates:
        stem = path.stem
        if _YYYYMMDD_RE.match(stem) and any(stem.startswith(prefix) for prefix in prefixes):
            matching.append(path)
    if not matching:
        return None
    return sorted(matching, key=lambda p: p.stem, reverse=True)[0]


def _select_flat_patch_candidate(
    root: Path, sensor_type: str, patch_id: str, prefixes: List[str]
) -> Optional[Path]:
    """Resolve extracted ``SENSOR_YYYYMMDD_PATCH.tif`` training archives."""
    if not root.is_dir():
        return None
    matches = []
    storage_key = _sensor_storage_key(sensor_type)
    for candidate_dir in (root, root / storage_key):
        if not candidate_dir.is_dir():
            continue
        for path in candidate_dir.glob(f"{storage_key}_*_{patch_id}.tif"):
            date_match = re.search(r"(?<!\d)(\d{8})(?!\d)", path.stem)
            if date_match and any(date_match.group(1).startswith(p) for p in prefixes):
                matches.append((date_match.group(1), path))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1].name))[1]


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
    storage_key = _sensor_storage_key(sensor_type)
    # Derive request-scoped prefixes once for exact and daily-scene matching.
    fuzzy_prefixes = _candidate_period_prefixes(periods)
    fallback_exact_periods = [p for p in periods if _YYYY_QUARTER_RE.match(p)]

    for root in roots:
        if not root:
            continue
        layouts = [
            Path(root) / region_id / storage_key / patch_id,
            Path(root) / patch_id / storage_key,
            # A configured sensor root may already point at ``.../s2``.
            Path(root) / patch_id,
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
            # Legacy quarterly archives remain a last-resort compatibility
            # source. Monthly and daily files above always win.
            for period in fallback_exact_periods:
                path = layout_dir / f"{period}.tif"
                if path.exists() and path.is_file():
                    return str(path)
        flat = _select_flat_patch_candidate(
            Path(root), sensor_type, patch_id, fuzzy_prefixes
        )
        if flat:
            return str(flat)
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

    Reads raw per-patch TIFFs from /workspace/data/raw/{region_id}/{sensor_type}.
    The `version` parameter is kept for API compatibility but is ignored for
    raw satellite sensors.

    Returns:
        (image_bytes, mime_type)
    """
    config = get_config()
    if region_id not in config.list_regions():
        raise DataValidationError(f"Region '{region_id}' does not exist")

    sensor_type = sensor_type.lower()
    # Allow each sensor to use its configured per-region scene directory.
    region_cfg = config.get_region(region_id) or {}
    roots = _configured_sensor_roots(region_cfg, sensor_type)

    if sensor_type == "embedding":
        if fmt.lower() != "png":
            raise DataValidationError("embedding mosaic only supports format='png'")
        return _build_embedding_mosaic(
            region_id, date, version, patch_ids, cache_dir, config.get_patches(region_id)
        )

    del version  # raw sensors do not use embedding versions
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
    legacy_quarter = _legacy_quarter_key(date)
    if legacy_quarter:
        periods.append(legacy_quarter)
    cache_period = periods[0]

    allowed_ids = set(patch_ids) if patch_ids else None
    cache_suffix = ""
    if allowed_ids:
        cache_suffix = "_" + "_".join(sorted(allowed_ids))[:64]
    ext = "tif" if fmt in ("tif", "tiff") else "png"
    cache_path = (
        Path(cache_dir)
        / f"{region_id}_{sensor_type}_{cache_period}_{_MOSAIC_CACHE_VERSION}{cache_suffix}.{ext}"
    )
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
        available_months = _available_sensor_months(
            roots, region_id, sensor_type
        )
        availability = (
            f" Available months for {sensor_type}: {', '.join(available_months)}."
            if available_months
            else ""
        )
        raise DataNotFoundError(
            f"No raw {sensor_type} images found for {region_id}/{date}; "
            "check date/sensor_type. Supported formats: YYYY-MM, YYYYMM, YYYYMMDD."
            f"{availability}"
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

    rgb = _to_mosaic_rgba(merged, sensor_type)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data = buf.getvalue()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data, "image/png"


def _legacy_quarter_key(date: str) -> Optional[str]:
    """Map a month to an old archive key without changing public semantics."""
    match = _YYYY_HYPHEN_MM_RE.match(date) or _YYYYMM_RE.match(date)
    if not match:
        return None
    year, month = match.groups()
    quarter = (int(month) - 1) // 3 + 1
    return f"{year}Q{quarter}"


def _build_embedding_mosaic(
    region_id: str,
    date: str,
    version: Optional[str],
    patch_ids: Optional[List[str]],
    cache_dir: str,
    patches: List[dict],
) -> Tuple[bytes, str]:
    """Compose globally normalized per-Patch embedding PNGs by map bounds."""
    periods = normalize_quarter_date(date)
    if not periods:
        raise DataValidationError(f"Invalid date format: '{date}'")
    effective_version = version or ("v1" if region_id == "haidian" else "v2")
    allowed_ids = set(patch_ids) if patch_ids else None
    suffix = ""
    if allowed_ids:
        suffix = "_" + "_".join(sorted(allowed_ids))[:64]
    cache_path = (
        Path(cache_dir)
        / f"{region_id}_embedding_{effective_version}_{periods[0]}{suffix}.png"
    )
    if cache_path.exists():
        return cache_path.read_bytes(), "image/png"

    tiles = []
    for patch in patches:
        patch_id = patch.get("patch_id")
        bounds = patch.get("bounds")
        if not patch_id or not bounds or len(bounds) != 4:
            continue
        if allowed_ids is not None and patch_id not in allowed_ids:
            continue
        path = DataService.get_embedding_path(
            region_id, patch_id, "png", version=effective_version, month=date
        )
        if path:
            with Image.open(path) as image:
                tiles.append((tuple(float(v) for v in bounds), image.convert("RGBA").copy()))

    if not tiles:
        raise DataNotFoundError(
            f"No embedding PNGs found for {region_id}/{date}, version {effective_version}"
        )

    first_bounds, first_image = tiles[0]
    tile_width = first_bounds[2] - first_bounds[0]
    tile_height = first_bounds[3] - first_bounds[1]
    if tile_width <= 0 or tile_height <= 0:
        raise DataValidationError("Patch bounds must have positive width and height")
    px_per_x = first_image.width / tile_width
    px_per_y = first_image.height / tile_height
    min_x = min(bounds[0] for bounds, _ in tiles)
    min_y = min(bounds[1] for bounds, _ in tiles)
    max_x = max(bounds[2] for bounds, _ in tiles)
    max_y = max(bounds[3] for bounds, _ in tiles)
    width = max(1, round((max_x - min_x) * px_per_x))
    height = max(1, round((max_y - min_y) * px_per_y))
    if width * height > 150_000_000:
        raise DataValidationError("Embedding mosaic is too large; filter with patch_ids")

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for bounds, image in tiles:
        expected_width = max(1, round((bounds[2] - bounds[0]) * px_per_x))
        expected_height = max(1, round((bounds[3] - bounds[1]) * px_per_y))
        if image.size != (expected_width, expected_height):
            image = image.resize((expected_width, expected_height), Image.Resampling.BILINEAR)
        x = round((bounds[0] - min_x) * px_per_x)
        y = round((max_y - bounds[3]) * px_per_y)
        canvas.alpha_composite(image, (x, y))

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    data = buffer.getvalue()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data, "image/png"


def _to_rgb(arr: np.ndarray, sensor_type: str) -> np.ndarray:
    """Convert a multi-band float array to an 8-bit RGBA image."""
    count, height, width = arr.shape
    red_i, green_i, blue_i = _SENSOR_RGB[sensor_type]

    required_indices = [i for i in (red_i, green_i, blue_i) if i is not None]
    if not required_indices or max(required_indices) >= count:
        raise DataValidationError(
            f"{sensor_type} image has {count} band(s), but its RGB mapping "
            f"requires at least {max(required_indices) + 1} band(s)"
        )

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


def _to_mosaic_rgba(arr: np.ndarray, sensor_type: str) -> np.ndarray:
    """Convert raw bands to PNG with a fixed sensor scale and no auto enhancement."""
    count, _, _ = arr.shape
    red_i, green_i, blue_i = _SENSOR_RGB[sensor_type]
    required = [index for index in (red_i, green_i, blue_i) if index is not None]
    if not required or max(required) >= count:
        raise DataValidationError(
            f"{sensor_type} image has {count} band(s), but its RGB mapping "
            f"requires at least {max(required) + 1} band(s)"
        )

    red = arr[red_i].astype(np.float32, copy=False)
    green = arr[green_i].astype(np.float32, copy=False)
    if blue_i is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            blue = np.where(red != 0, green / red, 0)
    else:
        blue = arr[blue_i].astype(np.float32, copy=False)

    source_rgb = np.stack([red, green, blue], axis=0)
    low, high = _SENSOR_DISPLAY_RANGES[sensor_type]
    scaled = (source_rgb - low) / (high - low)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    # PNG is a display derivative: raw zero/NaN pixels are rendered black,
    # not transparent, so a browser's white page cannot appear as noise.
    alpha = np.full(rgb.shape[1:], 255, dtype=np.uint8)
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
