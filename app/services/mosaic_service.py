"""Build region-wide mosaic PNG/GeoTIFF from per-patch preview images."""

import io
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from app.config import get_config
from app.services.data_service import DataNotFoundError, DataValidationError

logger = logging.getLogger(__name__)


def _get_preview_path(
    region_id: str,
    patch_id: str,
    date: str,
    version: str,
) -> Optional[str]:
    """Resolve a per-patch preview PNG path.

    Currently supports harbin-style directory layout:
        data/{region}/embeddings/{version}/{date}/{patch_id}.png
    """
    base = f"data/{region_id}/embeddings/{version}/{date}"
    path = os.path.join(base, f"{patch_id}.png")
    if os.path.exists(path) and os.path.isfile(path):
        return path
    return None


def _sensor_to_version(sensor_type: str, version: Optional[str]) -> str:
    """Map sensor_type to a default embedding version when version is omitted."""
    if version:
        return version
    if sensor_type == "s2":
        return "v2"
    # Future sensors can be mapped here.
    raise DataValidationError(
        f"sensor_type '{sensor_type}' is not supported; use 's2' (Sentinel-2)"
    )


def build_mosaic(
    region_id: str,
    date: str,
    sensor_type: str = "s2",
    version: Optional[str] = None,
    fmt: str = "png",
    cache_dir: str = "users/default/mosaic",
) -> Tuple[bytes, str]:
    """Build a mosaic image for the given region, date and sensor.

    Returns:
        (image_bytes, mime_type)
    """
    config = get_config()
    if region_id not in config.list_regions():
        raise DataValidationError(f"Region '{region_id}' does not exist")

    version = _sensor_to_version(sensor_type, version)

    if sensor_type not in ("s2",):
        raise DataValidationError(
            f"sensor_type '{sensor_type}' is not supported; currently only 's2' is available"
        )

    fmt = fmt.lower()
    if fmt not in ("png", "tif", "tiff"):
        raise DataValidationError("format must be 'png' or 'tif'")

    ext = "tif" if fmt in ("tif", "tiff") else "png"
    cache_path = Path(cache_dir) / f"{region_id}_{sensor_type}_{version}_{date}.{ext}"
    if cache_path.exists():
        return cache_path.read_bytes(), f"image/{ext}"

    patches = config.get_patches(region_id)
    if not patches:
        raise DataNotFoundError(f"No patches found for region '{region_id}'")

    loaded: list[Tuple[dict, np.ndarray, float, float]] = []
    for patch in patches:
        patch_id = patch.get("patch_id")
        bbox = patch.get("bounds_wgs84")
        if not patch_id or not bbox or len(bbox) != 4:
            continue
        path = _get_preview_path(region_id, patch_id, date, version)
        if not path:
            continue
        try:
            img = np.array(Image.open(path).convert("RGBA"))
        except Exception as e:
            logger.warning(f"Failed to open preview {path}: {e}")
            continue
        h, w = img.shape[:2]
        res_x = (bbox[2] - bbox[0]) / w
        res_y = (bbox[3] - bbox[1]) / h
        loaded.append((patch, img, res_x, res_y))

    if not loaded:
        raise DataNotFoundError(
            f"No preview images found for {region_id}/{version}/{date}; "
            "check date, version and sensor_type"
        )

    # Use average resolution across loaded patches.
    avg_res_x = float(np.mean([res_x for _, _, res_x, _ in loaded]))
    avg_res_y = float(np.mean([res_y for _, _, _, res_y in loaded]))

    minx = min(p["bounds_wgs84"][0] for p, _, _, _ in loaded)
    miny = min(p["bounds_wgs84"][1] for p, _, _, _ in loaded)
    maxx = max(p["bounds_wgs84"][2] for p, _, _, _ in loaded)
    maxy = max(p["bounds_wgs84"][3] for p, _, _, _ in loaded)

    out_width = max(1, int(round((maxx - minx) / avg_res_x)))
    out_height = max(1, int(round((maxy - miny) / avg_res_y)))

    mosaic = np.zeros((out_height, out_width, 4), dtype=np.uint8)

    for patch, img, res_x, res_y in loaded:
        bbox = patch["bounds_wgs84"]
        h, w = img.shape[:2]
        col = int(round((bbox[0] - minx) / avg_res_x))
        row = int(round((maxy - bbox[3]) / avg_res_y))
        # Clip to mosaic bounds to guard against rounding overflow.
        row_end = min(row + h, out_height)
        col_end = min(col + w, out_width)
        img_h = row_end - row
        img_w = col_end - col
        if img_h <= 0 or img_w <= 0:
            continue
        mosaic[row:row_end, col:col_end] = img[:img_h, :img_w]

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ("tif", "tiff"):
        return _write_geotiff(mosaic, minx, maxy, avg_res_x, avg_res_y, cache_path)

    # PNG output
    pil_img = Image.fromarray(mosaic)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data = buf.getvalue()
    cache_path.write_bytes(data)
    return data, "image/png"


def _write_geotiff(
    arr: np.ndarray,
    minx: float,
    maxy: float,
    res_x: float,
    res_y: float,
    cache_path: Path,
) -> Tuple[bytes, str]:
    """Write a GeoTIFF with WGS84 CRS."""
    import rasterio
    from rasterio.transform import Affine

    height, width = arr.shape[:2]
    transform = Affine.translation(minx, maxy) * Affine.scale(res_x, -res_y)

    # Some environments have a PROJ database mismatch; gracefully fall back
    # to writing a plain TIFF if the EPSG CRS cannot be resolved.
    crs_kwargs = {}
    try:
        crs_kwargs["crs"] = rasterio.CRS.from_epsg(4326)
    except Exception as e:
        logger.warning(f"Could not set WGS84 CRS, writing plain TIFF: {e}")

    with rasterio.open(
        str(cache_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=arr.dtype,
        transform=transform,
        **crs_kwargs,
    ) as dst:
        for i in range(4):
            dst.write(arr[:, :, i], i + 1)

    return cache_path.read_bytes(), "image/tiff"
