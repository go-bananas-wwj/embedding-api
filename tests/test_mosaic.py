"""Tests for the mosaic big-image endpoint."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.mosaic_service import build_mosaic
from app.services.data_service import DataNotFoundError, DataValidationError


def test_build_mosaic_png_returns_rgba_image():
    data, mime = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s2",
        version="v2",
        fmt="png",
    )
    assert mime == "image/png"
    img = Image.open(io.BytesIO(data))
    assert img.mode == "RGBA"
    assert img.width > 0 and img.height > 0


def test_build_mosaic_uses_cache(tmp_path):
    cache_dir = tmp_path / "mosaic"
    data1, _ = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s2",
        version="v2",
        fmt="png",
        cache_dir=str(cache_dir),
    )
    cache_file = cache_dir / "harbin_s2_v2_2025-04.png"
    assert cache_file.exists()
    data2, _ = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s2",
        version="v2",
        fmt="png",
        cache_dir=str(cache_dir),
    )
    assert data1 == data2


def test_build_mosaic_unsupported_sensor():
    with pytest.raises(DataValidationError):
        build_mosaic(
            region_id="harbin",
            date="2025-04",
            sensor_type="s1",
            fmt="png",
        )


def test_build_mosaic_unknown_region():
    with pytest.raises(DataValidationError):
        build_mosaic(
            region_id="unknown_region",
            date="2025-04",
            fmt="png",
        )


def test_build_mosaic_missing_date():
    with pytest.raises(DataNotFoundError):
        build_mosaic(
            region_id="harbin",
            date="2099-01",
            sensor_type="s2",
            version="v2",
            fmt="png",
        )


def test_build_mosaic_geotiff(tmp_path):
    try:
        import rasterio  # noqa: F401
    except ImportError:
        pytest.skip("rasterio not installed")

    cache_dir = tmp_path / "mosaic"
    data, mime = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s2",
        version="v2",
        fmt="tif",
        cache_dir=str(cache_dir),
    )
    assert mime == "image/tiff"
    assert len(data) > 0
