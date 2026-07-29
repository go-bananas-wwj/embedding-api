"""Tests for the mosaic big-image endpoint."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.mosaic_service import _to_mosaic_rgba, _to_rgb, build_mosaic
from app.services.data_service import DataNotFoundError, DataValidationError


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("mosaic")


def _sample_patches(n: int = 5):
    return [f"patch_{i:06d}" for i in range(n)]


def test_build_mosaic_png_returns_rgba_image(shared_cache):
    data, mime = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s2",
        fmt="png",
        patch_ids=_sample_patches(),
        cache_dir=str(shared_cache),
    )
    assert mime == "image/png"
    img = Image.open(io.BytesIO(data))
    assert img.mode == "RGBA"
    assert img.width > 0 and img.height > 0


def test_build_mosaic_s1(shared_cache):
    data, mime = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s1",
        fmt="png",
        patch_ids=_sample_patches(),
        cache_dir=str(shared_cache),
    )
    assert mime == "image/png"
    img = Image.open(io.BytesIO(data))
    assert img.mode == "RGBA"


def test_build_mosaic_landsat(shared_cache):
    data, mime = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="landsat",
        fmt="png",
        patch_ids=_sample_patches(),
        cache_dir=str(shared_cache),
    )
    assert mime == "image/png"
    img = Image.open(io.BytesIO(data))
    assert img.mode == "RGBA"


def test_build_mosaic_uses_cache(shared_cache):
    sample = _sample_patches()
    cache_file = (
        shared_cache
        / f"harbin_s2_2025-04_raw-v3_{'_'.join(sorted(sample))}.png"
    )
    assert cache_file.exists()
    data1 = cache_file.read_bytes()
    data2, _ = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s2",
        fmt="png",
        patch_ids=sample,
        cache_dir=str(shared_cache),
    )
    assert data1 == data2


def test_build_embedding_mosaic_returns_visible_png(shared_cache):
    data, mime = build_mosaic(
        region_id="haidian",
        date="202512",
        sensor_type="embedding",
        fmt="png",
        patch_ids=["patch_000000"],
        cache_dir=str(shared_cache),
    )

    assert mime == "image/png"
    image = Image.open(io.BytesIO(data))
    assert image.width > 1 and image.height > 1


def test_build_mosaic_unsupported_sensor():
    with pytest.raises(DataValidationError):
        build_mosaic(
            region_id="harbin",
            date="2025-04",
            sensor_type="modis",
            fmt="png",
        )


def test_highres_rgb_mapping_uses_first_three_bands():
    arr = np.stack(
        [
            np.arange(16, dtype=np.float32).reshape(4, 4),
            np.arange(16, dtype=np.float32).reshape(4, 4) * 2,
            np.arange(16, dtype=np.float32).reshape(4, 4) * 3,
        ]
    )
    rgba = _to_rgb(arr, "highres")
    assert rgba.shape == (4, 4, 4)


def test_highres_rgb_mapping_rejects_two_band_image():
    with pytest.raises(DataValidationError, match="requires at least 3"):
        _to_rgb(np.ones((2, 4, 4), dtype=np.float32), "highres")


def test_mosaic_png_uses_fixed_sensor_scale_instead_of_percentile_stretch():
    arr = np.zeros((12, 1, 2), dtype=np.float32)
    arr[2, 0] = [1000, 2000]
    arr[1, 0] = [2000, 3000]
    arr[0, 0] = [3000, 4000]

    rgba = _to_mosaic_rgba(arr, "s2")

    assert rgba[0, 0].tolist() == [26, 51, 76, 255]
    assert rgba[0, 1].tolist() == [51, 76, 102, 255]


def test_mosaic_png_keeps_original_zero_pixels_opaque():
    arr = np.zeros((7, 1, 2), dtype=np.float32)
    arr[0:3, 0, 1] = [0.1, 0.2, 0.3]

    rgba = _to_mosaic_rgba(arr, "landsat")

    assert rgba[0, 0].tolist() == [0, 0, 0, 255]
    assert rgba[0, 1, 3] == 255


def test_mosaic_png_renders_non_finite_source_as_opaque_black():
    arr = np.full((12, 1, 1), np.nan, dtype=np.float32)

    rgba = _to_mosaic_rgba(arr, "s2")

    assert rgba[0, 0].tolist() == [0, 0, 0, 255]


def test_s1_hr_mosaic_accepts_single_band_source():
    arr = np.array([[[-3.0, 0.0, 3.0]]], dtype=np.float32)

    rgba = _to_mosaic_rgba(arr, "s1_hr")

    assert rgba[0, 0].tolist() == [0, 0, 0, 255]
    assert rgba[0, 1].tolist() == [128, 128, 128, 255]
    assert rgba[0, 2].tolist() == [255, 255, 255, 255]


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
            fmt="png",
        )


@pytest.mark.slow
def test_build_mosaic_geotiff(shared_cache):
    try:
        import rasterio  # noqa: F401
    except ImportError:
        pytest.skip("rasterio not installed")

    data, mime = build_mosaic(
        region_id="harbin",
        date="2025-04",
        sensor_type="s2",
        fmt="tif",
        patch_ids=_sample_patches(),
        cache_dir=str(shared_cache),
    )
    assert mime == "image/tiff"
    assert len(data) > 0
