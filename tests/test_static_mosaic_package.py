"""Tests for the offline frontend mosaic package builder."""

import json
import zipfile

from PIL import Image

from scripts.build_static_mosaic_package import (
    asset_relative_path,
    build_zip_package,
    validate_png,
)


def test_asset_path_uses_region_sensor_date_layout():
    assert asset_relative_path("haidian", "s2", "202604").as_posix() == (
        "haidian/s2/202604/mosaic.png"
    )
    assert asset_relative_path("harbin", "embedding-v2", "202605").as_posix() == (
        "harbin/embedding-v2/202605/mosaic.png"
    )


def test_validate_png_requires_and_reports_transparency(tmp_path):
    path = tmp_path / "mosaic.png"
    image = Image.new("RGBA", (4, 3), (10, 20, 30, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    image.save(path)

    result = validate_png(path)

    assert result["width"] == 4
    assert result["height"] == 3
    assert result["transparent_pixels"] == 1
    assert len(result["sha256"]) == 64


def test_zip_package_keeps_png_layout_and_manifest(tmp_path):
    staging = tmp_path / "staging"
    relative = asset_relative_path("haidian", "s2", "202604")
    path = staging / relative
    path.parent.mkdir(parents=True)
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(path)
    manifest = {
        "package_filename": "regional-mosaics.zip",
        "total_assets": 1,
        "assets": [{"path": relative.as_posix()}],
    }
    output = tmp_path / "regional-mosaics.zip"

    build_zip_package(staging, output, manifest)

    with zipfile.ZipFile(output) as archive:
        assert relative.as_posix() in archive.namelist()
        packaged_manifest = json.loads(archive.read("manifest.json"))
    assert packaged_manifest["total_assets"] == 1
