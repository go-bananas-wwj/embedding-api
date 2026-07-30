"""HTTP regression tests for the region mosaic response."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.routers import regions


client = TestClient(app)


def test_mosaic_returns_the_complete_png(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    expected = buffer.getvalue()
    monkeypatch.setattr(
        regions,
        "build_mosaic_artifact",
        lambda **kwargs: (
            expected,
            "image/png",
            {
                "crs": "EPSG:4326",
                "bounds_wgs84": [1, 2, 3, 4],
                "width": 8,
                "height": 8,
            },
        ),
    )

    response = client.get(
        "/regions/haidian/mosaic",
        params={"date": "202512", "sensor_type": "s2", "format": "png"},
    )

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.content) > 8
    assert response.headers["x-mosaic-crs"] == "EPSG:4326"


def test_mosaic_json_returns_image_and_wgs84_footprints(monkeypatch):
    metadata = {
        "region_id": "haidian",
        "date": "202603",
        "sensor_type": "s2",
        "width": 256,
        "height": 128,
        "crs": "EPSG:4326",
        "bounds_wgs84": [116.2, 39.8, 116.3, 39.9],
        "footprint_wgs84": {"type": "Polygon", "coordinates": []},
        "corner_coordinates_wgs84": {
            "top_left": [116.2, 39.9],
            "top_right": [116.3, 39.9],
            "bottom_right": [116.3, 39.8],
            "bottom_left": [116.2, 39.8],
        },
        "patches": [
            {
                "patch_id": "patch_000000",
                "footprint_wgs84": {"type": "Polygon", "coordinates": []},
                "pixel_bounds": [0, 0, 128, 128],
                "source_date": "20260331",
            }
        ],
    }
    monkeypatch.setattr(
        regions,
        "build_mosaic_artifact",
        lambda **kwargs: (b"png", "image/png", metadata),
    )

    response = client.get(
        "/regions/haidian/mosaic",
        params={
            "date": "202603",
            "sensor_type": "s2",
            "format": "json",
            "patch_ids": "patch_000000",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["crs"] == "EPSG:4326"
    assert body["patches"][0]["patch_id"] == "patch_000000"
    assert "format=png" in body["image_url"]


def test_mosaic_rejects_unknown_region_before_building():
    response = client.get(
        "/regions/no_such_region/mosaic",
        params={"date": "202512", "sensor_type": "s2"},
    )
    assert response.status_code == 404
    assert "Region 'no_such_region' not found" in response.text


def test_mosaic_rejects_impossible_month():
    response = client.get(
        "/regions/haidian/mosaic",
        params={"date": "202513", "sensor_type": "s2"},
    )
    assert response.status_code == 422
    assert "Invalid date" in response.text


def test_mosaic_rejects_unknown_patch_explicitly():
    response = client.get(
        "/regions/haidian/mosaic",
        params={"date": "202512", "patch_ids": "patch_999999"},
    )
    assert response.status_code == 404
    assert "Patch 'patch_999999' not found" in response.text


def test_mosaic_openapi_documents_region_sensor_month_coverage():
    operation = client.get("/openapi.json").json()["paths"][
        "/regions/{region_id}/mosaic"
    ]["get"]
    description = operation["description"]

    assert "哈尔滨新区" in description
    assert "`s1_hr`" in description
    assert "`202506`、`202508`、`202509`、`202510`" in description
    assert "海淀区" in description
    assert "`highres_sar`" in description
    assert "`202512`～`202605`" in description
