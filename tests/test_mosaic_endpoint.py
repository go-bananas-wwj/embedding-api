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
    monkeypatch.setattr(regions, "build_mosaic", lambda **kwargs: (expected, "image/png"))

    response = client.get(
        "/regions/haidian/mosaic",
        params={"date": "202512", "sensor_type": "s2", "format": "png"},
    )

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.content) > 8


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
