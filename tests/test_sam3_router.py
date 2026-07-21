"""Tests for SAM3 router endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.services.data_service import DataNotFoundError, DataValidationError

client = TestClient(app)


class TestSAM3Embed:
    def test_embed_invalid_region(self):
        response = client.post("/regions/beijing/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })
        assert response.status_code == 404

    def test_embed_invalid_patch_id(self):
        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "../../etc/passwd",
            "month": "2025-10",
        })
        assert response.status_code == 422

    def test_embed_invalid_month(self):
        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "../etc/passwd",
        })
        assert response.status_code == 422

    @pytest.mark.parametrize("month", ["not-a-date", "2025-13", "20250230"])
    def test_embed_rejects_nonexistent_dates(self, month):
        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": month,
        })

        assert response.status_code == 422

    @patch("app.routers.sam3.SAM3Service.embed", new_callable=AsyncMock)
    def test_embed_missing_image_returns_404(self, mock_embed):
        mock_embed.side_effect = DataNotFoundError("raw image not found")

        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })

        assert response.status_code == 404
        assert response.json()["detail"] == "raw image not found"

    @patch("app.routers.sam3.SAM3Service.embed", new_callable=AsyncMock)
    def test_embed_invalid_georeference_returns_400(self, mock_embed):
        mock_embed.side_effect = DataValidationError("source image has no CRS")

        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })

        assert response.status_code == 400
        assert response.json()["detail"] == "source image has no CRS"


class TestSAM3Segment:
    def test_segment_invalid_region(self):
        response = client.post("/regions/beijing/sam3/segment", json={
            "date": "2025-10",
            "sensor_type": "s2",
            "point_coords": [[126.52, 45.75]],
        })
        assert response.status_code == 404

    def test_segment_rejects_legacy_month(self):
        response = client.post("/regions/harbin/sam3/segment", json={
            "date": "2025-10",
            "month": "2025-10",
            "sensor_type": "s2",
            "point_coords": [[126.52, 45.75]],
        })
        assert response.status_code == 422

    def test_segment_coords_invalid_wgs84(self):
        response = client.post("/regions/harbin/sam3/segment", json={
            "date": "2025-10",
            "sensor_type": "s2",
            "point_coords": [[181.0, 45.75]],
        })
        assert response.status_code == 422

    def test_segment_mismatched_labels(self):
        response = client.post("/regions/harbin/sam3/segment", json={
            "date": "2025-10",
            "sensor_type": "s2",
            "point_coords": [[126.52, 45.75], [126.521, 45.751]],
            "point_labels": [1],
        })
        assert response.status_code == 422

    @pytest.mark.parametrize("date", ["not-a-date", "2025-00", "20251301"])
    def test_segment_rejects_nonexistent_dates(self, date):
        response = client.post("/regions/harbin/sam3/segment", json={
            "date": date,
            "sensor_type": "s2",
            "point_coords": [[126.52, 45.75]],
        })

        assert response.status_code == 422


class TestSAM3Status:
    def test_status_invalid_region(self):
        response = client.get("/regions/beijing/sam3/status")
        assert response.status_code == 404

    def test_status_valid_region(self):
        response = client.get("/regions/harbin/sam3/status")
        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data
        assert "cache" in data
