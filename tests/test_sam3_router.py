"""Tests for SAM3 router endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

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


class TestSAM3Segment:
    def test_segment_invalid_region(self):
        response = client.post("/regions/beijing/sam3/segment", json={
            "embedding_id": "test",
            "point_coords": [[0.5, 0.5]],
            "point_labels": [1],
        })
        assert response.status_code == 404

    def test_segment_coords_out_of_range(self):
        response = client.post("/regions/harbin/sam3/segment", json={
            "embedding_id": "test",
            "point_coords": [[1.5, 0.5]],
            "point_labels": [1],
        })
        assert response.status_code == 422

    def test_segment_mismatched_labels(self):
        response = client.post("/regions/harbin/sam3/segment", json={
            "embedding_id": "test",
            "point_coords": [[0.5, 0.5], [0.3, 0.3]],
            "point_labels": [1],
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
