"""Integration tests for SAM3 with real model loading.

These tests require GPU and the full SAM3 model (3.45GB).
Run with: pytest tests/test_sam3_integration.py -v -m slow
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.slow

client = TestClient(app)


class TestSAM3RealModel:
    @pytest.fixture(scope="class", autouse=True)
    def reset_singleton(self):
        from app.services.sam3_service import SAM3Service
        SAM3Service._instance = None
        yield
        SAM3Service._instance = None

    def test_status_before_load(self):
        response = client.get("/regions/harbin/sam3/status")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False

    def test_embed_real(self):
        """Test embed with real model (loads ~4GB to GPU)."""
        response = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })
        # May be 200 (success) or 404 (S2 image not found) or 503 (GPU OOM)
        assert response.status_code in (200, 404, 503)
        if response.status_code == 200:
            data = response.json()
            assert "embedding_id" in data
            assert "image" in data
            assert data["image"]["width"] == 256
            assert data["image"]["height"] == 256

    def test_segment_real(self):
        """Test segment after embed."""
        # First embed
        embed_resp = client.post("/regions/harbin/sam3/embed", json={
            "patch_id": "patch_000000",
            "month": "2025-10",
        })
        if embed_resp.status_code != 200:
            pytest.skip("Embed failed, skipping segment test")

        embedding_id = embed_resp.json()["embedding_id"]

        # Then segment
        response = client.post("/regions/harbin/sam3/segment", json={
            "embedding_id": embedding_id,
            "point_coords": [[0.5, 0.5]],
            "point_labels": [1],
            "multimask_output": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "masks" in data
        assert len(data["masks"]) > 0
        for mask in data["masks"]:
            assert "data" in mask
            assert "score" in mask
            assert "bbox" in mask

    def test_status_after_load(self):
        response = client.get("/regions/harbin/sam3/status")
        assert response.status_code == 200
        data = response.json()
        # Model may or may not be loaded depending on prior tests
        assert "gpu_memory" in data
        assert "cache" in data
