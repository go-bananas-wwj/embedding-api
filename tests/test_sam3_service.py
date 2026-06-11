"""Unit tests for SAM3Service."""

import pytest
from unittest.mock import MagicMock, patch

from app.services.sam3_service import SAM3Service


@pytest.fixture(autouse=True)
def reset_service():
    """Reset singleton between tests."""
    SAM3Service._instance = None
    yield
    SAM3Service._instance = None


class TestSAM3ServiceStatus:
    def test_status_not_loaded(self):
        svc = SAM3Service()
        status = svc.get_status()
        assert status["model_loaded"] is False
        assert status["device"] == "not_loaded"
        assert status["cache"]["size"] == 0


class TestSAM3ServiceCache:
    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_s2_image")
    @patch("app.services.sam3_service.get_config")
    def test_embed_caches_result(self, mock_get_config, mock_load_img, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        mock_img = MagicMock()
        mock_load_img.return_value = mock_img

        svc = SAM3Service()
        svc._processor = MagicMock()
        state = {"original_height": 256, "original_width": 256}
        svc._processor.set_image.return_value = state

        import asyncio
        result = asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))

        assert result["embedding_id"] == "harbin_patch_000_2025-10"
        assert "image" in result
        assert "data" in result["image"]
        assert len(svc._cache) == 1

    @patch.object(SAM3Service, "_ensure_model")
    @patch("app.services.sam3_service.get_config")
    def test_segment_missing_embedding(self, mock_get_config, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        svc = SAM3Service()

        import asyncio
        with pytest.raises(ValueError, match="Embedding"):
            asyncio.run(svc.segment("missing_id", [[0.5, 0.5]], [1]))

    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_s2_image")
    @patch("app.services.sam3_service.get_config")
    def test_lru_eviction(self, mock_get_config, mock_load_img, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        mock_img = MagicMock()
        mock_load_img.return_value = mock_img

        svc = SAM3Service()
        svc._processor = MagicMock()
        state = {"original_height": 256, "original_width": 256}
        svc._processor.set_image.return_value = state

        import asyncio
        asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_001", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_002", "2025-10"))

        assert len(svc._cache) == 2
        assert "harbin_patch_000_2025-10" not in svc._cache
