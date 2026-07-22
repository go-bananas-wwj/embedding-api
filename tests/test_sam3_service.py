"""Unit tests for SAM3Service."""

import asyncio

import numpy as np
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


class TestSAM3HighresCandidateSelection:
    def test_rejects_near_full_frame_when_plausible_candidate_exists(self):
        full = np.ones((10, 10), dtype=bool)
        useful = np.zeros((10, 10), dtype=bool)
        useful[3:6, 4:7] = True

        selected = SAM3Service._select_single_highres_prediction(
            [(full, 0.95, [0, 0, 10, 10]), (useful, 0.75, [4, 3, 3, 3])]
        )

        assert len(selected) == 1
        assert selected[0][2] == [4, 3, 3, 3]

    def test_uses_highest_score_among_plausible_candidates(self):
        first = np.eye(10, dtype=bool)
        second = np.fliplr(first)
        selected = SAM3Service._select_single_highres_prediction(
            [(first, 0.4, [0, 0, 10, 10]), (second, 0.8, [0, 0, 10, 10])]
        )
        assert selected[0][1] == 0.8

    def test_prefers_local_instance_over_large_low_quality_region(self):
        local = np.zeros((100, 100), dtype=bool)
        local[48:53, 48:53] = True
        broad = np.zeros((100, 100), dtype=bool)
        broad[20:80, 20:80] = True
        selected = SAM3Service._select_single_highres_prediction(
            [(local, 0.03, [48, 48, 5, 5]), (broad, 0.11, [20, 20, 60, 60])]
        )
        assert selected[0][2] == [48, 48, 5, 5]


class TestSAM3ServiceCache:
    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_geo_image")
    @patch("app.services.sam3_service.get_config")
    def test_embed_caches_result(self, mock_get_config, mock_load_img, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        mock_img = MagicMock()
        mock_load_img.return_value = (
            mock_img,
            {"sam_width": 256, "sam_height": 256},
        )

        svc = SAM3Service()
        svc._processor = MagicMock()
        state = {"original_height": 256, "original_width": 256}
        svc._processor.set_image.return_value = state

        result = asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))

        assert result["embedding_id"] == "harbin_patch_000_s2_202510"
        assert "image" in result
        assert "data" in result["image"]
        assert len(svc._cache) == 1

    @patch.object(SAM3Service, "_ensure_model")
    @patch("app.services.sam3_service.get_config")
    def test_segment_missing_embedding(self, mock_get_config, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        svc = SAM3Service()

        with pytest.raises(ValueError, match="Embedding"):
            asyncio.run(svc._predict_from_cache("missing_id", [[0.5, 0.5]], [1]))

    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_geo_image")
    @patch("app.services.sam3_service.get_config")
    def test_lru_eviction(self, mock_get_config, mock_load_img, mock_ensure):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        mock_img = MagicMock()
        mock_load_img.return_value = (
            mock_img,
            {"sam_width": 256, "sam_height": 256},
        )

        svc = SAM3Service()
        svc._processor = MagicMock()
        state = {"original_height": 256, "original_width": 256}
        svc._processor.set_image.return_value = state

        asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_001", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_002", "2025-10"))

        assert len(svc._cache) == 2
        assert "harbin_patch_000_s2_202510" not in svc._cache

    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_geo_image")
    @patch("app.services.sam3_service.get_config")
    def test_cache_hit_refreshes_lru_order(
        self, mock_get_config, mock_load_img, mock_ensure
    ):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        image = MagicMock()
        mock_load_img.return_value = (
            image,
            {"sam_width": 8, "sam_height": 8},
        )
        svc = SAM3Service()
        svc._processor = MagicMock()
        svc._processor.set_image.return_value = {
            "original_height": 8,
            "original_width": 8,
        }
        svc._model = MagicMock()
        svc._model.predict_inst.return_value = (
            np.zeros((1, 8, 8), dtype=bool),
            np.array([0.9], dtype=np.float32),
            None,
        )

        asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_001", "2025-10"))
        asyncio.run(
            svc._predict_from_cache(
                "harbin_patch_000_s2_202510", [[0.5, 0.5]], [1]
            )
        )
        asyncio.run(svc.embed("harbin", "patch_002", "2025-10"))

        assert "harbin_patch_000_s2_202510" in svc._cache
        assert "harbin_patch_001_s2_202510" not in svc._cache

    @patch.object(SAM3Service, "_ensure_model")
    @patch.object(SAM3Service, "_load_geo_image")
    @patch("app.services.sam3_service.get_config")
    def test_reembedding_replaces_existing_state(
        self, mock_get_config, mock_load_img, mock_ensure
    ):
        mock_get_config.return_value.get_sam3_config.return_value = {"max_cache_size": 2}
        mock_load_img.return_value = (
            MagicMock(),
            {"sam_width": 8, "sam_height": 8},
        )
        first_state = {
            "original_height": 8,
            "original_width": 8,
            "feature": MagicMock(),
        }
        second_state = {"original_height": 8, "original_width": 8}
        svc = SAM3Service()
        svc._processor = MagicMock()
        svc._processor.set_image.side_effect = [first_state, second_state]

        asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))
        asyncio.run(svc.embed("harbin", "patch_000", "2025-10"))

        assert len(svc._cache) == 1
        assert first_state == {}
        assert svc._cache["harbin_patch_000_s2_202510"]["state"] is second_state


class TestSAM3DeviceHandling:
    @patch("torch.autocast")
    def test_indexed_cuda_device_uses_cuda_device_type(self, mock_autocast):
        svc = SAM3Service()
        svc._device = "cuda:4"

        svc._autocast_context()

        mock_autocast.assert_called_once()
        assert mock_autocast.call_args.kwargs["device_type"] == "cuda"


class TestSAM3HighResolutionAssets:
    @patch("app.services.sam3_service.get_config")
    def test_flat_highres_archive_selects_latest_month_scene(
        self, mock_get_config, tmp_path
    ):
        older = tmp_path / "highres_optical_20260301_patch_000212.tif"
        latest = tmp_path / "highres_optical_20260317_patch_000212.tif"
        older.touch()
        latest.touch()
        config = MagicMock()
        config.get_region.return_value = {"highres_dir": str(tmp_path)}
        mock_get_config.return_value = config

        path = SAM3Service()._resolve_image_path(
            "haidian", "patch_000212", "202603", "highres"
        )

        assert path == str(latest)

    @patch("app.services.sam3_service.get_config")
    def test_flat_highres_archive_honors_exact_day(self, mock_get_config, tmp_path):
        selected = tmp_path / "highres_optical_20260301_patch_000212.tif"
        other = tmp_path / "highres_optical_20260317_patch_000212.tif"
        selected.touch()
        other.touch()
        config = MagicMock()
        config.get_region.return_value = {"highres_dir": str(tmp_path)}
        mock_get_config.return_value = config

        path = SAM3Service()._resolve_image_path(
            "haidian", "patch_000212", "20260301", "highres"
        )

        assert path == str(selected)
