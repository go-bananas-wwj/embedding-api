"""Integration tests for cross-region date/period format robustness."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestCrossRegionTimeFormats:
    """Verify that both YYYY-MM and YYYYMM forms work for both regions."""

    def test_haidian_building_extraction_accepts_hyphen_month(self):
        response = client.get(
            "/regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&month=2026-01"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_haidian_road_extraction_accepts_ymd_month(self):
        response = client.get(
            "/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&month=20260115"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_harbin_building_extraction_accepts_compact_month(self):
        response = client.get(
            "/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&month=202510&version=v1"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_harbin_change_detection_accepts_mixed_period(self):
        response = client.get(
            "/regions/harbin/patches/patch_000000/tasks/change_detection/result?format=png&period=202504_vs_2025-06&version=v1"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_haidian_embedding_accepts_hyphen_month(self):
        response = client.get(
            "/regions/haidian/patches/patch_000000/embedding?format=png&month=2026-01"
        )
        # PNG may not exist for this exact month, but normalization should not crash.
        assert response.status_code in (200, 404)
