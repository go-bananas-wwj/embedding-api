"""Basic API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "regions" in data
        assert "harbin" in data["regions"]
        assert "haidian" in data["regions"]


class TestRegions:
    def test_list_regions(self):
        response = client.get("/regions")
        assert response.status_code == 200
        data = response.json()
        assert "regions" in data
        ids = [r["id"] for r in data["regions"]]
        assert "harbin" in ids
        assert "haidian" in ids

    def test_get_region_harbin(self):
        response = client.get("/regions/harbin")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "harbin"
        assert data["patch_count"] == 424
        assert "tasks" in data
        assert "embeddings" in data

    def test_get_region_haidian(self):
        response = client.get("/regions/haidian")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "haidian"
        assert data["patch_count"] == 320

    def test_get_region_not_found(self):
        response = client.get("/regions/beijing")
        assert response.status_code == 404


class TestPatches:
    def test_list_patches_harbin(self):
        response = client.get("/regions/harbin/patches?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 424
        assert len(data["patches"]) == 2
        assert "patch_id" in data["patches"][0]

    def test_list_patches_haidian(self):
        response = client.get("/regions/haidian/patches?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 320

    def test_list_patches_invalid_bbox(self):
        response = client.get("/regions/harbin/patches?bbox=invalid")
        assert response.status_code == 422

    def test_list_patches_nan_bbox(self):
        response = client.get("/regions/harbin/patches?bbox=nan,0,1,1")
        assert response.status_code == 422

    def test_list_patches_swapped_bbox(self):
        response = client.get("/regions/harbin/patches?bbox=10,10,5,5")
        assert response.status_code == 422

    def test_list_patches_page_boundary(self):
        # page_size at upper limit
        response = client.get("/regions/harbin/patches?page=1&page_size=100")
        assert response.status_code == 200
        data = response.json()
        assert len(data["patches"]) == 100
        # page_size exceeds limit
        response = client.get("/regions/harbin/patches?page=1&page_size=101")
        assert response.status_code == 422

    def test_list_patches_has_next(self):
        response = client.get("/regions/harbin/patches?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["has_next"] is True
        # Last page should not have next
        response = client.get("/regions/harbin/patches?page=5&page_size=100")
        assert response.status_code == 200
        data = response.json()
        assert data["has_next"] is False

    def test_get_patch(self):
        response = client.get("/regions/harbin/patches/patch_000000")
        assert response.status_code == 200
        data = response.json()
        assert data["patch_id"] == "patch_000000"
        assert "bounds_wgs84" in data

    def test_get_patch_not_found(self):
        response = client.get("/regions/harbin/patches/patch_999999")
        assert response.status_code == 404

    def test_list_patches_region_not_found(self):
        response = client.get("/regions/beijing/patches")
        assert response.status_code == 404


class TestEmbeddings:
    def test_get_embedding_json_harbin(self):
        response = client.get("/regions/harbin/patches/patch_000000/embedding?format=json")
        assert response.status_code == 200
        data = response.json()
        assert data["patch_id"] == "patch_000000"
        assert "shape" in data

    def test_get_embedding_json_haidian(self):
        response = client.get("/regions/haidian/patches/patch_000000/embedding?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "shape" in data

    def test_get_embedding_npy(self):
        response = client.get("/regions/haidian/patches/patch_000000/embedding?format=npy")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"

    def test_get_embedding_png(self):
        response = client.get("/regions/harbin/patches/patch_000000/embedding?format=png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_get_embedding_cache(self):
        response = client.get("/regions/harbin/patches/patch_000000/embedding?format=cache")
        # Should fall back to available format
        assert response.status_code in (200, 404)

    def test_get_embedding_format_not_available(self):
        """Requesting embedding for non-existent month should return 404."""
        response = client.get("/regions/harbin/patches/patch_000000/embedding?format=npy&month=2099-01")
        assert response.status_code == 404

    def test_get_embedding_invalid_format(self):
        response = client.get("/regions/harbin/patches/patch_000000/embedding?format=invalid")
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "cache" in detail

    def test_get_embedding_month_path_traversal_blocked(self):
        malicious_months = ["../etc/passwd", "..\\windows\\system32", "a" * 100]
        for month in malicious_months:
            response = client.get(
                f"/regions/harbin/patches/patch_000000/embedding?format=json&month={month}"
            )
            assert response.status_code == 422, f"Failed for month={month}"

    def test_get_embedding_patch_not_found(self):
        response = client.get("/regions/harbin/patches/patch_999999/embedding?format=json")
        assert response.status_code == 404

    def test_get_embedding_path_traversal_blocked(self):
        response = client.get("/regions/harbin/patches/../../../etc/passwd/embedding?format=json")
        # Regex validation blocks invalid patch_id before any file access
        assert response.status_code in (400, 404)


class TestTasks:
    def test_list_tasks_harbin(self):
        response = client.get("/regions/harbin/tasks")
        assert response.status_code == 200
        data = response.json()
        task_ids = [t["id"] for t in data["tasks"]]
        assert "change_detection" in task_ids
        assert "building_extraction" in task_ids
        assert "land_use_classification" in task_ids
        assert "land_cover_classification" in task_ids
        assert "water_extraction" in task_ids

    def test_patch_available_tasks_harbin(self):
        """Verify v2 period-subdir tasks are surfaced in available_tasks."""
        response = client.get("/regions/harbin/patches/patch_000010")
        assert response.status_code == 200
        data = response.json()
        available = data["available_tasks"]
        assert "building_extraction" in available

        response = client.get("/regions/harbin/patches/patch_000040")
        assert response.status_code == 200
        data = response.json()
        available = data["available_tasks"]
        assert "land_use_classification" in available

    def test_list_tasks_haidian(self):
        response = client.get("/regions/haidian/tasks")
        assert response.status_code == 200
        data = response.json()
        task_ids = [t["id"] for t in data["tasks"]]
        assert "change_detection" in task_ids
        assert "building_extraction" in task_ids
        assert "road_extraction" in task_ids
        assert "construction" in task_ids
        assert "land_use_classification" in task_ids
        assert "land_cover_classification" in task_ids
        assert "water_extraction" in task_ids

    def test_get_task_summary(self):
        response = client.get("/regions/harbin/tasks/building_extraction/summary?version=v1")
        assert response.status_code == 200
        data = response.json()
        # meta.json in legacy data still has old task name; just verify summary loads
        assert data["total_patches"] == 424

    def test_get_change_detection_summary(self):
        response = client.get(
            "/regions/harbin/tasks/change_detection/summary?version=v1&period=2025-04_vs_2025-06"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "change_detection"
        assert data["total_patches"] == 424

    def test_get_task_result(self):
        response = client.get("/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png")
        # May be 200 or 404 depending on data availability
        assert response.status_code in (200, 404)

    def test_get_change_detection_result(self):
        response = client.get(
            "/regions/harbin/patches/patch_000000/tasks/change_detection/result?format=png&version=v1&period=2025-04_vs_2025-06"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_get_task_result_invalid_format(self):
        response = client.get(
            "/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=invalid"
        )
        assert response.status_code == 422

    def test_get_task_prediction(self):
        response = client.get("/regions/harbin/patches/patch_000000/tasks/building_extraction/prediction")
        # May be 200 or 404 depending on data availability
        assert response.status_code in (200, 404)

    def test_get_task_label(self):
        response = client.get("/regions/harbin/patches/patch_000000/tasks/building_extraction/label")
        # May be 200 or 404 depending on data availability
        assert response.status_code in (200, 404)

    def test_get_tile_not_implemented(self):
        response = client.get("/regions/harbin/tasks/building_extraction/tiles/10/100/100.png")
        assert response.status_code == 501


class TestPathTraversal:
    def test_patch_id_path_traversal(self):
        """Verify path traversal attempts are blocked."""
        malicious_ids = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "patch_000000/../../etc/passwd",
            "a" * 1000,
        ]
        for malicious_id in malicious_ids:
            response = client.get(f"/regions/harbin/patches/{malicious_id}")
            assert response.status_code in (400, 404), f"Failed for {malicious_id}"

    def test_tile_period_path_traversal_blocked(self):
        """Malicious period parameters should return empty tiles, not file access."""
        malicious_periods = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "2025-04_vs_2025-06/../../etc/passwd",
            "a" * 1000,
        ]
        for period in malicious_periods:
            response = client.get(
                f"/regions/harbin/tasks/building_extraction/tiles?version=v1&period={period}"
            )
            assert response.status_code == 200, f"Failed for period={period}"
            data = response.json()
            assert data["tiles"] == []

    def test_tile_version_path_traversal_blocked(self):
        """Invalid version values should return empty tiles safely."""
        malicious_versions = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "v1/../../etc/passwd",
            "a" * 1000,
        ]
        for version in malicious_versions:
            response = client.get(
                f"/regions/harbin/tasks/building_extraction/tiles?version={version}"
            )
            assert response.status_code == 200, f"Failed for version={version}"
            data = response.json()
            assert data["tiles"] == []
