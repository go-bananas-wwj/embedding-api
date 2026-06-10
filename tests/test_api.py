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
        assert "construction" in task_ids
        assert "building_change" in task_ids
        assert "change_detection" in task_ids

    def test_list_tasks_haidian(self):
        response = client.get("/regions/haidian/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["tasks"] == []

    def test_get_task_summary(self):
        response = client.get("/regions/harbin/tasks/construction/summary?version=v1")
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "construction"
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
        response = client.get("/regions/harbin/patches/patch_000000/tasks/construction/result?format=png")
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
            "/regions/harbin/patches/patch_000000/tasks/construction/result?format=invalid"
        )
        assert response.status_code == 422

    def test_get_task_prediction(self):
        response = client.get("/regions/harbin/patches/patch_000000/tasks/construction/prediction")
        # May be 200 or 404 depending on data availability
        assert response.status_code in (200, 404)

    def test_get_task_label(self):
        response = client.get("/regions/harbin/patches/patch_000000/tasks/construction/label")
        # May be 200 or 404 depending on data availability
        assert response.status_code in (200, 404)

    def test_get_tile_not_implemented(self):
        response = client.get("/regions/harbin/tasks/construction/tiles/10/100/100.png")
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
