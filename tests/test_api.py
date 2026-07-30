"""Basic API endpoint tests."""

import io

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.routers.embeddings import _default_embedding_version, _latest_available_month

client = TestClient(app)


def test_default_embedding_version_is_region_specific():
    assert _default_embedding_version("haidian") == "v1"
    assert _default_embedding_version("harbin") == "v2"


def test_latest_available_embedding_month_uses_last_month():
    assert _latest_available_month(["2025-10", "2026-01", "2026-05"]) == "2026-05"
    assert _latest_available_month([]) is None


def test_haidian_building_summary_ignores_legacy_change_period():
    response = client.get(
        "/regions/haidian/tasks/building_extraction/summary",
        params={"version": "v1", "period": "2025-04_vs_2025-06"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "building_extraction"
    assert data["total_patches"] == 320
    assert data["schema_version"] == "2.0"
    assert data["status"] == "ready"
    assert 30 <= len(data["summary_text"]) <= 400
    assert data["data_coverage"]["prediction_patches"] == 320
    assert data["data_coverage"]["coverage_rate"] == 1.0
    assert "quality_metrics" not in data
    assert data["color_legend"]
    assert all({"color", "name", "meaning"} <= set(item) for item in data["color_legend"])
    assert isinstance(data["insights"], list)
    assert isinstance(data["warnings"], list)


def test_summary_explains_partial_task_in_human_readable_text():
    response = client.get(
        "/regions/haidian/tasks/road_extraction/summary",
        params={"version": "v1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "partial"
    assert data["data_coverage"]["label_patches"] == 320
    assert data["data_coverage"]["prediction_patches"] == 0
    assert "0 个已有结果" in data["summary_text"]
    assert any(item["code"] == "PREDICTIONS_MISSING" for item in data["warnings"])


def test_task_summary_can_filter_multiple_patches():
    response = client.get(
        "/regions/haidian/tasks/building_extraction/summary",
        params=[
            ("patch_ids", "patch_000000"),
            ("patch_ids", "patch_000001"),
            ("month", "202512"),
        ],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["data_coverage"]["configured_patches"] == 2
    assert data["data_coverage"]["available_result_patches"] == 2
    assert data["analysis_scope"]["patch_ids"] == ["patch_000000", "patch_000001"]
    assert data["analysis_scope"]["month"] == "202512"


def test_task_summary_generates_missing_selected_patch_results(monkeypatch, tmp_path):
    generated = tmp_path / "road_extraction_haidian_patch_000010_202604.png"

    def fake_infer(region_id, task_type, patch_id, month, version, results_dir):
        assert (region_id, task_type, patch_id, month, version) == (
            "haidian", "road_extraction", "patch_000010", "202604", "v1"
        )
        Image.new("RGB", (128, 128), "white").save(generated)
        return generated

    monkeypatch.setattr("app.routers.tasks.infer_system_model", fake_infer)
    response = client.get(
        "/regions/haidian/tasks/road_extraction/summary",
        params={"version": "v1", "month": "202604", "patch_ids": "patch_000010"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["data_coverage"]["result_tiles"] >= 1
    assert data["data_coverage"]["coverage_rate"] == 1.0
    assert data["analysis_scope"]["generated_results"] == 1
    assert "quality_metrics" not in data
    assert data["color_legend"]
    assert not any(item["code"] == "PREDICTIONS_MISSING" for item in data["warnings"])


def test_task_summary_does_not_require_prebuilt_summary_file(monkeypatch, tmp_path):
    generated = tmp_path / "water_extraction_haidian_patch_000010_202604.png"

    def fake_infer(*args, **kwargs):
        Image.new("RGB", (128, 128), "black").save(generated)
        return generated

    monkeypatch.setattr("app.routers.tasks.infer_system_model", fake_infer)
    monkeypatch.setattr("app.routers.tasks.DataService.load_task_summary", lambda *args: None)
    response = client.get(
        "/regions/haidian/tasks/water_extraction/summary",
        params={"version": "v1", "month": "202604", "patch_ids": "patch_000010"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "water_extraction"
    assert data["status"] == "ready"
    assert data["data_coverage"]["result_tiles"] >= 1
    assert data["prediction_statistics"]["mean_positive_pixel_ratio"] > 0
    assert {item["name"] for item in data["color_legend"]} >= {"背景", "水体"}
    assert "颜色说明" in data["summary_text"]
    assert data["image_analysis"]["image_count"] == 1
    assert data["image_analysis"]["total_pixels"] == 128 * 128
    assert data["image_analysis"]["target_pixels"] == 5
    assert data["result_images"][0]["cleanup_interval_seconds"] == 7200
    image_url = data["result_images"][0]["image_url"]
    assert image_url.startswith("http://60.31.21.42:22065/task-summary/results/")
    assert client.get(image_url.removeprefix("http://60.31.21.42:22065")).status_code == 200


def test_task_summary_default_version_falls_back_to_configured_task_assets():
    response = client.get(
        "/regions/harbin/tasks/road_extraction/summary",
        params={"month": "202510", "patch_ids": "patch_000010"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1"
    assert data["status"] == "ready"
    assert data["data_coverage"]["coverage_rate"] == 1.0


def test_change_summary_default_version_uses_configured_result_assets():
    response = client.get(
        "/regions/harbin/change-detection/summary",
        params=[
            ("before_month", "202504"),
            ("after_month", "202506"),
            ("patch_ids", "patch_000404"),
            ("patch_ids", "patch_000402"),
        ],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1"
    assert data["status"] == "ready"
    assert data["data_coverage"]["coverage_rate"] == 1.0


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
        for region in data["regions"]:
            mosaic = region["mosaic"]
            assert mosaic["crs"] == "EPSG:4326"
            assert len(mosaic["bounds_wgs84"]) == 4
            assert mosaic["footprint_wgs84"]["type"] in {"Polygon", "MultiPolygon"}
            assert set(mosaic["corner_coordinates_wgs84"]) == {
                "top_left",
                "top_right",
                "bottom_right",
                "bottom_left",
            }
            assert mosaic["image_format"] == "png"
            assert mosaic["transparent_background"] is True
            assert mosaic["package_filename"] == "regional-mosaics.zip"
            assert mosaic["assets"]
            for asset in mosaic["assets"]:
                assert asset["path_template"] == (
                    f"{region['id']}/{asset['sensor_type']}/{{date}}/mosaic.png"
                )
                assert asset["start_date"] == min(asset["available_dates"])
                assert asset["end_date"] == max(asset["available_dates"])
                assert asset["date_count"] == len(asset["available_dates"])

    def test_get_region_harbin(self):
        response = client.get("/regions/harbin")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "harbin"
        assert data["patch_count"] == 424
        assert "tasks" in data
        assert "embeddings" in data
        assert data["mosaic"]["assets"]

    def test_get_region_haidian(self):
        response = client.get("/regions/haidian")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "haidian"
        assert data["patch_count"] == 320
        assert data["mosaic"]["bounds_wgs84"]

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
        assert data["footprint_wgs84"]["type"] == "Polygon"
        assert len(data["footprint_wgs84"]["coordinates"][0]) == 5

    def test_patch_footprints_share_projected_edges(self):
        left = client.get("/regions/harbin/patches/patch_000000").json()
        right = client.get("/regions/harbin/patches/patch_000001").json()

        # WGS84 envelopes are not exact patch footprints for a projected grid.
        bbox_gap = right["bounds_wgs84"][0] - left["bounds_wgs84"][2]
        assert bbox_gap > 0

        left_ring = left["footprint_wgs84"]["coordinates"][0]
        right_ring = right["footprint_wgs84"]["coordinates"][0]
        left_right_edge = [left_ring[1], left_ring[2]]
        right_left_edge = [right_ring[0], right_ring[3]]
        assert left_right_edge == right_left_edge

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
        assert "building_extraction" in task_ids
        assert "road_extraction" in task_ids
        assert "construction" in task_ids
        assert "land_use_classification" in task_ids
        assert "land_cover_classification" in task_ids
        assert "water_extraction" in task_ids
        assert "change_detection" in task_ids
        assert "construction_joint" not in task_ids

    def test_haidian_change_detection_result_is_generated_on_demand(self):
        response = client.get(
            "/regions/haidian/patches/patch_000010/tasks/change_detection/result",
            params={
                "format": "npy",
                "before_month": "202512",
                "after_month": "202604",
            },
        )

        assert response.status_code == 200
        prediction = np.load(io.BytesIO(response.content), allow_pickle=False)
        assert prediction.shape == (128, 128)
        assert prediction.dtype == np.uint8
        assert set(np.unique(prediction)).issubset({0, 1})
        assert response.headers["x-change-algorithm"] == (
            "p10c-cosine-bidirectional-5x5-mean"
        )

    def test_haidian_change_detection_png_documents_threshold(self):
        response = client.get(
            "/regions/haidian/patches/patch_000010/tasks/change_detection/result",
            params={
                "format": "png",
                "before_month": "2025-12",
                "after_month": "2026-04",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-change-threshold"] == "0.771541"
        image = Image.open(io.BytesIO(response.content))
        assert image.size == (128, 128)

    def test_haidian_change_detection_requires_both_months(self):
        response = client.get(
            "/regions/haidian/patches/patch_000010/tasks/change_detection/result",
            params={"before_month": "202512"},
        )

        assert response.status_code == 422
        assert "before_month" in response.json()["detail"]
        assert "after_month" in response.json()["detail"]

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

    def test_result_default_version_uses_harbin_road_assets(self):
        response = client.get(
            "/regions/harbin/patches/patch_000010/tasks/road_extraction/result",
            params={"format": "png", "month": "202510"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_result_default_version_uses_change_detection_assets(self):
        response = client.get(
            "/regions/harbin/patches/patch_000404/tasks/change_detection/result",
            params={"format": "png", "before_month": "202504", "after_month": "202506"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_result_can_generate_npy_for_binary_system_head(self):
        response = client.get(
            "/regions/haidian/patches/patch_000010/tasks/water_extraction/result",
            params={"format": "npy", "month": "202604"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        array = np.load(io.BytesIO(response.content), allow_pickle=False)
        assert array.shape == (128, 128)
        assert set(np.unique(array)).issubset({0, 1})
        assert int(array.sum()) == 5

    def test_get_task_prediction(self):
        response = client.get("/regions/harbin/patches/patch_000000/tasks/building_extraction/prediction")
        # May be 200 or 404 depending on data availability
        assert response.status_code in (200, 404)

    def test_haidian_road_prediction_is_reconstructed_from_monthly_result(self):
        response = client.get(
            "/regions/haidian/patches/patch_000000/tasks/road_extraction/prediction",
            params={"version": "v1", "period": "202604"},
        )

        assert response.status_code == 200
        prediction = np.load(io.BytesIO(response.content), allow_pickle=False)
        assert prediction.shape == (128, 128)
        assert prediction.dtype == np.uint8
        assert set(np.unique(prediction)).issubset({0, 1})

    def test_get_task_label(self):
        response = client.get("/regions/harbin/patches/patch_000000/tasks/building_extraction/label")
        # May be 200 or 404 depending on data availability
        assert response.status_code in (200, 404)

    def test_get_tile_not_implemented(self):
        response = client.get("/regions/harbin/tasks/building_extraction/tiles/10/100/100.png")
        assert response.status_code == 501

    def test_haidian_road_patch_tile_uses_on_demand_system_result(self):
        response = client.get(
            "/regions/haidian/tasks/road_extraction/tiles/patch_000000.png",
            params={"version": "v1", "period": "202512"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(response.content)) as image:
            assert image.size == (128, 128)


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
        """Invalid version values should be rejected safely."""
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
            assert response.status_code == 422, f"Failed for version={version}"
