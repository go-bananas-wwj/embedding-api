"""Custom model training and inference route tests."""

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.models import _resolve_embedding_version


client = TestClient(app)


def _geojson_annotation(
    patch_id="patch_000000",
    cls_id="cls_001",
    month="2025-04",
    task_type="building_extraction",
):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "patch_id": patch_id,
                    "region_id": "harbin",
                    "class_id": cls_id,
                    "class_name": "建筑",
                    "color": "#FF0000",
                    "task_type": task_type,
                    "month": month,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [126.50, 45.74],
                            [126.52, 45.74],
                            [126.52, 45.76],
                            [126.50, 45.76],
                            [126.50, 45.74],
                        ]
                    ],
                },
            }
        ],
    }


def _model_payload(name="test-model", model_type="single_time_detection", task_type="building_extraction"):
    return {
        "name": name,
        "model_type": model_type,
        "region_id": "harbin",
        "annotations": _geojson_annotation(task_type=task_type),
        "classes": [{"id": "cls_001", "name": "建筑", "color": "#FF0000"}],
        "epochs": 1,
        "description": "test description",
    }


class TestModels:
    def test_list_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_model_requires_annotations(self):
        response = client.post(
            "/models",
            json={
                "name": "test",
                "model_type": "single_time_detection",
                "region_id": "harbin",
            },
        )
        assert response.status_code == 422

    def test_create_model_with_geojson(self):
        response = client.post("/models", json=_model_payload("test-geojson"))
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-geojson"
        assert data["status"] in {"training", "completed"}
        assert data["job_id"].startswith("job_")
        assert data["description"] == "test description"

    def test_create_model_infers_missing_feature_task_type(self):
        payload = _model_payload("test-infer-task")
        payload["annotations"]["features"][0]["properties"].pop("task_type")

        response = client.post("/models", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["task_type"] == "building_extraction"
        assert data["job_id"].startswith("job_")

    def test_create_model_accepts_legacy_classification_model_type(self):
        payload = _model_payload("test-legacy-model-type", model_type="classification")

        response = client.post("/models", json=payload)

        assert response.status_code == 200
        assert response.json()["type"] == "single_time_detection"

    def test_haidian_embedding_version_falls_back_to_available_v1(self):
        assert _resolve_embedding_version("haidian", "v2") == "v1"

    def test_create_model_invalid_type(self):
        payload = _model_payload("test-invalid")
        payload["model_type"] = "unsupported"
        response = client.post("/models", json=payload)
        assert response.status_code == 422

    def test_get_model(self):
        r = client.post("/models", json=_model_payload("test-get"))
        model_id = r.json()["id"]
        r = client.get(f"/models/{model_id}")
        assert r.status_code == 200
        assert r.json()["id"] == model_id

    def test_delete_model(self):
        r = client.post("/models", json=_model_payload("test-del"))
        model_id = r.json()["id"]
        r = client.delete(f"/models/{model_id}")
        assert r.status_code == 200
        r = client.get(f"/models/{model_id}")
        assert r.status_code == 404

    def test_infer_untrained_model(self):
        payload = _model_payload("test-inf")
        payload["annotations"]["features"][0]["properties"]["patch_id"] = "patch_999999"
        r = client.post("/models", json=payload)
        model_id = r.json()["id"]
        r = client.post(
            f"/models/{model_id}/infer",
            json={"region_id": "harbin", "patch_id": "patch_000000", "month": "2025-04"},
        )
        assert r.status_code == 400

    def test_infer_batch_exceeds_limit(self):
        r = client.post("/models", json=_model_payload("test-batch"))
        model_id = r.json()["id"]
        r = client.post(
            f"/models/{model_id}/infer_batch",
            json={
                "region_id": "harbin",
                "patch_ids": [f"patch_{i:06d}" for i in range(101)],
                "month": "2025-04",
            },
        )
        assert r.status_code == 422

    def test_job_status_for_missing_job(self):
        r = client.get("/models/jobs/job_nonexistent")
        assert r.status_code == 404

    def test_create_change_detection_model(self):
        payload = {
            "name": "test-cd",
            "model_type": "change_detection",
            "region_id": "harbin",
            "annotations": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "patch_id": "patch_000000",
                            "region_id": "harbin",
                            "class_id": "cls_001",
                            "class_name": "变化区域",
                            "color": "#FF0000",
                            "task_type": "change_detection",
                            "before_month": "2025-04",
                            "after_month": "2025-06",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [126.50, 45.74],
                                    [126.52, 45.74],
                                    [126.52, 45.76],
                                    [126.50, 45.76],
                                    [126.50, 45.74],
                                ]
                            ],
                        },
                    }
                ],
            },
            "classes": [{"id": "cls_001", "name": "变化区域", "color": "#FF0000"}],
            "epochs": 1,
        }
        r = client.post("/models", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in {"training", "completed"}
        assert data["job_id"].startswith("job_")
        assert data["type"] == "change_detection"

    def test_create_model_with_class_ids_subset(self):
        payload = _model_payload("test-subset")
        payload["classes"] = [
            {"id": "cls_001", "name": "建筑", "color": "#FF0000"},
            {"id": "cls_002", "name": "道路", "color": "#00FF00"},
        ]
        payload["class_ids"] = ["cls_001"]
        r = client.post("/models", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in {"training", "completed"}
        assert data["job_id"].startswith("job_")

    def test_create_model_rejects_unknown_class_ids(self):
        payload = _model_payload("test-unknown-class")
        payload["class_ids"] = ["cls_missing"]
        r = client.post("/models", json=payload)
        assert r.status_code == 422

    def test_create_model_rejects_mismatched_region_id(self):
        payload = _model_payload("test-region-mismatch")
        payload["annotations"]["features"][0]["properties"]["region_id"] = "haidian"
        r = client.post("/models", json=payload)
        assert r.status_code == 422

    def test_create_model_rejects_unknown_feature_class_id(self):
        payload = _model_payload("test-unknown-feature-class")
        payload["annotations"]["features"][0]["properties"]["class_id"] = "cls_missing"
        r = client.post("/models", json=payload)
        assert r.status_code == 422

    def test_list_models_includes_system_models_with_region(self):
        r = client.get("/models?region_id=harbin")
        assert r.status_code == 200
        data = r.json()
        system_ids = {m["id"] for m in data if m.get("source") == "system"}
        assert "building_extraction" in system_ids
        for m in data:
            if m.get("source") == "system":
                assert m["status"] == "ready"
                assert "versions" in m

    def test_get_system_model_via_models_endpoint(self):
        r = client.get("/models/building_extraction?region_id=harbin")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "building_extraction"
        assert data["source"] == "system"
        assert data["status"] == "ready"
        assert len(data["classes"]) > 0

    def test_infer_system_model_via_models_endpoint(self):
        r = client.post(
            "/models/building_extraction/infer",
            json={
                "region_id": "harbin",
                "patch_id": "patch_000000",
                "month": "2025-04",
                "version": "v2",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["result_url"].startswith("/system-models/results/")

        # Result image should be downloadable
        filename = data["result_url"].split("/")[-1]
        r2 = client.get(f"/system-models/results/{filename}")
        assert r2.status_code == 200
        assert r2.headers["content-type"] == "image/png"

    def test_infer_batch_system_model_via_models_endpoint(self):
        r = client.post(
            "/models/building_extraction/infer_batch",
            json={
                "region_id": "harbin",
                "patch_ids": ["patch_000000", "patch_000001"],
                "month": "2025-04",
                "version": "v2",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert data["success_count"] == 2
        assert data["error_count"] == 0
        for item in data["results"]:
            assert item["status"] == "success"
            assert item["result_url"].startswith("/system-models/results/")
