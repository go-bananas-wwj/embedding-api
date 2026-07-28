"""Custom model training and inference route tests."""

import time
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.models import _completion_metadata
from app.routers.models import _resolve_embedding_version
from app.services.model_registry import get_model_registry


client = TestClient(app)


def test_completion_metadata_does_not_duplicate_binding_fields():
    metadata = _completion_metadata(
        {
            "accuracy": 0.8,
            "n_samples": 4,
            "feature_source": "xuannv_embedding",
            "resolved_training_method": "prototype_pu_query",
        },
        {
            "feature_source": "P10C 64D embedding",
            "foundation_model_id": "p10c",
            "head_type": "prototype_pu_query",
        },
    )

    assert metadata["feature_source"] == "P10C 64D embedding"
    assert metadata["foundation_model_id"] == "p10c"
    assert metadata["resolved_training_method"] == "prototype_pu_query"


def test_persisted_job_response_backfills_checkpoint_binding(monkeypatch):
    monkeypatch.setattr("app.routers.models._training_jobs", {})
    monkeypatch.setattr(
        "app.routers.models.load_job",
        lambda _user, _job: {
            "job_id": "job_old",
            "user_id": "default",
            "status": "completed",
            "model_id": "model_old",
            "model_path": "users/default/models/model_old.pkl",
        },
    )
    monkeypatch.setattr(
        "app.routers.models.load_model_binding",
        lambda _path: {
            "foundation_model_id": "xuannv_earth",
            "foundation_model_version": "v2",
            "feature_dimension": 128,
            "preprocessing_version": "p10c_embedding_v2",
            "head_type": "pu_query_retrieval",
            "checkpoint_format": "pu_query_retrieval_v1",
            "compatible_regions": ["harbin"],
        },
    )

    response = client.get("/models/jobs/job_old")

    assert response.status_code == 200
    assert response.json()["foundation_model_id"] == "xuannv_earth"
    assert response.json()["compatible_regions"] == ["harbin"]


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
    def test_analysis_endpoint_is_disabled(self):
        response = client.post(
            "/models/model_test/analysis",
            json={
                "analysis_type": "single_time",
                "region_id": "haidian",
                "patch_ids": ["patch_000018"],
                "month": "202604",
            },
        )

        assert response.status_code == 404

    def test_training_capabilities_are_machine_readable(self):
        response = client.get("/models/capabilities?region_id=haidian")
        assert response.status_code == 200
        data = response.json()
        assert data["default_training_method"] == "xuannv_earth"
        methods = {item["id"]: item for item in data["methods"]}
        assert methods["traditional_ml"]["required_sensor"] == "s2"
        assert methods["traditional_ml"]["trainer"] == "random_forest"
        assert methods["aef"]["trainer"] == "pixel_mlp"
        assert methods["dinov3_sat493m"]["trainer"] == "pixel_mlp"

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
        assert data["requested_training_method"] == "xuannv_earth"
        assert data["feature_source"] == "xuannv_embedding"

    def test_missing_aef_assets_are_rejected_before_job_creation(self, monkeypatch):
        monkeypatch.setattr(
            "app.routers.models.aef_assets_available_for_region", lambda _region: False
        )
        payload = _model_payload("test-aef-unavailable")
        payload["training_method"] = "aef"
        response = client.post("/models", json=payload)
        assert response.status_code == 409
        assert "AEF" in response.text

    def test_traditional_ml_rejects_change_detection(self):
        payload = _model_payload("test-traditional-change")
        payload["training_method"] = "traditional_ml"
        payload["model_type"] = "change_detection"
        response = client.post("/models", json=payload)
        assert response.status_code == 422
        assert "single_time_detection" in response.text

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

    def test_create_model_trains_one_head_per_annotated_class(self):
        payload = _model_payload("test-multi-class-heads")
        payload["classes"].append({"id": "cls_002", "name": "道路", "color": "#00FF00"})
        payload["annotations"]["features"].append(
            {
                "type": "Feature",
                "properties": {
                    "patch_id": "patch_000000",
                    "region_id": "harbin",
                    "class_id": "cls_002",
                    "class_name": "道路",
                    "color": "#00FF00",
                    "task_type": "building_extraction",
                    "month": "2025-04",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [126.52, 45.74],
                            [126.53, 45.74],
                            [126.53, 45.75],
                            [126.52, 45.75],
                            [126.52, 45.74],
                        ]
                    ],
                },
            }
        )

        response = client.post("/models", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"training", "completed"}
        assert {item["id"] for item in data["classes"]} == {"cls_001", "cls_002"}
        model = client.get(f"/models/{data['id']}").json()
        assert model["status"] == "completed"
        assert model["head_type"] == "multi_binary_heads"
        assert {item["class_id"] for item in model["class_heads"]} == {
            "cls_001",
            "cls_002",
        }
        assert {
            item["training_strategy"] for item in model["class_heads"]
        } == {"pu_query_retrieval"}

        inference = client.post(
            f"/models/{data['id']}/infer",
            json={
                "region_id": "harbin",
                "patch_id": "patch_000000",
                "month": "2025-04",
            },
        )
        assert inference.status_code == 200

    def test_create_model_skips_class_without_annotations(self):
        payload = _model_payload("test-skip-unannotated-class")
        payload["classes"].append(
            {"id": "cls_002", "name": "道路", "color": "#00FF00"}
        )
        payload["class_ids"] = ["cls_001", "cls_002"]

        response = client.post("/models", json=payload)

        assert response.status_code == 200
        assert [item["id"] for item in response.json()["classes"]] == ["cls_001"]

    def test_create_model_selects_strategy_per_annotated_class(self):
        payload = _model_payload("test-per-class-training-strategy")
        payload["epochs"] = 1
        payload["classes"].append(
            {"id": "cls_002", "name": "道路", "color": "#00FF00"}
        )
        road_feature = {
            "type": "Feature",
            "properties": {
                "patch_id": "patch_000000",
                "region_id": "harbin",
                "class_id": "cls_002",
                "class_name": "道路",
                "color": "#00FF00",
                "task_type": "building_extraction",
                "month": "2025-04",
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
        payload["annotations"]["features"].extend(
            [road_feature for _ in range(10)]
        )

        response = client.post("/models", json=payload)

        assert response.status_code == 200
        model = client.get(f"/models/{response.json()['id']}").json()
        strategies = {
            item["class_id"]: item["training_strategy"]
            for item in model["class_heads"]
        }
        assert strategies == {
            "cls_001": "pu_query_retrieval",
            "cls_002": "binary_conv3x3",
        }

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

    def test_get_legacy_model_backfills_checkpoint_binding(self, monkeypatch, tmp_path):
        model_id = "model_legacy_binding"
        registry = get_model_registry("default")
        monkeypatch.setattr(
            registry,
            "get_model",
            lambda requested: {
                "id": model_id,
                "name": "legacy",
                "type": "single_time_detection",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00",
                "classes": [],
                "model_path": str(tmp_path / "legacy.pkl"),
                "compatible_regions": [],
            }
            if requested == model_id
            else None,
        )
        monkeypatch.setattr(
            "app.routers.models.get_model_registry", lambda _user_id: registry
        )
        monkeypatch.setattr(
            "app.routers.models.load_model_binding",
            lambda _path: {
                "foundation_model_id": "xuannv_earth",
                "foundation_model_version": "v2",
                "feature_source": "xuannv_embedding",
                "feature_dimension": 128,
                "preprocessing_version": "p10c_embedding_v2",
                "head_type": "pu_query_retrieval",
                "checkpoint_format": "pu_query_retrieval_v1",
                "compatible_regions": ["harbin"],
            },
        )

        body = client.get(f"/models/{model_id}").json()

        assert body["foundation_model_id"] == "xuannv_earth"
        assert body["feature_dimension"] == 128
        assert body["compatible_regions"] == ["harbin"]

    def test_rename_model_with_put(self):
        r = client.post("/models", json=_model_payload("test-rename-put"))
        model_id = r.json()["id"]

        r = client.put(f"/models/{model_id}", json={"name": "renamed-by-put"})

        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        r = client.get(f"/models/{model_id}")
        assert r.json()["name"] == "renamed-by-put"

    def test_rename_model_patch_still_supported(self):
        r = client.post("/models", json=_model_payload("test-rename-patch"))
        model_id = r.json()["id"]

        r = client.patch(f"/models/{model_id}", json={"name": "renamed-by-patch"})

        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        r = client.get(f"/models/{model_id}")
        assert r.json()["name"] == "renamed-by-patch"

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

    def test_get_haidian_system_model_omitted_version_uses_v1(self):
        r = client.get("/models/building_extraction?region_id=haidian")

        assert r.status_code == 200
        assert r.json()["foundation_model_version"] == "v1"

    def test_get_haidian_system_model_rejects_unavailable_explicit_version(self):
        r = client.get(
            "/models/building_extraction",
            params={"region_id": "haidian", "version": "v2"},
        )

        assert r.status_code == 404
        assert "Available versions: v1" in r.json()["detail"]

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
