"""Custom model training and inference route tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestModels:
    def test_list_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_model_requires_classes(self):
        response = client.post(
            "/models",
            json={
                "name": "test",
                "model_type": "classification",
                "task_type": "building_extraction",
                "region_id": "harbin",
            },
        )
        # Allowed to create with empty classes; training will fail later.
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test"
        assert data["status"] == "training"

    def test_create_model_invalid_type(self):
        response = client.post(
            "/models",
            json={
                "name": "test",
                "model_type": "unsupported",
                "task_type": "building_extraction",
                "region_id": "harbin",
            },
        )
        assert response.status_code == 422

    def test_get_model(self):
        r = client.post(
            "/models",
            json={
                "name": "test-get",
                "model_type": "classification",
                "task_type": "building_extraction",
                "region_id": "harbin",
            },
        )
        model_id = r.json()["id"]
        r = client.get(f"/models/{model_id}")
        assert r.status_code == 200
        assert r.json()["id"] == model_id

    def test_delete_model(self):
        r = client.post(
            "/models",
            json={
                "name": "test-del",
                "model_type": "classification",
                "task_type": "building_extraction",
                "region_id": "harbin",
            },
        )
        model_id = r.json()["id"]
        r = client.delete(f"/models/{model_id}")
        assert r.status_code == 200
        r = client.get(f"/models/{model_id}")
        assert r.status_code == 404

    def test_infer_untrained_model(self):
        r = client.post(
            "/models",
            json={
                "name": "test-inf",
                "model_type": "classification",
                "task_type": "building_extraction",
                "region_id": "harbin",
            },
        )
        model_id = r.json()["id"]
        r = client.post(
            f"/models/{model_id}/infer",
            json={"region_id": "harbin", "patch_id": "patch_000000", "month": "2025-04"},
        )
        # Training has no annotations and will fail; inference is rejected.
        assert r.status_code == 400

    def test_infer_batch_exceeds_limit(self):
        r = client.post(
            "/models",
            json={
                "name": "test-batch",
                "model_type": "classification",
                "task_type": "building_extraction",
                "region_id": "harbin",
            },
        )
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
