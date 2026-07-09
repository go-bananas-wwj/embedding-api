"""System pre-trained model route tests."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestSystemModels:
    def test_list_system_models_harbin(self):
        response = client.get("/system-models?region_id=harbin")
        assert response.status_code == 200
        data = response.json()
        task_ids = {m["id"] for m in data}
        assert "land_cover_classification" in task_ids
        assert "water_extraction" in task_ids
        assert "building_extraction" in task_ids
        assert all(m["versions"] for m in data)

    def test_list_system_models_haidian_excludes_unavailable_models(self):
        response = client.get("/system-models?region_id=haidian")
        assert response.status_code == 200
        data = response.json()
        task_ids = {m["id"] for m in data}
        assert task_ids == {
            "water_extraction",
            "building_extraction",
            "road_extraction",
        }
        assert all(m["versions"] for m in data)

    def test_get_system_model_classes(self):
        response = client.get(
            "/system-models/land_cover_classification/classes?region_id=harbin&version=v2"
        )
        # 200 if checkpoint exists, 404 if model file missing
        assert response.status_code in (200, 404)

    def test_haidian_system_model_omitted_version_uses_v1(self):
        response = client.get(
            "/system-models/road_extraction/classes?region_id=haidian"
        )
        assert response.status_code == 200
        assert response.json()[1]["name"] == "道路"

    def test_haidian_system_model_infer_omitted_version_uses_v1(self):
        response = client.post(
            "/system-models/road_extraction/infer"
            "?region_id=haidian&patch_id=patch_000000&month=202512"
        )
        assert response.status_code == 200
        assert response.json()["result_url"].endswith(
            "road_extraction_haidian_patch_000000_202512.png"
        )

    def test_infer_system_model_missing_embedding(self):
        response = client.post(
            "/system-models/land_cover_classification/infer?region_id=harbin&patch_id=patch_000000&month=2099-01&version=v2"
        )
        assert response.status_code == 404
