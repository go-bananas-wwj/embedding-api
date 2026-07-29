"""System pre-trained model route tests."""

from pathlib import Path

import numpy as np
import torch

from fastapi.testclient import TestClient

from app.main import app
from app.services.fewshot_heads import BinaryConv3x3ProbeHead
from app.services.system_model_service import _infer_torch_head


client = TestClient(app)


def test_infer_self_describing_binary_conv3x3_checkpoint(tmp_path: Path):
    model = BinaryConv3x3ProbeHead(embed_dim=2, hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(2.0)

    checkpoint = tmp_path / "conv.pt"
    torch.save(
        {
            "__format__": "embedding-api.system-head.v1",
            "head_type": "binary_conv3x3",
            "state_dict": model.state_dict(),
            "embed_dim": 2,
            "hidden_dim": 4,
            "dropout": 0.0,
            "threshold": 0.8,
        },
        checkpoint,
    )

    prediction = _infer_torch_head(checkpoint, np.zeros((2, 3, 5), dtype=np.float32))

    assert prediction.shape == (3, 5)
    assert np.all(prediction == 1)


def test_infer_legacy_mlp_checkpoint(tmp_path: Path):
    state = {
        "net.0.weight": torch.zeros((3, 2)),
        "net.0.bias": torch.zeros(3),
        "net.2.weight": torch.zeros((1, 3)),
        "net.2.bias": torch.tensor([-2.0]),
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save(state, checkpoint)

    prediction = _infer_torch_head(checkpoint, np.zeros((2, 2, 4), dtype=np.float32))

    assert prediction.shape == (2, 4)
    assert np.all(prediction == 0)


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
        assert all(m["head_type"] == "binary_conv3x3" for m in data)
        assert all(m["feature_source"] == "P10C 64D embedding" for m in data)

    def test_get_system_model_classes(self):
        response = client.get(
            "/system-models/land_cover_classification/classes?region_id=harbin&version=v2"
        )
        # 200 if checkpoint exists, 404 if model file missing
        assert response.status_code in (200, 404)

    def test_haidian_land_cover_classes_available_without_online_checkpoint(self):
        response = client.get(
            "/system-models/land_cover_classification/classes"
            "?region_id=haidian&version=v1"
        )
        assert response.status_code == 200
        classes = response.json()
        assert len(classes) == 7
        assert classes[0] == {
            "id": "sys_land_cover_classification_1",
            "name": "永久性水体",
            "color": "#1E64DC",
        }
        assert classes[-1] == {
            "id": "sys_land_cover_classification_8",
            "name": "树木覆盖",
            "color": "#006400",
        }

    def test_haidian_land_cover_classes_reject_unknown_version(self):
        response = client.get(
            "/system-models/land_cover_classification/classes"
            "?region_id=haidian&version=v2"
        )
        assert response.status_code == 404

    def test_haidian_land_use_exposes_only_reliably_supervised_classes(self):
        response = client.get(
            "/system-models/land_use_classification/classes"
            "?region_id=haidian&version=v1"
        )
        assert response.status_code == 200
        classes = response.json()
        assert len(classes) == 7
        assert {item["name"] for item in classes} == {
            "水体", "树木", "草地", "农作物", "灌木与矮林", "建成区", "裸地"
        }

    def test_haidian_system_model_omitted_version_uses_v1(self):
        response = client.get(
            "/system-models/road_extraction/classes?region_id=haidian"
        )
        assert response.status_code == 200
        assert response.json()[1]["name"] == "道路"

    def test_haidian_system_model_rejects_unavailable_explicit_version(self):
        response = client.get(
            "/system-models/road_extraction/classes"
            "?region_id=haidian&version=v2"
        )

        assert response.status_code == 404
        assert "Available versions: v1" in response.json()["detail"]

    def test_haidian_land_use_static_product_has_reliable_seven_class_legend(self):
        response = client.get(
            "/system-models/land_use_classification/classes",
            params={"region_id": "haidian", "version": "v1"},
        )
        assert response.status_code == 200
        classes = response.json()
        assert len(classes) == 7
        assert classes[0] == {
            "id": "sys_land_use_classification_0",
            "name": "水体",
            "color": "#286EE6",
        }

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
