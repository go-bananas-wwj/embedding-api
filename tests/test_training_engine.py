"""Tests for training engines."""

import numpy as np
import pytest
import torch
from PIL import Image

from app.schemas.models import GeoJSONFeature, GeoJSONFeatureCollection, ModelClass
from app.services.inference_engine import InferenceEngine
from app.services.model_registry import get_model_registry
from app.services.training_engine import (
    ChangeDetectionTrainingEngine,
    ClassificationTrainingEngine,
    ExternalEmbeddingMLPTrainingEngine,
)
from app.services.pu_query import CHECKPOINT_FORMAT as PU_QUERY_CHECKPOINT_FORMAT


@pytest.fixture
def user_id():
    return "test_training_engine"


@pytest.fixture
def classification_annotation():
    """A small annotation inside patch_000000 of harbin."""
    return GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=[
            GeoJSONFeature.model_validate(
                {
                    "type": "Feature",
                    "properties": {
                        "patch_id": "patch_000000",
                        "region_id": "harbin",
                        "class_id": "cls_001",
                        "class_name": "建筑",
                        "color": "#FF0000",
                        "task_type": "building_extraction",
                        "month": "2025-04",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [126.52, 45.75],
                                [126.525, 45.75],
                                [126.525, 45.753],
                                [126.52, 45.753],
                                [126.52, 45.75],
                            ]
                        ],
                    },
                }
            )
        ],
    )


@pytest.fixture
def classes():
    return [ModelClass(id="cls_001", name="建筑", color="#FF0000")]


def test_classification_training_saves_class_map(
    user_id, classification_annotation, classes
):
    registry = get_model_registry(user_id)
    model_id = registry.create_model(
        name="test-class-map",
        model_type="classification",
        classes=[c.model_dump() for c in classes],
        task_type="building_extraction",
        region_id="harbin",
    )

    engine = ClassificationTrainingEngine(user_id)
    result = engine.train(
        model_id=model_id,
        region_id="harbin",
        task_type="building_extraction",
        embedding_version="v2",
        annotations=classification_annotation,
        classes=classes,
        class_ids=["cls_001"],
    )

    model_data = torch.load(result["model_path"], map_location="cpu", weights_only=False)
    assert model_data["__format__"] == PU_QUERY_CHECKPOINT_FORMAT
    assert model_data["head_type"] == "pu_query_retrieval"
    assert model_data["training_strategy"] == "pu_query_retrieval"
    assert result["n_samples"] == 1
    assert "class_map" in model_data
    assert model_data["class_map"] == {"cls_001": 1}
    assert model_data["classes"] == [{"id": "cls_001", "name": "建筑", "color": "#FF0000"}]
    assert np.isfinite(model_data["threshold"])
    assert model_data["foreground_center"].shape == (model_data["embed_dim"],)
    assert model_data["background_center"].shape == (model_data["embed_dim"],)


def test_ten_polygons_keep_binary_conv_strategy(
    user_id, classification_annotation, classes
):
    """Ten valid polygons are the inclusive boundary for Binary Conv 3x3."""
    source = classification_annotation.features[0].model_dump()
    annotations = GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=[GeoJSONFeature.model_validate(source) for _ in range(10)],
    )
    registry = get_model_registry(user_id)
    model_id = registry.create_model(
        name="test-ten-polygons",
        model_type="classification",
        classes=[c.model_dump() for c in classes],
        task_type="building_extraction",
        region_id="harbin",
    )

    result = ClassificationTrainingEngine(user_id).train(
        model_id=model_id,
        region_id="harbin",
        task_type="building_extraction",
        embedding_version="v2",
        annotations=annotations,
        classes=classes,
        class_ids=["cls_001"],
        epochs=1,
    )

    model_data = torch.load(result["model_path"], map_location="cpu", weights_only=False)
    assert model_data["__format__"] == "torch_fewshot_head"
    assert model_data["head_type"] == "binary_conv3x3"
    assert model_data["training_strategy"] == "binary_conv3x3"
    assert result["n_samples"] == 10


def test_classification_inference_color_matches_class_map(
    user_id, classification_annotation, classes
):
    """Inference result should color pixels using the class color, not label index."""
    registry = get_model_registry(user_id)
    model_id = registry.create_model(
        name="test-infer-color",
        model_type="classification",
        classes=[c.model_dump() for c in classes],
        task_type="building_extraction",
        region_id="harbin",
    )

    engine = ClassificationTrainingEngine(user_id)
    engine.train(
        model_id=model_id,
        region_id="harbin",
        task_type="building_extraction",
        embedding_version="v2",
        annotations=classification_annotation,
        classes=classes,
        class_ids=["cls_001"],
    )

    infer_engine = InferenceEngine(user_id)
    result_path = infer_engine.infer(
        model_id=model_id,
        region_id="harbin",
        patch_id="patch_000000",
        month="2025-04",
    )

    img = np.array(Image.open(result_path))
    assert img.shape[:2] == (128, 128), "Inference result should be 128x128"
    # At least some pixels should be colored with the class color (#FF0000).
    red_pixels = np.all(img == [255, 0, 0], axis=-1)
    assert red_pixels.sum() > 0, "Expected some red pixels from class color encoding"


def test_external_embedding_mlp_training_and_inference(
    user_id, classification_annotation, classes, monkeypatch
):
    feature = np.zeros((12, 128, 128), dtype=np.float32)
    feature[:, 20:70, 20:70] = 2.0
    monkeypatch.setattr(
        "app.services.training_engine.load_external_embedding",
        lambda *args: feature.copy(),
    )
    monkeypatch.setattr(
        "app.services.inference_engine.load_external_embedding",
        lambda *args: feature.copy(),
    )
    registry = get_model_registry(user_id)
    model_id = registry.create_model(
        name="test-aef-mlp",
        model_type="single_time_detection",
        classes=[c.model_dump() for c in classes],
        task_type="building_extraction",
        region_id="harbin",
        requested_training_method="aef",
        feature_source="aef",
    )
    result = ExternalEmbeddingMLPTrainingEngine("aef", user_id).train(
        model_id=model_id,
        region_id="harbin",
        task_type="building_extraction",
        model_type="single_time_detection",
        annotations=classification_annotation,
        classes=classes,
        class_ids=["cls_001"],
        epochs=1,
    )
    checkpoint = torch.load(result["model_path"], map_location="cpu", weights_only=False)
    assert checkpoint["__format__"] == "external_embedding_mlp_v1"
    assert checkpoint["head_type"] == "pixel_mlp"
    assert checkpoint["training_method"] == "aef"
    assert checkpoint["foundation_model_id"] == "aef"
    assert checkpoint["foundation_model_version"] == "aef_annual_2025"
    assert checkpoint["preprocessing_version"] == "aef_annual_2025"
    assert checkpoint["feature_dimension"] == 12
    assert checkpoint["compatible_regions"] == ["harbin"]
    path = InferenceEngine(user_id).infer(
        model_id, "harbin", "patch_000000", month="2025-04"
    )
    assert Image.open(path).size == (128, 128)


@pytest.fixture
def change_detection_annotation():
    return GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=[
            GeoJSONFeature.model_validate(
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
                                [126.52, 45.75],
                                [126.525, 45.75],
                                [126.525, 45.753],
                                [126.52, 45.753],
                                [126.52, 45.75],
                            ]
                        ],
                    },
                }
            )
        ],
    )


def test_change_detection_inference_uses_before_and_after(
    user_id, change_detection_annotation, classes
):
    """CD inference must load both before and after embeddings."""
    registry = get_model_registry(user_id)
    model_id = registry.create_model(
        name="test-cd-infer",
        model_type="change_detection",
        classes=[c.model_dump() for c in classes],
        task_type="change_detection",
        region_id="harbin",
    )

    engine = ChangeDetectionTrainingEngine(user_id)
    engine.train(
        model_id=model_id,
        region_id="harbin",
        task_type="change_detection",
        embedding_version="v2",
        annotations=change_detection_annotation,
        classes=classes,
        class_ids=["cls_001"],
    )

    infer_engine = InferenceEngine(user_id)
    result_path = infer_engine.infer(
        model_id=model_id,
        region_id="harbin",
        patch_id="patch_000000",
        before_month="2025-04",
        after_month="2025-06",
    )

    assert result_path.endswith("_2025-04_vs_2025-06.png")
    img = np.array(Image.open(result_path))
    assert img.shape == (128, 128, 4)


def test_multi_class_multi_feature_training(user_id):
    """Multiple features and classes on the same patch should train successfully."""
    features = [
        {
            "type": "Feature",
            "properties": {
                "patch_id": "patch_000000",
                "region_id": "harbin",
                "class_id": "cls_001",
                "class_name": "建筑",
                "color": "#FF0000",
                "task_type": "building_extraction",
                "month": "2025-04",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [126.52, 45.75],
                        [126.525, 45.75],
                        [126.525, 45.753],
                        [126.52, 45.753],
                        [126.52, 45.75],
                    ]
                ],
            },
        },
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
                        [126.525, 45.75],
                        [126.53, 45.75],
                        [126.53, 45.753],
                        [126.525, 45.753],
                        [126.525, 45.75],
                    ]
                ],
            },
        },
        # Two features for cls_001 should be merged.
        {
            "type": "Feature",
            "properties": {
                "patch_id": "patch_000000",
                "region_id": "harbin",
                "class_id": "cls_001",
                "class_name": "建筑",
                "color": "#FF0000",
                "task_type": "building_extraction",
                "month": "2025-04",
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [126.52, 45.753],
                        [126.525, 45.753],
                        [126.525, 45.755],
                        [126.52, 45.755],
                        [126.52, 45.753],
                    ]
                ],
            },
        },
    ]
    annotations = GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=[GeoJSONFeature.model_validate(f) for f in features],
    )
    classes = [
        ModelClass(id="cls_001", name="建筑", color="#FF0000"),
        ModelClass(id="cls_002", name="道路", color="#00FF00"),
    ]

    registry = get_model_registry(user_id)
    model_id = registry.create_model(
        name="test-multi",
        model_type="classification",
        classes=[c.model_dump() for c in classes],
        task_type="building_extraction",
        region_id="harbin",
    )

    engine = ClassificationTrainingEngine(user_id)
    with pytest.raises(ValueError, match="exactly one target class_id"):
        engine.train(
            model_id=model_id,
            region_id="harbin",
            task_type="building_extraction",
            embedding_version="v2",
            annotations=annotations,
            classes=classes,
            class_ids=["cls_001", "cls_002"],
        )
