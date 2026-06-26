"""Tests for GeoJSON annotation adapter."""

import numpy as np
import pytest

from app.schemas.models import GeoJSONFeature, GeoJSONFeatureCollection, ModelClass
from app.services.geojson_adapter import (
    build_class_map,
    group_features_by_patch,
    parse_annotations_for_training,
    rasterize_geometry,
)


def _make_feature(patch_id="patch_000000", cls_id="cls_001", month="2025-04", coords=None):
    if coords is None:
        coords = [
            [126.50, 45.74],
            [126.52, 45.74],
            [126.52, 45.76],
            [126.50, 45.76],
            [126.50, 45.74],
        ]
    return {
        "type": "Feature",
        "properties": {
            "patch_id": patch_id,
            "region_id": "harbin",
            "class_id": cls_id,
            "class_name": "建筑",
            "color": "#FF0000",
            "task_type": "building_extraction",
            "month": month,
        },
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def test_build_class_map():
    classes = [
        ModelClass(id="cls_b", name="B", color="#00FF00"),
        ModelClass(id="cls_a", name="A", color="#FF0000"),
    ]
    assert build_class_map(classes) == {"cls_a": 1, "cls_b": 2}


def test_group_features_by_patch():
    features = [
        _make_feature(patch_id="patch_000000"),
        _make_feature(patch_id="patch_000000"),
        _make_feature(patch_id="patch_000001"),
    ]
    groups = group_features_by_patch(features)
    assert len(groups["patch_000000"]) == 2
    assert len(groups["patch_000001"]) == 1


def test_rasterize_geometry():
    bbox = (126.50, 45.74, 126.52, 45.76)
    geometry = {
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
    }
    mask = rasterize_geometry(geometry, bbox)
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


def test_parse_annotations_for_training():
    annotations = GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=[
            GeoJSONFeature.model_validate(_make_feature()),
        ],
    )
    classes = [ModelClass(id="cls_001", name="建筑", color="#FF0000")]
    records = parse_annotations_for_training(
        annotations=annotations,
        classes=classes,
        class_ids=["cls_001"],
        model_type="classification",
    )
    assert len(records) == 1
    assert records[0]["patch_id"] == "patch_000000"
    assert records[0]["month"] == "2025-04"
    assert records[0]["label_index"] == 1
    assert records[0]["mask"].shape == (256, 256)


def test_parse_change_detection_annotations():
    feature = _make_feature()
    feature["properties"].update(
        {
            "month": None,
            "before_month": "2025-04",
            "after_month": "2025-06",
            "task_type": "change_detection",
        }
    )
    annotations = GeoJSONFeatureCollection(
        type="FeatureCollection",
        features=[GeoJSONFeature.model_validate(feature)],
    )
    classes = [ModelClass(id="cls_001", name="变化区域", color="#FF0000")]
    records = parse_annotations_for_training(
        annotations=annotations,
        classes=classes,
        class_ids=["cls_001"],
        model_type="change_detection",
    )
    assert len(records) == 1
    assert records[0]["before_month"] == "2025-04"
    assert records[0]["after_month"] == "2025-06"
