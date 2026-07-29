import numpy as np

from scripts.retrain_haidian_land_labels import (
    LAND_COVER_SPEC,
    LAND_USE_SPEC,
    load_land_use_label,
    remove_tiny_water,
)


def test_verified_worldcover_mapping_keeps_water_and_tree_distinct():
    assert LAND_COVER_SPEC.source_to_index[1] == 0
    assert LAND_COVER_SPEC.source_to_index[8] == 6


def test_land_cover_and_land_use_are_separate_models():
    assert LAND_COVER_SPEC.checkpoint_name != LAND_USE_SPEC.checkpoint_name
    assert LAND_COVER_SPEC.output_values != LAND_USE_SPEC.output_values
    assert LAND_COVER_SPEC.label_source != LAND_USE_SPEC.label_source


def test_land_use_human_activity_masks_override_cover_label(monkeypatch):
    source = np.full((8, 8), 8, dtype=np.uint8)
    buildings = np.zeros_like(source, dtype=bool)
    roads = np.zeros_like(source, dtype=bool)
    buildings[1:3, 1:3] = True
    roads[5, 2:6] = True
    monkeypatch.setattr(
        "scripts.retrain_haidian_land_labels.load_worldcover",
        lambda _patch_id: source,
    )
    monkeypatch.setattr(
        "scripts.retrain_haidian_land_labels.load_optional_mask",
        lambda root, _patch_id: buildings if root.name == "labels" and "building" in str(root) else roads,
    )

    label = load_land_use_label("patch_test")
    built = LAND_USE_SPEC.output_values.index(6)

    assert np.all(label[buildings] == built)
    assert np.all(label[roads] == built)


def test_remove_tiny_water_replaces_only_isolated_components():
    built = LAND_COVER_SPEC.source_to_index[5]
    water = LAND_COVER_SPEC.source_to_index[1]
    prediction = np.full((8, 8), built, dtype=np.int64)
    prediction[1, 1] = water
    prediction[4:6, 4:6] = water

    filtered = remove_tiny_water(prediction, minimum_pixels=4)

    assert filtered[1, 1] == built
    assert np.all(filtered[4:6, 4:6] == water)
