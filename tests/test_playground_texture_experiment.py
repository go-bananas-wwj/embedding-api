"""Tests for optical texture-boundary constrained playground retrieval."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts.experiment_playground_texture import (
    assess_texture_improvement,
    calibrate_texture_experiment,
    compute_highres_texture_boundary,
    select_typical_patches,
    texture_boundary_area_prediction,
)


def _write_optical(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[2],
        height=values.shape[1],
        count=3,
        dtype=values.dtype,
        transform=from_origin(0, values.shape[1], 1, 1),
    ) as dataset:
        dataset.write(values)


def test_highres_texture_boundary_detects_real_optical_edge(tmp_path):
    optical = np.zeros((3, 96, 96), dtype=np.uint16)
    optical[:, :, :48] = 500
    optical[:, :, 48:] = 4000
    optical[:, 20:76, 20:44] += (
        np.indices((56, 24))[0] * 20
    ).astype(np.uint16)
    path = tmp_path / "optical.tif"
    _write_optical(path, optical)

    boundary = compute_highres_texture_boundary(path, output_shape=(24, 24))

    assert boundary.shape == (24, 24)
    assert np.isfinite(boundary).all()
    assert 0.0 <= float(boundary.min()) <= float(boundary.max()) <= 1.0
    assert boundary[:, 11:13].mean() > boundary[:, :5].mean()


def test_texture_boundary_blocks_cross_texture_growth_before_area_filter():
    score = np.full((12, 12), 0.55, dtype=np.float32)
    score[5:7, 2:4] = 1.2
    boundary = np.zeros((12, 12), dtype=np.float32)
    boundary[:, 6] = 1.0

    prediction = texture_boundary_area_prediction(
        score,
        boundary,
        seed_quantile=0.99,
        low=0.30,
        boundary_quantile=0.85,
        min_pixels=4,
        max_component_pixels=128,
        max_total_ratio=1.0,
    )

    assert prediction[:, :6].any()
    assert not prediction[:, 7:].any()


def test_texture_calibration_never_uses_independent_osm_patch():
    scores = {
        "patch_000059": np.array(
            [[0.1, 0.4, 0.8], [0.1, 0.7, 1.0]], dtype=np.float32
        ),
        "patch_000060": np.array(
            [[0.2, 0.5, 0.9], [0.1, 0.6, 1.1]], dtype=np.float32
        ),
        "patch_000076": np.full((2, 3), 99.0, dtype=np.float32),
    }
    references = {
        "patch_000059": np.array(
            [[False, False, True], [False, True, True]]
        ),
        "patch_000060": np.array(
            [[False, False, True], [False, True, True]]
        ),
        "patch_000076": np.ones((2, 3), dtype=bool),
    }
    boundaries = {
        patch_id: np.zeros((2, 3), dtype=np.float32) for patch_id in scores
    }

    parameters = calibrate_texture_experiment(
        scores,
        boundaries,
        references,
        calibration_patch_ids=["patch_000059", "patch_000060"],
        excluded_patch_ids=["patch_000076"],
        production_threshold=0.247057,
    )

    assert parameters["calibration_patch_ids"] == [
        "patch_000059",
        "patch_000060",
    ]
    assert parameters["excluded_patch_ids"] == ["patch_000076"]
    assert "patch_000076" not in json.dumps(
        parameters["search_evidence"], ensure_ascii=False
    )


def test_texture_calibration_rejects_holdout_leakage():
    values = {"patch_000076": np.ones((4, 4), dtype=np.float32)}

    with pytest.raises(ValueError, match="must not enter calibration"):
        calibrate_texture_experiment(
            values,
            values,
            {"patch_000076": values["patch_000076"].astype(bool)},
            calibration_patch_ids=["patch_000076"],
            excluded_patch_ids=["patch_000076"],
            production_threshold=0.247057,
        )


def test_typical_patch_selection_records_spatial_and_score_reasons():
    metadata = {
        "patches": [
            {
                "patch_id": f"patch_{index:06d}",
                "bounds_wgs84": [float(index), float(index), index + 1.0, index + 1.0],
            }
            for index in range(320)
        ]
    }
    ratios = {
        f"patch_{index:06d}": float(index) / 320.0 for index in range(320)
    }
    ratios["patch_000139"] = 0.99

    selected = select_typical_patches(metadata, ratios)

    assert selected["training"] == [
        "patch_000059",
        "patch_000060",
        "patch_000064",
    ]
    assert selected["independent_osm"] == ["patch_000076"]
    assert selected["global_high_false_positive"] == [
        "patch_000232",
        "patch_000249",
        "patch_000154",
    ]
    assert len(selected["spatial_high_score"]) == 1
    assert selected["selection_evidence"]
    assert all(
        "baseline_positive_ratio" in item
        and "center_wgs84" in item
        and "reason" in item
        for item in selected["selection_evidence"]
    )


def test_texture_assessment_rejects_numerically_trivial_gain():
    assessment = assess_texture_improvement(
        area_osm={"f1": 0.4023, "recall": 0.5309},
        texture_osm={"f1": 0.4039, "recall": 0.5309},
        area_high_false_ratio=0.0,
        texture_high_false_ratio=0.0,
        area_training={"recall": 0.9291},
        texture_training={"recall": 0.8865},
        legacy_osm={"recall": 0.0},
    )

    assert assessment["materially_improved_over_area_guard"] is False
    assert assessment["improved_over_legacy_guarded"] is True
    assert assessment["verdict"] == "纹理边界未证明有实质增益"
