"""Tests for the existing ``playground_xuannv`` offline experiment."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.experiment_playground_xuannv import (
    REFERENCE_LIMITATION,
    _combined_metrics,
    calibrate_postprocessing,
    extract_training_request,
    load_registered_model,
    predict_variants,
    rasterize_training_masks,
    write_strict_json,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registered_model_resolves_exact_name_and_id(tmp_path):
    checkpoint = tmp_path / "head.pkl"
    torch.save({"threshold": 0.4}, checkpoint)
    registry = tmp_path / "models_index.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "id": "model_expected",
                    "name": "playground_xuannv",
                    "model_path": str(checkpoint),
                },
                {
                    "id": "model_other",
                    "name": "another-head",
                    "model_path": str(checkpoint),
                },
            ]
        ),
        encoding="utf-8",
    )

    record, model_data = load_registered_model(
        registry, "playground_xuannv", "model_expected"
    )

    assert record["id"] == "model_expected"
    assert model_data["threshold"] == pytest.approx(0.4)


def test_registered_model_rejects_duplicate_name(tmp_path):
    registry = tmp_path / "models_index.json"
    registry.write_text(
        json.dumps(
            [
                {"id": "model_a", "name": "playground_xuannv", "model_path": "a"},
                {"id": "model_b", "name": "playground_xuannv", "model_path": "b"},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        load_registered_model(registry, "playground_xuannv")


def test_audit_training_polygons_are_stably_extracted_and_rasterized():
    request = extract_training_request(
        ROOT / "logs/request_audit.jsonl", "playground_xuannv"
    )
    masks = rasterize_training_masks(
        request, ROOT / "data/haidian/patches_meta_v2.json"
    )

    assert request["region_id"] == "haidian"
    assert sorted(masks) == [
        "patch_000059",
        "patch_000060",
        "patch_000064",
    ]
    assert all(mask.shape == (128, 128) for mask in masks.values())
    assert all(mask.dtype == np.bool_ and mask.any() for mask in masks.values())


def test_calibration_excludes_independent_osm_patch():
    calibration_scores = {
        "patch_000059": np.array(
            [[0.1, 0.3, 0.8], [0.1, 0.7, 0.9]], dtype=np.float32
        ),
        "patch_000060": np.array(
            [[0.2, 0.4, 0.75], [0.1, 0.65, 0.85]], dtype=np.float32
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

    parameters = calibrate_postprocessing(
        calibration_scores,
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
    assert "patch_000076" not in parameters["calibration_patch_ids"]
    assert parameters["strict_threshold"] < 2.0


def test_calibration_rejects_holdout_in_calibration_set():
    scores = {"patch_000076": np.ones((3, 3), dtype=np.float32)}
    references = {"patch_000076": np.ones((3, 3), dtype=bool)}

    with pytest.raises(ValueError, match="must not enter calibration"):
        calibrate_postprocessing(
            scores,
            references,
            calibration_patch_ids=["patch_000076"],
            excluded_patch_ids=["patch_000076"],
            production_threshold=0.247057,
        )


def test_prediction_variants_have_stable_shape_type_and_guards():
    score = np.zeros((16, 16), dtype=np.float32)
    score[1:9, 1:9] = 0.55
    score[3, 3] = 0.95
    parameters = {
        "strict_threshold": 0.8,
        "guarded": {
            "high": 0.8,
            "low": 0.5,
            "min_pixels": 4,
            "max_component_pixels": 48,
            "max_total_ratio": 0.25,
        },
    }

    variants = predict_variants(score, 0.247057, parameters)

    assert set(variants) == {"baseline", "strict", "guarded"}
    assert all(value.shape == score.shape for value in variants.values())
    assert all(value.dtype == np.bool_ for value in variants.values())
    assert variants["baseline"].sum() == 64
    assert variants["strict"].sum() == 1
    assert variants["guarded"].sum() == 0


def test_combined_reference_metrics_do_not_join_components_across_patches():
    first = np.array([[False, True]], dtype=bool)
    second = np.array([[True, False]], dtype=bool)

    metrics = _combined_metrics(
        [first, second],
        [first.copy(), second.copy()],
    )

    assert metrics["component_count"] == 2


def test_json_writer_rejects_non_finite_and_records_reference_limitation(tmp_path):
    output = tmp_path / "metrics.json"
    payload = {
        "reference_policy": {
            "limitation": REFERENCE_LIMITATION,
            "unlabeled_pixels_are_reliable_negatives": False,
        },
        "value": 1.0,
    }

    write_strict_json(output, payload)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["reference_policy"]["limitation"] == REFERENCE_LIMITATION
    assert loaded["reference_policy"]["unlabeled_pixels_are_reliable_negatives"] is False
    with pytest.raises(ValueError):
        write_strict_json(output, {"bad": float("nan")})


def test_script_entrypoint_runs_from_repository_root():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiment_playground_xuannv.py",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--month" in result.stdout
