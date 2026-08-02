"""Numerical tests for sparse positive-unlabeled retrieval."""

import numpy as np

from app.services.pu_query import predict_pu_query, score_pu_query, train_pu_query


def _toy_feature() -> np.ndarray:
    rng = np.random.default_rng(7)
    feature = rng.normal(0, 0.08, size=(4, 24, 24)).astype(np.float32)
    feature[0, 7:13, 8:14] += 2.5
    feature[1, 7:13, 8:14] += 1.0
    return feature


def test_pu_query_trains_from_polygon_and_scores_foreground():
    feature = _toy_feature()
    mask = np.zeros((24, 24), dtype=bool)
    mask[7:13, 8:14] = True

    model = train_pu_query([("support", feature, mask)])
    score, _ = score_pu_query(feature, model)

    assert np.isfinite(score).all()
    assert float(score[mask].mean()) > float(score[~mask].mean())
    assert np.isclose(np.linalg.norm(model["foreground_center"]), 1.0, atol=1e-5)
    assert model["postprocess"]["method"] in {
        "fixed_threshold",
        "relative_seed_area_guard",
    }
    assert model["postprocess"]["enabled"] is False
    if model["postprocess"]["method"] == "relative_seed_area_guard":
        assert model["postprocess"]["calibration_positive_recall"] >= (
            model["postprocess"]["recall_floor"]
        )


def test_query_adaptation_rejects_unbounded_growth():
    feature = _toy_feature()
    mask = np.zeros((24, 24), dtype=bool)
    mask[7:13, 8:14] = True
    model = train_pu_query([("support", feature, mask)])
    model["threshold"] = -10.0

    _, adapted = score_pu_query(feature, model)

    assert adapted is False


def test_old_checkpoint_prediction_keeps_fixed_threshold_behavior():
    score = np.array([[0.2, 0.5], [0.7, 0.1]], dtype=np.float32)

    prediction = predict_pu_query(score, {"threshold": 0.5})

    assert prediction.tolist() == [[False, True], [True, False]]


def test_relative_seed_area_guard_removes_unseeded_components():
    score = np.zeros((12, 12), dtype=np.float32)
    score[1:4, 1:4] = 0.65
    score[2, 2] = 0.99
    score[8:10, 8:10] = 0.61
    model = {
        "threshold": 0.5,
        "postprocess": {
            "enabled": True,
            "method": "relative_seed_area_guard",
            "low": 0.6,
            "seed_quantile": 0.98,
            "min_pixels": 2,
            "max_component_pixels": 20,
            "max_total_ratio": 0.2,
        },
    }

    prediction = predict_pu_query(score, model)

    assert prediction[1:4, 1:4].all()
    assert not prediction[8:10, 8:10].any()
