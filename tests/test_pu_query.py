"""Numerical tests for sparse positive-unlabeled retrieval."""

import numpy as np

from app.services.pu_query import score_pu_query, train_pu_query


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


def test_query_adaptation_rejects_unbounded_growth():
    feature = _toy_feature()
    mask = np.zeros((24, 24), dtype=bool)
    mask[7:13, 8:14] = True
    model = train_pu_query([("support", feature, mask)])
    model["threshold"] = -10.0

    _, adapted = score_pu_query(feature, model)

    assert adapted is False
