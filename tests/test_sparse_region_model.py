"""Tests for the category-agnostic sparse region model."""

import numpy as np

from app.services.sparse_region_model import (
    CHECKPOINT_FORMAT,
    local_descriptors,
    predict_sparse_region_model,
    score_sparse_region_model,
    select_sparse_strategy,
    train_sparse_region_model,
)


def _sample(key: str, offset: float = 0.0):
    rng = np.random.default_rng(abs(hash(key)) % (2**32))
    feature = rng.normal(0, 0.08, (8, 24, 24)).astype(np.float32)
    mask = np.zeros((24, 24), dtype=bool)
    mask[7:17, 8:18] = True
    feature[:, mask] += 1.0 + offset
    optical = np.full((24, 24, 3), 0.2, dtype=np.float32)
    optical[mask] = (0.65, 0.60, 0.55)
    return (key, feature, mask), optical


def test_local_descriptors_are_finite_and_spatially_aligned():
    sample, optical = _sample("patch_000001:202604")
    descriptor = local_descriptors(
        np.moveaxis(sample[1], 0, -1), optical
    )
    assert descriptor.shape[:2] == sample[2].shape
    assert descriptor.shape[2] > sample[1].shape[0]
    assert np.isfinite(descriptor).all()


def test_sparse_region_model_trains_and_predicts_foreground():
    first, first_optical = _sample("patch_000001:202604")
    second, second_optical = _sample("patch_000002:202604", 0.05)
    model = train_sparse_region_model(
        [first, second],
        {first[0]: first_optical, second[0]: second_optical},
        n_estimators=24,
    )
    score = score_sparse_region_model(first[1], first_optical, model)
    prediction = predict_sparse_region_model(score, model)
    assert model["__format__"] == CHECKPOINT_FORMAT
    assert float(score[first[2]].mean()) > float(score[~first[2]].mean())
    assert prediction[first[2]].mean() > 0.7


def test_strategy_guard_keeps_pu_for_thin_annotations():
    sample, optical = _sample("patch_000003:202604")
    thin_mask = np.zeros_like(sample[2])
    thin_mask[11:13, 3:21] = True
    thin_sample = (sample[0], sample[1], thin_mask)
    strategy, model = select_sparse_strategy(
        [thin_sample], {sample[0]: optical}
    )
    assert strategy == "pu_query_retrieval"
    assert model["selection"]["reason"] == "thin_annotation_guard"
