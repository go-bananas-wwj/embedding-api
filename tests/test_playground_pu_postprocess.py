"""Tests for offline PU + Query score postprocessing experiments."""

import numpy as np
import pytest

from scripts.playground_pu_postprocess import (
    binary_metrics,
    component_statistics,
    hysteresis_prediction,
    strict_threshold,
)


def test_hysteresis_keeps_low_score_pixels_only_when_connected_to_seed():
    score = np.zeros((12, 12), dtype=np.float32)
    score[2:5, 2:5] = 0.62
    score[3, 3] = 0.91
    score[8:10, 8:10] = 0.70

    result = hysteresis_prediction(score, high=0.85, low=0.60, min_pixels=4)

    assert result[2:5, 2:5].all()
    assert not result[8:10, 8:10].any()


def test_hysteresis_removes_tiny_seeded_component():
    score = np.zeros((8, 8), dtype=np.float32)
    score[1, 1] = 0.95

    result = hysteresis_prediction(score, high=0.90, low=0.60, min_pixels=4)

    assert not result.any()


def test_hysteresis_rejects_an_oversized_seeded_component():
    score = np.zeros((12, 12), dtype=np.float32)
    score[1:7, 1:7] = 0.65
    score[3, 3] = 0.95
    score[9:11, 9:11] = 0.65
    score[9, 9] = 0.91

    result = hysteresis_prediction(
        score,
        high=0.90,
        low=0.60,
        min_pixels=4,
        max_component_pixels=20,
    )

    assert not result[1:7, 1:7].any()
    assert result[9:11, 9:11].all()


def test_hysteresis_caps_total_area_by_seed_confidence():
    score = np.zeros((10, 10), dtype=np.float32)
    score[0:3, 0:3] = 0.65
    score[1, 1] = 0.99
    score[0:2, 5:7] = 0.65
    score[0, 5] = 0.95
    score[5:7, 8:10] = 0.65
    score[5, 8] = 0.90

    result = hysteresis_prediction(
        score,
        high=0.90,
        low=0.60,
        min_pixels=4,
        max_total_ratio=0.13,
    )

    assert result.sum() == 13
    assert result[0:3, 0:3].all()
    assert result[0:2, 5:7].all()
    assert not result[5:7, 8:10].any()


@pytest.mark.parametrize("high, low", [(np.nan, 0.60), (0.90, np.inf)])
def test_hysteresis_rejects_non_finite_thresholds(high, low):
    with pytest.raises(ValueError, match="thresholds must be finite"):
        hysteresis_prediction(
            np.zeros((4, 4), dtype=np.float32),
            high=high,
            low=low,
            min_pixels=1,
        )


@pytest.mark.parametrize("min_pixels", [True, np.bool_(True), 1.5, np.nan, np.inf])
def test_hysteresis_rejects_non_integer_min_pixels(min_pixels):
    with pytest.raises(ValueError, match="min_pixels must be a finite integer"):
        hysteresis_prediction(
            np.zeros((4, 4), dtype=np.float32),
            high=0.90,
            low=0.60,
            min_pixels=min_pixels,
        )


@pytest.mark.parametrize(
    "max_component_pixels", [True, np.bool_(True), 1.5, np.nan, np.inf]
)
def test_hysteresis_rejects_non_integer_max_component_pixels(max_component_pixels):
    with pytest.raises(
        ValueError, match="max_component_pixels must be a finite integer"
    ):
        hysteresis_prediction(
            np.zeros((4, 4), dtype=np.float32),
            high=0.90,
            low=0.60,
            min_pixels=1,
            max_component_pixels=max_component_pixels,
        )


def test_strict_threshold_prefers_higher_precision_then_smaller_area():
    scores = [np.array([[0.20, 0.30, 0.40, 0.95]], dtype=np.float32)]
    labels = [np.array([[False, True, False, True]])]

    threshold = strict_threshold(scores, labels)

    prediction = scores[0] >= threshold
    assert threshold >= 0.247057
    assert prediction.tolist() == [[False, True, True, True]]


def test_binary_metrics_and_component_statistics_describe_prediction():
    prediction = np.zeros((6, 7), dtype=bool)
    prediction[1:3, 2:4] = True
    prediction[4, 5:7] = True
    score = np.zeros((6, 7), dtype=np.float32)
    score[1:3, 2:4] = 0.75
    score[4, 5:7] = [0.80, 0.90]
    reference = np.zeros_like(prediction)
    reference[1:3, 2:4] = True
    reference[4, 5] = True

    metrics = binary_metrics(prediction, reference)
    statistics = component_statistics(prediction, score)

    assert metrics == pytest.approx({
        "precision": 5 / 6,
        "recall": 1.0,
        "f1": 10 / 11,
        "iou": 5 / 6,
        "positive_ratio": 6 / 42,
        "component_count": 2,
    })
    assert [item["area"] for item in statistics] == [4, 2]
    assert [item["bbox"] for item in statistics] == [[1, 2, 3, 4], [4, 5, 5, 7]]
    assert [item["mean_score"] for item in statistics] == pytest.approx([0.75, 0.85])
    assert [item["max_score"] for item in statistics] == pytest.approx([0.75, 0.90])
