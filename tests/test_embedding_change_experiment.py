from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.experiment_haidian_embedding_change import (
    bidirectional_neighborhood_cosine_change,
    change_scores,
    estimate_translation,
    global_limits,
    neighborhood_cosine_change,
    robust_temporal_normalize,
    robust_rgb,
    select_representative_patches,
    smooth_embedding,
    symmetric_neighborhood_cosine_change,
    threshold_colored_map,
)


def test_change_scores_identical_and_orthogonal_vectors():
    before = np.zeros((2, 1, 2), dtype=np.float32)
    after = np.zeros_like(before)
    before[:, 0, 0] = [1.0, 0.0]
    after[:, 0, 0] = [1.0, 0.0]
    before[:, 0, 1] = [1.0, 0.0]
    after[:, 0, 1] = [0.0, 1.0]

    cosine, euclidean, valid = change_scores(before, after)

    np.testing.assert_allclose(cosine[0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(euclidean[0], [0.0, np.sqrt(2.0)], atol=1e-6)
    assert valid.all()


def test_change_scores_marks_zero_vectors_invalid():
    before = np.zeros((2, 1, 1), dtype=np.float32)
    after = np.ones_like(before)

    cosine, euclidean, valid = change_scores(before, after)

    assert not valid[0, 0]
    assert np.isnan(cosine[0, 0])
    assert np.isnan(euclidean[0, 0])


def test_change_scores_rejects_mismatched_shapes():
    before = np.ones((2, 2, 2), dtype=np.float32)
    after = np.ones((2, 3, 2), dtype=np.float32)

    try:
        change_scores(before, after)
    except ValueError as exc:
        assert "same C,H,W shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched embedding shapes to fail")


def test_neighborhood_change_compensates_one_pixel_translation():
    before = np.zeros((2, 3, 3), dtype=np.float32)
    after = np.zeros_like(before)
    before[0] = 1.0
    before[:, 1, 1] = [0.0, 1.0]
    after[0] = 1.0
    after[:, 1, 2] = [0.0, 1.0]

    direct, _, _ = change_scores(before, after)
    local, valid = neighborhood_cosine_change(before, after, radius=1)

    assert direct[1, 1] > 0.9
    assert valid[1, 1]
    assert local[1, 1] < 1e-6


def test_neighborhood_change_requires_supported_radius():
    embedding = np.ones((2, 2, 2), dtype=np.float32)

    try:
        neighborhood_cosine_change(embedding, embedding, radius=0)
    except ValueError as exc:
        assert "radius" in str(exc)
    else:
        raise AssertionError("Expected radius=0 to fail")


def test_symmetric_neighborhood_change_is_swap_invariant():
    before = np.zeros((3, 4, 4), dtype=np.float32)
    after = np.zeros_like(before)
    before[0] = 1.0
    after[0] = 1.0
    before[:, 1, 1] = [0.0, 1.0, 0.0]
    after[:, 1, 2] = [0.0, 1.0, 0.0]

    forward, valid_forward = symmetric_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.05
    )
    backward, valid_backward = symmetric_neighborhood_cosine_change(
        after, before, radius=2, displacement_penalty=0.05
    )

    np.testing.assert_allclose(forward, backward, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(valid_forward, valid_backward)
    assert forward[1, 1] >= 0.0


def test_symmetric_neighborhood_change_penalizes_distant_match():
    before = np.zeros((2, 5, 5), dtype=np.float32)
    after = np.zeros_like(before)
    before[0] = 1.0
    after[0] = 1.0
    before[:, 2, 1] = [0.0, 1.0]
    after[:, 2, 3] = [0.0, 1.0]

    no_penalty, _ = symmetric_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.0
    )
    penalized, _ = symmetric_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.2
    )

    assert penalized[2, 1] > no_penalty[2, 1]


def test_bidirectional_mean_is_no_greater_than_max_fusion():
    before = np.zeros((2, 5, 5), dtype=np.float32)
    after = np.zeros_like(before)
    before[0] = 1.0
    after[0] = 1.0
    before[:, 2, 2] = [0.0, 1.0]

    maximum, valid = bidirectional_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.05, fusion="max"
    )
    mean, mean_valid = bidirectional_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.05, fusion="mean"
    )

    np.testing.assert_array_equal(valid, mean_valid)
    assert np.all(mean[valid] <= maximum[valid] + 1e-6)


def test_estimate_translation_returns_shift_to_align_moved_image():
    before = np.zeros((32, 32), dtype=np.float32)
    before[8:14, 11:19] = 1.0
    after = np.roll(before, shift=(3, -4), axis=(0, 1))

    shift = estimate_translation(before, after, max_shift=8)

    np.testing.assert_allclose(shift, (-3.0, 4.0), atol=0.1)


def test_robust_temporal_normalize_removes_channelwise_style_shift():
    grid = np.arange(16, dtype=np.float32).reshape(4, 4)
    before = np.stack([grid, grid[::-1]], axis=0)
    after = np.stack([grid * 3.0 + 7.0, grid[::-1] * 0.5 - 2.0], axis=0)

    normalized_before, normalized_after = robust_temporal_normalize(before, after)

    np.testing.assert_allclose(normalized_before, normalized_after, atol=1e-5)


def test_smooth_embedding_only_filters_spatial_dimensions():
    embedding = np.zeros((2, 5, 5), dtype=np.float32)
    embedding[0, 2, 2] = 9.0
    embedding[1] = 7.0

    smoothed = smooth_embedding(embedding, method="mean", size=3)

    np.testing.assert_allclose(smoothed[0, 1:4, 1:4], 1.0, atol=1e-6)
    np.testing.assert_allclose(smoothed[1], 7.0, atol=1e-6)


def test_smooth_embedding_rejects_unknown_method():
    embedding = np.ones((2, 3, 3), dtype=np.float32)

    try:
        smooth_embedding(embedding, method="bilateral", size=3)
    except ValueError as exc:
        assert "method" in str(exc)
    else:
        raise AssertionError("Expected unknown smoothing method to fail")


def test_threshold_colored_map_only_marks_values_at_threshold_red():
    scores = np.array([[0.1, 0.7, 0.8, np.nan]], dtype=np.float32)

    colored = threshold_colored_map(scores, low=0.0, threshold=0.8)

    assert colored[0, 0, 2] > colored[0, 0, 0]
    assert colored[0, 1, 1] > 150
    assert colored[0, 2, 0] > colored[0, 2, 2]
    np.testing.assert_array_equal(colored[0, 3], [238, 241, 244])


def test_global_limits_ignore_non_finite_values():
    maps = [
        np.array([[0.0, 1.0, np.nan]], dtype=np.float32),
        np.array([[2.0, np.inf, 3.0]], dtype=np.float32),
    ]

    low, high = global_limits(maps, low_quantile=0.0, high_quantile=1.0)

    assert low == 0.0
    assert high == 3.0


def test_select_representative_patches_spans_score_distribution():
    stats = [
        {"patch_id": f"patch_{index:06d}", "cosine_p95": float(index)}
        for index in range(20)
    ]

    selected = select_representative_patches(stats, count=6)

    assert len(selected) == 6
    assert len(set(selected)) == 6
    selected_scores = [
        next(item["cosine_p95"] for item in stats if item["patch_id"] == patch_id)
        for patch_id in selected
    ]
    assert min(selected_scores) <= 2
    assert 8 <= sorted(selected_scores)[len(selected_scores) // 2] <= 15
    assert max(selected_scores) >= 18


def test_robust_rgb_reads_named_sentinel_bands(tmp_path: Path):
    path = tmp_path / "scene.tif"
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[0] = np.arange(16, dtype=np.float32).reshape(4, 4)  # B02
    data[1] = data[0] + 10  # B03
    data[2] = data[0] + 20  # B04
    data[3] = 99
    data[0, 0, 0] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=4,
        dtype="float32",
        transform=from_origin(0, 4, 1, 1),
    ) as dataset:
        dataset.write(data)
        dataset.descriptions = ("B02", "B03", "B04", "SCL")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rgb = robust_rgb(path)

    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    assert np.isfinite(rgb).all()
    assert rgb[0, 0, 2] == 0
