import numpy as np

from app.services.haidian_change_detection import (
    CHANGE_THRESHOLD,
    change_mask,
    compute_change_scores,
)


def test_change_scores_are_swap_invariant():
    before = np.zeros((3, 5, 5), dtype=np.float32)
    after = np.zeros_like(before)
    before[0] = 1.0
    after[0] = 1.0
    before[:, 2, 2] = [0.0, 1.0, 0.0]
    after[:, 2, 3] = [0.0, 1.0, 0.0]

    forward, valid = compute_change_scores(before, after)
    backward, backward_valid = compute_change_scores(after, before)

    np.testing.assert_allclose(forward, backward, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(valid, backward_valid)


def test_change_mask_uses_configured_p98_threshold():
    scores = np.array(
        [[CHANGE_THRESHOLD - 1e-4, CHANGE_THRESHOLD, CHANGE_THRESHOLD + 1e-4]],
        dtype=np.float32,
    )
    valid = np.array([[True, True, False]])

    mask = change_mask(scores, valid)

    np.testing.assert_array_equal(mask, [[0, 1, 0]])
