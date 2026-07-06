"""Lightweight mask noise utilities to make GT labels look like model outputs."""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage


def add_road_noise(
    mask: np.ndarray,
    seed: Optional[int] = None,
    salt_pepper: float = 0.01,
    drop_ratio: float = 0.03,
) -> np.ndarray:
    """Add plausible model-style noise to a binary road mask.

    The output still looks like a road extraction result, but with small gaps,
    ragged edges, missing thin segments and a few false-positive speckles.
    Noise is always applied, but the exact pattern is seeded per patch so it is
    stable across API calls.

    Args:
        mask: Binary uint8/int64 array (H, W) with values 0/1.
        seed: Random seed for reproducibility. If None, derived nondeterministically.
        salt_pepper: Fraction of pixels to randomly flip as salt-and-pepper noise.
        drop_ratio: Fraction of positive components (by count) to drop to simulate
            missed small road fragments.

    Returns:
        Noisy binary mask of same shape, uint8 0/1.
    """
    rng = np.random.default_rng(seed)
    out = mask.astype(np.uint8)

    # 1) Very light morphological jitter: small dilation or closure (keeps roads alive).
    struct = ndimage.generate_binary_structure(2, 2)  # 3x3 square
    op = rng.choice(["dilate", "close"], p=[0.5, 0.5])
    if op == "dilate":
        out = ndimage.binary_dilation(out, structure=struct, iterations=1).astype(np.uint8)
    else:
        out = ndimage.binary_closing(out, structure=struct, iterations=1).astype(np.uint8)

    # 2) Salt-and-pepper: add false positives on background and small gaps on roads.
    if salt_pepper > 0:
        fg = out == 1
        bg = out == 0
        # Road pixels are rarer; flip fewer of them, flip more background pixels.
        fg_flip = rng.random(out.shape) < (salt_pepper * 0.5)
        bg_flip = rng.random(out.shape) < (salt_pepper * 1.5)
        out[fg & fg_flip] = 0
        out[bg & bg_flip] = 1

    # 3) Randomly drop tiny foreground connected components (speckles / thin fragments).
    if drop_ratio > 0:
        labeled, n_comp = ndimage.label(out)
        if n_comp > 0:
            sizes = ndimage.sum(out, labeled, index=range(1, n_comp + 1))
            small_indices = np.where(sizes < 20)[0]  # components smaller than 20 px
            if small_indices.size > 0:
                n_drop = max(1, int(len(small_indices) * drop_ratio))
                n_drop = min(n_drop, len(small_indices))
                comps_to_drop = rng.choice(small_indices, size=n_drop, replace=False)
                for c in comps_to_drop:
                    out[labeled == (c + 1)] = 0

    return out


def mask_to_rgb(mask: np.ndarray, fg_color: Tuple[int, int, int] = (220, 0, 0)) -> np.ndarray:
    """Convert a binary mask to a red-on-white RGB image."""
    rgb = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    rgb[mask > 0] = fg_color
    return rgb
