"""On-demand Haidian change detection from monthly P10C embeddings."""
from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

from app.services.data_service import DataService


CHANGE_THRESHOLD = 0.7715411186218262
DISPLACEMENT_PENALTY = 0.05
NEIGHBORHOOD_RADIUS = 2
EPSILON = 1e-8


def _directional_change(
    before: np.ndarray,
    after: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    before_pixels = np.moveaxis(np.asarray(before, dtype=np.float32), 0, -1)
    after_pixels = np.moveaxis(np.asarray(after, dtype=np.float32), 0, -1)
    if before_pixels.shape != after_pixels.shape or before_pixels.ndim != 3:
        raise ValueError("Embeddings must have the same C,H,W shape")

    before_norm = np.linalg.norm(before_pixels, axis=-1)
    after_norm = np.linalg.norm(after_pixels, axis=-1)
    before_valid = np.isfinite(before_pixels).all(axis=-1) & (before_norm > EPSILON)
    after_valid = np.isfinite(after_pixels).all(axis=-1) & (after_norm > EPSILON)
    before_unit = before_pixels / np.maximum(before_norm[..., None], EPSILON)
    after_unit = after_pixels / np.maximum(after_norm[..., None], EPSILON)

    radius = NEIGHBORHOOD_RADIUS
    height, width = before_valid.shape
    padded_after = np.pad(
        after_unit,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="constant",
    )
    padded_valid = np.pad(
        after_valid,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=False,
    )
    best = np.full((height, width), np.inf, dtype=np.float32)
    has_match = np.zeros((height, width), dtype=bool)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            candidate = padded_after[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            candidate_valid = padded_valid[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            pair_valid = before_valid & candidate_valid
            similarity = np.sum(before_unit * candidate, axis=-1)
            row_distance = row_offset - radius
            column_distance = column_offset - radius
            normalized_distance = (
                row_distance * row_distance + column_distance * column_distance
            ) / float(radius * radius)
            cost = (
                1.0
                - similarity
                + DISPLACEMENT_PENALTY * normalized_distance
            )
            best[pair_valid] = np.minimum(best[pair_valid], cost[pair_valid])
            has_match |= pair_valid
    best[~has_match] = np.nan
    return np.clip(best, 0.0, 2.0).astype(np.float32), has_match


def compute_change_scores(
    before: np.ndarray,
    after: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean-fused bidirectional 5x5 cosine-change scores."""
    forward, forward_valid = _directional_change(before, after)
    backward, backward_valid = _directional_change(after, before)
    valid = forward_valid & backward_valid
    scores = (forward + backward) * 0.5
    scores[~valid] = np.nan
    return scores.astype(np.float32), valid


def load_change_scores(
    patch_id: str,
    before_month: str,
    after_month: str,
    version: str = "v1",
) -> Tuple[np.ndarray, np.ndarray]:
    before_path = DataService.get_embedding_path(
        "haidian", patch_id, "npy", version, before_month
    )
    after_path = DataService.get_embedding_path(
        "haidian", patch_id, "npy", version, after_month
    )
    if not before_path:
        raise FileNotFoundError(
            f"Embedding not found for haidian/{patch_id}/{before_month}"
        )
    if not after_path:
        raise FileNotFoundError(
            f"Embedding not found for haidian/{patch_id}/{after_month}"
        )
    return compute_change_scores(
        np.load(before_path, allow_pickle=False),
        np.load(after_path, allow_pickle=False),
    )


def change_mask(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return ((scores >= CHANGE_THRESHOLD) & valid).astype(np.uint8)


def render_change_png(scores: np.ndarray, valid: np.ndarray) -> bytes:
    low = 0.07
    scaled = np.clip((scores - low) / (CHANGE_THRESHOLD - low), 0.0, 1.0)
    blue = np.array([49.0, 86.0, 166.0], dtype=np.float32)
    light_blue = np.array([213.0, 234.0, 247.0], dtype=np.float32)
    pale_yellow = np.array([255.0, 247.0, 188.0], dtype=np.float32)
    red = np.array([196.0, 52.0, 46.0], dtype=np.float32)
    first = np.clip(scaled / 0.75, 0.0, 1.0)[..., None]
    second = np.clip((scaled - 0.75) / 0.25, 0.0, 1.0)[..., None]
    below = blue + first * (light_blue - blue)
    below += second * (pale_yellow - light_blue)
    rgb = np.where((scores >= CHANGE_THRESHOLD)[..., None], red, below)
    rgb = np.rint(np.nan_to_num(rgb, nan=0.0)).astype(np.uint8)
    rgb[~valid] = (238, 241, 244)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()
