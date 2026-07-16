#!/usr/bin/env python3
"""Regenerate seam-consistent PCA-RGB previews from deployed NPY embeddings.

The PCA basis and percentile stretch are fitted once across a deterministic
sample from every input patch. Every output tile then uses exactly that same
color transform, so patch boundaries do not receive artificial color jumps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "embedding_root",
        type=Path,
        help="Version root containing month/patch_XXXXXX.npy files",
    )
    parser.add_argument("--max-samples", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patches-meta",
        type=Path,
        help="Optional patch metadata JSON used for cross-boundary feathering",
    )
    parser.add_argument(
        "--feather-pixels",
        type=int,
        default=0,
        help="Blend this many pixels on each side of adjacent patch boundaries",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _as_hwd(array: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D embedding, got shape {array.shape}")
    channels, height, width = array.shape
    return array.reshape(channels, height * width).T, channels, height, width


def fit_shared_transform(
    paths: list[Path], max_samples: int, seed: int
) -> tuple[PCA, np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("No NPY embeddings found")
    rng = np.random.default_rng(seed)
    per_file = max(1, max_samples // len(paths))
    samples: list[np.ndarray] = []
    expected_channels: int | None = None
    for path in paths:
        flat, channels, _, _ = _as_hwd(np.load(path, mmap_mode="r"))
        if expected_channels is None:
            expected_channels = channels
        elif channels != expected_channels:
            raise ValueError(
                f"Embedding channel mismatch: {path} has {channels}, expected {expected_channels}"
            )
        count = min(per_file, flat.shape[0])
        indices = rng.choice(flat.shape[0], size=count, replace=False)
        samples.append(np.asarray(flat[indices], dtype=np.float32))

    sample = np.concatenate(samples, axis=0)
    pca = PCA(n_components=3, random_state=seed)
    projected = pca.fit_transform(sample)
    low = np.percentile(projected, 2, axis=0).astype(np.float32)
    high = np.percentile(projected, 98, axis=0).astype(np.float32)
    return pca, low, high


def render(path: Path, pca: PCA, low: np.ndarray, high: np.ndarray) -> Image.Image:
    flat, _, height, width = _as_hwd(np.load(path, mmap_mode="r"))
    rgb = pca.transform(np.asarray(flat, dtype=np.float32)).reshape(height, width, 3)
    rgb = np.clip((rgb - low.reshape(1, 1, 3)) / (high - low).reshape(1, 1, 3), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def feather_month_tiles(
    month_dir: Path,
    patches: list[dict],
    feather_pixels: int,
) -> int:
    """Blend narrow strips across real adjacent patch boundaries.

    This is display-only post-processing. NPY embeddings remain untouched.
    """
    if feather_pixels <= 0:
        return 0
    by_native_origin = {
        (round(float(p["bounds"][0]), 6), round(float(p["bounds"][1]), 6)): p
        for p in patches
        if len(p.get("bounds", [])) == 4
    }
    arrays: dict[str, np.ndarray] = {}
    for patch in patches:
        path = month_dir / f'{patch["patch_id"]}.png'
        if path.exists():
            arrays[patch["patch_id"]] = np.asarray(Image.open(path).convert("RGB")).copy()
    if not arrays:
        return 0

    blended_boundaries = 0
    for patch in patches:
        patch_id = patch["patch_id"]
        current = arrays.get(patch_id)
        bounds = patch.get("bounds", [])
        if current is None or len(bounds) != 4:
            continue
        minx, miny, maxx, maxy = map(float, bounds)

        right_patch = by_native_origin.get((round(maxx, 6), round(miny, 6)))
        if right_patch and right_patch["patch_id"] in arrays:
            right = arrays[right_patch["patch_id"]]
            width = min(feather_pixels, current.shape[1] // 4, right.shape[1] // 4)
            left_strip = current[:, -width:, :].astype(np.float32).copy()
            right_strip = right[:, :width, :].astype(np.float32).copy()
            for depth in range(width):
                weight = 0.25 * float(width - depth) / float(width)
                left_col = width - depth - 1
                right_col = depth
                current[:, -depth - 1, :] = (
                    left_strip[:, left_col, :] * (1 - weight)
                    + right_strip[:, right_col, :] * weight
                ).astype(np.uint8)
                right[:, depth, :] = (
                    right_strip[:, right_col, :] * (1 - weight)
                    + left_strip[:, left_col, :] * weight
                ).astype(np.uint8)
            blended_boundaries += 1

        top_patch = by_native_origin.get((round(minx, 6), round(maxy, 6)))
        if top_patch and top_patch["patch_id"] in arrays:
            top = arrays[top_patch["patch_id"]]
            width = min(feather_pixels, current.shape[0] // 4, top.shape[0] // 4)
            bottom_strip = current[-width:, :, :].astype(np.float32).copy()
            top_strip = top[:width, :, :].astype(np.float32).copy()
            for depth in range(width):
                weight = 0.25 * float(width - depth) / float(width)
                bottom_row = width - depth - 1
                top_row = depth
                current[-depth - 1, :, :] = (
                    bottom_strip[bottom_row, :, :] * (1 - weight)
                    + top_strip[top_row, :, :] * weight
                ).astype(np.uint8)
                top[depth, :, :] = (
                    top_strip[top_row, :, :] * (1 - weight)
                    + bottom_strip[bottom_row, :, :] * weight
                ).astype(np.uint8)
            blended_boundaries += 1

    for patch_id, array in arrays.items():
        Image.fromarray(array).save(month_dir / f"{patch_id}.png")
    return blended_boundaries


def main() -> None:
    args = parse_args()
    root = args.embedding_root.resolve()
    paths = sorted(root.glob("*/*.npy"))
    pca, low, high = fit_shared_transform(paths, args.max_samples, args.seed)
    metadata = {
        "method": "global_pca_rgb",
        "normalization": "global_percentile_2_98",
        "input_files": len(paths),
        "sample_pixels": min(args.max_samples, len(paths) * max(1, args.max_samples // len(paths))),
        "channels": int(pca.n_features_in_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    for index, path in enumerate(paths, start=1):
        render(path, pca, low, high).save(path.with_suffix(".png"))
        if index % 100 == 0 or index == len(paths):
            print(f"Rendered {index}/{len(paths)}")
    if args.feather_pixels:
        if not args.patches_meta:
            raise ValueError("--patches-meta is required when --feather-pixels is set")
        patches = json.loads(args.patches_meta.read_text(encoding="utf-8"))
        boundary_count = 0
        for month_dir in sorted({path.parent for path in paths}):
            boundary_count += feather_month_tiles(month_dir, patches, args.feather_pixels)
        metadata["feather_pixels"] = args.feather_pixels
        metadata["feathered_boundaries"] = boundary_count
        print(f"Feathered {boundary_count} adjacent boundaries")
    (root / "pca_visualization.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
