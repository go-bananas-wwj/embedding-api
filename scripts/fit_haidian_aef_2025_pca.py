#!/usr/bin/env python3
"""Fit and optionally precompute Haidian AEF 2025 global-PCA PNGs."""

import argparse

from app.services.aef_pca_service import (
    AEF_ROOT,
    fit_global_pca,
    get_or_create_pca_png,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-patch", type=int, default=512)
    parser.add_argument("--precompute", action="store_true")
    args = parser.parse_args()
    metadata = fit_global_pca(samples_per_patch=args.samples_per_patch)
    print(metadata)
    if args.precompute:
        paths = sorted(AEF_ROOT.glob("patch_*.npy"))
        for index, path in enumerate(paths, 1):
            get_or_create_pca_png(path.stem)
            if index % 40 == 0 or index == len(paths):
                print(f"precomputed {index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
