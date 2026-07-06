#!/usr/bin/env python3
"""Download and install only Haidian V1 embedding assets from ModelScope.

This lightweight path uses the latest dataset
``WeijieWu/xuannv_haidian_embdding``. It downloads raw ``*_embedding_map.pt``
files from ``artifacts/haidian-embedding-v1/embeddings`` and converts them into
the API layout:

``data/haidian/embeddings/v1/{month}/{patch_id}.npy|png|json``.

Token is read from ``MODELSCOPE_TOKEN`` when required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from download_modelscope_assets import (
    download_with_cli,
    install_artifacts,
)
from paths import (
    DEFAULT_EMBEDDING_ARTIFACT,
    DEFAULT_MODELSCOPE_PREFIX,
    DEFAULT_MODELSCOPE_REPO,
    PROJECT_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_MODELSCOPE_REPO)
    parser.add_argument("--prefix", default=DEFAULT_MODELSCOPE_PREFIX)
    parser.add_argument("--embedding-artifact", default=DEFAULT_EMBEDDING_ARTIFACT)
    parser.add_argument(
        "--target",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root where data/haidian will be placed.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".modelscope_cache/haidian_v1_embeddings"),
        help="Temporary dataset download cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing converted embeddings and remove old patch-subdir layout.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Install from cache-dir without invoking the ModelScope downloader.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_with_cli(
            args.repo,
            args.cache_dir,
            [f"{args.prefix.rstrip('/')}/embeddings/**"],
            args.max_workers,
        )

    src = args.cache_dir / args.prefix
    install_artifacts(src, args.target, args.embedding_artifact, args.force)
    print(f"Haidian V1 embeddings installed into {args.target}/data/haidian")


if __name__ == "__main__":
    main()
