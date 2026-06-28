#!/usr/bin/env python3
"""Download only Haidian V1 embedding assets from ModelScope.

This is a lightweight alternative to ``download_modelscope_assets.py`` for
deployments that only need the embedding API and don't have space for raw
scenes or task results.

Downloads:
- ``data/haidian/embeddings/v1/**``
- ``data/haidian/patches_meta_v1.json``

Token is read from ``MODELSCOPE_TOKEN`` environment variable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from paths import DEFAULT_MODELSCOPE_PREFIX, DEFAULT_MODELSCOPE_REPO, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_MODELSCOPE_REPO)
    parser.add_argument("--prefix", default=DEFAULT_MODELSCOPE_PREFIX)
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
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    printable = list(cmd)
    for idx, item in enumerate(printable[:-1]):
        if item == "--token":
            printable[idx + 1] = "***"
    print("+", " ".join(printable))
    subprocess.run(cmd, check=True)


def download_files(repo: str, prefix: str, cache_dir: Path, rel_paths: list[str]) -> Path:
    """Download specific relative paths under prefix into cache_dir."""
    token = os.environ.get("MODELSCOPE_TOKEN")
    cmd = [
        "modelscope", "download", "--dataset", repo,
        "--local_dir", str(cache_dir),
    ]
    if token:
        cmd.extend(["--token", token])
    for p in rel_paths:
        cmd.extend(["--include", f"{prefix}/{p}"])
    run(cmd)
    return cache_dir / prefix / "data" / "haidian"


def download_single_file(repo: str, prefix: str, cache_dir: Path, rel_path: str) -> Path:
    """Download a single file by its relative path under prefix."""
    token = os.environ.get("MODELSCOPE_TOKEN")
    cmd = [
        "modelscope", "download", "--dataset", repo,
        "--local_dir", str(cache_dir),
    ]
    if token:
        cmd.extend(["--token", token])
    cmd.append(f"{prefix}/{rel_path}")
    run(cmd)
    return cache_dir / prefix / "data" / "haidian"


def copy_to_target(src: Path, target: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Downloaded Haidian data not found: {src}")
    dst_root = target / "data" / "haidian"
    dst_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_bytes = 0
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)
        total_files += 1
        total_bytes += item.stat().st_size

    print(f"Copied {total_files} files ({_human_size(total_bytes)}) to {dst_root}")


def _human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download embeddings (modelscope download picks a narrow root for **).
    download_files(
        args.repo, args.prefix, args.cache_dir,
        ["data/haidian/embeddings/v1/**"],
    )
    # 2. Download patches_meta separately because it sits outside the root used above.
    download_single_file(
        args.repo, args.prefix, args.cache_dir,
        "data/haidian/patches_meta_v1.json",
    )

    src = args.cache_dir / args.prefix / "data" / "haidian"
    copy_to_target(src, args.target)
    print(f"Haidian V1 embeddings installed into {args.target}/data/haidian")


if __name__ == "__main__":
    main()
