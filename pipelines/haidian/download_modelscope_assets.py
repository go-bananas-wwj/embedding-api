#!/usr/bin/env python3
"""Download Haidian V1 assets from ModelScope into embedding-api.

The preferred source is the ModelScope dataset
``WeijieWu/xuannv_embdding_api`` under ``haidian/v1/api_ready``.  The script
keeps credentials out of the repository: pass a token via
``MODELSCOPE_TOKEN`` when the dataset is private.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from paths import DEFAULT_MODELSCOPE_PREFIX, DEFAULT_MODELSCOPE_REPO, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_MODELSCOPE_REPO)
    parser.add_argument("--prefix", default=DEFAULT_MODELSCOPE_PREFIX)
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".modelscope_cache/haidian_v1"),
        help="Temporary dataset download cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files under the target directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ModelScope download command without downloading files.",
    )
    return parser.parse_args()


def masked_command(cmd: list[str]) -> list[str]:
    printable = list(cmd)
    for idx, item in enumerate(printable[:-1]):
        if item == "--token":
            printable[idx + 1] = "***"
    return printable


def run(cmd: list[str], dry_run: bool = False) -> None:
    printable = masked_command(cmd)
    print("+", " ".join(printable))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def download_with_cli(repo: str, prefix: str, cache_dir: Path, dry_run: bool) -> Path:
    token = os.environ.get("MODELSCOPE_TOKEN")
    include = [f"{prefix.rstrip('/')}/**"]
    cmd = [
        "modelscope",
        "download",
        "--dataset",
        repo,
        "--include",
        *include,
        "--local_dir",
        str(cache_dir),
    ]
    if token:
        cmd.extend(["--token", token])
    run(cmd, dry_run=dry_run)
    return cache_dir


def copy_tree(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"ModelScope prefix not found after download: {src}")
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        out = dst / rel
        if out.exists() and not force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, out)


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_root = download_with_cli(args.repo, args.prefix, args.cache_dir, args.dry_run)
    if args.dry_run:
        return
    src = cache_root / args.prefix
    copy_tree(src, args.target, args.force)
    print(f"Haidian V1 assets installed into {args.target}")


if __name__ == "__main__":
    main()
