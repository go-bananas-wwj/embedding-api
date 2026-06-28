#!/usr/bin/env python3
"""Download Harbin static assets from ModelScope into embedding-api.

The preferred source is the ModelScope dataset
``WeijieWu/xuannv_embdding_api`` under ``harbin/v1/api_ready``.  The assets are
stored as a small number of tar archives (because ModelScope datasets have a
file-count limit).  This script downloads the archives, verifies checksums, and
extracts them to their original locations.

Credentials stay out of the repository: pass a token via ``MODELSCOPE_TOKEN``
when the dataset is private.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from paths import DEFAULT_MODELSCOPE_PREFIX, DEFAULT_MODELSCOPE_REPO, PROJECT_ROOT


ARCHIVE_NAMES = [
    "data_harbin",
    "models_harbin",
    "models_sam3",
    "raw_harbin",
    "raw_harbin_scenes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_MODELSCOPE_REPO)
    parser.add_argument("--prefix", default=DEFAULT_MODELSCOPE_PREFIX)
    parser.add_argument(
        "--target",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root where data/, models/ will be extracted.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".modelscope_cache/harbin_v1"),
        help="Temporary dataset download cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files under the target directory.",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify checksums.sha256 before extraction.",
    )
    parser.add_argument(
        "--skip-raw-scenes",
        action="store_true",
        help="Skip raw satellite scene archives.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    printable = list(cmd)
    for idx, item in enumerate(printable[:-1]):
        if item == "--token":
            printable[idx + 1] = "***"
    print("+", " ".join(printable))
    subprocess.run(cmd, check=True)


def download_with_cli(repo: str, cache_dir: Path) -> Path:
    token = os.environ.get("MODELSCOPE_TOKEN")
    cmd = ["modelscope", "download", "--dataset", repo, "--local_dir", str(cache_dir)]
    if token:
        cmd.extend(["--token", token])
    run(cmd)
    return cache_dir


def verify_checksums(src: Path) -> bool:
    checksum_file = src / "checksums.sha256"
    if not checksum_file.exists():
        print("checksums.sha256 not found, skipping verification")
        return True

    lines = checksum_file.read_text(encoding="utf-8").strip().splitlines()
    mismatches = 0
    for line in lines:
        expected, name = line.split("  ", 1)
        path = src / name
        if not path.exists():
            print(f"MISS: {name}")
            mismatches += 1
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            print(f"MISMATCH: {name}")
            mismatches += 1
    if mismatches:
        print(f"Checksum verification failed: {mismatches} issue(s)")
        return False
    print("Checksum verification passed")
    return True


def extract_archive(archive: Path, target: Path, force: bool) -> None:
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            rel = Path(member.name)
            parts = rel.parts

            # Raw scenes are archived under workspace/raw/... and should be
            # extracted to the real filesystem root.
            if len(parts) >= 3 and parts[0] == "workspace" and parts[1] == "raw":
                out = Path("/") / rel
            else:
                out = target / rel

            if out.exists() and not force:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src_f:
                out.write_bytes(src_f.read())
    print(f"  done: {archive.name}")


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_root = download_with_cli(args.repo, args.cache_dir)
    src = cache_root / args.prefix

    if not src.exists():
        raise FileNotFoundError(f"ModelScope prefix not found after download: {src}")

    if args.verify_checksums:
        if not verify_checksums(src):
            raise SystemExit(1)

    for name in ARCHIVE_NAMES:
        if args.skip_raw_scenes and name.startswith("raw_"):
            print(f"Skipping {name}.tar (raw scenes disabled)")
            continue
        archive = src / f"{name}.tar"
        if not archive.exists():
            print(f"Archive not found: {archive}")
            continue
        extract_archive(archive, args.target, args.force)

    print(f"Harbin assets extracted into {args.target} and /workspace/raw")


if __name__ == "__main__":
    main()
