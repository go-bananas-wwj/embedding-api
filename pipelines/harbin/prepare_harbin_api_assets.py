#!/usr/bin/env python3
"""Prepare Harbin static assets for upload to ModelScope.

ModelScope datasets have a recursive item count limit per directory tree
(typically 50,000).  Because Harbin has ~150,000 small files, this script
packages each major asset group into a single tar archive.  The resulting
``api_ready`` directory contains only a handful of archives plus a manifest
and checksums, which can be uploaded without hitting the limit.

On the deployment side ``download_modelscope_assets.py`` downloads these
archives and extracts them back to their original locations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

from paths import (
    DATA_DIR,
    DEFAULT_MODELSCOPE_PREFIX,
    HARBIN_DIR,
    HARBIN_MODELS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RAW_HARBIN_DIR,
    RAW_HARBIN_SCENES_DIR,
    SAM3_MODELS_DIR,
)


ASSET_GROUPS = [
    ("data_harbin", "data/harbin", HARBIN_DIR),
    ("models_harbin", "models/harbin", HARBIN_MODELS_DIR),
    ("models_sam3", "models/sam3", SAM3_MODELS_DIR),
    ("raw_harbin", "workspace/raw/harbin", RAW_HARBIN_DIR),
    ("raw_harbin_scenes", "workspace/raw/harbin_scenes", RAW_HARBIN_SCENES_DIR),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/modelscope_upload/harbin/v1"),
        help="Root directory where api_ready/ will be created.",
    )
    parser.add_argument(
        "--skip-raw-scenes",
        action="store_true",
        help="Skip raw satellite scenes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be archived without writing anything.",
    )
    return parser.parse_args()


def _resolve_symlink(src: Path) -> Path:
    """Return the real file/dir for a symlink."""
    if src.is_symlink():
        target = src.readlink()
        if not target.is_absolute():
            target = src.parent / target
        return target.resolve()
    return src


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() or path.is_symlink():
            yield path


def _human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def _archive_group(
    archive_name: str,
    rel_prefix: str,
    src: Path,
    api_root: Path,
    dry_run: bool,
) -> dict[str, object]:
    if not src.exists():
        print(f"  SKIP (missing): {rel_prefix}")
        return {"source": str(src), "files": 0, "bytes": 0, "archive": None}

    archive_path = api_root / f"{archive_name}.tar"
    file_count = 0
    byte_count = 0

    if dry_run:
        for p in _iter_files(src):
            real_p = _resolve_symlink(p)
            try:
                byte_count += real_p.stat().st_size
            except OSError:
                pass
            file_count += 1
        print(f"  WOULD ARCHIVE {archive_name}.tar: {file_count} files, {_human_size(byte_count)}")
        return {
            "source": str(src),
            "files": file_count,
            "bytes": byte_count,
            "archive": str(archive_path),
        }

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(archive_path, "w") as tar:
        for p in _iter_files(src):
            real_p = _resolve_symlink(p)
            arcname = f"{rel_prefix}/{p.relative_to(src).as_posix()}"
            tar.add(real_p, arcname=arcname)
            try:
                byte_count += real_p.stat().st_size
            except OSError:
                pass
            file_count += 1

    print(f"  ARCHIVED {archive_name}.tar: {file_count} files, {_human_size(byte_count)}")
    return {
        "source": str(src),
        "files": file_count,
        "bytes": byte_count,
        "archive": str(archive_path),
        "human_size": _human_size(byte_count),
    }


def write_manifest(root: Path, summary: dict) -> None:
    manifest = {
        "region": "harbin",
        "api_version": "v1/v2",
        "model_family": "xuannv_show",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "api_ready_prefix": DEFAULT_MODELSCOPE_PREFIX,
        "summary": summary,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()} {path.name}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    api_root = args.output_root / "api_ready"

    if not args.dry_run:
        api_root.mkdir(parents=True, exist_ok=True)
        # Remove any previous per-file copy results if present.
        for child in list(api_root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            elif child.name not in {"manifest.json", "checksums.sha256"}:
                child.unlink()

    summary = {
        "output_root": str(args.output_root),
        "dry_run": args.dry_run,
        "assets": {},
    }

    total_files = 0
    total_bytes = 0

    for archive_name, rel_prefix, src in ASSET_GROUPS:
        if args.skip_raw_scenes and archive_name.startswith("raw_"):
            print(f"  SKIP (raw scenes disabled): {archive_name}")
            continue
        info = _archive_group(archive_name, rel_prefix, src, api_root, args.dry_run)
        summary["assets"][archive_name] = info
        total_files += info.get("files", 0)
        total_bytes += info.get("bytes", 0)

    summary["total"] = {
        "files": total_files,
        "bytes": total_bytes,
        "human_size": _human_size(total_bytes),
    }

    if args.dry_run:
        print(f"\nDRY RUN total: {total_files} files, {_human_size(total_bytes)}")
        return

    write_manifest(api_root, summary)
    write_checksums(api_root)
    print(f"\nPrepared {api_root}")
    print(f"Total: {total_files} files, {_human_size(total_bytes)}")
    print("Upload with:")
    print(f"  modelscope upload --repo-type dataset WeijieWu/xuannv_embdding_api {api_root} harbin/v1/api_ready")


if __name__ == "__main__":
    main()
