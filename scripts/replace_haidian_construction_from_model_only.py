#!/usr/bin/env python3
"""
Replace Haidian construction result tiles with the model-only oracle-threshold
mask set (no OSM, no satellite background).

Archive layout inside the zip:
    construction_model_only_oracle_threshold/mask_only/patch_XXXXXX_mask_model_only.png

These are already 128x128 red-on-white masks. We replicate them across all
monthly period directories so the API works for any requested month.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PIL import Image

TASK = "construction"
OUT_ROOT = Path("data/haidian/tasks")
PERIODS = ["202512", "202601", "202602", "202603", "202604", "202605"]
MASK_PATTERN = re.compile(r"patch_(\d{6})_mask_model_only\.png$")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace Haidian construction tiles with model-only oracle masks."
    )
    parser.add_argument(
        "--zip",
        default="/tmp/hd_construction_only/haidian_v1_construction_model_only_no_osm_oracle_threshold.zip",
        help="Path to the downloaded zip archive",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview actions")
    return parser.parse_args(argv or sys.argv[1:])


def backup_existing_results(dry_run: bool) -> Path:
    task_root = OUT_ROOT / TASK / "v1" / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = task_root / f"backup_{timestamp}"

    png_files = list(task_root.rglob("patch_*.png"))
    if not png_files:
        print("[backup] No existing construction tiles to back up")
        return backup_dir

    if dry_run:
        print(f"[dry-run] Would back up {len(png_files)} files to {backup_dir}")
        return backup_dir

    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in png_files:
        dst = backup_dir / src.relative_to(task_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"[backup] Copied {len(png_files)} files to {backup_dir}")
    return backup_dir


def clear_existing_results(dry_run: bool) -> None:
    """Clear only the live results directories, never the backup dirs."""
    task_root = OUT_ROOT / TASK / "v1" / "results"
    live_dirs = [task_root / "tiles"] + [task_root / p / "tiles" for p in PERIODS]
    png_files = []
    for d in live_dirs:
        if d.exists():
            png_files.extend(d.glob("patch_*.png"))
    if not png_files:
        return
    if dry_run:
        print(f"[dry-run] Would delete {len(png_files)} existing tiles")
        return
    for src in png_files:
        src.unlink()
    print(f"[clear] Deleted {len(png_files)} existing tiles")


def extract_masks(zip_path: Path, dry_run: bool) -> dict[str, Image.Image]:
    masks: dict[str, Image.Image] = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if MASK_PATTERN.search(n)]
        print(f"[extract] Found {len(names)} mask files in zip")
        for name in names:
            match = MASK_PATTERN.search(name)
            patch_id = f"patch_{match.group(1)}"
            if dry_run:
                masks[patch_id] = None
                continue
            data = z.read(name)
            img = Image.open(__import__("io", fromlist=["BytesIO"]).BytesIO(data)).convert("RGB")
            if img.size != (128, 128):
                raise ValueError(f"Unexpected size {img.size} for {name}")
            masks[patch_id] = img
    return masks


def write_results(masks: dict[str, Image.Image], dry_run: bool) -> int:
    task_root = OUT_ROOT / TASK / "v1" / "results"
    dirs = [task_root / "tiles"] + [task_root / p / "tiles" for p in PERIODS]
    count = 0
    for d in dirs:
        if dry_run:
            print(f"[dry-run] Would write {len(masks)} masks to {d}")
            continue
        d.mkdir(parents=True, exist_ok=True)
        for patch_id, img in masks.items():
            out_path = d / f"{patch_id}.png"
            img.save(out_path, "PNG")
            count += 1
    if not dry_run:
        print(f"[write] Wrote {count} tiles across {len(dirs)} directories")
    return count


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    zip_path = Path(args.zip).resolve()
    dry_run = args.dry_run
    print(f"Zip: {zip_path}")
    print(f"Dry run: {dry_run}")

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    backup_existing_results(dry_run)
    clear_existing_results(dry_run)
    masks = extract_masks(zip_path, dry_run)
    write_results(masks, dry_run)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
