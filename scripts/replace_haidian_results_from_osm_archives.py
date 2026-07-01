#!/usr/bin/env python3
"""
Replace Haidian downstream-task result tiles with the OSM-assisted diagnostic
post-processed panels extracted from monthly tar archives.

Archive layout:
    {YYYYMM}/{task}/haidian_{YYYYMM}_{task}_patch_{XXXXXX}_osm_diagnostic.png

New archive tasks are mapped to the existing on-disk task IDs:
    building_extraction       -> building_extraction
    road_extraction           -> road_extraction
    construction_site_extraction -> construction
    water_extraction          -> water_extraction

The diagnostic PNG is a 1312x342 strip containing panels of approximately
256x256 with their top-left y at 64 and x positions [16, 278, 540, 802, 1064].
Panel index 3 (the 4th panel, labeled "后处理结果") is fully contained.
The post-processed panel is converted to a binary mask: foreground pixels in
the task color (red for building/road/construction, blue for water) on a white
background, then resized to 128x128.

Fallback tiles in `results/tiles/` are hardlinked (or copied) from the default
period `202605` after all months have been processed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Geometry of the diagnostic strip, verified from archive samples.
DIAG_SIZE = (1312, 342)
PANEL_SIZE = (256, 256)
PANEL_Y = 64
PANEL_XS = [16, 278, 540, 802, 1064]
POST_PROCESS_PANEL_INDEX = 3  # 4th panel, "后处理结果"

OUTPUT_SIZE = (128, 128)
RESAMPLE = Image.LANCZOS

# Mapping from the task directory names inside the archives to the on-disk task
# IDs used by the Embedding API for Haidian.
ARCHIVE_TO_TASK: Dict[str, str] = {
    "building_extraction": "building_extraction",
    "road_extraction": "road_extraction",
    "construction_site_extraction": "construction",
    "water_extraction": "water_extraction",
}

# Tasks whose old `results/tiles/` directory should be backed up and cleared.
# `construction_joint` is intentionally omitted: it is not present in the new
# archives and its old results must remain untouched.
TASKS_TO_REPLACE = list(ARCHIVE_TO_TASK.values())

# The default period used to create fallback tiles in `results/tiles/`.
DEFAULT_PERIOD = "202605"

# Archive filename pattern.
ARCHIVE_GLOB = "haidian_v1_*_monthly_osm_assisted_patch_tiles.tar"

# Regex extracting month and patch id from archive member paths.
MEMBER_RE = re.compile(
    r"^(?P<period>\d{6})/(?P<archive_task>[a-z_]+)/"
    r"haidian_\d{6}_(?P<archive_task2>[a-z_]+)_patch_(?P<patch_id>\d{6})_osm_diagnostic\.png$"
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace Haidian result tiles from OSM-assisted diagnostic archives."
    )
    parser.add_argument(
        "--archives-dir",
        default="/tmp/hd_osm_archives",
        help="Directory containing the monthly tar archives (default: /tmp/hd_osm_archives)",
    )
    parser.add_argument(
        "--out-root",
        default="data/haidian/tasks",
        help="Root directory for Haidian task outputs (default: data/haidian/tasks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without writing files.",
    )
    return parser.parse_args(argv)


def find_archives(archives_dir: Path) -> List[Path]:
    """Return sorted list of monthly tar archives."""
    archives = sorted(archives_dir.glob(ARCHIVE_GLOB))
    if not archives:
        raise FileNotFoundError(
            f"No archives matching {ARCHIVE_GLOB!r} found in {archives_dir}"
        )
    return archives


def backup_old_tiles(out_root: Path, dry_run: bool) -> List[Tuple[str, Path]]:
    """
    Back up existing `results/tiles/*.png` for every task being replaced.
    Also back up `construction_joint` tiles if they exist, without deleting
    them afterwards.

    Returns a list of (task_id, backup_dir) tuples.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dirs: List[Tuple[str, Path]] = []

    # Tasks whose old tiles will be replaced.
    tasks_to_backup = set(TASKS_TO_REPLACE)
    # construction_joint gets a backup too, but is not cleared later.
    tasks_to_backup.add("construction_joint")

    for task in sorted(tasks_to_backup):
        src_dir = out_root / task / "v1" / "results" / "tiles"
        if not src_dir.exists():
            print(f"[backup] No source dir for {task}, skipping.")
            continue

        png_files = list(src_dir.glob("patch_*.png"))
        if not png_files:
            print(f"[backup] No patch_*.png files in {src_dir}, skipping.")
            continue

        backup_dir = out_root / task / "v1" / "results" / f"tiles_backup_{timestamp}"
        backup_dirs.append((task, backup_dir))

        if dry_run:
            print(f"[dry-run] Would back up {len(png_files)} files from {src_dir} to {backup_dir}")
            continue

        backup_dir.mkdir(parents=True, exist_ok=True)
        for src in png_files:
            shutil.copy2(src, backup_dir / src.name)
        print(f"[backup] {task}: copied {len(png_files)} files to {backup_dir}")

    return backup_dirs


def clear_old_tiles(out_root: Path, dry_run: bool) -> None:
    """Delete old `patch_*.png` files in the tasks being replaced."""
    for task in TASKS_TO_REPLACE:
        src_dir = out_root / task / "v1" / "results" / "tiles"
        if not src_dir.exists():
            continue
        png_files = list(src_dir.glob("patch_*.png"))
        if not png_files:
            continue

        if dry_run:
            print(f"[dry-run] Would delete {len(png_files)} files from {src_dir}")
            continue

        for src in png_files:
            src.unlink()
        print(f"[clear] {task}: deleted {len(png_files)} old tiles")


def _panel_to_mask(panel: Image.Image, task: str) -> Image.Image:
    """Convert the post-processed panel to a binary mask on white background."""
    arr = np.array(panel.convert("RGB"))
    hsv = np.array(panel.convert("HSV"))
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if task == "water_extraction":
        # Blue foreground used in the water post-processed panel.
        mask = ((hue > 90) & (hue < 140) & (sat > 50) & (val > 50))
        fg = np.array([0, 0, 255], dtype=np.uint8)
    else:
        # Red foreground used in building/road/construction panels.
        mask = (((hue < 15) | (hue > 165)) & (sat > 50) & (val > 50))
        fg = np.array([255, 0, 0], dtype=np.uint8)

    out = np.full_like(arr, 255)
    out[mask] = fg
    return Image.fromarray(out)


def extract_post_processed_mask(
    tar: tarfile.TarFile, member: tarfile.TarInfo, task: str
) -> Image.Image:
    """Extract a diagnostic strip member and return the resized binary mask."""
    f = tar.extractfile(member)
    if f is None:
        raise ValueError(f"Member {member.name} is not a regular file")

    img = Image.open(f).convert("RGB")
    if img.size != DIAG_SIZE:
        raise ValueError(
            f"Unexpected diagnostic size {img.size} for {member.name}, expected {DIAG_SIZE}"
        )

    x = PANEL_XS[POST_PROCESS_PANEL_INDEX]
    y = PANEL_Y
    panel = img.crop((x, y, x + PANEL_SIZE[0], y + PANEL_SIZE[1]))
    mask = _panel_to_mask(panel, task)
    return mask.resize(OUTPUT_SIZE, Image.Resampling.NEAREST)


def ensure_water_extraction_dirs(out_root: Path, dry_run: bool) -> None:
    """Create empty predictions/ and labels/ dirs for water_extraction if absent."""
    task_root = out_root / "water_extraction" / "v1"
    for sub in ("predictions", "labels"):
        d = task_root / sub
        if dry_run:
            if not d.exists():
                print(f"[dry-run] Would create {d}")
            continue
        d.mkdir(parents=True, exist_ok=True)
        print(f"[mkdir] {d}")


def process_archives(archives: List[Path], out_root: Path, dry_run: bool) -> int:
    """
    Extract and convert all archive members. Returns the number of tiles written.
    """
    processed = 0
    skipped = 0

    for archive_path in archives:
        print(f"[archive] Processing {archive_path.name}")
        with tarfile.open(archive_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            for member in members:
                match = MEMBER_RE.match(member.name)
                if not match:
                    print(f"[warn] Skipping unexpected member: {member.name}")
                    skipped += 1
                    continue

                period = match.group("period")
                archive_task = match.group("archive_task")
                archive_task2 = match.group("archive_task2")
                patch_id = f"patch_{match.group('patch_id')}"

                if archive_task != archive_task2:
                    print(
                        f"[warn] Task mismatch in {member.name}: "
                        f"{archive_task} vs {archive_task2}"
                    )
                    skipped += 1
                    continue

                task = ARCHIVE_TO_TASK.get(archive_task)
                if task is None:
                    # construction_site_extraction is mapped to construction;
                    # any other unknown task is skipped.
                    print(f"[warn] Unknown archive task {archive_task!r}, skipping {member.name}")
                    skipped += 1
                    continue

                out_dir = out_root / task / "v1" / "results" / period / "tiles"
                out_path = out_dir / f"{patch_id}.png"

                if dry_run:
                    print(f"[dry-run] Would write {out_path}")
                    processed += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                mask = extract_post_processed_mask(tar, member, task)
                mask.save(out_path, "PNG")
                processed += 1

        print(f"[archive] Finished {archive_path.name}")

    print(f"[summary] Converted {processed} tiles, skipped {skipped} members")
    return processed


def create_fallback_tiles(out_root: Path, dry_run: bool) -> int:
    """
    Hardlink (or copy, if cross-device) fallback tiles from the default period
    into each task's `results/tiles/` directory.
    """
    total_created = 0
    for task in TASKS_TO_REPLACE:
        period_dir = out_root / task / "v1" / "results" / DEFAULT_PERIOD / "tiles"
        fallback_dir = out_root / task / "v1" / "results" / "tiles"

        if not period_dir.exists():
            print(f"[warn] Period dir {period_dir} missing, cannot create fallback for {task}")
            continue

        if dry_run:
            count = len(list(period_dir.glob("patch_*.png")))
            print(f"[dry-run] Would create {count} fallback tiles in {fallback_dir}")
            total_created += count
            continue

        fallback_dir.mkdir(parents=True, exist_ok=True)
        task_created = 0
        for src in sorted(period_dir.glob("patch_*.png")):
            dst = fallback_dir / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            task_created += 1
        total_created += task_created
        print(f"[fallback] {task}: created {task_created} fallback tiles from {DEFAULT_PERIOD}")

    return total_created


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    archives_dir = Path(args.archives_dir).resolve()
    out_root = Path(args.out_root).resolve()
    dry_run = args.dry_run

    print(f"Archives dir: {archives_dir}")
    print(f"Output root:  {out_root}")
    print(f"Dry run:      {dry_run}")

    archives = find_archives(archives_dir)
    print(f"Found {len(archives)} archive(s)")

    backup_dirs = backup_old_tiles(out_root, dry_run)
    clear_old_tiles(out_root, dry_run)
    ensure_water_extraction_dirs(out_root, dry_run)
    process_archives(archives, out_root, dry_run)

    if dry_run:
        print("[dry-run] Would create fallback tiles from period 202605")
    else:
        create_fallback_tiles(out_root, dry_run)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
