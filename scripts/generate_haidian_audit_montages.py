#!/usr/bin/env python3
"""Generate spatial full-domain montages for Haidian task results.

Reads patch bounds from ``data/haidian/patches_meta_v1.json`` and arranges each
per-patch result tile at its true geographic location on a regular grid. Missing
cells are filled with gray.

Outputs are written to ``test_output/haidian_audit_montages/``.
"""

import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHES_META = PROJECT_ROOT / "data/haidian/patches_meta_v1.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "haidian_audit_montages"

# Task label -> (period name, source PNG resolver)
TASK_SOURCES: List[Tuple[str, str, Callable[[str], Optional[Path]]]] = [
    (
        "building_extraction",
        "v1",
        lambda pid: PROJECT_ROOT / "data/haidian/tasks/building_extraction/v1/results/tiles" / f"{pid}.png",
    ),
    (
        "road_extraction",
        "v1",
        lambda pid: PROJECT_ROOT / "data/haidian/tasks/road_extraction/v1/results/tiles" / f"{pid}.png",
    ),
    (
        "construction",
        "gt_20260701",
        lambda pid: PROJECT_ROOT
        / "data/haidian/v1/reports/haidian_construction_gt_patch_labels_20260701/labels"
        / f"haidian_construction_gt_{pid}.png",
    ),
    (
        "land_use_classification",
        "v1",
        lambda pid: PROJECT_ROOT / "data/haidian/tasks/land_use_classification/v1/results/tiles" / f"{pid}.png",
    ),
    (
        "land_cover_classification",
        "v1",
        lambda pid: PROJECT_ROOT / "data/haidian/tasks/land_cover_classification/v1/results/tiles" / f"{pid}.png",
    ),
    (
        "water_extraction",
        "v1",
        lambda pid: PROJECT_ROOT / "data/haidian/tasks/water_extraction/v1/results/tiles" / f"{pid}.png",
    ),
]


def load_patches() -> List[Dict]:
    with open(PATCHES_META, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("patches", [])
    return data


def build_grid(patches: List[Dict]) -> Tuple[Dict[float, int], Dict[float, int], int, int]:
    xs = sorted({p["bounds"][0] for p in patches})
    ys = sorted({p["bounds"][1] for p in patches})
    col_of = {x: i for i, x in enumerate(xs)}
    row_of = {y: i for i, y in enumerate(ys)}
    return col_of, row_of, len(xs), len(ys)


def make_montage(
    patches: List[Dict],
    source_resolver: Callable[[str], Optional[Path]],
    output_path: Path,
    title: str,
) -> Dict:
    col_of, row_of, ncols, nrows = build_grid(patches)
    tile_w = tile_h = 128
    canvas = Image.new("RGB", (ncols * tile_w, nrows * tile_h), (220, 220, 220))
    found = 0
    missing = 0
    for p in patches:
        src = source_resolver(p["patch_id"])
        if not src or not src.exists():
            missing += 1
            continue
        img = Image.open(src).convert("RGB")
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), Image.NEAREST)
        x = col_of[p["bounds"][0]] * tile_w
        y = (nrows - 1 - row_of[p["bounds"][1]]) * tile_h
        canvas.paste(img, (x, y))
        found += 1

    # Title banner at the bottom so it doesn't obscure the geographic layout
    banner_h = 32
    full = Image.new("RGB", (canvas.width, canvas.height + banner_h), (255, 255, 255))
    full.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(full)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, canvas.height + 6), title, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full.save(output_path, "PNG")
    return {
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "found_tiles": found,
        "missing_tiles": missing,
        "grid": f"{ncols}x{nrows}",
        "canvas": f"{full.width}x{full.height}",
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    patches = load_patches()
    summary = []
    for task, period, resolver in TASK_SOURCES:
        output_path = OUTPUT_DIR / f"{task}_{period}_spatial.png"
        info = make_montage(patches, resolver, output_path, f"Haidian {task} ({period})")
        info["task"] = task
        info["period"] = period
        summary.append(info)
        print(f"{task}: {info['found_tiles']}/{len(patches)} tiles -> {info['output']}")

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
