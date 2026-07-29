#!/usr/bin/env python3
"""Export the three-class comparison as a high-DPI PPT-ready figure."""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

import experiment_no_positive_memory as experiment
from experiment_sparse_retrieval_vs_conv import ROOT


SOURCE = ROOT / "Tmp/pu_query_aef_traditional_single_patch_20260726"
PNG = SOURCE / "玄女少量标注四模型对比_300DPI.png"
PDF = SOURCE / "玄女少量标注四模型对比_高清.pdf"

CASES = [
    (
        "建筑检测 · patch_000205 · 3 个 Polygon",
        "building_extraction",
        [
            ("完整 Patch 光学影像", ""),
            ("P10C 嵌入投影", ""),
            ("模型实际看见的标注", "绿色高亮 3 个 Polygon"),
            ("玄女检测结果", "F1 0.688 · IoU 0.525"),
            ("AEF Pixel MLP", "F1 0.633 · IoU 0.463"),
            ("传统 Sentinel-2 RF", "F1 0.538 · IoU 0.368"),
            ("DINOv3-SAT493M", "F1 0.592 · IoU 0.421"),
        ],
    ),
    (
        "道路检测 · patch_000264 · 2 个 Polygon",
        "road_extraction",
        [
            ("完整 Patch 光学影像", ""),
            ("P10C 嵌入投影", ""),
            ("模型实际看见的标注", "绿色高亮 2 个 Polygon"),
            ("玄女检测结果", "F1 0.706 · IoU 0.545"),
            ("AEF Pixel MLP", "F1 0.512 · IoU 0.344"),
            ("传统 Sentinel-2 RF", "F1 0.424 · IoU 0.269"),
            ("DINOv3-SAT493M", "F1 0.450 · IoU 0.290"),
        ],
    ),
    (
        "水体检测 · patch_000106 · 1 个 Polygon",
        "water",
        [
            ("完整 Patch 光学影像", ""),
            ("P10C 嵌入投影", ""),
            ("模型实际看见的标注", "绿色高亮 1 个 Polygon"),
            ("玄女检测结果", "F1 0.923 · IoU 0.857"),
            ("AEF Pixel MLP", "F1 0.855 · IoU 0.747"),
            ("传统 Sentinel-2 RF", "F1 0.912 · IoU 0.838"),
            ("DINOv3-SAT493M", "F1 0.743 · IoU 0.590"),
        ],
    ),
]

SUFFIXES = ("optical", "pca", "visible", "xuannv", "aef", "traditional", "dino")


def main() -> None:
    experiment.configure_chinese_font()
    figure, axes = plt.subplots(3, 7, figsize=(24, 10.6), constrained_layout=True)
    figure.patch.set_facecolor("#f3f5f7")
    figure.suptitle(
        "同一 Patch 少量标注下的四模型对比",
        fontsize=22,
        fontweight="bold",
    )
    for row, (row_title, prefix, captions) in enumerate(CASES):
        for column, (suffix, caption) in enumerate(zip(SUFFIXES, captions)):
            axis = axes[row, column]
            image = Image.open(SOURCE / f"{prefix}_{suffix}.png").convert("RGB")
            axis.imshow(image)
            axis.axis("off")
            title, detail = caption
            axis.set_title(
                title + (f"\n{detail}" if detail else ""),
                fontsize=11,
                fontweight="bold" if column in (2, 3) else "normal",
                color="#147a52" if column == 3 else "#17212b",
                pad=6,
            )
            if column == 0:
                axis.text(
                    0,
                    1.18,
                    row_title,
                    transform=axis.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    color="#17212b",
                )
    figure.savefig(PNG, dpi=300, facecolor=figure.get_facecolor())
    figure.savefig(PDF, dpi=300, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
