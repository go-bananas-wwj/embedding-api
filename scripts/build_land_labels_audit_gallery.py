#!/usr/bin/env python3
"""Build a visual audit gallery for Haidian land-cover and land-use results."""
from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Tmp/land_labels_audit_20260724"
MONTH = "202604"
ASSET_VERSION = "20260724-independent-conv3x3-v3"
PATCH_IDS = (
    "patch_000000",
    "patch_000024",
    "patch_000043",
    "patch_000071",
    "patch_000107",
    "patch_000128",
)

LAND_COVER = [
    ("永久性水体", "#1E64DC"),
    ("灌木地", "#B4D250"),
    ("草地", "#F5DC5A"),
    ("耕地", "#D23C3C"),
    ("建成区", "#BEAA82"),
    ("裸地/稀疏植被", "#A0DCDC"),
    ("树木覆盖", "#006400"),
]
LAND_USE = [
    ("水体", "#286EE6"),
    ("树木", "#46B450"),
    ("草地", "#F5DC5A"),
    ("农作物", "#FFB496"),
    ("灌木与矮林", "#E63C28"),
    ("建成区", "#6E6E6E"),
    ("裸地", "#965A46"),
]


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _save_optical(path: Path, output: Path) -> None:
    with rasterio.open(path) as dataset:
        data = dataset.read()
    bands = data[:3].astype(np.float32)
    rendered = np.zeros_like(bands, dtype=np.uint8)
    for index, band in enumerate(bands):
        valid = np.isfinite(band)
        low, high = np.percentile(band[valid], (2, 98))
        rendered[index] = np.clip((band - low) / max(high - low, 1e-6) * 255, 0, 255)
    Image.fromarray(np.moveaxis(rendered, 0, -1)).save(output)


def _distribution(path: Path, classes: list[tuple[str, str]]) -> str:
    pixels = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)
    counts = Counter(map(tuple, pixels.tolist()))
    total = len(pixels)
    rows = []
    for name, color in classes:
        count = counts.get(_rgb(color), 0)
        rows.append(
            f'<span class="legend{" absent" if not count else ""}">'
            f'<i style="background:{color}"></i>'
            f"{html.escape(name)} <b>{count / total:.1%}</b></span>"
        )
    return "".join(rows)


def _audit(root: Path, classes: list[tuple[str, str]]) -> tuple[int, list[str]]:
    allowed = {_rgb(color) for _, color in classes}
    files = list(root.rglob("*.png"))
    unexpected = set()
    for path in files:
        pixels = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)
        unexpected.update(map(tuple, np.unique(pixels, axis=0).tolist()))
    return len(files), sorted(
        "#" + "".join(f"{channel:02X}" for channel in color)
        for color in unexpected - allowed
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cover_root = ROOT / "data/haidian/tasks/land_cover_classification/v1/results"
    use_root = ROOT / "data/haidian/tasks/land_use_classification/v1/results"
    optical_root = (
        ROOT
        / "data/haidian/archive/processed_training_data/extracted/patches/highres_optical"
    )
    cover_count, cover_unexpected = _audit(cover_root, LAND_COVER)
    use_count, use_unexpected = _audit(use_root, LAND_USE)

    sections = []
    for patch_id in PATCH_IDS:
        optical = optical_root / f"highres_optical_20260401_{patch_id}.tif"
        cover = cover_root / MONTH / "tiles" / f"{patch_id}.png"
        use = use_root / MONTH / "tiles" / f"{patch_id}.png"
        optical_output = OUTPUT / f"{patch_id}_optical.png"
        _save_optical(optical, optical_output)
        sections.append(
            f"""
            <section>
              <h2>{patch_id} <small>2026 年 4 月</small></h2>
              <div class="grid">
                <figure><img src="{optical_output.name}" alt="{patch_id} 高分辨率光学影像">
                  <figcaption>高分辨率光学影像</figcaption></figure>
                <figure><img src="../../{cover.relative_to(ROOT)}?v={ASSET_VERSION}" alt="{patch_id} 土地覆盖">
                  <figcaption>土地覆盖（独立 Conv 3×3 模型）</figcaption>
                  <div class="legends">{_distribution(cover, LAND_COVER)}</div></figure>
                <figure><img src="../../{use.relative_to(ROOT)}?v={ASSET_VERSION}" alt="{patch_id} 土地利用">
                  <figcaption>土地利用（独立 Conv 3×3 模型）</figcaption>
                  <div class="legends">{_distribution(use, LAND_USE)}</div></figure>
              </div>
            </section>
            """
        )

    cover_status = "通过" if not cover_unexpected else f"异常颜色：{cover_unexpected}"
    use_status = "通过" if not use_unexpected else f"异常颜色：{use_unexpected}"
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>海淀土地覆盖与土地利用标签审查</title>
<style>
body{{margin:0;background:#f5f7f8;color:#172026;font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1680px;margin:auto;padding:24px}}h1{{font-size:28px;margin:0 0 8px}}h2{{font-size:19px}}
small{{font-weight:400;color:#66737c}}.summary{{background:#fff;border-left:4px solid #16845b;padding:14px 18px}}
section{{background:#fff;border-top:1px solid #dfe5e8;padding:18px 0 28px;margin-top:22px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}
figure{{margin:0}}img{{width:100%;aspect-ratio:1;object-fit:cover;image-rendering:auto;cursor:zoom-in;border:1px solid #d8dee2}}
figcaption{{font-weight:700;margin:8px 0}}.legends{{display:flex;flex-wrap:wrap;gap:6px 12px}}
.legend{{white-space:nowrap;color:#4b5860}}.legend.absent{{opacity:.48}}.legend i{{display:inline-block;width:11px;height:11px;margin-right:5px;border:1px solid #77838a}}
dialog{{border:0;padding:0;background:transparent;max-width:96vw;max-height:96vh}}dialog img{{max-width:96vw;max-height:92vh;width:auto}}
dialog::backdrop{{background:rgba(8,13,16,.9)}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>海淀土地覆盖与土地利用标签审查</h1>
<p>同一 Patch 横向对照高分辨率光学影像、土地覆盖结果与土地利用结果。点击图片可放大。</p>
<div class="summary"><b>自动核验：</b>土地覆盖 {cover_count} 张 PNG，{cover_status}；土地利用 {use_count} 张 PNG，{use_status}。<br>
土地覆盖已修正：类别 1 为蓝色永久性水体，类别 8 为绿色树木覆盖。土地利用仅保留有可靠监督的 7 类，不再输出淹水植被和冰雪。<br>
<b>两版独立模型：</b>两者均输入海淀 P10C v1 的 64 维月度 embedding，但分别随机初始化、训练和保存权重。
土地覆盖模型使用 WorldCover 物理地表覆盖监督；土地利用模型使用独立类别编码，并由 OSM 建筑和道路约束人类活动建成区。
两个模型的 checkpoint 分别为 <code>land_cover_conv3x3_best.pt</code> 和
<code>land_use_conv3x3_best.pt</code>。模型结构均为
<code>64 → Conv 3×3(128) → GroupNorm → GELU → Conv 1×1(7)</code>，
采用类别加权交叉熵训练 20 轮，固定 256 个训练 Patch、64 个验证 Patch。
推理后仅移除小于 4 像素且不与较大水体相连的孤立水体区域。</div>
{''.join(sections)}
</main><dialog id="viewer"><img alt="放大预览"></dialog>
<script>const d=document.querySelector('#viewer'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script>
</body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
