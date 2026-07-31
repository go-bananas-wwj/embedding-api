"""Build a focused Chinese gallery for the optical texture experiment."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import font_manager
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
ROLE_LABELS = {
    "training": "原训练标注（疑似错标）",
    "independent_osm": "独立 OSM 测试",
    "global_high_false_positive": "极端高误检复核",
    "spatial_high_score": "中右下高分复核",
}
GROUP_LABELS = (
    ("training", "原训练标注（疑似错标）"),
    ("independent_osm", "独立 OSM 操场"),
    ("global_high_false_positive", "全域极端高误检代表"),
    ("spatial_high_score", "区域中右下高分代表"),
)
TRAINING_VISUAL_REVIEWS = {
    "patch_000059": "视觉核验：标注主要落在院落和建筑区域，疑似不是操场。",
    "patch_000060": "视觉核验：标注位于建筑旁的小块区域，疑似不是操场。",
    "patch_000064": (
        "视觉核验：标注落在左侧建成区；真实蓝绿操场位于影像下方，"
        "原标注疑似错位。"
    ),
}
VIEW_TITLES = (
    ("optical", "高分辨率光学影像"),
    ("reference_overlay", "参考 Polygon 叠加"),
    ("score_heatmap", "PU 连续得分"),
    ("texture_boundary", "光学纹理边界"),
    ("baseline", "原始阈值"),
    ("guarded", "现有全局面积保护"),
    ("area_guard", "相对种子 + 面积筛选"),
    ("texture_boundary_area_guard", "纹理边界 + 面积筛选"),
    ("texture_overlay", "纹理方案叠加"),
)
VARIANT_LABELS = {
    "baseline": "原始阈值",
    "guarded": "现有全局面积保护",
    "area_guard": "相对种子 + 面积筛选",
    "texture_boundary_area_guard": "纹理边界 + 面积筛选",
}


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_array(input_dir: Path, relative_path: str) -> np.ndarray:
    path = input_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Gallery array not found: {path}")
    return np.load(path, allow_pickle=False)


def _stretch(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    finite = values[np.isfinite(values)]
    nonzero = finite[finite != 0]
    sample = nonzero if len(nonzero) >= max(32, int(finite.size * 0.02)) else finite
    if not len(sample):
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.quantile(sample, [0.02, 0.98])
    if high <= low:
        high = low + 1.0
    return np.round(
        np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _read_optical(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Optical image not found: {path}")
    with rasterio.open(path) as dataset:
        count = min(3, dataset.count)
        values = dataset.read(list(range(1, count + 1)))
    if count == 1:
        values = np.repeat(values, 3, axis=0)
    elif count == 2:
        values = np.concatenate([values, values[-1:]], axis=0)
    return np.stack([_stretch(values[index]) for index in range(3)], axis=-1)


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    source = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(
        source.resize(size, resample=Image.Resampling.NEAREST)
    ) > 0


def _save_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path, optimize=True)


def _mask_image(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    resized = _resize_mask(mask, size)
    image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    image[resized] = np.array([218, 42, 52], dtype=np.uint8)
    return image


def _reference_overlay(optical: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = optical.copy()
    if not np.asarray(reference, dtype=bool).any():
        return result
    boundary = np.logical_xor(
        ndimage.binary_dilation(reference, iterations=1),
        ndimage.binary_erosion(reference, iterations=1),
    )
    boundary = _resize_mask(
        boundary,
        (optical.shape[1], optical.shape[0]),
    )
    boundary = ndimage.binary_dilation(boundary, iterations=2)
    result[boundary] = np.array([255, 218, 52], dtype=np.uint8)
    return result


def _prediction_overlay(optical: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = optical.astype(np.float32)
    mask = _resize_mask(
        prediction,
        (optical.shape[1], optical.shape[0]),
    )
    red = np.array([232.0, 37.0, 56.0], dtype=np.float32)
    result[mask] = 0.48 * result[mask] + 0.52 * red
    boundary = np.logical_xor(
        ndimage.binary_dilation(mask, iterations=1),
        ndimage.binary_erosion(mask, iterations=1),
    )
    result[boundary] = red
    return np.clip(result, 0, 255).astype(np.uint8)


def _colored_map(
    values: np.ndarray,
    low: float,
    high: float,
    size: Tuple[int, int],
    cmap_name: str,
) -> np.ndarray:
    normalized = np.clip(
        (np.asarray(values, dtype=np.float32) - low) / max(high - low, 1e-6),
        0.0,
        1.0,
    )
    rgb = np.round(
        matplotlib.colormaps[cmap_name](normalized)[..., :3] * 255.0
    ).astype(np.uint8)
    return np.asarray(
        Image.fromarray(rgb).resize(size, resample=Image.Resampling.BILINEAR)
    )


def _font(assets_dir: Path) -> Optional[font_manager.FontProperties]:
    candidates = (
        assets_dir / "NotoSansCJKsc-Regular.otf",
        ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf",
        ROOT
        / "Tmp/playground_pu_query_20260731/assets/NotoSansCJKsc-Regular.otf",
    )
    for candidate in candidates:
        if candidate.is_file():
            return font_manager.FontProperties(fname=str(candidate))
    return None


def _plot_selected_area(
    path: Path,
    metrics: Mapping[str, Any],
    font: Optional[font_manager.FontProperties],
) -> None:
    patch_ids = list(metrics["per_patch"])
    x = np.arange(len(patch_ids))
    width = 0.19
    colors = ("#C93B30", "#767676", "#176B87", "#2E8B57")
    fig, axis = plt.subplots(figsize=(11.5, 4.3), dpi=150)
    for index, (name, label) in enumerate(VARIANT_LABELS.items()):
        values = [
            metrics["per_patch"][patch_id]["variants"][name]["positive_ratio"]
            * 100.0
            for patch_id in patch_ids
        ]
        axis.bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            color=colors[index],
            label=label,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(patch_ids, rotation=24, ha="right")
    axis.set_ylabel("预测面积占比（%）", fontproperties=font)
    axis.set_title("典型 Patch 方案对比", fontproperties=font)
    axis.legend(frameon=False, prop=font, ncols=2)
    axis.grid(axis="y", color="#D5D5D5", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_reference_metrics(
    path: Path,
    metrics: Mapping[str, Any],
    font: Optional[font_manager.FontProperties],
) -> None:
    variants = list(VARIANT_LABELS)
    labels = [VARIANT_LABELS[name] for name in variants]
    training = metrics["reference_relative_metrics"]["training_polygons"]
    osm = metrics["reference_relative_metrics"]["independent_osm_polygon"]
    x = np.arange(len(variants))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 4.1), dpi=150)
    axis.bar(
        x - width / 2,
        [training[name]["f1"] for name in variants],
        width,
        label="原训练标注（疑似错标）参考相对 F1",
        color="#176B87",
    )
    axis.bar(
        x + width / 2,
        [osm[name]["f1"] for name in variants],
        width,
        label="独立 OSM 参考相对 F1",
        color="#2E8B57",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontproperties=font, rotation=12, ha="right")
    axis.set_ylim(0, 0.65)
    axis.set_ylabel("参考相对 F1", fontproperties=font)
    axis.set_title("已知 Polygon 上的参考相对指标", fontproperties=font)
    axis.legend(frameon=False, prop=font)
    axis.grid(axis="y", color="#D5D5D5", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _percent(value: float) -> str:
    return f"{float(value) * 100.0:.2f}%"


def _selection_reason(
    selection: Mapping[str, Any],
    patch_id: str,
) -> Mapping[str, Any]:
    return next(
        item
        for item in selection["selection_evidence"]
        if item["patch_id"] == patch_id
    )


def _patch_section(
    patch_id: str,
    artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    urls: Mapping[str, str],
    reason: Mapping[str, Any],
) -> str:
    role = ROLE_LABELS[artifact["role"]]
    patch_metrics = metrics["per_patch"][patch_id]["variants"]
    figures = []
    for key, title in VIEW_TITLES:
        if artifact["role"] == "training" and key == "reference_overlay":
            title = "原训练标注（疑似错标）叠加"
        suffix = ""
        if key in patch_metrics:
            suffix = " · " + _percent(patch_metrics[key]["positive_ratio"])
        url = html.escape(urls[key], quote=True)
        label = html.escape(f"{patch_id} {title}", quote=True)
        figures.append(
            f"""
            <figure>
              <button class="image-button" type="button"
                      data-full-image="{url}" data-image-label="{label}"
                      aria-label="点击放大 {label}">
                <img src="{url}" alt="{label}" loading="lazy">
              </button>
              <figcaption>{html.escape(title)}{suffix}</figcaption>
            </figure>"""
        )
    center = reason["center_wgs84"]
    visual_review = TRAINING_VISUAL_REVIEWS.get(patch_id)
    review_html = (
        f'<p class="visual-review">{html.escape(visual_review)}</p>'
        if visual_review
        else ""
    )
    return f"""
    <section class="patch-section" data-patch-id="{html.escape(patch_id, quote=True)}">
      <div class="patch-heading">
        <h3>{html.escape(patch_id)}</h3>
        <span class="role">{html.escape(role)}</span>
        <span class="reason">{html.escape(reason["reason"])}</span>
        <span class="coords">{center[0]:.5f}, {center[1]:.5f}</span>
      </div>
      {review_html}
      <div class="patch-grid">{''.join(figures)}</div>
    </section>"""


def _build_html(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    image_urls: Mapping[str, Mapping[str, str]],
    chart_urls: Mapping[str, str],
    score_range: Tuple[float, float],
) -> str:
    assessment = metrics["texture_assessment"]
    sections = []
    for group_key, group_title in GROUP_LABELS:
        rows = []
        for patch_id in manifest["selection"][group_key]:
            rows.append(
                _patch_section(
                    patch_id,
                    manifest["artifacts"]["arrays"][patch_id],
                    metrics,
                    image_urls[patch_id],
                    _selection_reason(manifest["selection"], patch_id),
                )
            )
        sections.append(
            f'<div class="sample-group"><h2>{group_title}</h2>'
            + "".join(rows)
            + "</div>"
        )
    osm = metrics["reference_relative_metrics"]["independent_osm_polygon"]
    area = osm["area_guard"]
    texture = osm["texture_boundary_area_guard"]
    low, high = score_range
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>操场纹理边界实验</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026; --muted: #59676f; --line: #d9dfe2;
      --page: #f4f7f8; --surface: #fff; --red: #da2a34;
      --yellow: #ffda34; --teal: #176b87; --green: #2e8b57;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; color: var(--ink); background: var(--page);
      font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
      letter-spacing: 0;
    }}
    main {{ max-width: 1900px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 500; }}
    h2 {{ margin: 30px 0 10px; font-size: 21px; font-weight: 500; }}
    h3 {{ margin: 0; font-size: 16px; font-weight: 500; }}
    p {{ margin: 0; }}
    .subtitle, .reason, .coords {{ color: var(--muted); }}
    .summary {{
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px; margin: 20px 0;
    }}
    .stat {{
      background: var(--surface); border: 1px solid var(--line);
      border-radius: 6px; padding: 14px;
    }}
    .stat small {{ display: block; color: var(--muted); }}
    .stat strong {{ display: block; margin-top: 4px; font-size: 21px; font-weight: 500; }}
    .notice {{
      background: var(--surface); border-left: 4px solid var(--yellow);
      padding: 12px 14px; margin: 12px 0;
    }}
    .p1-warning {{
      background: #fff0f0; border: 2px solid #b4232d;
      color: #771721; padding: 15px 16px; margin: 16px 0;
      font-size: 16px; line-height: 1.55;
    }}
    .verdict {{
      background: #fff6dd; border: 1px solid #e4c56b;
      border-radius: 6px; padding: 14px; margin: 14px 0;
    }}
    .methods {{
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px; margin: 16px 0;
    }}
    .method {{ background: var(--surface); border-top: 3px solid var(--teal); padding: 12px; }}
    .method strong {{ display: block; margin-bottom: 4px; font-weight: 500; }}
    .method span {{ color: var(--muted); font-size: 14px; }}
    .charts {{
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px; margin: 22px 0;
    }}
    .chart {{
      margin: 0; background: var(--surface); border: 1px solid var(--line);
      border-radius: 6px; padding: 9px;
    }}
    .chart img {{ display: block; width: 100%; height: auto; }}
    .chart figcaption {{ padding: 5px; color: var(--muted); }}
    .legend {{
      display: flex; align-items: center; flex-wrap: wrap;
      gap: 9px; margin: 10px 0 18px; color: var(--muted); font-size: 13px;
    }}
    .score-bar {{
      width: min(360px, 55vw); height: 12px;
      background: linear-gradient(90deg,#30123b,#466be3,#1ae4b6,#a4fc3c,#faba39,#e31a1c,#7a0403);
      border: 1px solid var(--line);
    }}
    .swatch-red {{ width: 15px; height: 15px; background: var(--red); }}
    .swatch-yellow {{ width: 15px; height: 15px; background: var(--yellow); }}
    .patch-section {{
      background: var(--surface); border-top: 1px solid var(--line);
      padding: 13px 0 17px;
    }}
    .patch-heading {{
      display: flex; align-items: center; flex-wrap: wrap;
      gap: 9px; padding: 0 11px 10px;
    }}
    .role {{
      padding: 2px 8px; border-radius: 999px;
      background: #e2edf0; color: #244c59; font-size: 12px;
    }}
    .reason {{ font-size: 13px; }}
    .coords {{ margin-left: auto; font-size: 12px; }}
    .visual-review {{
      margin: 0 11px 11px; padding: 9px 11px;
      background: #fff0f0; border-left: 3px solid #b4232d;
      color: #771721; font-size: 13px;
    }}
    .patch-grid {{
      display: grid; grid-template-columns: repeat(9, minmax(0, 1fr));
      gap: 6px; padding: 0 7px;
    }}
    figure {{ margin: 0; min-width: 0; }}
    .image-button {{
      width: 100%; padding: 0; border: 1px solid var(--line);
      background: #fff; cursor: zoom-in; border-radius: 4px; overflow: hidden;
    }}
    .image-button:focus-visible {{ outline: 3px solid var(--teal); outline-offset: 2px; }}
    .image-button img {{
      display: block; width: 100%; aspect-ratio: 1;
      object-fit: contain; background: #fff;
    }}
    figcaption {{
      margin-top: 4px; min-height: 35px; color: var(--muted);
      font-size: 12px; line-height: 1.35;
    }}
    dialog {{
      width: min(94vw, 1100px); max-height: 94vh; border: 0;
      border-radius: 6px; padding: 12px; background: #fff;
    }}
    dialog::backdrop {{ background: rgba(0,0,0,.78); }}
    .dialog-head {{
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px; margin-bottom: 8px;
    }}
    .dialog-head button {{
      border: 1px solid var(--line); background: #fff;
      border-radius: 4px; padding: 7px 12px; cursor: pointer;
    }}
    #dialog-image {{ display: block; max-width: 100%; max-height: 80vh; margin: 0 auto; }}
    .footer {{ margin: 25px 0 8px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 1300px) {{
      .patch-grid {{ grid-template-columns: repeat(5, minmax(0, 1fr)); }}
    }}
    @media (max-width: 800px) {{
      main {{ padding: 15px 10px; }}
      .summary, .methods, .charts {{ grid-template-columns: 1fr 1fr; }}
      .patch-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .coords {{ margin-left: 0; }}
    }}
    @media (max-width: 470px) {{
      .summary, .methods, .charts, .patch-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>操场纹理边界实验</h1>
    <p class="subtitle">playground_xuannv · 海淀 2026 年 4 月 · 只展示 8 个典型 Patch</p>
  </header>
  <div class="p1-warning" role="alert">
    <strong>P1 数据质量警告：原训练标注疑似错标，当前模型不可上线。</strong><br>
    纹理与面积实验仅用于评估错误标注模型的风险抑制，不能证明操场识别能力，
    也不能作为生产部署依据。
  </div>
  <div class="summary">
    <div class="stat"><small>独立 OSM · 纯面积 F1</small><strong>{area["f1"]:.4f}</strong></div>
    <div class="stat"><small>独立 OSM · 纹理方案 F1</small><strong>{texture["f1"]:.4f}</strong></div>
    <div class="stat"><small>纹理 F1 增量</small><strong>{assessment["independent_osm_f1_delta"]:+.4f}</strong></div>
    <div class="stat"><small>独立 OSM 召回</small><strong>{_percent(texture["recall"])}</strong></div>
  </div>
  <div class="verdict"><strong>{html.escape(assessment["verdict"])}</strong><br>{html.escape(assessment["interpretation"])}</div>
  <div class="notice"><strong>参考标签不完整：</strong>{html.escape(manifest["reference_policy"]["limitation"])}</div>
  <div class="notice"><strong>无类别作弊：</strong>{html.escape(manifest["reference_policy"]["texture_prior"])} 页面中的纹理边界没有人工类别先验。</div>
  <div class="methods">
    <div class="method"><strong>现有全局面积保护</strong><span>固定全域高低阈值，再按连通域与面积过滤；独立操场会被完全漏掉。</span></div>
    <div class="method"><strong>相对种子 + 面积筛选</strong><span>用每个 Patch 的高分位作种子，只做 PU 连通扩张和面积限制，是纹理实验的公平对照。</span></div>
    <div class="method"><strong>纹理边界 + 面积筛选</strong><span>在公平对照上增加真实高分辨率 RGB 梯度和局部纹理边界，阻止跨边界扩张。</span></div>
  </div>
  <div class="charts">
    <figure class="chart"><img src="{chart_urls["area"]}" alt="典型 Patch 方案面积对比"><figcaption>典型 Patch 方案面积对比</figcaption></figure>
    <figure class="chart"><img src="{chart_urls["reference"]}" alt="参考相对指标对比"><figcaption>参考相对指标只用于有限对照，不代表全域真值精度</figcaption></figure>
  </div>
  <div class="legend">
    <span class="swatch-red"></span><span>预测目标</span>
    <span class="swatch-yellow"></span><span>参考 Polygon 边界</span>
    <span>{low:.2f}</span><span class="score-bar"></span><span>{high:.2f}</span>
    <span>PU 热力图统一色标</span>
  </div>
  {''.join(sections)}
  <p class="footer">点击放大任意图片。光学纹理边界越亮表示局部梯度或纹理变化越强；它不代表任何预先知道的地物类别。</p>
</main>
<dialog id="image-dialog">
  <div class="dialog-head"><strong id="dialog-label"></strong><button id="dialog-close" type="button">关闭</button></div>
  <img id="dialog-image" alt="">
</dialog>
<script>
  const dialog = document.getElementById("image-dialog");
  const dialogImage = document.getElementById("dialog-image");
  const dialogLabel = document.getElementById("dialog-label");
  document.querySelectorAll(".image-button").forEach((button) => {{
    button.addEventListener("click", () => {{
      dialogImage.src = button.dataset.fullImage;
      dialogImage.alt = button.dataset.imageLabel;
      dialogLabel.textContent = button.dataset.imageLabel;
      dialog.showModal();
    }});
  }});
  document.getElementById("dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {{
    if (event.target === dialog) dialog.close();
  }});
</script>
</body>
</html>
"""


def build_gallery(input_dir: Path, repo_root: Path = ROOT) -> Path:
    """Render the eight-patch texture experiment as a standalone page."""
    input_dir = Path(input_dir).resolve()
    repo_root = Path(repo_root).resolve()
    manifest = _load_json(input_dir / "experiment_manifest.json")
    metrics = _load_json(input_dir / "metrics.json")
    assets = input_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    selected = [
        patch_id
        for group, _ in GROUP_LABELS
        for patch_id in manifest["selection"][group]
    ]
    if len(selected) != 8:
        raise ValueError(f"Focused gallery requires exactly 8 patches, found {len(selected)}")
    scores = {
        patch_id: _load_array(
            input_dir,
            manifest["artifacts"]["arrays"][patch_id]["arrays"]["score_query"],
        )
        for patch_id in selected
    }
    all_scores = np.concatenate([score.reshape(-1) for score in scores.values()])
    score_low, score_high = [
        float(value) for value in np.quantile(all_scores, [0.01, 0.99])
    ]
    image_urls: Dict[str, Dict[str, str]] = {}
    for patch_id in selected:
        artifact = manifest["artifacts"]["arrays"][patch_id]
        arrays = artifact["arrays"]
        optical = _read_optical(repo_root / artifact["optical_path"])
        size = (optical.shape[1], optical.shape[0])
        reference = _load_array(input_dir, arrays["reference"]).astype(bool)
        boundary = _load_array(input_dir, arrays["texture_boundary"])
        predictions = {
            name: _load_array(input_dir, arrays[name]).astype(bool)
            for name in (
                "baseline",
                "guarded",
                "area_guard",
                "texture_boundary_area_guard",
            )
        }
        images = {
            "optical": optical,
            "reference_overlay": _reference_overlay(optical, reference),
            "score_heatmap": _colored_map(
                scores[patch_id],
                score_low,
                score_high,
                size,
                "turbo",
            ),
            "texture_boundary": _colored_map(boundary, 0.0, 1.0, size, "magma"),
            "baseline": _mask_image(predictions["baseline"], size),
            "guarded": _mask_image(predictions["guarded"], size),
            "area_guard": _mask_image(predictions["area_guard"], size),
            "texture_boundary_area_guard": _mask_image(
                predictions["texture_boundary_area_guard"],
                size,
            ),
            "texture_overlay": _prediction_overlay(
                optical,
                predictions["texture_boundary_area_guard"],
            ),
        }
        image_urls[patch_id] = {}
        for name, image in images.items():
            path = assets / f"{patch_id}-{name}.png"
            _save_rgb(path, image)
            image_urls[patch_id][name] = path.relative_to(input_dir).as_posix()

    font = _font(assets)
    area_chart = assets / "selected-patch-area-comparison.png"
    reference_chart = assets / "reference-metric-comparison.png"
    _plot_selected_area(area_chart, metrics, font)
    _plot_reference_metrics(reference_chart, metrics, font)
    chart_urls = {
        "area": area_chart.relative_to(input_dir).as_posix(),
        "reference": reference_chart.relative_to(input_dir).as_posix(),
    }
    output = input_dir / "index.html"
    output.write_text(
        _build_html(
            manifest,
            metrics,
            image_urls,
            chart_urls,
            (score_low, score_high),
        ),
        encoding="utf-8",
    )
    (input_dir / "README.md").write_text(
        """# 操场纹理边界实验

该页面只展示 8 个典型 Patch。原有 320 Patch 统计保留在上一阶段作为诊断证据，
不在本页面继续铺开。

- 红色为预测目标，黄色边线为已有 Polygon。
- P1 数据质量警告：三个原训练标注疑似错标，当前模型不可上线。
- `area_guard` 是 Patch 相对种子加面积筛选，不使用光学纹理。
- `texture_boundary_area_guard` 在同一对照上增加真实高分辨率光学纹理边界。
- 纹理边界没有使用建筑、道路或操场类别掩膜。
- 当前实验判定：纹理边界相对公平面积对照没有证明实质增益。
- 纹理和面积结果仅是错误标注模型上的风险抑制对照，不可用于上线结论。
""",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "Tmp/playground_texture_20260731",
    )
    args = parser.parse_args()
    print(build_gallery(args.input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
