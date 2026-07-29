"""Compare bidirectional fusion and image-guided registration."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    import scripts.experiment_haidian_construction_change as construction
    import scripts.experiment_haidian_embedding_change as base
    import scripts.experiment_haidian_smoothing_change as smoothing
except ModuleNotFoundError:
    import experiment_haidian_construction_change as construction
    import experiment_haidian_embedding_change as base
    import experiment_haidian_smoothing_change as smoothing


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ("maximum", "当前双向最大值"),
    ("q75", "双向稳健融合 Q75"),
    ("mean", "双向平均值"),
    ("aligned_mean", "高分配准 + 双向平均值"),
)


def _registration_feature(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    gray = (
        values[..., 0] * 0.299
        + values[..., 1] * 0.587
        + values[..., 2] * 0.114
    )
    local_mean = ndimage.gaussian_filter(gray, sigma=8.0)
    local_square = ndimage.gaussian_filter(gray * gray, sigma=8.0)
    local_std = np.sqrt(np.maximum(local_square - local_mean * local_mean, 1e-4))
    normalized = (gray - local_mean) / local_std
    horizontal = ndimage.sobel(normalized, axis=1)
    vertical = ndimage.sobel(normalized, axis=0)
    return np.hypot(horizontal, vertical).astype(np.float32)


def _scores(
    before: np.ndarray,
    after: np.ndarray,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], tuple[float, float]]:
    maximum = base.bidirectional_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.05, fusion="max"
    )
    q75 = base.bidirectional_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.05, fusion="q75"
    )
    mean = base.bidirectional_neighborhood_cosine_change(
        before, after, radius=2, displacement_penalty=0.05, fusion="mean"
    )
    highres_shift = base.estimate_translation(
        _registration_feature(before_rgb),
        _registration_feature(after_rgb),
        max_shift=12,
    )
    embedding_scale = before.shape[1] / before_rgb.shape[0]
    embedding_shift = (
        highres_shift[0] * embedding_scale,
        highres_shift[1] * embedding_scale,
    )
    aligned_after = ndimage.shift(
        after,
        shift=(0.0, embedding_shift[0], embedding_shift[1]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    aligned_mean = base.bidirectional_neighborhood_cosine_change(
        before,
        aligned_after,
        radius=2,
        displacement_penalty=0.05,
        fusion="mean",
    )
    return {
        "maximum": maximum,
        "q75": q75,
        "mean": mean,
        "aligned_mean": aligned_mean,
    }, highres_shift


def _original_p98() -> float:
    values = []
    for patch_id in base._paired_patch_ids()[::5]:
        before = np.load(base.EMBEDDING_ROOT / base.BEFORE_MONTH / f"{patch_id}.npy")
        after = np.load(base.EMBEDDING_ROOT / base.AFTER_MONTH / f"{patch_id}.npy")
        scores, valid = base.symmetric_neighborhood_cosine_change(
            before, after, radius=2, displacement_penalty=0.05
        )
        values.append(scores[valid])
    return float(np.quantile(np.concatenate(values), 0.98))


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    threshold = _original_p98()
    patch_ids, construction_ids = smoothing._sample_patch_ids()
    rows = []
    for patch_id in patch_ids:
        before = np.load(base.EMBEDDING_ROOT / base.BEFORE_MONTH / f"{patch_id}.npy")
        after = np.load(base.EMBEDDING_ROOT / base.AFTER_MONTH / f"{patch_id}.npy")
        before_rgb, before_scene = construction._highres_rgb(
            base.BEFORE_MONTH, patch_id
        )
        after_rgb, after_scene = construction._highres_rgb(base.AFTER_MONTH, patch_id)
        methods, highres_shift = _scores(before, after, before_rgb, after_rgb)
        display_shape = after_rgb.shape[:2]
        files = {"before": f"{patch_id}_before.png", "after": f"{patch_id}_after.png"}
        construction._save(output / files["before"], before_rgb)
        construction._save(output / files["after"], after_rgb)
        stats = {}
        for key, _ in METHODS:
            scores, valid = methods[key]
            heatmap = base.threshold_colored_map(scores, low=0.07, threshold=threshold)
            display_heatmap = construction._resize_rgb(heatmap, display_shape)
            display_valid = construction._resize_mask(valid, display_shape)
            files[f"{key}_change"] = f"{patch_id}_{key}_change.png"
            files[f"{key}_overlay"] = f"{patch_id}_{key}_overlay.png"
            construction._save(output / files[f"{key}_change"], display_heatmap)
            construction._save(
                output / files[f"{key}_overlay"],
                base._overlay(after_rgb, display_heatmap, display_valid),
            )
            values = scores[valid]
            stats[key] = {
                "mean": float(values.mean()),
                "p98": float(np.quantile(values, 0.98)),
                "red_share": float((values >= threshold).mean()),
            }
        rows.append(
            {
                "patch_id": patch_id,
                "sample_type": (
                    "施工变化样本" if patch_id in construction_ids else "固定随机样本"
                ),
                "highres_shift_to_align_after": list(highres_shift),
                "before_scene": before_scene.stem,
                "after_scene": after_scene.stem,
                "stats": stats,
                **files,
            }
        )
    result = {
        "red_threshold": threshold,
        "methods": [{"key": key, "name": name} for key, name in METHODS],
        "patches": rows,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_html(output, result)
    return result


def _write_html(output: Path, result: dict) -> None:
    sections = []
    for row in result["patches"]:
        methods = []
        for method in result["methods"]:
            key = method["key"]
            stat = row["stats"][key]
            methods.append(
                f"<article><h3>{method['name']}</h3><div class='pair'>"
                f"<figure><button data-src='{row[f'{key}_change']}'><img src='{row[f'{key}_change']}' alt='{method['name']}变化图'></button><figcaption>变化图</figcaption></figure>"
                f"<figure><button data-src='{row[f'{key}_overlay']}'><img src='{row[f'{key}_overlay']}' alt='{method['name']}叠加图'></button><figcaption>叠加图</figcaption></figure></div>"
                f"<p>均值 {stat['mean']:.4f}　P98 {stat['p98']:.4f}　红色 {stat['red_share']:.2%}</p></article>"
            )
        shift = row["highres_shift_to_align_after"]
        sections.append(
            f"<section><div class='head'><h2>{row['patch_id']}</h2><span>{row['sample_type']}</span></div>"
            f"<p>高分影像估计校正位移：行 {shift[0]:+.1f}、列 {shift[1]:+.1f} 个 3 米像素。</p>"
            "<div class='optical'>"
            f"<figure><button data-src='{row['before']}'><img src='{row['before']}' alt='变化前'></button><figcaption>2025 年 12 月</figcaption></figure>"
            f"<figure><button data-src='{row['after']}'><img src='{row['after']}' alt='变化后'></button><figcaption>2026 年 4 月</figcaption></figure>"
            f"</div><div class='methods'>{''.join(methods)}</div></section>"
        )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>海淀配准与双向融合对照</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:#18212a;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1880px;margin:auto;padding:24px}}
header,section{{background:#fff;border:1px solid #dce2e7;border-radius:8px;padding:20px;margin-bottom:18px}}h1{{margin:0 0 8px;font-size:27px}}h2{{margin:0;font-size:20px}}h3{{font-size:16px;margin:0 0 9px}}p{{margin:6px 0}}
.head{{display:flex;gap:12px;align-items:center}}.head span{{background:#eef2f5;padding:2px 8px;border-radius:4px}}.optical{{display:grid;grid-template-columns:repeat(2,minmax(220px,360px));gap:12px;margin:14px 0 18px}}
.methods{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:12px}}article{{border:1px solid #dfe5ea;padding:12px;border-radius:6px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
figure{{margin:0;min-width:0}}button{{border:0;padding:0;background:none;width:100%;cursor:zoom-in}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;border:1px solid #d8dee3;background:#eef1f3}}
figcaption{{font-weight:600;text-align:center;font-size:13px;margin-top:5px}}article p{{font-size:13px}}dialog{{border:0;border-radius:8px;padding:12px;background:#111;max-width:94vw;max-height:94vh}}dialog img{{width:auto;max-width:90vw;max-height:88vh;border:0}}dialog::backdrop{{background:rgba(0,0,0,.78)}}
@media(max-width:1300px){{.methods{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:720px){{main{{padding:10px}}.methods,.optical{{grid-template-columns:1fr}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>海淀配准与双向融合对照</h1>
<p>四组使用同一红色阈值 {result['red_threshold']:.4f}（当前双向最大值的全区 P98），因此红色面积可以直接比较。</p>
<p>配准只使用两期 RGB 的局部亮度标准化和梯度结构，通过相位相关估计平移；没有使用建筑、道路、施工或其他语义掩膜。</p>
<p>双向最大值最敏感；Q75 保留较强方向但降低单向异常；平均值要求两个方向共同支持。配准版先校正整体位移，再取双向平均。</p>
</header>{''.join(sections)}</main><dialog id="viewer"><img alt="放大预览"></dialog>
<script>const d=document.querySelector('#viewer'),i=d.querySelector('img');document.querySelectorAll('button[data-src]').forEach(b=>b.onclick=()=>{{i.src=b.dataset.src;d.showModal()}});d.onclick=()=>d.close();</script>
</body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    output = ROOT / "Tmp" / f"haidian_registration_fusion_{datetime.now():%Y%m%d_%H%M%S}"
    result = run(output)
    print(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
