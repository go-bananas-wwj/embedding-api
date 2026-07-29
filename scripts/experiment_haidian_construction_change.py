"""Build a focused gallery of construction-site embedding changes."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import scripts.experiment_haidian_embedding_change as base
except ModuleNotFoundError:
    import experiment_haidian_embedding_change as base


ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = ROOT / "data/haidian/tasks/construction/v1/labels"
HIGHRES_ROOT = (
    ROOT / "data/haidian/archive/processed_training_data/extracted/patches/highres_optical"
)
PATCH_IDS = (
    "patch_000017",
    "patch_000111",
    "patch_000302",
    "patch_000303",
    "patch_000263",
    "patch_000024",
)


def _save(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).resize((512, 512), Image.Resampling.NEAREST).save(path)


def _highres_rgb(month: str, patch_id: str) -> tuple[np.ndarray, Path]:
    path = HIGHRES_ROOT / f"highres_optical_{month}01_{patch_id}.tif"
    if not path.exists():
        raise FileNotFoundError(f"High-resolution image not found: {path}")
    with base.rasterio.open(path) as dataset:
        bands = dataset.read([1, 2, 3]).astype(np.float32)
    rgb = np.moveaxis(bands, 0, -1)
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel]
        finite = np.isfinite(values)
        low, high = np.percentile(values[finite], [2, 98])
        stretched = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
        stretched[~finite] = 0.0
        output[..., channel] = np.rint(np.power(stretched, 0.85) * 255).astype(
            np.uint8
        )
    return output, path


def _resize_rgb(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(array).resize(
            (shape[1], shape[0]),
            Image.Resampling.BILINEAR,
        )
    )


def _resize_mask(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(array.astype(np.uint8)).resize(
            (shape[1], shape[0]),
            Image.Resampling.NEAREST,
        )
    ) > 0


def _label_view(mask: np.ndarray) -> np.ndarray:
    view = np.full((*mask.shape, 3), 246, dtype=np.uint8)
    view[mask] = (48, 194, 94)
    return view


def _outlined_overlay(
    optical: np.ndarray,
    heatmap: np.ndarray,
    valid: np.ndarray,
    label: np.ndarray,
) -> np.ndarray:
    overlay = base._overlay(optical, heatmap, valid)
    edge = label & ~ndimage.binary_erosion(label)
    overlay[edge] = (35, 245, 92)
    return overlay


def _pixel_baseline() -> np.ndarray:
    """Collect mask-free normalized pixel scores for empirical calibration."""
    values = []
    for patch_id in base._paired_patch_ids()[::5]:
        before = np.load(base.EMBEDDING_ROOT / base.BEFORE_MONTH / f"{patch_id}.npy")
        after = np.load(base.EMBEDDING_ROOT / base.AFTER_MONTH / f"{patch_id}.npy")
        before, after = base.robust_temporal_normalize(before, after)
        scores, valid = base.symmetric_neighborhood_cosine_change(
            before,
            after,
            radius=2,
            displacement_penalty=0.05,
        )
        selected = scores[valid]
        if selected.size:
            values.append(selected)
    if not values:
        raise RuntimeError("No valid pixels available for calibration")
    return np.sort(np.concatenate(values).astype(np.float32))


def _pixel_anomaly_percentile(
    scores: np.ndarray,
    valid: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    confidence = np.full(scores.shape, np.nan, dtype=np.float32)
    confidence[valid] = (
        np.searchsorted(baseline, scores[valid], side="right") / baseline.size
    )
    return confidence


def _area_consistent_confidence(
    percentile: np.ndarray,
) -> np.ndarray:
    """Suppress isolated or one-pixel-wide high-distance responses."""
    valid = np.isfinite(percentile)
    adjusted = percentile.copy()
    high_support = ndimage.uniform_filter(
        np.where(valid, adjusted >= 0.90, 0.0).astype(np.float32),
        size=3,
        mode="nearest",
    )
    anomaly_strength = np.clip(
        (np.nan_to_num(adjusted, nan=0.0) - 0.70) / 0.30,
        0.0,
        1.0,
    )
    confidence = anomaly_strength * np.clip(high_support / 0.45, 0.0, 1.0)
    confidence[~valid] = np.nan
    return confidence.astype(np.float32)


def _confidence_map(confidence: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.nan_to_num(confidence, nan=0.0), 0.0, 1.0)
    rgba = base.matplotlib.colormaps["RdYlBu_r"](normalized, bytes=True)
    rgb = rgba[..., :3]
    rgb[~np.isfinite(confidence)] = (238, 241, 244)
    return rgb


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    all_cosine = []
    for patch_id in base._paired_patch_ids():
        before = np.load(base.EMBEDDING_ROOT / base.BEFORE_MONTH / f"{patch_id}.npy")
        after = np.load(base.EMBEDDING_ROOT / base.AFTER_MONTH / f"{patch_id}.npy")
        cosine, _, _ = base.change_scores(before, after)
        all_cosine.append(cosine)
    limits = base.global_limits(all_cosine)
    high_threshold = limits[1]
    pixel_baseline = _pixel_baseline()
    pixel_quantiles = {
        f"p{quantile}": float(np.quantile(pixel_baseline, quantile / 100))
        for quantile in (70, 90, 97, 99)
    }

    rows = []
    for patch_id in PATCH_IDS:
        before = np.load(base.EMBEDDING_ROOT / base.BEFORE_MONTH / f"{patch_id}.npy")
        after = np.load(base.EMBEDDING_ROOT / base.AFTER_MONTH / f"{patch_id}.npy")
        local_5x5, valid_5x5 = base.neighborhood_cosine_change(
            before, after, radius=2
        )
        symmetric_5x5, symmetric_valid = (
            base.symmetric_neighborhood_cosine_change(
                before,
                after,
                radius=2,
                displacement_penalty=0.05,
            )
        )
        normalized_before, normalized_after = base.robust_temporal_normalize(
            before,
            after,
        )
        normalized_5x5, normalized_valid = (
            base.symmetric_neighborhood_cosine_change(
                normalized_before,
                normalized_after,
                radius=2,
                displacement_penalty=0.05,
            )
        )
        percentile = _pixel_anomaly_percentile(
            normalized_5x5,
            normalized_valid,
            pixel_baseline,
        )
        corrected = _area_consistent_confidence(percentile)
        label = np.load(LABEL_ROOT / f"{patch_id}.npy") > 0
        before_rgb, before_scene = _highres_rgb(base.BEFORE_MONTH, patch_id)
        after_rgb, after_scene = _highres_rgb(base.AFTER_MONTH, patch_id)
        display_shape = after_rgb.shape[:2]
        display_label = _resize_mask(label, display_shape)
        method_scores = {
            "local_5x5": (local_5x5, valid_5x5),
            "symmetric_5x5": (symmetric_5x5, symmetric_valid),
            "normalized_corrected": (corrected, np.isfinite(corrected)),
        }

        files = {
            "before": f"{patch_id}_202512.png",
            "after": f"{patch_id}_202604.png",
            "label": f"{patch_id}_construction_label.png",
        }
        _save(output / files["before"], before_rgb)
        _save(output / files["after"], after_rgb)
        _save(output / files["label"], _label_view(label))
        metrics = {}
        for method, (scores, method_valid) in method_scores.items():
            heatmap = (
                _confidence_map(scores)
                if method == "normalized_corrected"
                else base._colored_map(scores, limits)
            )
            display_heatmap = _resize_rgb(heatmap, display_shape)
            display_valid = _resize_mask(method_valid, display_shape)
            files[f"{method}_change"] = f"{patch_id}_{method}_change.png"
            files[f"{method}_overlay"] = f"{patch_id}_{method}_overlay.png"
            _save(output / files[f"{method}_change"], display_heatmap)
            _save(
                output / files[f"{method}_overlay"],
                _outlined_overlay(
                    after_rgb,
                    display_heatmap,
                    display_valid,
                    display_label,
                ),
            )
            inside = label & method_valid
            outside = ~label & method_valid
            inside_values = scores[inside]
            outside_values = scores[outside]
            metrics[method] = {
                "inside_mean": float(inside_values.mean()),
                "inside_p95": float(np.quantile(inside_values, 0.95)),
                "outside_mean": float(outside_values.mean()),
                "inside_minus_outside": float(
                    inside_values.mean() - outside_values.mean()
                ),
                "inside_high_change_share": float(
                    (
                        inside_values
                        >= (
                            0.90
                            if method == "normalized_corrected"
                            else high_threshold
                        )
                    ).mean()
                ),
            }
        rows.append(
            {
                "patch_id": patch_id,
                "before_scene": before_scene.stem,
                "after_scene": after_scene.stem,
                "construction_pixels": int(label.sum()),
                "metrics": metrics,
                **files,
            }
        )

    result = {
        "before_month": base.BEFORE_MONTH,
        "after_month": base.AFTER_MONTH,
        "metric": "cosine_change_distance",
        "global_limits": limits,
        "high_change_threshold": high_threshold,
        "pixel_baseline_quantiles": pixel_quantiles,
        "pixel_baseline_pixel_count": int(pixel_baseline.size),
        "patches": rows,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_html(output, result)
    return result


def _write_html(output: Path, result: dict) -> None:
    sections = []
    context_titles = (
        ("before", "2025 年 12 月光学影像"),
        ("after", "2026 年 4 月光学影像"),
        ("label", "施工工地参考标签"),
    )
    method_titles = (
        ("local_5x5", "当前单向 5×5"),
        ("symmetric_5x5", "双向 5×5 + 位移惩罚"),
        ("normalized_corrected", "光照/风格归一化 + 空间连续性"),
    )
    for row in result["patches"]:
        context = "".join(
            f"<figure><button data-src='{row[key]}'><img src='{row[key]}' "
            f"alt='{title}'></button><figcaption>{title}</figcaption></figure>"
            for key, title in context_titles
        )
        methods = []
        for key, title in method_titles:
            metric = row["metrics"][key]
            methods.append(
                f"<article><h3>{title}</h3><div class='pair'>"
                f"<figure><button data-src='{row[f'{key}_change']}'><img "
                f"src='{row[f'{key}_change']}' alt='{title}热力图'></button>"
                "<figcaption>变化距离</figcaption></figure>"
                f"<figure><button data-src='{row[f'{key}_overlay']}'><img "
                f"src='{row[f'{key}_overlay']}' alt='{title}叠加图'></button>"
                "<figcaption>叠加图（绿线为施工标签）</figcaption></figure></div>"
                "<div class='method-metrics'>"
                f"<span>施工区均值 <b>{metric['inside_mean']:.4f}</b></span>"
                f"<span>区域外均值 <b>{metric['outside_mean']:.4f}</b></span>"
                f"<span>区内−区外 <b>{metric['inside_minus_outside']:+.4f}</b></span>"
                f"<span>施工区 P95 <b>{metric['inside_p95']:.4f}</b></span>"
                f"</div></article>"
            )
        sections.append(
            f"<section><h2>{row['patch_id']}</h2>"
            f"<p class='scene'>{row['before_scene']} → {row['after_scene']}</p>"
            f"<p>施工标签像素：<b>{row['construction_pixels']}</b></p>"
            f"<div class='context'>{context}</div>"
            f"<div class='methods'>{''.join(methods)}</div></section>"
        )

    low, high = result["global_limits"]
    quantiles = result["pixel_baseline_quantiles"]
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>海淀施工工地变化距离</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:#18212a;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1680px;margin:auto;padding:24px}}header,section{{background:#fff;border:1px solid #dce2e7;border-radius:8px;padding:20px;margin-bottom:18px}}
h1{{margin:0 0 8px;font-size:27px}}h2{{margin:0;font-size:20px}}p{{margin:6px 0}}.scene{{font-size:13px;color:#64717c}}
.legend{{height:14px;max-width:460px;background:linear-gradient(90deg,#4575b4,#fee090,#a50026);border:1px solid #bdc6ce;margin-top:12px}}
.labels{{display:flex;justify-content:space-between;max-width:460px;color:#5d6974;font-size:13px}}.context{{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:12px;max-width:900px;margin:14px 0 18px}}
.methods{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:14px}}article{{border:1px solid #dfe5ea;padding:12px;border-radius:6px}}h3{{font-size:16px;margin:0 0 9px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:9px}}
.method-metrics{{display:grid;grid-template-columns:1fr 1fr;gap:5px 12px;margin-top:10px;font-size:13px}}.method-metrics span{{border-left:3px solid #56788f;padding-left:7px}}
figure{{margin:0;min-width:0}}button{{border:0;padding:0;background:none;width:100%;cursor:zoom-in}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;border:1px solid #d8dee3;background:#eef1f3}}
figcaption{{font-weight:600;text-align:center;font-size:13px;margin-top:5px}}dialog{{border:0;border-radius:8px;padding:12px;background:#111;max-width:94vw;max-height:94vh}}dialog img{{width:auto;max-width:90vw;max-height:88vh;border:0}}dialog::backdrop{{background:rgba(0,0,0,.78)}}
@media(max-width:1100px){{.methods{{grid-template-columns:1fr}}}}@media(max-width:620px){{main{{padding:10px}}.context{{grid-template-columns:1fr 1fr}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>海淀施工工地变化距离</h1>
<p>对比 <b>2025 年 12 月</b> 与 <b>2026 年 4 月</b>，使用 P10C 64D embedding 余弦变化距离。</p>
<p>光学底图使用两期 427×427 高分辨率 RGB 影像；变化距离来自 128×128 embedding，并按相同地理范围插值后叠加。高分底图只改善观察细节，不会增加 embedding 本身的变化信息。</p>
<p>前两种方法统一使用原始距离色标 {low:.4f}–{high:.4f}。校正版只使用原始 embedding 像素：分别按月份做逐通道中位数/IQR 归一化，再执行双向 5×5 位移补偿和空间连续性过滤。</p>
<p>全区归一化像素距离：P70={quantiles['p70']:.4f}，P90={quantiles['p90']:.4f}，P97={quantiles['p97']:.4f}，P99={quantiles['p99']:.4f}。校正版蓝色约为 P70 以下，黄色约为 P90，红色表示高异常且形成连续区域。</p>
<p>“区内−区外”大于 0 表示施工区域比周围更突出。算法没有读取建筑物掩膜或施工掩膜；施工标签只用于统计和绿框展示。</p>
<div class="legend"></div><div class="labels"><span>稳定 / 低异常</span><span>疑似变化</span><span>连续高异常</span></div>
</header>{''.join(sections)}</main><dialog id="viewer"><img alt="放大预览"></dialog>
<script>const d=document.querySelector('#viewer'),i=d.querySelector('img');
document.querySelectorAll('button[data-src]').forEach(b=>b.onclick=()=>{{i.src=b.dataset.src;d.showModal()}});
d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    output = ROOT / "Tmp" / f"haidian_construction_change_{datetime.now():%Y%m%d_%H%M%S}"
    result = run(output)
    print(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
