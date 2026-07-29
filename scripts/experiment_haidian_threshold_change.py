"""Compare hard red thresholds for Haidian 5x5 embedding change maps."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import scripts.experiment_haidian_construction_change as construction
    import scripts.experiment_haidian_embedding_change as base
    import scripts.experiment_haidian_smoothing_change as smoothing
except ModuleNotFoundError:
    import experiment_haidian_construction_change as construction
    import experiment_haidian_embedding_change as base
    import experiment_haidian_smoothing_change as smoothing


ROOT = Path(__file__).resolve().parents[1]
PERCENTILES = (90, 95, 98, 99)


def _score(patch_id: str) -> tuple[np.ndarray, np.ndarray]:
    before = np.load(base.EMBEDDING_ROOT / base.BEFORE_MONTH / f"{patch_id}.npy")
    after = np.load(base.EMBEDDING_ROOT / base.AFTER_MONTH / f"{patch_id}.npy")
    return base.symmetric_neighborhood_cosine_change(
        before,
        after,
        radius=2,
        displacement_penalty=0.05,
    )


def _calibration_values() -> np.ndarray:
    values = []
    for patch_id in base._paired_patch_ids()[::5]:
        scores, valid = _score(patch_id)
        values.append(scores[valid])
    return np.concatenate(values).astype(np.float32)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    calibration = _calibration_values()
    low = float(np.quantile(calibration, 0.02))
    thresholds = {
        f"p{percentile}": float(np.quantile(calibration, percentile / 100))
        for percentile in PERCENTILES
    }
    patch_ids, construction_ids = smoothing._sample_patch_ids()
    rows = []
    for patch_id in patch_ids:
        scores, valid = _score(patch_id)
        before_rgb, before_scene = construction._highres_rgb(
            base.BEFORE_MONTH, patch_id
        )
        after_rgb, after_scene = construction._highres_rgb(base.AFTER_MONTH, patch_id)
        display_shape = after_rgb.shape[:2]
        display_valid = construction._resize_mask(valid, display_shape)
        files = {
            "before": f"{patch_id}_before.png",
            "after": f"{patch_id}_after.png",
        }
        construction._save(output / files["before"], before_rgb)
        construction._save(output / files["after"], after_rgb)
        shares = {}
        for key, threshold in thresholds.items():
            heatmap = base.threshold_colored_map(scores, low, threshold)
            display_heatmap = construction._resize_rgb(heatmap, display_shape)
            files[f"{key}_change"] = f"{patch_id}_{key}_change.png"
            files[f"{key}_overlay"] = f"{patch_id}_{key}_overlay.png"
            construction._save(output / files[f"{key}_change"], display_heatmap)
            construction._save(
                output / files[f"{key}_overlay"],
                base._overlay(after_rgb, display_heatmap, display_valid),
            )
            shares[key] = float((scores[valid] >= threshold).mean())
        rows.append(
            {
                "patch_id": patch_id,
                "sample_type": (
                    "施工变化样本" if patch_id in construction_ids else "固定随机样本"
                ),
                "before_scene": before_scene.stem,
                "after_scene": after_scene.stem,
                "red_shares": shares,
                **files,
            }
        )
    result = {
        "before_month": base.BEFORE_MONTH,
        "after_month": base.AFTER_MONTH,
        "calibration_patch_count": len(base._paired_patch_ids()[::5]),
        "calibration_pixel_count": int(calibration.size),
        "low": low,
        "thresholds": thresholds,
        "patches": rows,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_html(output, result)
    return result


def _write_html(output: Path, result: dict) -> None:
    sections = []
    for row in result["patches"]:
        comparisons = []
        for key, threshold in result["thresholds"].items():
            comparisons.append(
                f"<article><h3>红色阈值 {key.upper()} = {threshold:.4f}</h3>"
                "<div class='pair'>"
                f"<figure><button data-src='{row[f'{key}_change']}'><img "
                f"src='{row[f'{key}_change']}' alt='{key}变化图'></button>"
                "<figcaption>阈值变化图</figcaption></figure>"
                f"<figure><button data-src='{row[f'{key}_overlay']}'><img "
                f"src='{row[f'{key}_overlay']}' alt='{key}叠加图'></button>"
                "<figcaption>叠加到 2026 年 4 月</figcaption></figure></div>"
                f"<p>该 Patch 红色像素占比：<b>{row['red_shares'][key]:.2%}</b></p>"
                "</article>"
            )
        sections.append(
            f"<section><div class='head'><h2>{row['patch_id']}</h2>"
            f"<span>{row['sample_type']}</span></div>"
            "<div class='optical'>"
            f"<figure><button data-src='{row['before']}'><img src='{row['before']}' "
            "alt='变化前'></button><figcaption>2025 年 12 月高分影像</figcaption></figure>"
            f"<figure><button data-src='{row['after']}'><img src='{row['after']}' "
            "alt='变化后'></button><figcaption>2026 年 4 月高分影像</figcaption></figure>"
            f"</div><div class='methods'>{''.join(comparisons)}</div></section>"
        )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>海淀变化检测红色阈值对照</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:#18212a;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1880px;margin:auto;padding:24px}}header,section{{background:#fff;border:1px solid #dce2e7;border-radius:8px;padding:20px;margin-bottom:18px}}
h1{{margin:0 0 8px;font-size:27px}}h2{{margin:0;font-size:20px}}h3{{font-size:16px;margin:0 0 9px}}
p{{margin:6px 0}}.head{{display:flex;gap:12px;align-items:center}}.head span{{background:#eef2f5;padding:2px 8px;border-radius:4px}}
.legend{{height:14px;max-width:460px;background:linear-gradient(90deg,#3156a6,#d5eaf7,#fff7bc,#c4342e);border:1px solid #bdc6ce;margin-top:12px}}
.labels{{display:flex;justify-content:space-between;max-width:460px;color:#5d6974;font-size:13px}}
.optical{{display:grid;grid-template-columns:repeat(2,minmax(220px,360px));gap:12px;margin:14px 0 18px}}
.methods{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:12px}}article{{border:1px solid #dfe5ea;padding:12px;border-radius:6px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}figure{{margin:0;min-width:0}}button{{border:0;padding:0;background:none;width:100%;cursor:zoom-in}}
img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;border:1px solid #d8dee3;background:#eef1f3}}figcaption{{font-weight:600;text-align:center;font-size:13px;margin-top:5px}}
article p{{font-size:13px}}dialog{{border:0;border-radius:8px;padding:12px;background:#111;max-width:94vw;max-height:94vh}}dialog img{{width:auto;max-width:90vw;max-height:88vh;border:0}}dialog::backdrop{{background:rgba(0,0,0,.78)}}
@media(max-width:1300px){{.methods{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:720px){{main{{padding:10px}}.methods,.optical{{grid-template-columns:1fr}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>海淀变化检测红色阈值对照</h1>
<p>四组全部使用同一份原始双向 5×5 embedding 变化距离，只改变“从哪里开始标红”。</p>
<p>阈值来自全区固定抽样的 {result['calibration_patch_count']} 个 Patch、{result['calibration_pixel_count']:,} 个有效像素。P90/P95/P98/P99 分别表示只有全区距离最高的约 10%/5%/2%/1% 像素会标红。</p>
<p>阈值以下为蓝色到浅黄色，达到阈值才直接进入红色。所有 Patch 共用同一组数值，不按单图自动拉伸。</p>
<div class="legend"></div><div class="labels"><span>低变化</span><span>接近阈值</span><span>达到阈值：红色</span></div>
</header>{''.join(sections)}</main><dialog id="viewer"><img alt="放大预览"></dialog>
<script>const d=document.querySelector('#viewer'),i=d.querySelector('img');
document.querySelectorAll('button[data-src]').forEach(b=>b.onclick=()=>{{i.src=b.dataset.src;d.showModal()}});
d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    output = ROOT / "Tmp" / f"haidian_threshold_change_{datetime.now():%Y%m%d_%H%M%S}"
    result = run(output)
    print(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
