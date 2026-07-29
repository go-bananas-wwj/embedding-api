"""Compare spatial smoothing choices before 5x5 embedding matching."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import scripts.experiment_haidian_construction_change as construction
    import scripts.experiment_haidian_embedding_change as base
except ModuleNotFoundError:
    import experiment_haidian_construction_change as construction
    import experiment_haidian_embedding_change as base


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 20260723
RANDOM_COUNT = 10
METHODS = (
    ("raw", "原始双向 5×5（不平滑）"),
    ("mean3", "3×3 均值 + 双向 5×5"),
    ("mean5", "5×5 均值 + 双向 5×5"),
    ("gaussian1", "高斯 σ=1.0 + 双向 5×5"),
)


def _sample_patch_ids() -> tuple[list[str], set[str]]:
    construction_ids = list(construction.PATCH_IDS)
    candidates = sorted(set(base._paired_patch_ids()) - set(construction_ids))
    random_ids = (
        np.random.default_rng(RANDOM_SEED)
        .choice(candidates, size=RANDOM_COUNT, replace=False)
        .tolist()
    )
    return construction_ids + sorted(random_ids), set(construction_ids)


def _smoothed_pair(
    before: np.ndarray,
    after: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "raw":
        return before, after
    if method == "mean3":
        return (
            base.smooth_embedding(before, method="mean", size=3),
            base.smooth_embedding(after, method="mean", size=3),
        )
    if method == "mean5":
        return (
            base.smooth_embedding(before, method="mean", size=5),
            base.smooth_embedding(after, method="mean", size=5),
        )
    return (
        base.smooth_embedding(before, method="gaussian", sigma=1.0),
        base.smooth_embedding(after, method="gaussian", sigma=1.0),
    )


def _score_pair(
    before: np.ndarray,
    after: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    before, after = _smoothed_pair(before, after, method)
    return base.symmetric_neighborhood_cosine_change(
        before,
        after,
        radius=2,
        displacement_penalty=0.05,
    )


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    patch_ids, construction_ids = _sample_patch_ids()
    records = []
    all_scores = []

    for patch_id in patch_ids:
        before = np.load(base.EMBEDDING_ROOT / base.BEFORE_MONTH / f"{patch_id}.npy")
        after = np.load(base.EMBEDDING_ROOT / base.AFTER_MONTH / f"{patch_id}.npy")
        methods = {}
        for key, _ in METHODS:
            scores, valid = _score_pair(before, after, key)
            methods[key] = (scores, valid)
            all_scores.append(scores)
        records.append((patch_id, methods))

    limits = base.global_limits(all_scores, low_quantile=0.02, high_quantile=0.98)
    rows = []
    for patch_id, methods in records:
        before_rgb, before_scene = construction._highres_rgb(
            base.BEFORE_MONTH, patch_id
        )
        after_rgb, after_scene = construction._highres_rgb(base.AFTER_MONTH, patch_id)
        display_shape = after_rgb.shape[:2]
        files = {
            "before": f"{patch_id}_before.png",
            "after": f"{patch_id}_after.png",
        }
        construction._save(output / files["before"], before_rgb)
        construction._save(output / files["after"], after_rgb)

        method_stats = {}
        for key, _ in METHODS:
            scores, valid = methods[key]
            heatmap = base._colored_map(scores, limits)
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
            method_stats[key] = {
                "mean": float(values.mean()),
                "p90": float(np.quantile(values, 0.90)),
                "p98": float(np.quantile(values, 0.98)),
                "high_share": float((values >= limits[1]).mean()),
            }
        rows.append(
            {
                "patch_id": patch_id,
                "sample_type": (
                    "施工变化样本" if patch_id in construction_ids else "固定随机样本"
                ),
                "before_scene": before_scene.stem,
                "after_scene": after_scene.stem,
                "stats": method_stats,
                **files,
            }
        )

    result = {
        "before_month": base.BEFORE_MONTH,
        "after_month": base.AFTER_MONTH,
        "random_seed": RANDOM_SEED,
        "random_count": RANDOM_COUNT,
        "shared_limits": limits,
        "methods": [{"key": key, "name": name} for key, name in METHODS],
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
        methods = []
        for method in result["methods"]:
            key = method["key"]
            stats = row["stats"][key]
            methods.append(
                f"<article><h3>{method['name']}</h3><div class='pair'>"
                f"<figure><button data-src='{row[f'{key}_change']}'><img "
                f"src='{row[f'{key}_change']}' alt='{method['name']}变化图'></button>"
                "<figcaption>变化距离</figcaption></figure>"
                f"<figure><button data-src='{row[f'{key}_overlay']}'><img "
                f"src='{row[f'{key}_overlay']}' alt='{method['name']}叠加图'></button>"
                "<figcaption>叠加到 2026 年 4 月影像</figcaption></figure></div>"
                f"<p class='stats'>均值 {stats['mean']:.4f}　P90 "
                f"{stats['p90']:.4f}　P98 {stats['p98']:.4f}　"
                f"高变化占比 {stats['high_share']:.2%}</p></article>"
            )
        sections.append(
            f"<section><div class='section-head'><h2>{row['patch_id']}</h2>"
            f"<span>{row['sample_type']}</span></div>"
            f"<p class='scene'>{row['before_scene']} → {row['after_scene']}</p>"
            "<div class='optical'>"
            f"<figure><button data-src='{row['before']}'><img src='{row['before']}' "
            "alt='2025年12月影像'></button><figcaption>2025 年 12 月高分影像</figcaption></figure>"
            f"<figure><button data-src='{row['after']}'><img src='{row['after']}' "
            "alt='2026年4月影像'></button><figcaption>2026 年 4 月高分影像</figcaption></figure>"
            f"</div><div class='methods'>{''.join(methods)}</div></section>"
        )

    low, high = result["shared_limits"]
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>海淀 embedding 平滑方法对照</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:#18212a;font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1880px;margin:auto;padding:24px}}header,section{{background:#fff;border:1px solid #dce2e7;border-radius:8px;padding:20px;margin-bottom:18px}}
h1{{margin:0 0 8px;font-size:27px}}h2{{margin:0;font-size:20px}}h3{{font-size:16px;margin:0 0 9px}}
p{{margin:6px 0}}.section-head{{display:flex;gap:12px;align-items:center}}.section-head span{{color:#495865;background:#eef2f5;padding:2px 8px;border-radius:4px}}
.scene{{font-size:13px;color:#64717c}}.legend{{height:14px;max-width:460px;background:linear-gradient(90deg,#3b4cc0,#f7f7f7,#b40426);border:1px solid #bdc6ce;margin-top:12px}}
.labels{{display:flex;justify-content:space-between;max-width:460px;color:#5d6974;font-size:13px}}
.optical{{display:grid;grid-template-columns:repeat(2,minmax(220px,360px));gap:12px;margin:14px 0 18px}}
.methods{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:12px}}
article{{border:1px solid #dfe5ea;padding:12px;border-radius:6px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
figure{{margin:0;min-width:0}}button{{border:0;padding:0;background:none;width:100%;cursor:zoom-in}}
img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;border:1px solid #d8dee3;background:#eef1f3}}
figcaption{{font-weight:600;text-align:center;font-size:13px;margin-top:5px}}.stats{{font-size:12px;color:#52606b;margin-top:9px}}
dialog{{border:0;border-radius:8px;padding:12px;background:#111;max-width:94vw;max-height:94vh}}dialog img{{width:auto;max-width:90vw;max-height:88vh;border:0}}dialog::backdrop{{background:rgba(0,0,0,.78)}}
@media(max-width:1300px){{.methods{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:720px){{main{{padding:10px}}.methods,.optical{{grid-template-columns:1fr}}.pair{{grid-template-columns:1fr}}}}
</style></head><body><main><header><h1>海淀 embedding 平滑方法对照</h1>
<p>对比 2025 年 12 月与 2026 年 4 月。包含 6 个施工变化样本和 10 个固定随机样本，随机种子为 {result['random_seed']}。</p>
<p>第一组直接执行原始双向 5×5 邻域匹配；其余三组先分别平滑两期 64 维 embedding，再执行完全相同的双向 5×5 匹配和位移惩罚。没有使用建筑、施工或其他语义掩膜。</p>
<p><b>注意：</b>“3×3/5×5 均值”是匹配前的特征平滑窗口；“双向 5×5”是随后寻找对应像素的搜索窗口，两者作用不同。</p>
<p>四种方法共用同一绝对色标 {low:.4f}–{high:.4f}：蓝色表示距离较小，红色表示距离较大。平滑会降低局部噪声，也可能抹去小范围真实变化，因此必须结合两期光学影像判断。</p>
<div class="legend"></div><div class="labels"><span>低变化</span><span>中等变化</span><span>高变化</span></div>
</header>{''.join(sections)}</main><dialog id="viewer"><img alt="放大预览"></dialog>
<script>const d=document.querySelector('#viewer'),i=d.querySelector('img');
document.querySelectorAll('button[data-src]').forEach(b=>b.onclick=()=>{{i.src=b.dataset.src;d.showModal()}});
d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    output = ROOT / "Tmp" / f"haidian_smoothing_change_{datetime.now():%Y%m%d_%H%M%S}"
    result = run(output)
    print(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
