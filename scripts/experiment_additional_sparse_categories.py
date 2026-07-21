"""Sparse retrieval-vs-conv experiment for additional Haidian categories."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

import experiment_prototype_similarity as base
from experiment_local_prototype_memory import build_local_memory, local_score_map
from experiment_sparse_retrieval_vs_conv import (
    ROOT, calibrate_from_sparse_targets, conv_scores, evaluate,
    grouped_sparse_samples, support_montage, test_panel,
)
from app.services.training_engine import _train_binary_conv_head


WORLDCOVER = ROOT / "data/haidian/archive/processed_training_data/extracted/labels/worldcover"
CATEGORIES = {
    # The archived masks use the project's normalized class indices rather
    # than ESA's original 10/20/.../100 codes. Visual audit against the
    # co-registered optical imagery confirms 1=water and 8=tree cover.
    "water": {"name": "永久水体", "worldcover_class": 1, "source": "ESA WorldCover 粗标签"},
    "tree_cover": {"name": "树木覆盖", "worldcover_class": 8, "source": "ESA WorldCover 粗标签"},
    "cropland": {"name": "耕地", "worldcover_class": 4, "source": "ESA WorldCover 粗标签"},
    "construction": {"name": "施工地", "source": "项目人工标签"},
}
COUNTS = (1, 3, 5, 9)


def prepare_labels(root: Path) -> None:
    for category, spec in CATEGORIES.items():
        target = root / category / "v1/labels"
        target.mkdir(parents=True, exist_ok=True)
        if "worldcover_class" in spec:
            for path in WORLDCOVER.glob("worldcover_*_patch_*.tif"):
                patch_id = "patch_" + path.stem.rsplit("patch_", 1)[-1]
                with rasterio.open(path) as dataset:
                    label = dataset.read(1) == spec["worldcover_class"]
                np.save(target / f"{patch_id}.npy", label.astype(np.uint8))
        else:
            source = ROOT / "data/haidian/tasks/construction/v1/labels"
            for path in source.glob("patch_*.npy"):
                np.save(target / path.name, (np.load(path) > 0).astype(np.uint8))


def write_html(output: Path, results: dict) -> None:
    sections = []
    for category, data in results.items():
        rows = "".join(
            f"<tr><td>{r['polygon_count']}</td><td>{r['method']}</td><td>{r['precision']:.3f}</td>"
            f"<td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['iou']:.3f}</td></tr>"
            for r in data["runs"]
        )
        figures = f"<figure><img src='{category}_support_annotations.png'><figcaption>5 个紫红色稀疏 Polygon；这是模型能看到的全部标签</figcaption></figure>"
        figures += "".join(
            f"<figure><img src='{category}_{pid}.png'><figcaption>{pid}，独立测试 patch；完整标签只用于评分</figcaption></figure>"
            for pid in data["visualized"]
        )
        sections.append(
            f"<section><h2>{data['name']}</h2><p class='source'>标签来源：{data['source']}</p>"
            f"<table><thead><tr><th>Polygon 数</th><th>方法</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead>"
            f"<tbody>{rows}</tbody></table><div class='grid'>{figures}</div></section>"
        )
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>海淀多类别稀疏标注实验</title><style>body{{margin:0;background:#f3f5f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1800px;margin:auto;padding:24px}}.intro,section{{background:white;border:1px solid #dce2e6;border-radius:8px;padding:20px;margin:18px 0}}h1,h2{{margin-top:0}}.source{{color:#59636c}}table{{border-collapse:collapse;width:100%;max-width:900px}}th,td{{padding:8px 11px;border-bottom:1px solid #e4e8eb;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}.grid{{display:grid;grid-template-columns:1fr;gap:18px;margin-top:20px}}figure{{margin:0}}figure img{{display:block;width:100%;border:1px solid #d7dde1;cursor:zoom-in}}figcaption{{color:#59636c;margin-top:5px}}dialog{{border:0;padding:0;background:transparent;max-width:97vw;max-height:97vh}}dialog img{{max-width:97vw;max-height:93vh;display:block}}dialog::backdrop{{background:rgba(8,12,16,.9)}}</style></head><body><main><div class='intro'><h1>不同地物类别的稀疏标注效果</h1><p>向量检索和 Binary Conv 3x3 接收完全相同的 1、3、5、9 个 Polygon。阈值仅由稀疏输入确定；测试 patch 完整标签不参与训练。</p><p>光学图使用 202604 Sentinel-2 多景有效像素合成。点击任意图片放大。</p></div>{''.join(sections)}</main><dialog id='viewer'><img></dialog><script>const d=document.querySelector('#viewer'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path); args = parser.parse_args()
    output = args.output or ROOT / "Tmp" / f"additional_sparse_categories_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    label_root = output / "_evaluation_labels"
    prepare_labels(label_root)
    base.TASK_ROOT = label_root
    results = {}
    for category, spec in CATEGORIES.items():
        patches = base.available_patches(category)
        # Support and test remain disjoint; skip a gap to reduce neighboring-patch leakage.
        support_pool, test = patches[:10], patches[18:30]
        candidates = base.polygon_candidates(support_pool, category)
        if len(candidates) < 9 or len(test) < 4:
            raise RuntimeError(f"{category}: insufficient polygons={len(candidates)} or test patches={len(test)}")
        runs = []; display = None
        for count in COUNTS:
            selected = candidates[:count]
            memory = build_local_memory(selected, category)
            sparse_samples = grouped_sparse_samples(selected, category)
            retrieval = lambda embedding, m=memory: local_score_map(embedding, m, adaptive=False)
            retrieval_threshold, _ = calibrate_from_sparse_targets(retrieval, sparse_samples)
            retrieval_metrics = evaluate(retrieval, retrieval_threshold, category, test)
            runs.append({"polygon_count": count, "method": "局部向量检索", "threshold": retrieval_threshold, **retrieval_metrics})
            model, conv_threshold, _, _, epochs = _train_binary_conv_head(sparse_samples, 40)
            conv = lambda embedding, m=model: conv_scores(m, embedding)
            conv_metrics = evaluate(conv, conv_threshold, category, test)
            runs.append({"polygon_count": count, "method": "Binary Conv 3x3", "threshold": conv_threshold, "epochs": epochs, **conv_metrics})
            if count == 5:
                display = selected, memory, retrieval_threshold, model, conv_threshold
        selected, memory, retrieval_threshold, model, conv_threshold = display
        support_montage(output, category, selected)
        retrieval = lambda embedding, m=memory: local_score_map(embedding, m, adaptive=False)
        conv = lambda embedding, m=model: conv_scores(m, embedding)
        visualized = test[:4]
        for pid in visualized:
            test_panel(output, category, pid, retrieval, retrieval_threshold, conv, conv_threshold)
        results[category] = {"name": spec["name"], "source": spec["source"], "runs": runs, "visualized": visualized}
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output, results)
    print(output); print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
