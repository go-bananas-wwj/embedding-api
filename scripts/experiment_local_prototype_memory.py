"""Offline local-prototype memory experiment for Haidian P10C embeddings."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

from experiment_prototype_similarity import (
    EMBEDDING_ROOT,
    MONTH,
    ROOT,
    TASKS,
    available_patches,
    build_polygon_prototypes,
    confusion,
    load_pair,
    merge_prototypes,
    metrics,
    polygon_candidates,
    score_map as mean_score_map,
    unit_rows,
)

OPTICAL_ROOT = Path("/workspace/projects/xuannv-show/data/haidian/base_maps")


def spherical_clusters(pixels: np.ndarray, count: int) -> np.ndarray:
    pixels = unit_rows(pixels.astype(np.float32, copy=False))
    count = min(count, len(pixels))
    seeds = np.linspace(0, len(pixels) - 1, count, dtype=int)
    centers = pixels[seeds]
    for _ in range(25):
        groups = np.argmax(pixels @ centers.T, axis=1)
        updated = []
        for index in range(count):
            members = pixels[groups == index]
            updated.append(centers[index] if not len(members) else members.mean(axis=0))
        updated = unit_rows(np.asarray(updated))
        if np.allclose(updated, centers, atol=1e-5):
            break
        centers = updated
    return centers


def build_local_memory(polygons: list[tuple[str, np.ndarray, int]], task: str) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    local = []
    for pid, mask, area in polygons:
        if pid not in cache:
            cache[pid] = load_pair(task, pid)[0]
        pixels = np.moveaxis(cache[pid], 0, -1)[mask]
        # Preserve within-polygon appearance modes instead of one global mean.
        count = 2 if area < 64 else 3 if area < 256 else 4
        local.extend(spherical_clusters(pixels, count))
    memory = unit_rows(np.asarray(local, dtype=np.float32))
    if len(memory) > 32:
        memory = merge_prototypes(memory, count=32)
    return memory


def local_score_map(embedding: np.ndarray, memory: np.ndarray, adaptive: bool = False) -> np.ndarray:
    pixels = unit_rows(np.moveaxis(embedding, 0, -1))
    similarities = np.tensordot(pixels, memory.T, axes=([-1], [0]))
    nearest = np.partition(similarities, -min(3, len(memory)), axis=-1)[..., -min(3, len(memory)):]
    score = nearest.mean(axis=-1)
    if adaptive:
        # One conservative query-specific update; only the strongest 0.3% can vote.
        cutoff = max(float(np.quantile(score, 0.997)), 0.82)
        confident = score >= cutoff
        if 4 <= int(confident.sum()) <= 160:
            query_proto = unit_rows(pixels[confident].mean(axis=0, keepdims=True))[0]
            query_similarity = np.tensordot(pixels, query_proto, axes=([-1], [0]))
            score = 0.85 * score + 0.15 * query_similarity
    return ndimage.gaussian_filter(score, sigma=0.55)


def score_for(strategy: str, embedding: np.ndarray, representation: np.ndarray) -> np.ndarray:
    if strategy == "single_mean":
        return mean_score_map(embedding, representation)
    return local_score_map(embedding, representation, adaptive=strategy == "local_top3_adaptive")


def calibrate(strategy: str, representation: np.ndarray, task: str, patch_ids: list[str]) -> tuple[float, float]:
    scores, labels = [], []
    for pid in patch_ids:
        embedding, label, _ = load_pair(task, pid)
        scores.append(score_for(strategy, embedding, representation).reshape(-1))
        labels.append(label.reshape(-1))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    low, high = np.quantile(scores, [0.02, 0.995])
    best = (float(high), -1.0)
    for threshold in np.linspace(low, high, 260):
        tp, fp, fn = confusion(scores, labels, float(threshold))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f05 = 1.25 * precision * recall / max(1e-12, 0.25 * precision + recall)
        if f05 > best[1]:
            best = (float(threshold), float(f05))
    return best


def evaluate(strategy: str, representation: np.ndarray, threshold: float, task: str, patch_ids: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for pid in patch_ids:
        embedding, label, _ = load_pair(task, pid)
        current = confusion(score_for(strategy, embedding, representation), label, threshold)
        tp += current[0]
        fp += current[1]
        fn += current[2]
    return metrics(tp, fp, fn)


def optical_image(pid: str, fallback: np.ndarray) -> np.ndarray:
    path = OPTICAL_ROOT / pid / MONTH / "s2.png"
    return np.asarray(Image.open(path).convert("RGB")) if path.exists() else fallback


def save_panel(output: Path, task: str, pid: str, memory: np.ndarray, threshold: float) -> None:
    embedding, label, pca = load_pair(task, pid)
    optical = optical_image(pid, pca)
    if optical.shape[:2] != label.shape:
        optical = np.asarray(Image.fromarray(optical).resize(label.shape[::-1], Image.Resampling.BILINEAR))
    scores = local_score_map(embedding, memory, adaptive=True)
    predicted = scores >= threshold
    prediction = np.full((*label.shape, 3), 255, dtype=np.uint8)
    prediction[predicted] = (225, 38, 45)
    overlay = optical.copy()
    overlay[label] = (0.5 * overlay[label] + 0.5 * np.array([35, 210, 95])).astype(np.uint8)
    overlay[predicted] = (0.45 * overlay[predicted] + 0.55 * np.array([235, 45, 50])).astype(np.uint8)

    fig, axes = plt.subplots(1, 6, figsize=(19.4, 3.5), constrained_layout=True)
    content = [
        (optical, "Sentinel-2 optical"), (pca, "P10C embedding PCA"),
        (label, "Reference label"), (scores, f"Local prototype score\nthreshold={threshold:.3f}"),
        (prediction, "Prediction\nred = predicted target"), (overlay, "Optical overlay\nred prediction / green label"),
    ]
    for axis, (image, title) in zip(axes, content):
        axis.imshow(image, cmap="turbo" if image is scores else None)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{task} · {pid} · unseen test patch", fontsize=13)
    fig.savefig(output / f"{task}_{pid}.png", dpi=170)
    plt.close(fig)


def write_html(output: Path, results: dict) -> None:
    names = {"building_extraction": "建筑提取", "road_extraction": "道路提取"}
    strategy_names = {"single_mean": "单一平均原型", "local_top3": "局部原型 Top-3", "local_top3_adaptive": "局部原型 Top-3 + 自适应"}
    sections = []
    for task, data in results.items():
        rows = "".join(
            f"<tr><td>{strategy_names[r['strategy']]}</td><td>{r['prototype_count']}</td><td>{r['threshold']:.4f}</td>"
            f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['iou']:.3f}</td></tr>"
            for r in data["runs"]
        )
        images = "".join(f"<figure><img src='{task}_{pid}.png' alt='{pid}' loading='lazy'><figcaption>{pid}，独立测试 patch</figcaption></figure>" for pid in data["visualized_patches"])
        sections.append(f"<section><h2>{names[task]}</h2><table><thead><tr><th>策略</th><th>原型数</th><th>阈值</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead><tbody>{rows}</tbody></table><div class='grid'>{images}</div></section>")
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>海淀局部原型实验</title><style>
    body{{margin:0;background:#f3f5f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1700px;margin:auto;padding:26px}}section,.intro{{background:#fff;border:1px solid #dce2e6;border-radius:8px;padding:20px;margin:18px 0}}h1,h2{{margin-top:0}}table{{border-collapse:collapse;width:100%;max-width:980px}}th,td{{padding:9px 12px;border-bottom:1px solid #e4e8eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(700px,1fr));gap:18px;margin-top:20px}}figure{{margin:0}}figure img{{width:100%;display:block;border:1px solid #d7dde1;cursor:zoom-in}}figcaption{{color:#59636c;margin-top:5px}}dialog{{border:0;padding:0;background:transparent;max-width:96vw;max-height:96vh}}dialog img{{display:block;max-width:96vw;max-height:92vh;background:#fff}}dialog::backdrop{{background:rgba(8,12,16,.88)}}@media(max-width:760px){{main{{padding:12px}}.grid{{grid-template-columns:1fr}}}}
    </style></head><body><main><div class='intro'><h1>海淀 P10C 局部原型记忆实验</h1><p>9 个 Polygon；Polygon 内提取局部原型，查询像素使用最相似 3 个原型的平均分。阈值校准和最终测试使用不同 patch。</p><p>点击任意结果图可放大，再次点击或按 Esc 关闭。</p></div>{''.join(sections)}</main><dialog id='viewer'><img alt='放大结果'></dialog><script>const d=document.querySelector('#viewer'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.addEventListener('click',()=>{{v.src=i.src;d.showModal()}}));d.addEventListener('click',()=>d.close());</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "Tmp" / f"local_prototype_memory_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in TASKS:
        patches = available_patches(task)
        support, calibration, test = patches[:8], patches[8:16], patches[16:28]
        polygons = polygon_candidates(support, task)[:9]
        polygon_means = build_polygon_prototypes(polygons, task)
        single = merge_prototypes(polygon_means, 1)
        memory = build_local_memory(polygons, task)
        runs = []
        for strategy, representation in (("single_mean", single), ("local_top3", memory), ("local_top3_adaptive", memory)):
            threshold, calibration_f05 = calibrate(strategy, representation, task, calibration)
            result = evaluate(strategy, representation, threshold, task, test)
            runs.append({"strategy": strategy, "prototype_count": len(representation), "threshold": threshold, "calibration_f05": calibration_f05, **result})
        best = next(run for run in runs if run["strategy"] == "local_top3_adaptive")
        visualized = test[:4]
        for pid in visualized:
            save_panel(output, task, pid, memory, best["threshold"])
        results[task] = {"runs": runs, "visualized_patches": visualized, "support_polygons": [f"{p}({a}px)" for p, _, a in polygons]}
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output, results)
    print(output)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
