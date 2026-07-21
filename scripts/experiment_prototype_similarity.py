"""Evaluate polygon-prototype retrieval on the deployed Haidian embeddings.

This is an offline experiment only. It does not import or modify API training
code, model artifacts, or registry records.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_ROOT = ROOT / "data/haidian/embeddings/v1"
TASK_ROOT = ROOT / "data/haidian/tasks"
MONTH = "202604"
SUPPORT_COUNTS = (1, 3, 5, 9)
TASKS = ("building_extraction", "road_extraction")


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def load_pair(task: str, patch_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embedding = np.load(EMBEDDING_ROOT / MONTH / f"{patch_id}.npy").astype(np.float32)
    label = np.load(TASK_ROOT / task / "v1/labels" / f"{patch_id}.npy") > 0
    preview = np.asarray(Image.open(EMBEDDING_ROOT / MONTH / f"{patch_id}.png").convert("RGB"))
    return embedding, label, preview


def available_patches(task: str) -> list[str]:
    labels = TASK_ROOT / task / "v1/labels"
    result = []
    for path in sorted(labels.glob("patch_*.npy")):
        pid = path.stem
        if (EMBEDDING_ROOT / MONTH / f"{pid}.npy").exists():
            mask = np.load(path)
            positive = int((mask > 0).sum())
            if 80 <= positive <= 9000:
                result.append(pid)
    return result


def polygon_candidates(patch_ids: list[str], task: str) -> list[tuple[str, np.ndarray, int]]:
    by_patch: list[list[tuple[str, np.ndarray, int]]] = []
    for pid in patch_ids:
        _, mask, _ = load_pair(task, pid)
        components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        candidates = []
        for index in range(1, count + 1):
            component = components == index
            area = int(component.sum())
            if 8 <= area <= 1200:
                candidates.append((pid, component, area))
        candidates.sort(key=lambda item: item[2], reverse=True)
        if candidates:
            by_patch.append(candidates[:4])

    # Round-robin prevents one patch with many polygons from dominating support.
    result: list[tuple[str, np.ndarray, int]] = []
    for rank in range(4):
        for candidates in by_patch:
            if rank < len(candidates):
                result.append(candidates[rank])
    return result


def build_polygon_prototypes(polygons: list[tuple[str, np.ndarray, int]], task: str) -> np.ndarray:
    polygon_prototypes = []
    embedding_cache: dict[str, np.ndarray] = {}
    for pid, mask, _ in polygons:
        if pid not in embedding_cache:
            embedding_cache[pid] = load_pair(task, pid)[0]
        pixels = np.moveaxis(embedding_cache[pid], 0, -1)[mask]
        # Each polygon contributes equally, independent of its pixel area.
        polygon_prototypes.append(unit_rows(pixels).mean(axis=0))
    return unit_rows(np.asarray(polygon_prototypes, dtype=np.float32))


def merge_prototypes(polygon_prototypes: np.ndarray, count: int = 1) -> np.ndarray:
    """Deterministic spherical k-means over polygon-level prototypes."""
    count = min(count, len(polygon_prototypes))
    if count == 1:
        return unit_rows(polygon_prototypes.mean(axis=0, keepdims=True))
    centers = polygon_prototypes[np.linspace(0, len(polygon_prototypes) - 1, count, dtype=int)]
    for _ in range(30):
        groups = np.argmax(polygon_prototypes @ centers.T, axis=1)
        updated = []
        for index in range(count):
            members = polygon_prototypes[groups == index]
            updated.append(centers[index] if len(members) == 0 else members.mean(axis=0))
        updated = unit_rows(np.asarray(updated))
        if np.allclose(updated, centers, atol=1e-5):
            break
        centers = updated
    return centers


def score_map(embedding: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    pixels = np.moveaxis(embedding, 0, -1)
    prototypes = prototype[None, :] if prototype.ndim == 1 else prototype
    scores = np.tensordot(unit_rows(pixels), prototypes.T, axes=([-1], [0]))
    return scores.max(axis=-1)


def confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[int, int, int]:
    predicted = scores >= threshold
    tp = int(np.logical_and(predicted, labels).sum())
    fp = int(np.logical_and(predicted, ~labels).sum())
    fn = int(np.logical_and(~predicted, labels).sum())
    return tp, fp, fn


def metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "iou": tp / max(1, tp + fp + fn),
    }


def calibrate_threshold(prototype: np.ndarray, task: str, patch_ids: list[str]) -> tuple[float, float]:
    scores_all, labels_all = [], []
    for pid in patch_ids:
        embedding, label, _ = load_pair(task, pid)
        scores_all.append(score_map(embedding, prototype).reshape(-1))
        labels_all.append(label.reshape(-1))
    scores = np.concatenate(scores_all)
    labels = np.concatenate(labels_all)
    low, high = np.quantile(scores, [0.02, 0.995])
    best_threshold, best_f05 = float(high), -1.0
    for threshold in np.linspace(low, high, 240):
        tp, fp, fn = confusion(scores, labels, float(threshold))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f05 = 1.25 * precision * recall / max(1e-12, 0.25 * precision + recall)
        if f05 > best_f05:
            best_threshold, best_f05 = float(threshold), float(f05)
    return best_threshold, best_f05


def evaluate(prototype: np.ndarray, threshold: float, task: str, patch_ids: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for pid in patch_ids:
        embedding, label, _ = load_pair(task, pid)
        current = confusion(score_map(embedding, prototype), label, threshold)
        tp += current[0]
        fp += current[1]
        fn += current[2]
    return metrics(tp, fp, fn)


def save_panel(output: Path, task: str, pid: str, prototype: np.ndarray, threshold: float) -> None:
    embedding, label, preview = load_pair(task, pid)
    scores = score_map(embedding, prototype)
    predicted = scores >= threshold
    overlay = preview.copy()
    overlay[label] = (0.55 * overlay[label] + 0.45 * np.array([40, 210, 100])).astype(np.uint8)
    overlay[predicted] = (0.45 * overlay[predicted] + 0.55 * np.array([245, 75, 70])).astype(np.uint8)

    prediction_view = np.full((*predicted.shape, 3), 255, dtype=np.uint8)
    prediction_view[predicted] = np.array([225, 38, 45], dtype=np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(16.2, 3.5), constrained_layout=True)
    axes[0].imshow(preview)
    axes[0].set_title("P10C embedding PCA")
    axes[1].imshow(label, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Reference label")
    image = axes[2].imshow(scores, cmap="turbo", vmin=np.quantile(scores, 0.05), vmax=np.quantile(scores, 0.99))
    axes[2].set_title(f"Cosine similarity\nthreshold={threshold:.3f}")
    fig.colorbar(image, ax=axes[2], fraction=0.046)
    axes[3].imshow(prediction_view)
    axes[3].set_title("Prediction mask\nred = predicted target")
    axes[4].imshow(overlay)
    axes[4].set_title("Overlay: prediction red / label green")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"{task} · {pid} · unseen test patch", fontsize=13)
    fig.savefig(output / f"{task}_{pid}.png", dpi=160)
    plt.close(fig)


def write_html(output: Path, results: dict) -> None:
    task_names = {"building_extraction": "建筑提取", "road_extraction": "道路提取"}
    sections = []
    for task, data in results.items():
        rows = "".join(
            f"<tr><td>{item['polygon_count']}</td><td>{item.get('prototype_count', 1)}</td><td>{item['threshold']:.4f}</td>"
            f"<td>{item['precision']:.3f}</td><td>{item['recall']:.3f}</td>"
            f"<td>{item['f1']:.3f}</td><td>{item['iou']:.3f}</td></tr>"
            for item in data["runs"]
        )
        images = "".join(
            f"<figure><img src='{task}_{pid}.png'><figcaption>{pid}，未参与原型生成和阈值校准</figcaption></figure>"
            for pid in data["visualized_patches"]
        )
        sections.append(f"""
        <section><h2>{task_names[task]}</h2>
        <p>支持 Polygon：{', '.join(data['support_polygons'])}</p>
        <table><thead><tr><th>Polygon 数</th><th>原型数</th><th>自动阈值</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead><tbody>{rows}</tbody></table>
        <div class='grid'>{images}</div></section>""")
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>海淀少样本原型检索实验</title><style>
    body{{margin:0;background:#f4f6f8;color:#17202a;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}
    h1{{font-size:28px;margin:0 0 8px}}h2{{margin-top:0}}.intro,section{{background:white;border:1px solid #dfe4e8;border-radius:8px;padding:22px;margin:18px 0}}
    table{{border-collapse:collapse;width:100%;max-width:760px}}th,td{{padding:9px 12px;border-bottom:1px solid #e5e9ed;text-align:right}}th:first-child,td:first-child{{text-align:left}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:18px;margin-top:20px}}figure{{margin:0}}img{{width:100%;display:block;border:1px solid #d9dee3}}figcaption{{color:#56616b;margin-top:5px}}
    code{{background:#edf1f3;padding:2px 5px;border-radius:3px}}@media(max-width:700px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
    </style></head><body><main><div class='intro'><h1>海淀 P10C 少样本原型相似度实验</h1>
    <p>数据：最新海淀 P10C epoch800 embedding，月份 <code>{MONTH}</code>。每个连通标注区域模拟一个前端 Polygon；每个 Polygon 独立求平均向量并等权聚合。</p>
    <p>阈值只在校准 patch 上按 F0.5 自动选择；表格指标和图片来自完全独立的测试 patch。该实验没有修改 API 或现有模型。</p></div>{''.join(sections)}</main></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "Tmp" / f"prototype_similarity_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    results = {}

    for task in TASKS:
        patches = available_patches(task)
        support_patches, calibration_patches, test_patches = patches[:8], patches[8:16], patches[16:28]
        candidates = polygon_candidates(support_patches, task)
        if len(candidates) < max(SUPPORT_COUNTS):
            raise RuntimeError(f"Not enough polygon components for {task}: {len(candidates)}")
        runs = []
        final_prototype = None
        final_threshold = None
        for count in SUPPORT_COUNTS:
            polygon_prototypes = build_polygon_prototypes(candidates[:count], task)
            prototype = merge_prototypes(polygon_prototypes, count=1)
            threshold, calibration_f05 = calibrate_threshold(prototype, task, calibration_patches)
            result = evaluate(prototype, threshold, task, test_patches)
            runs.append({"polygon_count": count, "threshold": threshold, "calibration_f05": calibration_f05, **result})
            if count == max(SUPPORT_COUNTS):
                final_prototype, final_threshold = prototype, threshold
        polygon_prototypes = build_polygon_prototypes(candidates[:9], task)
        multi_prototype = merge_prototypes(polygon_prototypes, count=3)
        multi_threshold, multi_calibration_f05 = calibrate_threshold(multi_prototype, task, calibration_patches)
        multi_result = evaluate(multi_prototype, multi_threshold, task, test_patches)
        runs.append({
            "polygon_count": 9,
            "prototype_count": 3,
            "strategy": "three_prototypes",
            "threshold": multi_threshold,
            "calibration_f05": multi_calibration_f05,
            **multi_result,
        })
        final_prototype, final_threshold = multi_prototype, multi_threshold
        visualized = test_patches[:4]
        for pid in visualized:
            save_panel(output, task, pid, final_prototype, final_threshold)
        results[task] = {
            "runs": runs,
            "support_polygons": [f"{pid}({area}px)" for pid, _, area in candidates[:9]],
            "support_patches": support_patches,
            "calibration_patches": calibration_patches,
            "test_patches": test_patches,
            "visualized_patches": visualized,
        }

    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output, results)
    print(output)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
