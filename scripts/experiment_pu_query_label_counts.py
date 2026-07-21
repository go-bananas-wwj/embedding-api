"""Visualize PU + Query performance with one to five sparse labels."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import experiment_no_positive_memory as experiment
import experiment_prototype_similarity as base
from experiment_sparse_retrieval_vs_conv import ROOT


EXAMPLES = {
    "building_extraction": "patch_000205",
    "road_extraction": "patch_000264",
    "water": "patch_000276",
    "tree_cover": "patch_000169",
    "cropland": "patch_000297",
    "construction": "patch_000198",
}

PREVIOUS_FIXED_EXAMPLES = {
    "building_extraction": [
        "patch_000018", "patch_000019", "patch_000020", "patch_000021"
    ],
    "road_extraction": [
        "patch_000018", "patch_000019", "patch_000020", "patch_000021"
    ],
    "water": [
        "patch_000106", "patch_000275", "patch_000089", "patch_000107"
    ],
    "tree_cover": [
        "patch_000136", "patch_000172", "patch_000084", "patch_000054"
    ],
    "cropland": [
        "patch_000119", "patch_000121", "patch_000134", "patch_000141"
    ],
    "construction": [
        "patch_000113", "patch_000122", "patch_000123", "patch_000125"
    ],
}

# Selected on the independent fixed-validation patches, never on the displayed
# target label. The default largest-mask heuristic overfits patch_000003 for
# buildings and does not represent the repetitive residential roofs well.
SUPPORT_PATCH_OVERRIDES = {
    "building_extraction": "patch_000026",
}


def sparse_polygons_from_one_patch(task: str, support: list[str], count: int = 5):
    """Create small, separated positive polygons on one representative support image."""
    choices = []
    for pid in support:
        _, label, _ = base.load_pair(task, pid)
        choices.append((int(label.sum()), pid, label))
    _, pid, label = max(choices)

    distance = ndimage.distance_transform_edt(label)
    candidates = np.argwhere(np.logical_and(label, distance >= 2))
    if len(candidates) < count:
        candidates = np.argwhere(label)
    if len(candidates) < count:
        raise ValueError(f"Not enough positive pixels for {task}/{pid}")

    center = candidates.mean(axis=0)
    selected = [candidates[np.argmin(np.square(candidates - center).sum(axis=1))]]
    while len(selected) < count:
        distances = np.min(
            [np.square(candidates - point).sum(axis=1) for point in selected], axis=0
        )
        selected.append(candidates[int(np.argmax(distances))])

    yy, xx = np.ogrid[: label.shape[0], : label.shape[1]]
    polygons = []
    occupied = np.zeros_like(label, dtype=bool)
    for y, x in selected:
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= 7**2
        mask = np.logical_and.reduce((disk, label, ~occupied))
        if int(mask.sum()) < 8:
            disk = (yy - y) ** 2 + (xx - x) ** 2 <= 11**2
            mask = np.logical_and.reduce((disk, label, ~occupied))
        occupied |= mask
        polygons.append((pid, mask, None))
    return polygons


def save_comparison_panel(
    output: Path,
    task: str,
    task_name: str,
    target_pid: str,
    getter,
    threshold: float,
    polygons,
):
    embedding, label, pca = base.load_pair(task, target_pid)
    optical = experiment.true_color_composite(target_pid)
    predicted = getter(embedding) >= threshold
    overlay = optical.copy()
    overlay[label] = (
        0.5 * overlay[label] + 0.5 * np.array([35, 210, 95])
    ).astype(np.uint8)
    overlay[predicted] = (
        0.45 * overlay[predicted] + 0.55 * np.array([235, 45, 50])
    ).astype(np.uint8)
    model_visible, support_pid = experiment.model_visible_label_view(polygons)

    count = len(polygons)
    items = [
        (optical, "新的 Sentinel-2 光学影像"),
        (pca, "P10C 64 维嵌入\nPCA 三通道投影"),
        (experiment.true_label_view(label), "完整真实标签\n仅用于离线评估"),
        (
            model_visible,
            f"模型可见标签 · {support_pid}\n大图中紫红色为 {count} 个 Polygon",
        ),
        (experiment.mask_view(predicted), "PU + Query 预测结果"),
        (overlay, "结果叠加\n红色预测 / 绿色标签"),
    ]
    figure, axes = plt.subplots(1, 6, figsize=(19.2, 3.5), constrained_layout=True)
    for axis, (image, title) in zip(axes, items):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{task_name} · {target_pid} · 仅使用 {count} 个稀疏标注 Polygon"
    )
    filename = f"{task}_{target_pid}_labels_{count}.png"
    figure.savefig(output / filename, dpi=170)
    plt.close(figure)
    return filename


def write_html(output: Path, results: dict) -> None:
    sections = []
    for task, data in results.items():
        rows = "".join(
            f"<tr><td>{row['label_count']}</td><td>{row['precision']:.3f}</td>"
            f"<td>{row['recall']:.3f}</td><td>{row['f1']:.3f}</td>"
            f"<td>{row['iou']:.3f}</td></tr>"
            for row in data["runs"]
        )
        figures = "".join(
            f"<figure><img src='{row['image']}' alt='{data['name']} {row['label_count']} 个标签'>"
            f"<figcaption>{row['label_count']} 个 Polygon · F1 {row['f1']:.3f} · "
            f"IoU {row['iou']:.3f}</figcaption></figure>"
            for row in data["runs"]
        )
        previous_figures = "".join(
            f"<figure><img src='{row['image']}' alt='{data['name']} {row['patch_id']}'>"
            f"<figcaption>{row['patch_id']} · 固定使用 5 个 Polygon · "
            f"F1 {row['f1']:.3f} · IoU {row['iou']:.3f}</figcaption></figure>"
            for row in data["previous_examples"]
        )
        sections.append(
            f"<section><h2>{data['name']}</h2><p>固定测试案例：<code>{data['target_patch']}</code>；"
            f"固定支持影像：<code>{data['support_patch']}</code>。"
            "五行仅增加模型可见 Polygon 数量，其他条件保持不变。</p>"
            "<table><thead><tr><th>Polygon 数</th><th>Precision</th><th>Recall</th>"
            f"<th>F1</th><th>IoU</th></tr></thead><tbody>{rows}</tbody></table>"
            f"<div class='grid'>{figures}</div>"
            "<div class='previous'><h3>更多固定测试案例</h3>"
            "<p>以下是之前画廊中的其余案例，统一使用上方第 5 档的五个稀疏 Polygon 参数。"
            "跨片区精选 Patch 已作为上方 1–5 对照的固定目标，因此不重复。</p>"
            f"<div class='grid'>{previous_figures}</div></div></section>"
        )
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>PU + Query 1-5 个标签对照</title><style>
body{{margin:0;background:#f3f5f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1980px;margin:auto;padding:24px}}.intro,section{{background:#fff;border:1px solid #dce2e6;border-radius:8px;padding:20px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;max-width:700px}}th,td{{padding:8px 11px;border-bottom:1px solid #e4e8eb;text-align:right}}
.grid{{display:grid;grid-template-columns:1fr;gap:20px;margin-top:20px}}figure{{margin:0}}figure img{{width:100%;display:block;border:1px solid #d7dde1;cursor:zoom-in}}
figcaption{{color:#59636c;margin-top:5px}}.previous{{margin-top:30px;padding-top:22px;border-top:2px solid #dce2e6}}.previous h3{{margin:0 0 6px}}code{{background:#eef1f3;padding:2px 5px}}dialog{{border:0;padding:0;background:transparent;max-width:97vw;max-height:97vh}}
dialog img{{max-width:97vw;max-height:93vh}}dialog::backdrop{{background:rgba(8,12,16,.9)}}</style></head><body><main>
<div class='intro'><h1>PU + Query 少量标注效果</h1>
<p>每个类别只选择一个测试案例，比较模型仅看到 1–5 个稀疏 Polygon 时的效果。模型可见标签已直接画在完整支持影像上，紫红色区域就是当前训练阶段实际可见的全部正样本。</p>
<p>测试影像的完整真实标签仅用于评估，不参与训练。点击图片可放大。</p></div>{''.join(sections)}</main>
<dialog id='viewer'><img></dialog><script>const d=document.querySelector('#viewer'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script>
</body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    experiment.configure_chinese_font()
    output = ROOT / "Tmp" / f"pu_query_label_counts_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    labels = output / "_evaluation_labels"
    experiment.prepare_all_labels(labels)
    base.TASK_ROOT = labels
    mean, std = experiment.global_stats()

    results = {}
    for task, spec in experiment.CATEGORIES.items():
        patches = base.available_patches(task)
        target = EXAMPLES[task]
        if target not in patches:
            raise FileNotFoundError(f"Selected example is unavailable: {task}/{target}")
        support = [pid for pid in patches if pid != target][:18]
        support_override = SUPPORT_PATCH_OVERRIDES.get(task)
        polygon_support = [support_override] if support_override else support
        polygons = sparse_polygons_from_one_patch(task, polygon_support)
        runs = []
        five_label_model = None
        for count in range(1, 6):
            selected = polygons[:count]
            positive = experiment.polygon_center(selected, task, mean, std)
            negative, positive_pixels, negative_pixels = experiment.reliable_background_center(
                selected, task, positive, mean, std
            )
            positive_scores = positive_pixels @ positive - 0.65 * (positive_pixels @ negative)
            negative_scores = negative_pixels @ positive - 0.65 * (negative_pixels @ negative)
            threshold = experiment.tune_threshold(positive_scores, negative_scores)
            getter = lambda embedding, p=positive, n=negative, t=threshold: experiment.contrast_score(
                embedding, p, n, mean, std, True, t
            )
            metrics = experiment.evaluate(getter, threshold, task, [target])
            image = save_comparison_panel(
                output, task, spec["name"], target, getter, threshold, selected
            )
            runs.append({"label_count": count, "image": image, **metrics})
            if count == 5:
                five_label_model = (getter, threshold, selected)

        getter, threshold, selected = five_label_model
        previous_examples = []
        for previous_pid in PREVIOUS_FIXED_EXAMPLES[task]:
            if previous_pid not in patches:
                continue
            metrics = experiment.evaluate(getter, threshold, task, [previous_pid])
            image = save_comparison_panel(
                output,
                task,
                spec["name"],
                previous_pid,
                getter,
                threshold,
                selected,
            )
            previous_examples.append(
                {"patch_id": previous_pid, "image": image, **metrics}
            )
        results[task] = {
            "name": spec["name"],
            "target_patch": target,
            "support_patch": polygons[0][0],
            "runs": runs,
            "previous_examples": previous_examples,
        }

    (output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(output, results)
    print(output)


if __name__ == "__main__":
    main()
