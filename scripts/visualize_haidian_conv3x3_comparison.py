#!/usr/bin/env python3
"""Build a visual MLP-versus-Conv3x3 acceptance gallery."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.fewshot_heads import BinaryConv3x3ProbeHead
from experiment_sparse_retrieval_vs_conv import true_color_composite
from train_haidian_conv3x3_system_heads import load_label, resize_label


MONTH = "202604"
TASK_NAMES = {
    "building_extraction": "建筑物提取",
    "road_extraction": "道路提取",
    "water_extraction": "水体提取",
}
REFERENCE_LABEL_NAMES = {
    "building_extraction": "OSM 已标注建筑\n非完整真值",
    "road_extraction": "OSM 已标注道路\n非完整真值",
    "water_extraction": "WorldCover 水体参考标签",
}
CURATED_TEST_PATCHES = {
    "water_extraction": [
        "patch_000073",
        "patch_000274",
        "patch_000165",
        "patch_000293",
    ],
}
COLORS = {
    "building_extraction": np.array([239, 68, 68]),
    "road_extraction": np.array([245, 158, 11]),
    "water_extraction": np.array([37, 99, 235]),
}
FONT = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"


def configure_font() -> None:
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def load_models(task: str, training_root: Path):
    checkpoint = torch.load(
        training_root / task / f"{task}_conv3x3_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    conv = BinaryConv3x3ProbeHead(
        checkpoint["embed_dim"], checkpoint["hidden_dim"], checkpoint["dropout"]
    )
    conv.load_state_dict(checkpoint["state_dict"])
    conv.eval()

    short = task.removesuffix("_extraction")
    state = torch.load(
        ROOT / "models/haidian/v1/task_heads" / f"{short}_mlp_fold0_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    mlp = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
    )
    mlp.load_state_dict({key.replace("net.", "", 1): value for key, value in state.items()})
    mlp.eval()
    return checkpoint, conv, mlp


def infer(embedding: np.ndarray, checkpoint, conv, mlp):
    feature = torch.from_numpy(embedding.astype(np.float32, copy=False))[None]
    with torch.no_grad():
        conv_prob = torch.sigmoid(conv(feature))[0, 0].numpy()
        d, h, w = embedding.shape
        mlp_prob = torch.sigmoid(
            mlp(feature[0].permute(1, 2, 0).reshape(-1, d))
        ).reshape(h, w).numpy()
    return mlp_prob >= 0.5, conv_prob >= float(checkpoint["threshold"]), conv_prob


def resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def mask_rgb(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    output = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    output[mask] = color
    return output


def save_panel(output: Path, task: str, patch_id: str, checkpoint, conv, mlp):
    embedding = np.load(ROOT / f"data/haidian/embeddings/v1/{MONTH}/{patch_id}.npy")
    target = resize_label(load_label(task, patch_id), embedding.shape[-2:]) > 0
    mlp_pred, conv_pred, conv_prob = infer(embedding, checkpoint, conv, mlp)
    optical = true_color_composite(patch_id)
    h, w = optical.shape[:2]
    target_large = np.asarray(Image.fromarray(target).resize((w, h), Image.Resampling.NEAREST))
    conv_large = np.asarray(Image.fromarray(conv_pred).resize((w, h), Image.Resampling.NEAREST))
    overlay = optical.copy()
    overlay[target_large] = (0.55 * overlay[target_large] + 0.45 * np.array([35, 200, 90])).astype(np.uint8)
    overlay[conv_large] = (0.45 * overlay[conv_large] + 0.55 * COLORS[task]).astype(np.uint8)

    pca = np.asarray(Image.open(ROOT / f"data/haidian/embeddings/v1/{MONTH}/{patch_id}.png").convert("RGB"))
    items = [
        (optical, "Sentinel-2 光学影像"),
        (pca, "P10C 64维嵌入 PCA"),
        (mask_rgb(target, np.array([35, 190, 90])), REFERENCE_LABEL_NAMES[task]),
        (mask_rgb(mlp_pred, np.array([145, 145, 145])), "旧 MLP 预测"),
        (mask_rgb(conv_pred, COLORS[task]), "新 Conv 3×3 预测"),
        (overlay, "结果叠加\n彩色预测 / 绿色标签"),
    ]
    figure, axes = plt.subplots(1, 6, figsize=(19.2, 3.55), constrained_layout=True)
    for axis, (image, title) in zip(axes, items):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{TASK_NAMES[task]} · {patch_id} · {MONTH} · Conv阈值 {checkpoint['threshold']:.2f}"
    )
    filename = f"{task}_{patch_id}.png"
    figure.savefig(output / filename, dpi=170)
    plt.close(figure)
    return filename, float(conv_prob.mean())


def main() -> None:
    configure_font()
    training_root = ROOT / "Tmp/haidian_conv3x3_training_20260722"
    output = ROOT / "Tmp" / f"haidian_conv3x3_comparison_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    sections = []
    report = {}
    for task, name in TASK_NAMES.items():
        metrics = json.loads((training_root / task / "metrics.json").read_text())
        split = json.loads((training_root / task / "split.json").read_text())
        ranked = sorted(
            split["test"], key=lambda pid: float(load_label(task, pid).mean()), reverse=True
        )
        # Spread the examples across the ranked positives rather than showing only extremes.
        picks = CURATED_TEST_PATCHES.get(
            task,
            [
                ranked[index]
                for index in (
                    0,
                    len(ranked) // 5,
                    len(ranked) // 2,
                    3 * len(ranked) // 4,
                )
            ],
        )
        checkpoint, conv, mlp = load_models(task, training_root)
        figures = []
        for patch_id in picks:
            filename, mean_probability = save_panel(
                output, task, patch_id, checkpoint, conv, mlp
            )
            figures.append(
                f"<figure><img src='{filename}'><figcaption>{patch_id} · Conv平均概率 {mean_probability:.3f}</figcaption></figure>"
            )
        conv_metrics = metrics["conv3x3_test"]
        mlp_metrics = metrics["mlp_test"]
        sections.append(
            f"<section><h2>{name}</h2><p>相对现有参考标签的一致性 F1：MLP <strong>{mlp_metrics['f1']:.3f}</strong> → "
            f"Conv 3×3 <strong>{conv_metrics['f1']:.3f}</strong>；IoU：{mlp_metrics['iou']:.3f} → {conv_metrics['iou']:.3f}。</p>"
            f"<div class='grid'>{''.join(figures)}</div></section>"
        )
        report[task] = {"patches": picks, "metrics": metrics}

    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>海淀 Conv 3×3 下游头诊断</title><style>body{{margin:0;background:#f3f5f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}main{{max-width:2000px;margin:auto;padding:24px}}.intro,section{{background:#fff;border:1px solid #dce2e6;border-radius:8px;padding:20px;margin:18px 0}}.warning{{border-left:4px solid #d97706;padding:10px 14px;background:#fff7ed}}.grid{{display:grid;gap:20px}}figure{{margin:0}}figure img{{width:100%;display:block;border:1px solid #d7dde1;cursor:zoom-in}}figcaption{{color:#59636c;margin-top:5px}}dialog{{border:0;padding:0;background:transparent}}dialog img{{max-width:97vw;max-height:94vh}}dialog::backdrop{{background:rgba(8,12,16,.9)}}</style></head><body><main><div class='intro'><h1>海淀 P10C Binary Conv 3×3 下游头诊断</h1><p class='warning'><strong>标签质量提示：</strong>建筑和道路参考标签来自 OSM，只代表 OSM 已收录要素，不是完整真值。未被绿色覆盖的真实目标不能直接视为模型误检，页面中的 F1/IoU 也只表示与现有稀疏标签的一致程度。</p><p>每行依次展示光学影像、P10C PCA、来源明确的参考标签、旧 MLP、新 Conv 3×3 和结果叠加；点击图片可放大。</p><p>绿色是现有参考标签，彩色区域是 Conv 预测。阈值只在验证集选择，以下 Patch 均来自独立测试集。</p></div>{''.join(sections)}</main><dialog id='v'><img></dialog><script>const d=document.querySelector('#v'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script></body></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
