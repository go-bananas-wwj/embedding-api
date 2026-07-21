"""Compare prototype retrieval and Binary Conv 3x3 with sparse user polygons."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.training_engine import _build_binary_target, _normalize_feature_map, _train_binary_conv_head
from experiment_local_prototype_memory import build_local_memory, local_score_map
from experiment_prototype_similarity import (
    MONTH, ROOT, TASKS, available_patches, confusion, load_pair, metrics, polygon_candidates
)

S2_ROOT = ROOT / "data/haidian/archive/processed_training_data/extracted/patches/s2"
COUNTS = (1, 3, 5, 9)


def true_color_composite(pid: str) -> np.ndarray:
    scenes = sorted(S2_ROOT.glob(f"s2_{MONTH}*_" + pid + ".tif"))
    rgb_stack, valid_stack = [], []
    for scene in scenes:
        mask_path = scene.with_name(scene.stem + "_mask.tif")
        with rasterio.open(scene) as ds:
            data = ds.read([3, 2, 1]).astype(np.float32)  # B04, B03, B02
        valid = np.isfinite(data).all(axis=0) & (data.max(axis=0) > 0)
        if mask_path.exists():
            with rasterio.open(mask_path) as ds:
                valid &= ds.read(1) > 0
        data[:, ~valid] = np.nan
        rgb_stack.append(data)
        valid_stack.append(valid)
    if not rgb_stack:
        return np.full((128, 128, 3), 128, dtype=np.uint8)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(np.stack(rgb_stack), axis=0)
    valid = np.isfinite(composite).all(axis=0)
    rgb = np.moveaxis(composite, 0, -1)
    output = np.full(rgb.shape, 150, dtype=np.float32)
    if valid.any():
        values = rgb[valid]
        low, high = np.nanpercentile(values, [2, 98], axis=0)
        stretched = np.clip((rgb - low) / np.maximum(high - low, 1), 0, 1)
        stretched = np.power(stretched, 0.85)
        output[valid] = stretched[valid] * 255
    return output.astype(np.uint8)


def grouped_sparse_samples(polygons, task):
    grouped = defaultdict(list)
    for pid, mask, area in polygons:
        grouped[pid].append(mask)
    samples = []
    for pid, masks in grouped.items():
        embedding, _, _ = load_pair(task, pid)
        sparse = np.logical_or.reduce(masks)
        samples.append((embedding.astype(np.float32, copy=False), _build_binary_target(embedding, sparse)))
    return samples


def conv_scores(model, embedding):
    with torch.no_grad():
        x = torch.from_numpy(_normalize_feature_map(embedding)).float().unsqueeze(0)
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def calibrate(score_getter, task, patch_ids):
    scores, labels = [], []
    for pid in patch_ids:
        embedding, label, _ = load_pair(task, pid)
        scores.append(score_getter(embedding).reshape(-1))
        labels.append(label.reshape(-1))
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    low, high = np.quantile(scores, [0.01, 0.995])
    best = (float(high), -1.0)
    for threshold in np.linspace(low, high, 240):
        tp, fp, fn = confusion(scores, labels, float(threshold))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f05 = 1.25 * precision * recall / max(1e-12, 0.25 * precision + recall)
        if f05 > best[1]:
            best = float(threshold), float(f05)
    return best


def calibrate_from_sparse_targets(score_getter, samples):
    """Tune only from user positives and automatically mined weak negatives."""
    scores, labels = [], []
    for embedding, target in samples:
        valid = target >= 0
        scores.append(score_getter(embedding)[valid])
        labels.append(target[valid].astype(np.uint8))
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    low, high = float(scores.min()), float(scores.max())
    best = (high, -1.0)
    for threshold in np.linspace(low, high, 160):
        tp, fp, fn = confusion(scores, labels, float(threshold))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f05 = 1.25 * precision * recall / max(1e-12, 0.25 * precision + recall)
        if f05 > best[1]:
            best = float(threshold), float(f05)
    return best


def evaluate(score_getter, threshold, task, patch_ids):
    tp = fp = fn = 0
    for pid in patch_ids:
        embedding, label, _ = load_pair(task, pid)
        current = confusion(score_getter(embedding), label, threshold)
        tp += current[0]; fp += current[1]; fn += current[2]
    return metrics(tp, fp, fn)


def support_montage(output, task, polygons):
    grouped = defaultdict(list)
    for pid, mask, _ in polygons:
        grouped[pid].append(mask)
    fig, axes = plt.subplots(1, len(grouped), figsize=(3.2 * len(grouped), 3.4), squeeze=False, constrained_layout=True)
    for axis, (pid, masks) in zip(axes[0], grouped.items()):
        image = true_color_composite(pid)
        sparse = np.logical_or.reduce(masks)
        overlay = image.copy()
        overlay[sparse] = (0.35 * overlay[sparse] + 0.65 * np.array([250, 40, 190])).astype(np.uint8)
        axis.imshow(overlay); axis.set_title(f"{pid}\n{len(masks)} user polygon(s)"); axis.axis("off")
    fig.suptitle(f"{task}: only these sparse magenta polygons are visible to both methods")
    fig.savefig(output / f"{task}_support_annotations.png", dpi=170)
    plt.close(fig)


def test_panel(output, task, pid, retrieval_getter, retrieval_threshold, conv_getter, conv_threshold):
    embedding, label, pca = load_pair(task, pid)
    optical = true_color_composite(pid)
    retrieval = retrieval_getter(embedding) >= retrieval_threshold
    conv = conv_getter(embedding) >= conv_threshold
    masks = []
    for prediction in (retrieval, conv):
        view = np.full((*prediction.shape, 3), 255, dtype=np.uint8)
        view[prediction] = (225, 38, 45)
        masks.append(view)
    overlays = []
    for prediction in (retrieval, conv):
        view = optical.copy(); view[label] = (0.5 * view[label] + 0.5 * np.array([35, 210, 95])).astype(np.uint8)
        view[prediction] = (0.45 * view[prediction] + 0.55 * np.array([235, 45, 50])).astype(np.uint8)
        overlays.append(view)
    fig, axes = plt.subplots(1, 8, figsize=(25, 3.5), constrained_layout=True)
    items = [(optical,"Sentinel-2 optical"),(pca,"P10C PCA"),(label,"Full reference\nevaluation only"),
             (masks[0],"Vector retrieval\nred prediction"),(masks[1],"Binary Conv 3x3\nred prediction"),
             (overlays[0],"Retrieval overlay"),(overlays[1],"Conv overlay")]
    for axis, (image,title) in zip(axes,items): axis.imshow(image); axis.set_title(title); axis.axis("off")
    axes[-1].axis("off"); axes[-1].text(.5,.6,"Green = reference\nRed = prediction\n\nNo test labels were\nused for training",ha="center",va="center",fontsize=12)
    fig.suptitle(f"{task} · {pid} · unseen patch · 5 sparse training polygons")
    fig.savefig(output / f"{task}_{pid}.png", dpi=170); plt.close(fig)


def write_html(output, results):
    names={"building_extraction":"建筑提取","road_extraction":"道路提取"}; sections=[]
    for task,data in results.items():
        rows="".join(f"<tr><td>{r['polygon_count']}</td><td>{r['method']}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['iou']:.3f}</td></tr>" for r in data['runs'])
        imgs=f"<figure><img src='{task}_support_annotations.png'><figcaption>真实输入：稀疏用户 Polygon，完整标签不会提供给模型</figcaption></figure>"+"".join(f"<figure><img src='{task}_{p}.png'><figcaption>{p}，独立测试 patch</figcaption></figure>" for p in data['visualized'])
        sections.append(f"<section><h2>{names[task]}</h2><table><thead><tr><th>Polygon 数</th><th>方法</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead><tbody>{rows}</tbody></table><div class='grid'>{imgs}</div></section>")
    html=f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>稀疏标注：检索与训练对比</title><style>body{{margin:0;background:#f3f5f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1800px;margin:auto;padding:24px}}section,.intro{{background:white;border:1px solid #dce2e6;border-radius:8px;padding:20px;margin:18px 0}}table{{border-collapse:collapse;width:100%;max-width:900px}}th,td{{padding:8px 11px;border-bottom:1px solid #e4e8eb;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}.grid{{display:grid;grid-template-columns:1fr;gap:18px;margin-top:20px}}figure{{margin:0}}figure img{{display:block;width:100%;border:1px solid #d7dde1;cursor:zoom-in}}figcaption{{color:#59636c;margin-top:5px}}dialog{{border:0;padding:0;background:transparent;max-width:97vw;max-height:97vh}}dialog img{{max-width:97vw;max-height:93vh;display:block}}dialog::backdrop{{background:rgba(8,12,16,.9)}}</style></head><body><main><div class='intro'><h1>稀疏用户标注下的向量检索与模型训练对比</h1><p>两种方法接收完全相同的 1、3、5、9 个 Polygon，阈值也只根据这些稀疏正样本和自动挖掘的弱负样本确定。完整标签仅用于独立测试评分。光学图由 202604 原始 Sentinel-2 有效观测进行掩膜中位数合成，不使用存在黑块的旧 PNG。</p><p>点击图片放大，按 Esc 或点击遮罩关闭。</p></div>{''.join(sections)}</main><dialog id='v'><img></dialog><script>const d=document.querySelector('#v'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script></body></html>"""
    (output/'index.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path); args=parser.parse_args()
    output=args.output or ROOT/'Tmp'/f"sparse_retrieval_vs_conv_{datetime.now():%Y%m%d_%H%M%S}"; output.mkdir(parents=True,exist_ok=True)
    results={}
    for task in TASKS:
        patches=available_patches(task); support,test=patches[:8],patches[16:28]
        candidates=polygon_candidates(support,task); runs=[]; display=None
        for count in COUNTS:
            selected=candidates[:count]; memory=build_local_memory(selected,task); sparse_samples=grouped_sparse_samples(selected,task)
            retrieval_getter=lambda e,m=memory: local_score_map(e,m,adaptive=False)
            rt,_=calibrate_from_sparse_targets(retrieval_getter,sparse_samples); rm=evaluate(retrieval_getter,rt,task,test)
            runs.append({'polygon_count':count,'method':'局部向量检索','threshold':rt,**rm})
            model,ct,_,_,epochs=_train_binary_conv_head(sparse_samples,40)
            conv_getter=lambda e,m=model: conv_scores(m,e)
            cm=evaluate(conv_getter,ct,task,test)
            runs.append({'polygon_count':count,'method':'Binary Conv 3x3','threshold':ct,'epochs':epochs,**cm})
            if count==5: display=(selected,memory,rt,model,ct)
        selected,memory,rt,model,ct=display; support_montage(output,task,selected)
        rg=lambda e,m=memory: local_score_map(e,m,adaptive=False); cg=lambda e,m=model: conv_scores(m,e)
        visualized=test[:4]
        for pid in visualized: test_panel(output,task,pid,rg,rt,cg,ct)
        results[task]={'runs':runs,'visualized':visualized}
    (output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8'); write_html(output,results)
    print(output); print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
