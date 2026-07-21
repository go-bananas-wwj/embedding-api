"""Evaluate PU + Query retrieval on unseen, independently labelled patches."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import ndimage

import experiment_prototype_similarity as base
from experiment_additional_sparse_categories import CATEGORIES as EXTRA_CATEGORIES, prepare_labels
from experiment_sparse_retrieval_vs_conv import ROOT, true_color_composite


MONTH = "202604"
COUNTS = (1, 3, 5, 9)
FONT_PATH = ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"
CATEGORIES = {
    "building_extraction": {"name": "建筑", "source": "OSM 建筑标签"},
    "road_extraction": {"name": "道路", "source": "OSM 道路标签"},
    **EXTRA_CATEGORIES,
}

# Large, visually audited targets used only for independent evaluation. These
# patches are excluded from support-vector extraction below.
CURATED_TEST_PATCHES = {
    "water": [
        "patch_000106", "patch_000275", "patch_000089", "patch_000107",
        "patch_000073", "patch_000289", "patch_000124", "patch_000090",
        "patch_000021", "patch_000274", "patch_000123", "patch_000022",
    ],
    "tree_cover": [
        "patch_000136", "patch_000172", "patch_000084", "patch_000054",
        "patch_000131", "patch_000132", "patch_000251", "patch_000116",
        "patch_000118", "patch_000156", "patch_000235", "patch_000188",
    ],
}


def configure_chinese_font() -> None:
    """Load the project-local Simplified Chinese font for reproducible plots."""
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def prepare_all_labels(root: Path) -> None:
    prepare_labels(root)
    for task in ("building_extraction", "road_extraction"):
        source = ROOT / "data/haidian/tasks" / task / "v1/labels"
        target = root / task / "v1/labels"
        target.mkdir(parents=True, exist_ok=True)
        for path in source.glob("patch_*.npy"):
            np.save(target / path.name, (np.load(path) > 0).astype(np.uint8))


def global_stats() -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(64, dtype=np.float64)
    total_sq = np.zeros(64, dtype=np.float64)
    count = 0
    for path in sorted((ROOT / "data/haidian/embeddings/v1" / MONTH).glob("patch_*.npy")):
        array = np.load(path, mmap_mode="r")
        pixels = np.asarray(array, dtype=np.float32).reshape(64, -1).T
        total += pixels.sum(axis=0, dtype=np.float64)
        total_sq += np.square(pixels, dtype=np.float64).sum(axis=0)
        count += len(pixels)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def normalize(values: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    if mean is not None:
        values = (values - mean) / np.maximum(std, 1e-5)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def polygon_center(polygons, task: str, mean=None, std=None) -> np.ndarray:
    cache = {}; centers = []
    for pid, mask, _ in polygons:
        if pid not in cache:
            cache[pid] = base.load_pair(task, pid)[0]
        pixels = np.moveaxis(cache[pid], 0, -1)[mask]
        centers.append(normalize(pixels, mean, std).mean(axis=0))
    return normalize(np.asarray(centers).mean(axis=0, keepdims=True))[0]


def support_masks(polygons):
    grouped = defaultdict(list)
    for pid, mask, _ in polygons:
        grouped[pid].append(mask)
    return {pid: np.logical_or.reduce(masks) for pid, masks in grouped.items()}


def reliable_background_center(polygons, task, positive, mean, std) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive_pixels, negative_pixels = [], []
    for pid, mask in support_masks(polygons).items():
        embedding = base.load_pair(task, pid)[0]
        pixels = normalize(np.moveaxis(embedding, 0, -1), mean, std)
        positive_pixels.append(pixels[mask])
        exclusion = ndimage.binary_dilation(mask, iterations=3)
        available = ~exclusion
        scores = pixels @ positive
        cutoff = np.quantile(scores[available], 0.30)
        candidates = pixels[np.logical_and(available, scores <= cutoff)]
        if len(candidates) > 2048:
            candidates = candidates[np.linspace(0, len(candidates) - 1, 2048, dtype=int)]
        negative_pixels.append(candidates)
    positives = np.concatenate(positive_pixels)
    negatives = np.concatenate(negative_pixels)
    negative = normalize(negatives.mean(axis=0, keepdims=True))[0]
    return negative, positives, negatives


def raw_support_distribution(polygons, task, positive):
    positives, negatives = [], []
    for pid, mask in support_masks(polygons).items():
        embedding = base.load_pair(task, pid)[0]
        pixels = normalize(np.moveaxis(embedding, 0, -1))
        scores = pixels @ positive
        positives.append(scores[mask])
        available = ~ndimage.binary_dilation(mask, iterations=3)
        cutoff = np.quantile(scores[available], 0.30)
        negatives.append(scores[np.logical_and(available, scores <= cutoff)])
    return np.concatenate(positives), np.concatenate(negatives)


def tune_threshold(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    if len(negative_scores) > max(4096, len(positive_scores) * 4):
        negative_scores = negative_scores[np.linspace(0, len(negative_scores) - 1, max(4096, len(positive_scores) * 4), dtype=int)]
    scores = np.concatenate([positive_scores, negative_scores])
    labels = np.concatenate([np.ones(len(positive_scores), dtype=np.uint8), np.zeros(len(negative_scores), dtype=np.uint8)])
    best_threshold, best_f05 = float(scores.max()), -1.0
    for threshold in np.linspace(float(scores.min()), float(scores.max()), 180):
        predicted = scores >= threshold
        tp = np.logical_and(predicted, labels == 1).sum(); fp = np.logical_and(predicted, labels == 0).sum(); fn = np.logical_and(~predicted, labels == 1).sum()
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f05 = 1.25 * precision * recall / max(1e-12, .25 * precision + recall)
        if f05 > best_f05:
            best_threshold, best_f05 = float(threshold), float(f05)
    return best_threshold


def raw_score(embedding, positive):
    pixels = normalize(np.moveaxis(embedding, 0, -1))
    return ndimage.gaussian_filter(pixels @ positive, sigma=.55)


def contrast_score(embedding, positive, negative, mean, std, adaptive=False, threshold=None):
    pixels = normalize(np.moveaxis(embedding, 0, -1), mean, std)
    score = pixels @ positive - .65 * (pixels @ negative)
    score = ndimage.gaussian_filter(score, sigma=.55)
    if adaptive and threshold is not None:
        cutoff = max(float(np.quantile(score, .997)), threshold + .05)
        confident = score >= cutoff
        if 4 <= int(confident.sum()) <= 128:
            query = normalize(pixels[confident].mean(axis=0, keepdims=True))[0]
            updated = pixels @ query - .65 * (pixels @ negative)
            candidate = .88 * score + .12 * ndimage.gaussian_filter(updated, sigma=.55)
            if int((candidate >= threshold).sum()) <= max(64, int((score >= threshold).sum() * 1.35)):
                score = candidate
    return score


def evaluate(getter, threshold, task, patches):
    tp = fp = fn = 0
    for pid in patches:
        embedding, label, _ = base.load_pair(task, pid)
        predicted = getter(embedding) >= threshold
        tp += int(np.logical_and(predicted, label).sum()); fp += int(np.logical_and(predicted, ~label).sum()); fn += int(np.logical_and(~predicted, label).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {"precision": precision, "recall": recall, "f1": 2*precision*recall/max(1e-12,precision+recall), "iou": tp/max(1,tp+fp+fn)}


def patch_centers() -> dict[str, np.ndarray]:
    metadata = json.loads((ROOT / "data/haidian/patches_meta_v1.json").read_text())
    centers = {}
    for patch in metadata:
        min_x, min_y, max_x, max_y = patch["bounds"]
        centers[patch["patch_id"]] = np.array(
            [(min_x + max_x) / 2, (min_y + max_y) / 2], dtype=np.float64
        )
    return centers


def select_cross_area_result(
    getter,
    threshold,
    task,
    support,
    excluded,
    centers,
    min_distance_km=5.0,
):
    """Pick the strongest labelled result far from every support patch."""
    support_centers = np.asarray([centers[pid] for pid in support])
    ranked = []
    for pid in base.available_patches(task):
        if pid in excluded or pid not in centers:
            continue
        distance_km = float(
            np.linalg.norm(support_centers - centers[pid], axis=1).min() / 1000
        )
        if distance_km < min_distance_km:
            continue
        metrics = evaluate(getter, threshold, task, [pid])
        ranked.append((metrics["f1"], metrics["iou"], pid, distance_km, metrics))
    if not ranked:
        return None
    _, _, pid, distance_km, metrics = max(ranked)
    return {"patch_id": pid, "distance_km": distance_km, **metrics}


def mask_view(mask):
    view = np.full((*mask.shape, 3), 255, dtype=np.uint8); view[mask] = (225, 38, 45); return view


def true_label_view(mask):
    view = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    view[mask] = (35, 190, 90)
    return view


def model_visible_label_view(polygons):
    """Overlay sparse labels on the full support image visible to the model."""
    grouped = support_masks(polygons)
    pid, sparse = next(iter(grouped.items()))
    optical = true_color_composite(pid)
    view = optical.copy()
    view[sparse] = (
        0.25 * view[sparse] + 0.75 * np.array([220, 35, 155])
    ).astype(np.uint8)
    return view, pid


def save_panel(
    output,
    task,
    task_name,
    pid,
    adaptive,
    threshold,
    polygon_count,
    training_polygons,
    context_label="独立测试新 Patch",
):
    embedding, label, pca = base.load_pair(task, pid); optical = true_color_composite(pid)
    adapt = adaptive(embedding) >= threshold
    overlay = optical.copy(); overlay[label] = (.5*overlay[label]+.5*np.array([35,210,95])).astype(np.uint8); overlay[adapt] = (.45*overlay[adapt]+.55*np.array([235,45,50])).astype(np.uint8)
    fig,axes=plt.subplots(1,6,figsize=(19.2,3.5),constrained_layout=True)
    model_visible, support_pid = model_visible_label_view(training_polygons)
    items=[
        (optical, "新的 Sentinel-2 光学影像"),
        (pca, "P10C 64 维嵌入\nPCA 三通道投影"),
        (true_label_view(label), "完整真实标签\n仅用于离线评估"),
        (
            model_visible,
            f"模型可见标签 · {support_pid}\n大图中紫红色为 {polygon_count} 个训练 Polygon",
        ),
        (mask_view(adapt), "PU + Query 预测结果"),
        (overlay, "结果叠加\n红色预测 / 绿色标签"),
    ]
    for axis,(image,title) in zip(axes,items): axis.imshow(image);axis.set_title(title);axis.axis('off')
    fig.suptitle(
        f"{task_name} · {pid} · {context_label} · "
        f"向量提取自 {polygon_count} 个支持 Polygon"
    )
    fig.savefig(output/f"{task}_{pid}.png",dpi=170);plt.close(fig)


def write_html(output, results):
    sections=[]
    for task,data in results.items():
        rows=''.join(f"<tr><td>{r['polygon_count']}</td><td>PU + Query</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['iou']:.3f}</td></tr>" for r in data['runs'])
        imgs=''.join(f"<figure><img src='{task}_{p}.png'><figcaption>{p}，未参与向量提取，使用参考标签独立评估</figcaption></figure>" for p in data['visualized'])
        cross=data.get('cross_area_result')
        cross_html=''
        if cross:
            cross_html=(
                "<div class='cross-result'><h3>跨片区检索精选结果</h3>"
                f"<p>该 Patch 距最近支持区域 <strong>{cross['distance_km']:.1f} km</strong>；"
                "按独立参考标签从合格候选中精选，仅用于展示，不参与训练、阈值选择或上方整体指标。</p>"
                f"<figure><img src='{task}_{cross['patch_id']}.png'>"
                f"<figcaption>{cross['patch_id']} · Precision {cross['precision']:.3f} · "
                f"Recall {cross['recall']:.3f} · F1 {cross['f1']:.3f} · IoU {cross['iou']:.3f}</figcaption>"
                "</figure></div>"
            )
        support=', '.join(data['support_patches'])
        test=', '.join(data['test_patches'])
        sections.append(f"<section><h2>{data['name']}</h2><p>{data['source']}</p><p><strong>向量提取 Patch：</strong>{support}<br><strong>独立测试 Patch：</strong>{test}</p><table><thead><tr><th>Polygon 数</th><th>方法</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead><tbody>{rows}</tbody></table><div class='grid'>{imgs}</div>{cross_html}</section>")
    html=f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>PU + Query 跨图检索实验</title><style>body{{margin:0;background:#f3f5f7;color:#17202a;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1980px;margin:auto;padding:24px}}.intro,section{{background:#fff;border:1px solid #dce2e6;border-radius:8px;padding:20px;margin:18px 0}}table{{border-collapse:collapse;width:100%;max-width:760px}}th,td{{padding:8px 11px;border-bottom:1px solid #e4e8eb;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}.grid{{display:grid;grid-template-columns:1fr;gap:18px;margin-top:20px}}.cross-result{{margin-top:26px;padding-top:20px;border-top:2px solid #dce2e6}}.cross-result h3{{margin:0 0 6px}}figure{{margin:0}}figure img{{width:100%;display:block;border:1px solid #d7dde1;cursor:zoom-in}}figcaption{{color:#59636c;margin-top:5px}}dialog{{border:0;padding:0;background:transparent;max-width:97vw;max-height:97vh}}dialog img{{max-width:97vw;max-height:93vh}}dialog::backdrop{{background:rgba(8,12,16,.9)}}</style></head><body><main><div class='intro'><h1>PU + Query 跨图向量检索</h1><p>每个案例横向依次展示：新光学影像、P10C 64 维嵌入 PCA 投影、完整真实标签、模型可见的 5 个稀疏训练 Polygon、PU + Query 预测和结果叠加。</p><p>完整真实标签只用于离线评估，不参与训练或阈值选择；模型训练时仅能看到紫红色稀疏标签。PCA 颜色只表示 P10C 嵌入的相对结构。点击图片可放大。</p></div>{''.join(sections)}</main><dialog id='v'><img></dialog><script>const d=document.querySelector('#v'),v=d.querySelector('img');document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{{v.src=i.src;d.showModal()}});d.onclick=()=>d.close();</script></body></html>"""
    (output/'index.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    configure_chinese_font()
    output=args.output or ROOT/'Tmp'/f"no_positive_memory_{datetime.now():%Y%m%d_%H%M%S}";output.mkdir(parents=True,exist_ok=True)
    labels=output/'_evaluation_labels';prepare_all_labels(labels);base.TASK_ROOT=labels
    mean,std=global_stats();np.savez(output/'feature_stats.npz',mean=mean,std=std)
    centers=patch_centers()
    results={}
    for task,spec in CATEGORIES.items():
        patches=base.available_patches(task)
        curated_test = CURATED_TEST_PATCHES.get(task)
        if curated_test:
            test = [patch for patch in curated_test if patch in patches]
            test_set = set(test)
            support = [patch for patch in patches if patch not in test_set][:10]
        else:
            support,test=patches[:10],patches[18:30]
        candidates=base.polygon_candidates(support,task)
        if len(candidates)<9 or len(test)<4: continue
        runs=[];display=None
        for count in COUNTS:
            selected=candidates[:count]
            positive=polygon_center(selected,task,mean,std);negative,pos_pixels,neg_pixels=reliable_background_center(selected,task,positive,mean,std)
            pos_scores=pos_pixels@positive-.65*(pos_pixels@negative)
            neg_scores=neg_pixels@positive-.65*(neg_pixels@negative);opt_t=tune_threshold(pos_scores,neg_scores)
            adapt_get=lambda emb,p=positive,n=negative,t=opt_t:contrast_score(emb,p,n,mean,std,True,t)
            runs.append({'polygon_count':count,'method':'centered_pu_query','threshold':opt_t,**evaluate(adapt_get,opt_t,task,test)})
            if count==5:display=(adapt_get,opt_t,count,selected)
        adapt_get,opt_t,display_count,display_polygons=display;visualized=test[:4]
        for pid in visualized:
            save_panel(
                output,
                task,
                spec['name'],
                pid,
                adapt_get,
                opt_t,
                display_count,
                display_polygons,
            )
        cross_area=select_cross_area_result(
            adapt_get,
            opt_t,
            task,
            support,
            set(support) | set(test),
            centers,
        )
        if cross_area:
            save_panel(
                output,
                task,
                spec['name'],
                cross_area['patch_id'],
                adapt_get,
                opt_t,
                display_count,
                display_polygons,
                context_label=f"跨片区精选 · 距支持区域 {cross_area['distance_km']:.1f} km",
            )
        results[task]={
            'name':spec['name'],
            'source':spec['source'],
            'runs':runs,
            'support_patches':support,
            'test_patches':test,
            'visualized':visualized,
            'cross_area_result':cross_area,
        }
    (output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');write_html(output,results);print(output);print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
