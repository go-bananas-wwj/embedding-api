"""Build machine-readable and human-readable downstream task summaries."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image

from app.config import get_config
from app.services.data_service import DataService


_BINARY_TASKS = {
    "building_extraction",
    "road_extraction",
    "water_extraction",
    "construction",
}
_SUMMARY_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _files(path: Optional[str], suffix: str) -> list[Path]:
    if not path:
        return []
    root = Path(path)
    return sorted(root.rglob(f"*{suffix}")) if root.exists() else []


def _patch_files(
    path: Optional[str], suffix: str, allowed_patch_ids: Optional[set[str]] = None
) -> list[Path]:
    files = [p for p in _files(path, suffix) if re.match(r"^patch_\d{6}", p.stem)]
    if allowed_patch_ids is None:
        return files
    return [p for p in files if p.stem[:12] in allowed_patch_ids]


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def _patch_ids_from_files(files: Sequence[Path]) -> set[str]:
    patch_ids = set()
    for path in files:
        match = re.search(r"patch_\d{6}", path.stem)
        if match:
            patch_ids.add(match.group(0))
    return patch_ids


def _model_metadata(region_id: str, task_type: str, version: str) -> Dict[str, Any]:
    if region_id == "haidian":
        head = "binary_conv3x3" if task_type in _BINARY_TASKS else "precomputed_classifier"
        return {
            "foundation_model": "P10C",
            "api_version": version,
            "feature_source": "P10C 64D embedding",
            "feature_dimension": 64,
            "head_type": head,
        }
    return {
        "foundation_model": "V5" if version == "v2" else "V4",
        "api_version": version,
        "feature_source": f"Xuannv {version} embedding",
        "feature_dimension": 64,
        "head_type": "configured_system_head",
    }


def _prediction_statistics(
    prediction_files: list[Path],
    label_files: list[Path],
    threshold: float,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    positive_ratios = []
    tp = fp = fn = tn = evaluated = 0
    labels_by_name = {path.name: path for path in label_files}

    for prediction_path in prediction_files:
        try:
            prediction = np.load(prediction_path, allow_pickle=False)
        except (OSError, ValueError):
            continue
        prediction = np.asarray(prediction).squeeze()
        if prediction.ndim != 2:
            continue
        prediction_mask = np.nan_to_num(prediction, nan=0.0) >= threshold
        positive_ratios.append(float(prediction_mask.mean()))

        label_path = labels_by_name.get(prediction_path.name)
        if not label_path:
            continue
        try:
            label = np.load(label_path, allow_pickle=False)
        except (OSError, ValueError):
            continue
        label_mask = np.asarray(label).squeeze() > 0
        if label_mask.shape != prediction_mask.shape:
            continue
        tp += int(np.logical_and(prediction_mask, label_mask).sum())
        fp += int(np.logical_and(prediction_mask, ~label_mask).sum())
        fn += int(np.logical_and(~prediction_mask, label_mask).sum())
        tn += int(np.logical_and(~prediction_mask, ~label_mask).sum())
        evaluated += 1

    stats: Dict[str, Any] = {"threshold": round(float(threshold), 6)}
    if positive_ratios:
        values = np.asarray(positive_ratios)
        stats.update(
            {
                "analyzed_patches": len(positive_ratios),
                "positive_patches": int((values > 0).sum()),
                "negative_patches": int((values == 0).sum()),
                "mean_positive_pixel_ratio": round(float(values.mean()), 6),
                "median_positive_pixel_ratio": round(float(np.median(values)), 6),
                "p95_positive_pixel_ratio": round(float(np.percentile(values, 95)), 6),
                "min_positive_pixel_ratio": round(float(values.min()), 6),
                "max_positive_pixel_ratio": round(float(values.max()), 6),
            }
        )

    quality: Dict[str, Any] = {
        "reference_available": evaluated > 0,
        "evaluated_patches": evaluated,
        "reference_type": "project labels" if evaluated else None,
    }
    if evaluated:
        quality.update(
            {
                "iou": _safe_ratio(tp, tp + fp + fn),
                "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
                "precision": _safe_ratio(tp, tp + fp),
                "recall": _safe_ratio(tp, tp + fn),
                "pixel_accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn),
                "confusion_pixels": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            }
        )
    else:
        quality["unavailable_reason"] = "没有可与预测逐 Patch 对齐的参考标签。"
    return stats, quality


def _png_distribution(tile_files: list[Path]) -> Dict[str, Any]:
    colors: Counter = Counter()
    analyzed = 0
    for path in tile_files:
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, ValueError):
            continue
        analyzed += 1
        values, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
        for value, count in zip(values, counts):
            colors[tuple(int(v) for v in value)] += int(count)
    total = sum(colors.values())
    distribution = [
        {
            "color": "#%02X%02X%02X" % color,
            "pixels": count,
            "ratio": round(count / total, 6),
        }
        for color, count in colors.most_common(12)
    ]
    return {"analyzed_patches": analyzed, "color_distribution": distribution}


def _generated_binary_statistics(
    task_type: str, tile_files: Sequence[Path], label_files: list[Path]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate generated binary PNG outputs against aligned project labels."""
    labels_by_patch = {path.stem[:12]: path for path in label_files}
    ratios = []
    tp = fp = fn = tn = evaluated = 0
    for tile_path in tile_files:
        match = re.search(r"patch_\d{6}", tile_path.stem)
        if not match:
            continue
        try:
            with Image.open(tile_path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, ValueError):
            continue
        target_colors = {
            "building_extraction": (239, 68, 68),
            "road_extraction": (245, 158, 11),
            "water_extraction": (37, 99, 235),
        }
        target_color = target_colors.get(task_type)
        white_background = np.all(rgb == 255, axis=-1)
        black_background = np.all(rgb == 0, axis=-1)
        if white_background.any():
            prediction_mask = ~white_background
        elif black_background.any():
            prediction_mask = ~black_background
        elif target_color is not None:
            prediction_mask = np.all(rgb == target_color, axis=-1)
        else:
            continue
        ratios.append(float(prediction_mask.mean()))
        label_path = labels_by_patch.get(match.group(0))
        if not label_path:
            continue
        try:
            label_mask = np.asarray(np.load(label_path, allow_pickle=False)).squeeze() > 0
        except (OSError, ValueError):
            continue
        if label_mask.shape != prediction_mask.shape:
            continue
        tp += int(np.logical_and(prediction_mask, label_mask).sum())
        fp += int(np.logical_and(prediction_mask, ~label_mask).sum())
        fn += int(np.logical_and(~prediction_mask, label_mask).sum())
        tn += int(np.logical_and(~prediction_mask, ~label_mask).sum())
        evaluated += 1

    stats = _png_distribution(list(tile_files))
    stats["threshold"] = 0.5
    if ratios:
        values = np.asarray(ratios)
        stats.update({
            "positive_patches": int((values > 0).sum()),
            "negative_patches": int((values == 0).sum()),
            "mean_positive_pixel_ratio": round(float(values.mean()), 6),
            "median_positive_pixel_ratio": round(float(np.median(values)), 6),
            "p95_positive_pixel_ratio": round(float(np.percentile(values, 95)), 6),
            "min_positive_pixel_ratio": round(float(values.min()), 6),
            "max_positive_pixel_ratio": round(float(values.max()), 6),
        })
    quality: Dict[str, Any] = {
        "reference_available": evaluated > 0,
        "evaluated_patches": evaluated,
        "reference_type": "project labels" if evaluated else None,
    }
    if evaluated:
        quality.update({
            "iou": _safe_ratio(tp, tp + fp + fn),
            "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
            "precision": _safe_ratio(tp, tp + fp),
            "recall": _safe_ratio(tp, tp + fn),
            "pixel_accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn),
            "confusion_pixels": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        })
    else:
        quality["unavailable_reason"] = "生成结果与参考标签无法按 Patch 和尺寸对齐。"
    return stats, quality


def _narrative(
    name: str,
    region_id: str,
    model: Dict[str, Any],
    temporal: Dict[str, Any],
    coverage: Dict[str, Any],
    prediction: Dict[str, Any],
    color_legend: list[Dict[str, Any]],
    status: str,
) -> str:
    region_name = "海淀区" if region_id == "haidian" else "哈尔滨新区"
    months = temporal.get("available_months") or []
    status_label = {"ready": "结果齐全", "partial": "数据不完整", "unavailable": "暂无结果"}.get(status, status)
    scope_month = temporal.get("requested_month")
    time_text = f"月份为 {scope_month}" if scope_month else (
        f"可用月份为 {months[0]} 至 {months[-1]}" if months else "未指定月份"
    )
    text = (
        f"{region_name}{name}分析：{time_text}，共分析 {coverage['configured_patches']} 个 Patch，"
        f"其中 {coverage['available_result_patches']} 个已有结果，覆盖率 {coverage['coverage_rate']:.1%}，"
        f"当前状态为“{status_label}”。"
    )
    if prediction.get("mean_positive_pixel_ratio") is not None:
        text += (
            f"目标区域平均占图像 {prediction['mean_positive_pixel_ratio']:.2%}。"
        )
    if color_legend:
        color_text = "、".join(
            f"{item['color']} 表示{item['name']}（{item['ratio']:.2%}）"
            for item in color_legend[:8]
        )
        text += f"颜色说明：{color_text}。"
    return text


def _color_legend(
    region_id: str,
    task_type: str,
    task_name: str,
    version: str,
    prediction: Dict[str, Any],
) -> list[Dict[str, Any]]:
    distribution = prediction.get("color_distribution") or []
    if not distribution:
        return []
    class_names: Dict[str, str] = {}
    if task_type not in _BINARY_TASKS:
        try:
            from app.services.system_model_service import get_system_model_classes

            classes = get_system_model_classes(region_id, task_type, version)
            class_names = {str(item["color"]).upper(): str(item["name"]) for item in classes}
        except (FileNotFoundError, ValueError, KeyError):
            class_names = {}
    chinese_names = {
        "Water": "水体", "Crops": "耕地", "Built": "建成区", "Bare": "裸地",
        "Snow/Ice": "冰雪", "Tree": "树木", "Grassland": "草地",
        "Cropland": "耕地", "Built-up": "建成区", "Wetland": "湿地",
        "Building": "建筑物", "Non-building": "非建筑区域",
        "Non-water": "非水体区域",
    }
    legend = []
    for item in distribution:
        color = str(item["color"]).upper()
        if task_type in _BINARY_TASKS:
            name = "背景" if color in {"#FFFFFF", "#000000"} else task_name.replace("提取", "")
        else:
            name = class_names.get(color, "未命名类别")
            name = chinese_names.get(name, name)
        legend.append({
            "color": color,
            "name": name,
            "meaning": f"结果图中该颜色表示{name}。",
            "pixels": item.get("pixels"),
            "ratio": item.get("ratio", 0.0),
        })
    return legend


def _image_analysis(task_type: str, tile_files: Sequence[Path]) -> Dict[str, Any]:
    images = []
    total_pixels = target_pixels = 0
    for path in tile_files:
        match = re.search(r"patch_\d{6}", path.stem)
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, ValueError):
            continue
        height, width = rgb.shape[:2]
        pixels = int(width * height)
        item: Dict[str, Any] = {
            "patch_id": match.group(0) if match else None,
            "width": width,
            "height": height,
            "total_pixels": pixels,
        }
        total_pixels += pixels
        if task_type in _BINARY_TASKS:
            white = np.all(rgb == 255, axis=-1)
            black = np.all(rgb == 0, axis=-1)
            if white.any():
                mask = ~white
            elif black.any():
                mask = ~black
            else:
                mask = np.zeros((height, width), dtype=bool)
            count = int(mask.sum())
            item.update({"target_pixels": count, "target_ratio": round(count / pixels, 6)})
            target_pixels += count
        images.append(item)
    return {
        "image_count": len(images),
        "total_pixels": total_pixels,
        "target_pixels": target_pixels if task_type in _BINARY_TASKS else None,
        "target_ratio": round(target_pixels / total_pixels, 6)
        if task_type in _BINARY_TASKS and total_pixels
        else None,
        "images": images,
    }


def _asset_signature(*groups: list[Path]) -> tuple:
    signature = []
    for files in groups:
        latest = 0
        for path in files:
            try:
                latest = max(latest, path.stat().st_mtime_ns)
            except OSError:
                continue
        signature.extend((len(files), latest))
    return tuple(signature)


def build_task_summary(
    region_id: str,
    task_type: str,
    version: str,
    task_name: str,
    base_summary: Optional[Dict[str, Any]] = None,
    patch_ids: Optional[Sequence[str]] = None,
    month: Optional[str] = None,
    before_month: Optional[str] = None,
    after_month: Optional[str] = None,
    generated_tile_files: Optional[Sequence[Path]] = None,
    inference_errors: Optional[Sequence[Dict[str, str]]] = None,
    result_images: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute a rich summary from configured assets and local result files."""
    config = get_config()
    region = config.get_region(region_id) or {}
    task = (region.get("tasks") or {}).get(task_type) or {}
    version_config = (task.get("versions") or {}).get(version) or {}
    base_summary = base_summary or {}
    configured_patch_items = config.get_patches(region_id)
    configured_patch_ids = [item.get("patch_id") for item in configured_patch_items]
    selected_patch_ids = list(dict.fromkeys(patch_ids or configured_patch_ids))
    allowed_patch_ids = set(selected_patch_ids) if patch_ids is not None else None
    configured_patches = len(selected_patch_ids)

    prediction_files = _patch_files(version_config.get("predictions"), ".npy", allowed_patch_ids)
    label_files = _patch_files(version_config.get("labels"), ".npy", allowed_patch_ids)
    results_root = version_config.get("results")
    tile_files = _patch_files(
        str(Path(results_root) / "tiles") if results_root else None,
        ".png",
        allowed_patch_ids,
    )
    if not tile_files:
        tile_files = _patch_files(results_root, ".png", allowed_patch_ids)
    if generated_tile_files is not None:
        tile_files = list(dict.fromkeys(generated_tile_files))
        # Explicit month/Patch analysis must use the selected result images,
        # not timeless prediction arrays from another acquisition month.
        prediction_files = []
    if before_month and after_month:
        before = before_month.replace("-", "")
        after = after_month.replace("-", "")

        def matches_period(path: Path) -> bool:
            compact = str(path).replace("-", "")
            return f"{before}_vs_{after}" in compact

        prediction_files = [path for path in prediction_files if matches_period(path)]
        label_files = [path for path in label_files if matches_period(path)]
        tile_files = [path for path in tile_files if matches_period(path)]

    signature = _asset_signature(prediction_files, label_files, tile_files)
    scope = {
        "mode": "change_detection" if before_month and after_month else "single_time",
        "patch_ids": selected_patch_ids if patch_ids is not None else None,
        "patch_count": configured_patches,
        "month": month,
        "before_month": before_month,
        "after_month": after_month,
        "aggregation": "每个 Patch 独立推理后汇总统计",
        "generated_results": len(generated_tile_files or []),
    }
    cache_key = (
        region_id, task_type, version, task_name, tuple(selected_patch_ids),
        month, before_month, after_month, signature,
    )
    cached = _SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    prediction_patch_ids = _patch_ids_from_files(prediction_files)
    tile_patch_ids = _patch_ids_from_files(tile_files)
    label_patch_ids = _patch_ids_from_files(label_files)
    result_patch_ids = prediction_patch_ids | tile_patch_ids
    result_count = len(result_patch_ids)
    coverage_rate = _safe_ratio(result_count, configured_patches) or 0.0
    coverage = {
        "configured_patches": configured_patches,
        "prediction_patches": len(prediction_patch_ids),
        "result_tiles": len(tile_patch_ids),
        "label_patches": len(label_patch_ids),
        "available_result_patches": result_count,
        "missing_result_patches": max(0, configured_patches - result_count),
        "coverage_rate": coverage_rate,
    }
    status = "ready" if configured_patches and result_count >= configured_patches else (
        "partial" if result_count or label_files else "unavailable"
    )

    patch_id = selected_patch_ids[0] if selected_patch_ids else None
    months = DataService.get_available_months(region_id, patch_id) if patch_id else []
    temporal = {
        "available_months": months,
        "start_month": months[0] if months else None,
        "end_month": months[-1] if months else None,
        "month_count": len(months),
        "requested_month": month,
    }
    threshold = float(base_summary.get("visualization_threshold", 0.5))
    if prediction_files:
        prediction, quality = _prediction_statistics(
            prediction_files, label_files, threshold
        )
        if tile_files:
            prediction.update(_png_distribution(tile_files))
    elif generated_tile_files and task_type in _BINARY_TASKS:
        prediction, quality = _generated_binary_statistics(
            task_type, generated_tile_files, label_files
        )
    elif generated_tile_files:
        prediction = _png_distribution(list(generated_tile_files))
        quality = {
            "reference_available": False,
            "evaluated_patches": 0,
            "unavailable_reason": "分类结果缺少逐像素类别参考数组，当前只统计颜色分布。",
        }
    else:
        prediction = _png_distribution(tile_files) if tile_files else {}
        quality = {
            "reference_available": False,
            "evaluated_patches": 0,
            "unavailable_reason": "缺少可与标签对齐的数值预测数组。",
        }

    warnings = []
    for error in inference_errors or []:
        warnings.append(
            {
                "code": "PATCH_INFERENCE_FAILED",
                "severity": "warning",
                "message": f"{error['patch_id']} 推理失败：{error['error']}",
                "evidence": error,
            }
        )
    if not prediction_files and not generated_tile_files:
        warnings.append(
            {
                "code": "PREDICTIONS_MISSING",
                "severity": "warning",
                "message": "未发现逐 Patch 数值预测数组，无法计算阈值分布和标准质量指标。",
            }
        )
    if result_count < configured_patches:
        warnings.append(
            {
                "code": "RESULT_COVERAGE_INCOMPLETE",
                "severity": "warning",
                "message": f"仍有 {configured_patches - result_count} 个 Patch 缺少可用结果。",
            }
        )
    insights = [
        {
            "code": "RESULT_COVERAGE",
            "severity": "info",
            "message": f"可用结果覆盖 {result_count}/{configured_patches} 个 Patch。",
            "evidence": {"coverage_rate": coverage_rate},
        }
    ]
    if prediction.get("mean_positive_pixel_ratio") is not None:
        insights.append(
            {
                "code": "TARGET_DENSITY",
                "severity": "info",
                "message": "目标在区域内呈稀疏分布。" if prediction["mean_positive_pixel_ratio"] < 0.05 else "目标在区域内分布较广。",
                "evidence": {"mean_positive_pixel_ratio": prediction["mean_positive_pixel_ratio"]},
            }
        )

    model = _model_metadata(region_id, task_type, version)
    color_legend = _color_legend(
        region_id, task_type, task_name, version, prediction
    )
    image_analysis = _image_analysis(task_type, tile_files)
    result = {
        "schema_version": "2.0",
        "task": task_type,
        "name": task_name,
        "region_id": region_id,
        "version": version,
        "period": base_summary.get("period"),
        "status": status,
        "analysis_scope": scope,
        "summary_text": _narrative(
            task_name, region_id, model, temporal, coverage, prediction, color_legend, status
        ),
        "analysis_notes": [item["message"] for item in insights + warnings],
        "model": model,
        "temporal_coverage": temporal,
        "data_coverage": coverage,
        "prediction_statistics": prediction,
        "color_legend": color_legend,
        "image_analysis": image_analysis,
        "result_images": list(result_images or []),
        "insights": insights,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid_size": base_summary.get("grid_size"),
        "total_polygons": base_summary.get("total_polygons"),
        "total_patches": configured_patches,
        "positive_patches": prediction.get("positive_patches"),
        "negative_patches": prediction.get("negative_patches"),
    }
    if len(_SUMMARY_CACHE) >= 64:
        _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))
    _SUMMARY_CACHE[cache_key] = deepcopy(result)
    return result
