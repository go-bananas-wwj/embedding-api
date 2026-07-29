"""Downstream task router."""

import json
import re
import io
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from fastapi import APIRouter, HTTPException, Query, Path as PathParam
from fastapi.responses import FileResponse
from fastapi.responses import Response
import numpy as np
from PIL import Image

from app.config import get_config
from app.schemas.models import (
    TasksResponse, TaskInfo, TaskSummary, TilesResponse, TileInfo, ErrorResponse,
)
from app.services.data_service import DataService, DataServiceError, DataValidationError
from app.services.system_model_service import (
    get_system_model_classes,
    infer_system_model,
    infer_system_model_array,
    is_system_task,
)
from app.services.tile_service import TileService
from app.services.task_summary_service import build_task_summary
from app.services.summary_image_service import SUMMARY_IMAGE_DIR, publish_summary_images
from app.services.haidian_change_detection import (
    CHANGE_THRESHOLD as HAIDIAN_CHANGE_THRESHOLD,
    change_mask as haidian_change_mask,
    load_change_scores as load_haidian_change_scores,
    render_change_png as render_haidian_change_png,
)

router = APIRouter()
TASK_FORMATS = Literal["png", "npy"]
TASK_VERSIONS = Literal["v1", "v2"]
_BINARY_TASKS = {
    "change_detection",
    "building_extraction",
    "road_extraction",
    "water_extraction",
    "construction",
}
_HAIDIAN_LAND_RESULT_VERSION = "haidian-land-independent-conv3x3-20260724"


def _result_cache_headers(region_id: str, task_type: str) -> dict[str, str]:
    if region_id == "haidian" and task_type in {
        "land_use_classification",
        "land_cover_classification",
    }:
        return {
            "Cache-Control": "no-store",
            "X-Result-Version": _HAIDIAN_LAND_RESULT_VERSION,
        }
    return {}


def _resolve_task_version(region_id: str, task: dict, requested: Optional[str]) -> str:
    """Resolve one version consistently for every task-result endpoint."""
    configured = list((task.get("versions") or {}).keys())
    if requested:
        if requested not in ("v1", "v2"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid task version '{requested}'. Use v1 or v2.",
            )
        if configured and requested not in configured:
            raise HTTPException(
                status_code=404,
                detail=f"Task version '{requested}' is not available for region '{region_id}'",
            )
        return requested
    preferred = "v1" if region_id == "haidian" else "v2"
    if preferred in configured or not configured:
        return preferred
    if "v1" in configured:
        return "v1"
    return configured[0]

_TASK_OPENAPI_EXAMPLES = {
    "change_detection": {"summary": "Change detection", "value": "change_detection"},
    "building_extraction": {"summary": "Building extraction", "value": "building_extraction"},
    "road_extraction": {"summary": "Road extraction", "value": "road_extraction"},
    "construction": {"summary": "Construction detection", "value": "construction"},
    "land_use_classification": {"summary": "Land use classification", "value": "land_use_classification"},
    "land_cover_classification": {"summary": "Land cover classification", "value": "land_cover_classification"},
    "water_extraction": {"summary": "Water extraction", "value": "water_extraction"},
}

_VERSION_OPENAPI_EXAMPLES = {
    "haidian_p10c": {"summary": "海淀 P10C（API v1）", "value": "v1"},
    "harbin_v5": {"summary": "哈尔滨 V5（API v2）", "value": "v2"},
}

_MONTH_OPENAPI_EXAMPLES = {
    "haidian_latest": {
        "summary": "海淀最新数据默认月份",
        "description": "海淀新下载模型/embedding 可用月份之一，推荐前端联调优先使用。",
        "value": "202512",
    },
    "haidian_latest_hyphen": {
        "summary": "海淀月份（带横杠写法）",
        "value": "2025-12",
    },
    "harbin_available": {
        "summary": "哈尔滨月份（紧凑写法）",
        "value": "202510",
    },
    "harbin_available_hyphen": {
        "summary": "哈尔滨月份（带横杠写法）",
        "value": "2025-10",
    },
}

_BEFORE_MONTH_OPENAPI_EXAMPLES = {
    "haidian": {"summary": "海淀变化前月份", "value": "202512"},
    "compact": {"summary": "紧凑写法", "value": "202504"},
    "hyphen": {"summary": "带横杠写法", "value": "2025-04"},
}

_AFTER_MONTH_OPENAPI_EXAMPLES = {
    "haidian": {"summary": "海淀变化后月份", "value": "202604"},
    "compact": {"summary": "紧凑写法", "value": "202506"},
    "hyphen": {"summary": "带横杠写法", "value": "2025-06"},
}

_PATCH_IDS_OPENAPI_EXAMPLES = {
    "single": {
        "summary": "分析单个 Patch",
        "description": "只分析 patch_000000。",
        "value": ["patch_000000"],
    },
    "multiple": {
        "summary": "分析多个 Patch",
        "description": "两个 Patch 分别独立处理，最后汇总统计。",
        "value": ["patch_000000", "patch_000001"],
    },
}

_PERIOD_OPENAPI_EXAMPLES = {
    "harbin_change_detection": {
        "summary": "哈尔滨变化检测对比期",
        "value": "2025-04_vs_2025-06",
    },
    "harbin_building_v2": {
        "summary": "哈尔滨 building_extraction v2 对比期",
        "value": "2025-09_vs_2025-10",
    },
}

_TASK_RESULT_EXAMPLES = """
**推荐可直接运行示例（海淀最新数据）：**

```http
GET /regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=202512
```

**为什么 Swagger 里以前会 404：** 海淀最新数据的可用月份是
`202512`、`202601`、`202602`、`202603`、`202604`、`202605`。
如果 `region_id=haidian` 却填 `month=2025-04`，后端会按海淀数据查找，
这个月份不存在，所以返回 404 是合理的。
"""

_TASK_TIME_GUIDE = (
    "单期任务（building_extraction、road_extraction、construction、"
    "land_use_classification、land_cover_classification、water_extraction）"
    "填写 month；变化检测 change_detection 填 period，或 before_month + after_month。"
)

_HAIDIAN_VALID_MONTHS = "202512, 202601, 202602, 202603, 202604, 202605"
_HARBIN_VALID_MONTHS = "2025-04, 2025-06, 2025-08, 2025-09, 2025-10"


def _task_not_found_detail(
    kind: str,
    region_id: str,
    patch_id: str,
    task_type: str,
    version: str,
    period: Optional[str],
    month: Optional[str] = None,
) -> str:
    """Build an actionable not-found message for task result endpoints."""
    requested_time = month or period or "未填写"
    if region_id == "haidian":
        time_parameter = "period" if kind in {"Prediction", "Label"} else "month"
        return (
            f"{kind} not found for patch '{patch_id}', task '{task_type}', "
            f"version '{version}', time '{requested_time}'. "
            f"海淀最新数据请使用 {time_parameter}=202512/202601/202602/202603/202604/202605；"
            "例如：region_id=haidian, task_type=building_extraction, "
            f"version=v1, {time_parameter}=202512。"
        )
    if region_id == "harbin":
        return (
            f"{kind} not found for patch '{patch_id}', task '{task_type}', "
            f"version '{version}', time '{requested_time}'. "
            f"哈尔滨单期任务常用月份：{_HARBIN_VALID_MONTHS}；"
            "变化检测请使用 period=2025-04_vs_2025-06 等对比期。"
        )
    return (
        f"{kind} not found for patch '{patch_id}', task '{task_type}', "
        f"version '{version}', time '{requested_time}'."
    )

# Classification tasks whose visualizations are stored in xuannv_show static seg_tiles.
_XUANNV_SHOW_SEG_TILE_DIR = Path("/workspace/projects/xuannv-show/static_assets/data/seg_tiles")
_CLASS_TASK_TO_XUANNV_HEAD = {
    "building_extraction": "building_extraction",
    "land_use_classification": "dynamic_world",
    "land_cover_classification": "worldcover",
    "water_extraction": "jrc_water",
}

# Haidian ground-truth patch labels served as model outputs.
# Key: task_id; Value: (GT directory, PNG subdir, PNG name template, NPY subdir)
_HAIDIAN_GT_OVERRIDES: Dict[str, Tuple[Path, str, str, str]] = {
    "construction": (
        Path("data/haidian/v1/reports/haidian_construction_gt_patch_labels_20260701"),
        "labels",
        "haidian_construction_gt_{patch_id}.png",
        "predictions",
    ),
}


def _resolve_haidian_gt_override(
    region_id: str,
    task_type: str,
    patch_id: str,
    fmt: str,
    version: str,
    month: Optional[str] = None,
) -> Optional[str]:
    """Resolve Haidian GT result/prediction for curated tasks.

    Returns the path to a PNG label visualization or a binary NPY prediction mask
    when the caller requests a Haidian V1 task whose outputs are overridden by
    ground-truth labels.

    ``month`` is validated so that requests outside the Haidian V1 embedding
    time range (2025-12 ~ 2026-05) do not accidentally receive the override
    data for an unrelated date.
    """
    if region_id != "haidian" or version != "v1":
        return None
    override = _HAIDIAN_GT_OVERRIDES.get(task_type)
    if override is None:
        return None

    # Haidian V1 valid embedding months; the override labels are presented as
    # model outputs for this time window.
    if month is not None:
        from app.services.time_utils import normalize_month

        valid_months = {
            "2025-12", "202512",
            "2026-01", "202601",
            "2026-02", "202602",
            "2026-03", "202603",
            "2026-04", "202604",
            "2026-05", "202605",
        }
        if not any(m in valid_months for m in normalize_month(month)):
            return None

    gt_dir, png_subdir, png_template, npy_subdir = override
    if fmt == "png":
        filename = png_template.format(patch_id=patch_id)
        path = gt_dir / png_subdir / filename
    elif fmt == "npy":
        path = gt_dir / npy_subdir / f"{patch_id}.npy"
    else:
        return None
    return str(path) if path.exists() else None


def _list_haidian_gt_tiles(region_id: str, task_type: str, version: str) -> list:
    """Return tile entries for Haidian GT-overridden tasks.

    The GT directories hold curated labels/predictions but do not follow the
    standard ``results/tiles`` layout. This helper scans the PNG subdirectory
    and returns entries compatible with ``TileInfo``.
    """
    if region_id != "haidian" or version != "v1":
        return []
    override = _HAIDIAN_GT_OVERRIDES.get(task_type)
    if not override:
        return []
    gt_dir, png_subdir, png_template, _ = override
    png_dir = gt_dir / png_subdir
    if not png_dir.exists():
        return []

    result = []
    # Template stem is e.g. "haidian_construction_gt_{patch_id}" or "{patch_id}"
    prefix = png_template.split("{patch_id}")[0]
    suffix = png_template.split("{patch_id}")[-1].replace(".png", "")
    try:
        for path in sorted(png_dir.glob("*.png")):
            stem = path.stem
            if prefix and not stem.startswith(prefix):
                continue
            if suffix and not stem.endswith(suffix):
                continue
            patch_id = stem[len(prefix):] if prefix else stem
            if suffix:
                patch_id = patch_id[: -len(suffix)]
            if not patch_id:
                continue
            result.append(
                {
                    "patch_id": patch_id,
                    "period": "2026-05",
                    "filename": path.name,
                }
            )
    except OSError:
        pass
    return result


def _list_generated_system_tiles(
    region_id: str, task_type: str, period: Optional[str]
) -> List[Dict[str, Optional[str]]]:
    """List on-demand system results using canonical patch tile filenames."""
    if not period or not is_system_task(task_type):
        return []
    compact = period.replace("-", "")
    directory = Path("system_models/task_results")
    result: List[Dict[str, Optional[str]]] = []
    for path in sorted(
        directory.glob(f"{task_type}_{region_id}_patch_*_{compact}.png")
    ):
        match = re.search(r"patch_\d+", path.name)
        if not match:
            continue
        patch_id = match.group(0)
        result.append(
            {
                "patch_id": patch_id,
                "period": period,
                "filename": f"{patch_id}.png",
            }
        )
    return result


def _safe_filename(filename: str) -> bool:
    """Validate a tile filename to prevent path traversal."""
    import re
    return bool(re.match(r"^[\w\-\.]+\.png$", filename)) and ".." not in filename


def _binary_mask_from_result_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    white = np.all(rgb == 255, axis=-1)
    black = np.all(rgb == 0, axis=-1)
    if white.any():
        return (~white).astype(np.uint8)
    if black.any():
        return (~black).astype(np.uint8)
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    background = colors[int(np.argmax(counts))]
    return np.any(rgb != background, axis=-1).astype(np.uint8)


def _class_ids_from_result_image(
    path: str, region_id: str, task_type: str, version: str
) -> np.ndarray:
    """Decode a documented class-color PNG into integer class IDs."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    classes = get_system_model_classes(region_id, task_type, version)
    result = np.zeros(image.shape[:2], dtype=np.uint16)
    for index, item in enumerate(classes):
        color = str(item.get("color", "#000000")).lstrip("#")
        if len(color) != 6:
            continue
        rgb = np.array(
            [int(color[offset : offset + 2], 16) for offset in (0, 2, 4)],
            dtype=np.uint8,
        )
        result[np.all(image == rgb, axis=2)] = index
    return result


def _npy_result_response(prediction: np.ndarray, patch_id: str, task_type: str) -> Response:
    buffer = io.BytesIO()
    np.save(buffer, prediction, allow_pickle=False)
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{patch_id}_{task_type}_prediction.npy"'
            )
        },
    )


def _generate_summary_results(
    region_id: str,
    task_type: str,
    version: str,
    month: Optional[str],
    patch_ids: Optional[List[str]],
) -> tuple[Optional[list[Path]], list[Dict[str, str]]]:
    """Generate or reuse selected single-time task results for summary analysis."""
    if not month or not patch_ids or not is_system_task(task_type):
        return None, []
    results_dir = Path("system_models/task_results")
    task_config = (((get_config().get_region(region_id) or {}).get("tasks") or {}).get(task_type) or {})
    version_config = (task_config.get("versions") or {}).get(version) or {}
    configured_results = Path(version_config["results"]) if version_config.get("results") else None
    compact_month = month.replace("-", "")
    hyphen_month = f"{compact_month[:4]}-{compact_month[4:]}" if len(compact_month) == 6 else month
    generated: list[Path] = []
    errors: list[Dict[str, str]] = []
    for patch_id in patch_ids:
        configured_candidates = []
        if configured_results:
            for month_value in dict.fromkeys((month, compact_month, hyphen_month)):
                configured_candidates.extend([
                    configured_results / month_value / "tiles" / f"{patch_id}.png",
                    configured_results / "tiles" / f"{patch_id}_{month_value}.png",
                ])
            configured_candidates.append(configured_results / "tiles" / f"{patch_id}.png")
        existing = next((path for path in configured_candidates if path.exists()), None)
        if existing:
            generated.append(existing)
            continue
        cached = results_dir / f"{task_type}_{region_id}_{patch_id}_{month}.png"
        if cached.exists():
            generated.append(cached)
            continue
        try:
            generated.append(
                Path(infer_system_model(
                    region_id,
                    task_type,
                    patch_id,
                    month,
                    version=version,
                    results_dir=results_dir,
                ))
            )
        except Exception as exc:
            errors.append({"patch_id": patch_id, "error": str(exc)})
    return generated, errors


def _validate_summary_scope(
    region_id: str, month: Optional[str], patch_ids: Optional[List[str]]
) -> Optional[List[str]]:
    if month:
        match = re.fullmatch(r"(\d{4})-?(\d{2})", month)
        if not match or not 1 <= int(match.group(2)) <= 12:
            raise HTTPException(status_code=422, detail="month 必须是有效的 YYYYMM 或 YYYY-MM，例如 202604。")
    if not patch_ids:
        return patch_ids
    unique_patch_ids = list(dict.fromkeys(patch_ids))
    if len(unique_patch_ids) > 100 or any(not re.fullmatch(r"patch_\d{6}", value) for value in unique_patch_ids):
        raise HTTPException(status_code=422, detail="patch_ids 最多 100 个，格式应为 patch_000000。")
    region_patch_ids = {item.get("patch_id") for item in get_config().get_patches(region_id)}
    unknown = [value for value in unique_patch_ids if value not in region_patch_ids]
    if unknown:
        raise HTTPException(status_code=404, detail=f"以下 Patch 不属于区域 {region_id}：{', '.join(unknown)}")
    return unique_patch_ids


@router.get("/task-summary/results/{filename}", include_in_schema=False)
async def get_summary_result_image(filename: str):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.png", filename) or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid summary image filename")
    path = SUMMARY_IMAGE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary image expired or not found")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


def _resolve_classification_tile(
    region_id: str,
    task_type: str,
    patch_id: str,
    month: str,
    version: str,
) -> Optional[str]:
    """Resolve a semantic classification result tile.

    First look for a pre-generated xuannv_show static seg tile. If not found and
    the task is a supported system model, run inference on demand and return the
    generated PNG path.

    Haidian V1 uses its own pre-generated results first. If a configured tile is
    absent, supported system tasks fall back to the latest Haidian task heads.
    """
    head = _CLASS_TASK_TO_XUANNV_HEAD.get(task_type)
    if not head and not is_system_task(task_type):
        return None

    # Haidian has its own task result layout; avoid Harbin static tiles.
    if region_id == "haidian":
        static_path = None
    else:
        static_path = (
            _XUANNV_SHOW_SEG_TILE_DIR / head / month / f"{patch_id}.png"
            if head
            else None
        )
        if static_path and static_path.exists():
            return str(static_path)

    if is_system_task(task_type):
        try:
            result_path = infer_system_model(
                region_id=region_id,
                task_id=task_type,
                patch_id=patch_id,
                month=month,
                version=version,
                results_dir=Path("system_models/task_results"),
            )
            return str(result_path)
        except FileNotFoundError:
            return None

    return None


@router.get("/regions/{region_id}/tasks", response_model=TasksResponse)
async def list_tasks(
    region_id: str = PathParam(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
):
    """列出指定区域支持的所有下游监测任务。

    用于前端任务选择面板，展示该区域可执行的变化检测、建筑物提取等任务。
    返回任务 ID、名称、描述及可用版本的 JSON 列表。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    region = config.get_region(region_id)
    tasks = []
    for tid, tinfo in region.get("tasks", {}).items():
        tasks.append(
            TaskInfo(
                id=tid,
                name=tinfo.get("name", tid),
                description=tinfo.get("description"),
                versions=list(tinfo.get("versions", {}).keys()),
            )
        )
    return TasksResponse(tasks=tasks)


@router.get("/regions/{region_id}/tasks/{task_type}/summary", response_model=TaskSummary)
async def get_task_summary(
    region_id: str = PathParam(
        ...,
        description=(
            "区域 ID。可选 `harbin` 或 `haidian`。"
            "海淀区请配合最新月份 `202512` ~ `202605` 使用。"
        ),
        examples=["haidian"],
    ),
    task_type: str = PathParam(
        ...,
        description=(
            "任务类型。海淀最新数据推荐先用 `building_extraction` 验证；"
            "其他可选值见 examples。"
        ),
        examples=["building_extraction"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    version: Optional[str] = Query(
        None,
        description=(
            "模型版本，通常不需要填写。省略时海淀自动使用 P10C（API `v1`），"
            "哈尔滨自动使用 V5（API `v2`）。"
        ),
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        include_in_schema=False,
    ),
    month: Optional[str] = Query(
        None,
        description=(
            "单期任务影像月份。支持 `YYYYMM` 和 `YYYY-MM`；不填时分析现有结果。"
        ),
        openapi_examples=_MONTH_OPENAPI_EXAMPLES,
    ),
    patch_ids: Optional[List[str]] = Query(
        None,
        description=(
            "需要分析的 Patch ID，格式为 `patch_` 加六位数字，例如 `patch_000000`。"
            "可重复传入多个；每个 Patch 独立推理/统计，最后汇总。不填表示分析全部 Patch。"
            "URL 示例：`?patch_ids=patch_000000&patch_ids=patch_000001`。"
        ),
        openapi_examples=_PATCH_IDS_OPENAPI_EXAMPLES,
    ),
):
    """获取任务资产、预测分布、质量指标和中文综合分析。

    摘要从真实预测、标签和结果瓦片生成，适合前端仪表盘和智能体分析。
    没有参考标签时会说明指标不可用原因，不会用空值冒充模型质量。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    region = config.get_region(region_id)
    tasks = region.get("tasks", {})
    if task_type not in tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_type}' not found")
    if task_type == "change_detection" and not period:
        raise HTTPException(
            status_code=400,
            detail="变化检测请使用 /regions/{region_id}/change-detection/summary，并传 before_month、after_month。",
        )
    patch_ids = _validate_summary_scope(region_id, month, patch_ids)

    task = tasks[task_type]
    version = _resolve_task_version(region_id, task, version)
    effective_period = period if task_type == "change_detection" else None
    legacy_before = legacy_after = None
    if effective_period and "_vs_" in effective_period:
        legacy_before, legacy_after = effective_period.split("_vs_", 1)
    generated_tiles, inference_errors = _generate_summary_results(
        region_id, task_type, version, month, patch_ids
    )
    if patch_ids and month and inference_errors and not generated_tiles:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "请求的 Patch 均未能生成任务结果。",
                "failures": inference_errors,
            },
        )
    result_images = (
        publish_summary_images(
            region_id, task_type, version, month, generated_tiles
        )
        if month and generated_tiles is not None
        else []
    )

    # Haidian GT-overridden tasks use the curated manifest for summary stats.
    if region_id == "haidian" and version == "v1" and task_type in _HAIDIAN_GT_OVERRIDES:
        gt_dir, _, _, _ = _HAIDIAN_GT_OVERRIDES[task_type]
        manifest_path = gt_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return TaskSummary(**build_task_summary(
                region_id=region_id,
                task_type=task_type,
                version=version,
                task_name=task.get("name", task_type),
                base_summary={
                    "period": "20260701",
                    "total_patches": manifest.get("patch_count"),
                    "positive_patches": manifest.get("labeled_patch_count"),
                    "negative_patches": manifest.get("unlabeled_patch_count"),
                },
                patch_ids=patch_ids,
                month=month,
                generated_tile_files=generated_tiles,
                inference_errors=inference_errors,
                result_images=result_images,
            ))

    try:
        summary = DataService.load_task_summary(
            region_id, task_type, version, effective_period
        )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # A prebuilt summary is optional. Configured tasks can generate selected
    # Patch results on demand and derive their summary directly from assets.
    summary = dict(summary or {})
    summary.setdefault("period", effective_period)
    return TaskSummary(**build_task_summary(
        region_id=region_id,
        task_type=task_type,
        version=version,
        task_name=task.get("name", task_type),
        base_summary=summary,
        patch_ids=patch_ids,
        month=month,
        before_month=legacy_before,
        after_month=legacy_after,
        generated_tile_files=generated_tiles,
        inference_errors=inference_errors,
        result_images=result_images,
    ))


@router.get("/regions/{region_id}/change-detection/summary", response_model=TaskSummary)
async def get_change_detection_summary(
    region_id: str = PathParam(..., description="区域 ID，可选 harbin 或 haidian。", examples=["harbin"]),
    before_month: str = Query(..., description="变化前月份，支持 YYYYMM 或 YYYY-MM。", openapi_examples=_BEFORE_MONTH_OPENAPI_EXAMPLES),
    after_month: str = Query(..., description="变化后月份，支持 YYYYMM 或 YYYY-MM。", openapi_examples=_AFTER_MONTH_OPENAPI_EXAMPLES),
    patch_ids: Optional[List[str]] = Query(
        None,
        description=(
            "需要分析的 Patch ID，格式为 `patch_000000`。可重复传入多个；"
            "每个 Patch 只比较自己的前后月份，最后汇总。"
            "URL 示例：`?patch_ids=patch_000000&patch_ids=patch_000001`。"
        ),
        openapi_examples=_PATCH_IDS_OPENAPI_EXAMPLES,
    ),
    version: Optional[str] = Query(None, description="模型版本；不填时按区域自动选择。", openapi_examples=_VERSION_OPENAPI_EXAMPLES),
):
    """对一个或多个 Patch 分别执行双时相变化分析，再汇总结果。"""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    task = ((config.get_region(region_id) or {}).get("tasks") or {}).get("change_detection")
    if not task:
        raise HTTPException(status_code=404, detail="Change detection is not configured for this region")
    patch_ids = _validate_summary_scope(region_id, before_month, patch_ids)
    _validate_summary_scope(region_id, after_month, patch_ids)
    resolved_version = _resolve_task_version(region_id, task, version)
    period = f"{before_month}_vs_{after_month}"
    try:
        summary = DataService.load_task_summary(region_id, "change_detection", resolved_version, period) or {"period": period}
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TaskSummary(**build_task_summary(
        region_id=region_id,
        task_type="change_detection",
        version=resolved_version,
        task_name=task.get("name", "变化检测"),
        base_summary=dict(summary),
        patch_ids=patch_ids,
        before_month=before_month,
        after_month=after_month,
    ))


@router.get(
    "/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result",
    responses={
        200: {
            "description": "Task result",
            "content": {
                "image/png": {},
                "application/octet-stream": {},
            },
        },
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def get_task_result(
    region_id: str = PathParam(
        ...,
        description=(
            "区域 ID。可选 `harbin` 或 `haidian`。"
            "Swagger 默认示例使用海淀最新数据，点 Try it out 可直接成功。"
        ),
        examples=["haidian"],
    ),
    patch_id: str = PathParam(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    task_type: str = PathParam(
        ...,
        description=(
            "任务类型。推荐默认 `building_extraction`。"
            "`change_detection` 是变化检测，需同时填写 `before_month` 和 `after_month`。"
        ),
        examples=["building_extraction"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    format: TASK_FORMATS = Query(
        "png",
        description="Output format. Allowed values: png, npy.",
        examples=["png"],
        openapi_examples={
            "png": {"summary": "PNG image", "value": "png"},
            "npy": {"summary": "NumPy array", "value": "npy"},
        },
    ),
    version: Optional[str] = Query(
        None,
        description=(
            "模型版本，通常不需要填写。省略时海淀自动使用 P10C（API `v1`），"
            "哈尔滨自动使用 V5（API `v2`）。"
        ),
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        include_in_schema=False,
    ),
    month: Optional[str] = Query(
        None,
        description=(
            "单期任务月份。海淀和哈尔滨统一支持 `YYYYMM` 与 `YYYY-MM` 两种写法。"
            "海淀可用 `202512` ~ `202605`，哈尔滨可用 `202504` ~ `202510`。"
        ),
        examples=["202512"],
        openapi_examples=_MONTH_OPENAPI_EXAMPLES,
    ),
    before_month: Optional[str] = Query(
        None,
        description=(
            "仅变化检测填写：变化前月份。支持 `YYYYMM` 和 `YYYY-MM`，"
            "例如 `202504` 或 `2025-04`。"
        ),
        openapi_examples=_BEFORE_MONTH_OPENAPI_EXAMPLES,
    ),
    after_month: Optional[str] = Query(
        None,
        description=(
            "仅变化检测填写：变化后月份。支持 `YYYYMM` 和 `YYYY-MM`，"
            "例如 `202506` 或 `2025-06`。"
        ),
        openapi_examples=_AFTER_MONTH_OPENAPI_EXAMPLES,
    ),
):
    """获取单个 Patch 的任务结果，支持 `png` 和 `npy`。

    单期任务只填写 `month`；变化检测同时填写 `before_month` 和 `after_month`。
    海淀变化检测使用 P10C 64D embedding、双向 5×5 邻域平均融合；
    PNG 中达到全区 P98 阈值 `0.7715` 的像素显示为红色，NPY 返回对应的 0/1 掩膜。
    `version` 省略时按区域自动选择当前默认模型。
    """
    if format not in ("png", "npy"):
        raise HTTPException(
            status_code=422, detail=f"Invalid format '{format}'. Use: png, npy"
        )

    # Derive period from month / before+after when not explicitly provided.
    effective_period = period
    if not effective_period:
        if task_type == "change_detection":
            if before_month and after_month:
                effective_period = f"{before_month}_vs_{after_month}"
        elif month:
            effective_period = month

    # Single-month value for classification tasks.
    class_month = month or (
        effective_period if effective_period and "_vs_" not in effective_period else None
    )

    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    region = config.get_region(region_id) or {}
    task = (region.get("tasks") or {}).get(task_type)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_type}' not found")
    version = _resolve_task_version(region_id, task, version)

    if region_id == "haidian" and task_type == "change_detection":
        if not before_month or not after_month:
            raise HTTPException(
                status_code=422,
                detail=(
                    "海淀变化检测必须同时填写 before_month 和 after_month；"
                    "例如 before_month=202512&after_month=202604。"
                ),
            )
        try:
            scores, valid = load_haidian_change_scores(
                patch_id,
                before_month,
                after_month,
                version=task.get("embedding_version", "v1"),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        headers = {
            "X-Change-Algorithm": "p10c-cosine-bidirectional-5x5-mean",
            "X-Change-Threshold": f"{HAIDIAN_CHANGE_THRESHOLD:.6f}",
        }
        if format == "npy":
            response = _npy_result_response(
                haidian_change_mask(scores, valid), patch_id, task_type
            )
            response.headers.update(headers)
            return response
        return Response(
            content=render_haidian_change_png(scores, valid),
            media_type="image/png",
            headers=headers,
        )

    try:
        patch = DataService.get_patch(region_id, patch_id)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    try:
        # Haidian construction uses the curated GT patch labels as predictions.
        override_path = _resolve_haidian_gt_override(
            region_id, task_type, patch_id, format, version, month=class_month
        )
        if override_path:
            if format == "npy":
                return FileResponse(
                    override_path,
                    media_type="application/octet-stream",
                    filename=f"{patch_id}_{task_type}_prediction.npy",
                )
            return FileResponse(override_path, media_type="image/png")

        if format == "npy":
            matching_png = None
            if class_month and region_id != "haidian" and task_type in _CLASS_TASK_TO_XUANNV_HEAD:
                head = _CLASS_TASK_TO_XUANNV_HEAD[task_type]
                candidate = _XUANNV_SHOW_SEG_TILE_DIR / head / class_month / f"{patch_id}.png"
                if candidate.exists():
                    matching_png = str(candidate)
            if not matching_png:
                matching_png = DataService.get_task_result_path(
                    region_id, patch_id, task_type, "png", version, effective_period
                ) or DataService.get_task_result_path(
                    region_id, patch_id, task_type, "tile", version, effective_period
                )
            if matching_png and matching_png.lower().endswith(".png"):
                if task_type in _BINARY_TASKS:
                    return _npy_result_response(
                        _binary_mask_from_result_image(matching_png), patch_id, task_type
                    )
                if task_type in {
                    "land_use_classification",
                    "land_cover_classification",
                }:
                    return _npy_result_response(
                        _class_ids_from_result_image(
                            matching_png, region_id, task_type, version
                        ),
                        patch_id,
                        task_type,
                    )
            if class_month and is_system_task(task_type):
                try:
                    prediction = infer_system_model_array(
                        region_id, task_type, patch_id, class_month, version
                    )
                except FileNotFoundError:
                    prediction = None
                if prediction is not None:
                    return _npy_result_response(prediction, patch_id, task_type)
            path = DataService.get_task_result_path(
                region_id, patch_id, task_type, "npy", version, effective_period
            )
            if path:
                return FileResponse(
                    path,
                    media_type="application/octet-stream",
                    filename=f"{patch_id}_{task_type}_prediction.npy",
                )
        else:
            # Classification tasks: prefer xuannv_show static seg_tiles if available
            # (Harbin only; Haidian uses its own pre-generated results and GT overrides).
            if region_id != "haidian" and task_type in _CLASS_TASK_TO_XUANNV_HEAD and class_month:
                head = _CLASS_TASK_TO_XUANNV_HEAD[task_type]
                static_path = _XUANNV_SHOW_SEG_TILE_DIR / head / class_month / f"{patch_id}.png"
                if static_path.exists():
                    return FileResponse(str(static_path), media_type="image/png")

            # Try configured result files / tiles (e.g. haidian land use/cover)
            path = DataService.get_task_result_path(
                region_id, patch_id, task_type, "png", version, effective_period
            )
            if path and path.lower().endswith(".png"):
                return FileResponse(
                    path,
                    media_type="image/png",
                    headers=_result_cache_headers(region_id, task_type),
                )
            # Fallback: per-patch tile image (results/.../tiles/patch_*.png)
            path = DataService.get_task_result_path(
                region_id, patch_id, task_type, "tile", version, effective_period
            )
            if path and path.lower().endswith(".png"):
                return FileResponse(
                    path,
                    media_type="image/png",
                    headers=_result_cache_headers(region_id, task_type),
                )

            # Last resort: run system model inference. For Haidian this uses the
            # latest v1 task heads from models/haidian/v1/task_heads.
            if class_month and is_system_task(task_type):
                tile_path = _resolve_classification_tile(
                    region_id, task_type, patch_id, class_month, version
                )
                if tile_path:
                    return FileResponse(
                        tile_path,
                        media_type="image/png",
                        headers=_result_cache_headers(region_id, task_type),
                    )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(
        status_code=404,
        detail=_task_not_found_detail(
            "Result",
            region_id,
            patch_id,
            task_type,
            version,
            effective_period,
            month=class_month,
        ),
    )


@router.get("/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction")
async def get_task_prediction(
    region_id: str = PathParam(
        ...,
        description="区域 ID。可选 `harbin` 或 `haidian`；海淀最新数据推荐用 `haidian`。",
        examples=["haidian"],
    ),
    patch_id: str = PathParam(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    task_type: str = PathParam(
        ...,
        description="任务类型。海淀最新数据推荐默认 `building_extraction`。",
        examples=["building_extraction"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    version: Optional[str] = Query(
        None,
        description="Task result version. Allowed values: v1, v2.",
        examples=["v1"],
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        description=(
            "时间参数。单期任务可填月份如 `202512`；"
            "变化检测可填 `2025-04_vs_2025-06`。不填时会尝试查找 patch 默认预测。"
        ),
        examples=["202512"],
        openapi_examples={
            "haidian_latest": {"summary": "海淀最新月份", "value": "202512"},
            **_PERIOD_OPENAPI_EXAMPLES,
        },
    ),
):
    """获取某个 Patch 的原始预测数组（.npy）。

    用于将模型输出接入自定义分析或后处理流程。
    返回二进制 NumPy 数组，建议使用 curl 或程序代码下载处理。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    task = ((config.get_region(region_id) or {}).get("tasks") or {}).get(task_type)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_type}' not found")
    version = _resolve_task_version(region_id, task, version)

    try:
        patch = DataService.get_patch(region_id, patch_id)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    override_path = _resolve_haidian_gt_override(
        region_id, task_type, patch_id, "npy", version, month=period
    )
    if override_path:
        return FileResponse(
            override_path,
            media_type="application/octet-stream",
            filename=f"{patch_id}_{task_type}_prediction.npy",
        )

    try:
        path = DataService.get_task_result_path(
            region_id, patch_id, task_type, "npy", version, period
        )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if path:
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{patch_id}_{task_type}_prediction.npy",
        )

    # Some monthly products persist only a colorized PNG tile. Reconstruct
    # the numeric prediction so this endpoint matches `result?format=npy`.
    try:
        png_path = DataService.get_task_result_path(
            region_id, patch_id, task_type, "png", version, period
        ) or DataService.get_task_result_path(
            region_id, patch_id, task_type, "tile", version, period
        )
        if png_path and png_path.lower().endswith(".png"):
            if task_type in _BINARY_TASKS:
                prediction = _binary_mask_from_result_image(png_path)
            elif task_type in {
                "land_use_classification",
                "land_cover_classification",
            }:
                prediction = _class_ids_from_result_image(
                    png_path, region_id, task_type, version
                )
            else:
                prediction = None
            if prediction is not None:
                return _npy_result_response(prediction, patch_id, task_type)

        if period and is_system_task(task_type):
            prediction = infer_system_model_array(
                region_id, task_type, patch_id, period, version
            )
            return _npy_result_response(prediction, patch_id, task_type)
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    raise HTTPException(
        status_code=404,
        detail=_task_not_found_detail(
            "Prediction",
            region_id,
            patch_id,
            task_type,
            version,
            period,
        ),
    )


@router.get("/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label")
async def get_task_label(
    region_id: str = PathParam(
        ...,
        description="区域 ID。可选 `harbin` 或 `haidian`；海淀最新数据推荐用 `haidian`。",
        examples=["haidian"],
    ),
    patch_id: str = PathParam(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    task_type: str = PathParam(
        ...,
        description="任务类型。海淀最新数据推荐默认 `building_extraction`。",
        examples=["building_extraction"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    version: Optional[str] = Query(
        None,
        description="Task result version. Allowed values: v1, v2.",
        examples=["v1"],
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        description=(
            "时间参数。单期任务可填月份如 `202512`；"
            "变化检测可填 `2025-04_vs_2025-06`。不填时会查找 patch 默认标签。"
        ),
        examples=["202512"],
        openapi_examples={
            "haidian_latest": {"summary": "海淀最新月份", "value": "202512"},
            **_PERIOD_OPENAPI_EXAMPLES,
        },
    ),
):
    """获取某个 Patch 的标签数据（.npy 或 .json）。

    用于对比模型预测与标签数据、计算精度或制作训练样本。
    返回二进制 NumPy 数组或 JSON 元数据，具体取决于标签文件的存储格式。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    task = ((config.get_region(region_id) or {}).get("tasks") or {}).get(task_type)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_type}' not found")
    version = _resolve_task_version(region_id, task, version)

    try:
        patch = DataService.get_patch(region_id, patch_id)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    try:
        path = DataService.get_task_result_path(
            region_id, patch_id, task_type, "label", version, period
        )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if path:
        if path.lower().endswith(".npy"):
            return FileResponse(
                path,
                media_type="application/octet-stream",
                filename=f"{patch_id}_{task_type}_label.npy",
            )
        elif path.lower().endswith(".json"):
            return FileResponse(path, media_type="application/json")

    raise HTTPException(
        status_code=404,
        detail=_task_not_found_detail(
            "Label",
            region_id,
            patch_id,
            task_type,
            version,
            period,
        ),
    )


@router.get("/regions/{region_id}/tasks/{task_type}/tiles", response_model=TilesResponse)
async def list_tiles(
    region_id: str = PathParam(
        ...,
        description="区域 ID。可选 `harbin` 或 `haidian`；海淀最新数据推荐用 `haidian`。",
        examples=["haidian"],
    ),
    task_type: str = PathParam(
        ...,
        description="任务类型。海淀最新数据推荐默认 `building_extraction`。",
        examples=["building_extraction"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    version: Optional[str] = Query(
        None,
        description="Task result version. Allowed values: v1, v2.",
        examples=["v1"],
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        description=(
            "时间参数。海淀单期任务可填 `202512` ~ `202605`；"
            "不填时返回默认 tiles 目录。"
        ),
        examples=["202512"],
        openapi_examples={
            "haidian_latest": {"summary": "海淀最新月份", "value": "202512"},
            **_PERIOD_OPENAPI_EXAMPLES,
        },
    ),
):
    """列出某任务下可用的瓦片文件。

    用于前端构建瓦片图层索引，如结果大图叠加在地图上。
    返回瓦片文件名、所属 Patch 及对比期信息的 JSON 列表。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    task = ((config.get_region(region_id) or {}).get("tasks") or {}).get(task_type)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_type}' not found")
    version = _resolve_task_version(region_id, task, version)

    raw_tiles = await TileService.list_available_tiles(region_id, task_type, version, period)

    # Include Haidian GT-overridden task tiles (construction, road_extraction)
    # whose PNGs live outside the standard results/tiles directory.
    raw_tiles.extend(_list_haidian_gt_tiles(region_id, task_type, version))
    raw_tiles.extend(_list_generated_system_tiles(region_id, task_type, period))

    deduplicated = {}
    for tile in raw_tiles:
        key = (tile.get("patch_id"), tile.get("period"), tile.get("filename"))
        deduplicated[key] = tile
    raw_tiles = list(deduplicated.values())

    tiles = [
        TileInfo(
            patch_id=t.get("patch_id", ""),
            period=t.get("period"),
            filename=t.get("filename", ""),
        )
        for t in raw_tiles
    ]
    return TilesResponse(tiles=tiles, total=len(tiles))


@router.get(
    "/regions/{region_id}/tasks/{task_type}/tiles/{filename}",
    responses={
        200: {"content": {"image/png": {}}},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def get_patch_tile_by_filename(
    region_id: str = PathParam(
        ...,
        description="区域 ID。可选 `harbin` 或 `haidian`；海淀最新数据推荐用 `haidian`。",
        examples=["haidian"],
    ),
    task_type: str = PathParam(
        ...,
        description="任务类型。海淀最新数据推荐默认 `building_extraction`。",
        examples=["building_extraction"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    filename: str = PathParam(
        ...,
        description="Tile filename as returned by /regions/{region_id}/tasks/{task_type}/tiles.",
        examples=["patch_000000.png"],
    ),
    version: Optional[str] = Query(
        None,
        description="Task result version. Allowed values: v1, v2.",
        examples=["v1"],
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        description="时间参数。海淀单期任务可填 `202512` ~ `202605`；不填时查默认 tiles。",
        examples=["202512"],
        openapi_examples={
            "haidian_latest": {"summary": "海淀最新月份", "value": "202512"},
            **_PERIOD_OPENAPI_EXAMPLES,
        },
    ),
):
    """获取某个 Patch 的任务结果瓦片图片。

    ``filename`` 必须来自 ``/regions/{region_id}/tasks/{task_type}/tiles`` 列表。
    支持标准 ``results/tiles`` 布局以及海淀区 GT override 目录中的标签图片。
    """
    if not _safe_filename(filename):
        raise HTTPException(status_code=422, detail="Invalid tile filename")

    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    task = ((config.get_region(region_id) or {}).get("tasks") or {}).get(task_type)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_type}' not found")
    version = _resolve_task_version(region_id, task, version)

    # Derive patch_id from filename. All tile filenames contain a
    # ``patch_NNNNNN`` token (e.g. patch_000000.png, patch_000000_2025-04.png,
    # haidian_construction_gt_patch_000000.png).
    import re

    match = re.search(r"patch_\d+", filename)
    if not match:
        raise HTTPException(status_code=422, detail="Unrecognized tile filename")
    patch_id = match.group(0)

    # Haidian GT overrides take precedence when available.
    override_path = _resolve_haidian_gt_override(
        region_id, task_type, patch_id, "png", version, month=period
    )
    if override_path:
        return FileResponse(override_path, media_type="image/png")

    # Standard results/tiles layout.
    path = DataService.get_task_result_path(
        region_id, patch_id, task_type, "tile", version, period
    )
    if path and path.lower().endswith(".png"):
        return FileResponse(path, media_type="image/png")

    if period and is_system_task(task_type):
        path = _resolve_classification_tile(
            region_id, task_type, patch_id, period, version
        )
        if path and path.lower().endswith(".png"):
            return FileResponse(path, media_type="image/png")

    raise HTTPException(
        status_code=404,
        detail=f"Tile '{filename}' not found for task '{task_type}'",
    )


@router.get(
    "/regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png",
    include_in_schema=False,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)
async def get_tile(
    region_id: str = PathParam(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    task_type: str = PathParam(
        ...,
        description="Downstream task type.",
        examples=["change_detection"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    z: int = PathParam(..., ge=0, le=20, description="Tile zoom level.", examples=[12]),
    x: int = PathParam(..., ge=0, description="Tile X coordinate.", examples=[6828]),
    y: int = PathParam(..., ge=0, description="Tile Y coordinate.", examples=[3102]),
    version: str = Query(
        "v1",
        description="Task result version. Allowed values: v1, v2.",
        examples=["v1"],
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        description="Comparison period for time-series tasks, e.g. 2025-04_vs_2025-06.",
        examples=["2025-04_vs_2025-06"],
    ),
):
    """获取指定坐标的地图瓦片图片。

    当前端点尚未实现，仅返回 HTTP 501。
    后续可用于 Leaflet、Mapbox 等地图库叠加任务结果瓦片图层。
    """
    raise HTTPException(
        status_code=501,
        detail="Tile serving is not yet implemented. Use /regions/{region_id}/tasks/{task_type}/tiles for available patch tiles.",
    )
