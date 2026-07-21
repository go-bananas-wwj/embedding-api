"""Downstream task router."""

import json
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple
from fastapi import APIRouter, HTTPException, Query, Path as PathParam
from fastapi.responses import FileResponse

from app.config import get_config
from app.schemas.models import (
    TasksResponse, TaskInfo, TaskSummary, TilesResponse, TileInfo, ErrorResponse,
)
from app.services.data_service import DataService, DataServiceError, DataValidationError
from app.services.system_model_service import infer_system_model, is_system_task
from app.services.tile_service import TileService

router = APIRouter()
TASK_FORMATS = Literal["png", "npy"]

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
    "v1": {"summary": "v1（海淀最新模型与哈尔滨 V4）", "value": "v1"},
    "v2": {"summary": "v2（哈尔滨 V5 对比结果）", "value": "v2"},
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
        "summary": "哈尔滨可用月份",
        "value": "2025-10",
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
        return (
            f"{kind} not found for patch '{patch_id}', task '{task_type}', "
            f"version '{version}', time '{requested_time}'. "
            "海淀最新数据请使用 month=202512/202601/202602/202603/202604/202605；"
            "例如：region_id=haidian, task_type=building_extraction, "
            "version=v1, month=202512。"
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


def _safe_filename(filename: str) -> bool:
    """Validate a tile filename to prevent path traversal."""
    import re
    return bool(re.match(r"^[\w\-\.]+\.png$", filename)) and ".." not in filename


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
    version: str = Query(
        "v1",
        description="Task result version. Allowed values: v1, v2.",
        examples=["v1"],
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        description=(
            "对比周期，仅变化检测或部分 v2 对比任务使用。"
            "单期任务不要填这个字段，改填 `month`。"
        ),
        examples=["2025-04_vs_2025-06"],
        openapi_examples=_PERIOD_OPENAPI_EXAMPLES,
    ),
):
    """获取某任务的全局统计摘要。

    用于仪表板展示任务覆盖 Patch 数、正负样本数等概览信息。
    返回包含任务名称、版本、统计指标及对比周期的 JSON。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    region = config.get_region(region_id)
    tasks = region.get("tasks", {})
    if task_type not in tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_type}' not found")

    task = tasks[task_type]

    # Haidian GT-overridden tasks use the curated manifest for summary stats.
    if region_id == "haidian" and version == "v1" and task_type in _HAIDIAN_GT_OVERRIDES:
        gt_dir, _, _, _ = _HAIDIAN_GT_OVERRIDES[task_type]
        manifest_path = gt_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return TaskSummary(
                task=task_type,
                name=task.get("name", task_type),
                version=version,
                period="20260701",
                grid_size=None,
                total_polygons=None,
                total_patches=manifest.get("patch_count"),
                positive_patches=manifest.get("labeled_patch_count"),
                negative_patches=manifest.get("unlabeled_patch_count"),
            )

    try:
        summary = DataService.load_task_summary(region_id, task_type, version, period)
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not summary:
        raise HTTPException(
            status_code=404, detail=f"Summary not found for task '{task_type}'"
        )

    return TaskSummary(
        task=task_type,
        name=task.get("name", task_type),
        version=version,
        period=summary.get("period") or period,
        grid_size=summary.get("grid_size"),
        total_polygons=summary.get("total_polygons"),
        total_patches=summary.get("total_patches"),
        positive_patches=summary.get("positive_patches"),
        negative_patches=summary.get("negative_patches"),
    )


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
            "`change_detection` 是变化检测，需填写 `period` 或 `before_month` + `after_month`。"
        ),
        examples=["building_extraction"],
        openapi_examples=_TASK_OPENAPI_EXAMPLES,
    ),
    format: str = Query(
        "png",
        description="Output format. Allowed values: png, npy.",
        examples=["png"],
        openapi_examples={
            "png": {"summary": "PNG image", "value": "png"},
            "npy": {"summary": "NumPy array", "value": "npy"},
        },
    ),
    version: str = Query(
        "v1",
        description="Task result version. Allowed values: v1, v2.",
        examples=["v1"],
        openapi_examples=_VERSION_OPENAPI_EXAMPLES,
    ),
    period: Optional[str] = Query(
        None,
        description=(
            "对比周期。仅变化检测或部分 v2 对比任务填写；"
            "例如 `2025-04_vs_2025-06`。"
            "海淀 `building_extraction` 等单期任务不要填 period，填 `month=202512`。"
        ),
        examples=["2025-04_vs_2025-06"],
        openapi_examples=_PERIOD_OPENAPI_EXAMPLES,
    ),
    month: Optional[str] = Query(
        None,
        description=(
            "单期任务月份。海淀最新数据可填 `202512` ~ `202605`"
            "（也兼容 `2025-12` ~ `2026-05`）；哈尔滨可填 `2025-04` ~ `2025-10`。"
        ),
        examples=["202512"],
        openapi_examples=_MONTH_OPENAPI_EXAMPLES,
    ),
    before_month: Optional[str] = Query(
        None,
        description="变化检测任务的起始月份。哈尔滨可用：2025-04、2025-06、2025-08、2025-09。",
        examples=["2025-04"],
    ),
    after_month: Optional[str] = Query(
        None,
        description="变化检测任务的结束月份。哈尔滨可用：2025-06、2025-08、2025-09、2025-10。",
        examples=["2025-06"],
    ),
):
    """推荐可直接运行示例（海淀最新数据）：

    ```http
    GET /regions/haidian/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&month=202512
    ```

    为什么 Swagger 里以前会 404：海淀最新数据的可用月份是
    `202512`、`202601`、`202602`、`202603`、`202604`、`202605`。
    如果 `region_id=haidian` 却填 `month=2025-04`，后端会按海淀数据查找，
    这个月份不存在，所以返回 404 是合理的。

    获取某个 Patch 在指定任务下的结果图。

    支持 `png` 和 `npy` 两种格式。单期任务传 `month`；变化检测传
    `before_month` + `after_month`（或直接用 `period`）。

    单期任务（building_extraction、road_extraction、construction、
    land_use_classification、land_cover_classification、water_extraction）
    填写 month；变化检测 change_detection 填 period，或 before_month + after_month。

    时间格式统一为 `YYYY-MM`（如 `2025-04`、`2025-12`、`2026-05`），
    接口会自动兼容 `YYYYMM` 写法。

    | 区域 | 任务 | 版本 | 时间参数 | 可用时间范围 |
    |------|------|------|----------|--------------|
    | 哈尔滨 | change_detection | v1/v2 | period / before+after | 2025-04_vs_2025-06、2025-04_vs_2025-10、2025-06_vs_2025-08、2025-06_vs_2025-10、2025-08_vs_2025-09、2025-08_vs_2025-10、2025-09_vs_2025-10 |
    | 哈尔滨 | building_extraction | v1 | month | 2025-04 ~ 2025-10（仅 2025-10 预生成，其余实时推理） |
    | 哈尔滨 | building_extraction | v2 | period / before+after | 2025-04_vs_2025-06、2025-08_vs_2025-09、2025-09_vs_2025-10 |
    | 哈尔滨 | road_extraction | v1 | month | 2025-04 ~ 2025-10 |
    | 哈尔滨 | land_use_classification | v1 | month | 2025-04 ~ 2025-10（仅 2025-10 预生成，其余实时推理） |
    | 哈尔滨 | land_use_classification | v2 | period / before+after | 2025-04_vs_2025-06、2025-08_vs_2025-09、2025-09_vs_2025-10 |
    | 哈尔滨 | land_cover_classification | v1/v2 | month | 2025-04 ~ 2025-10（实时推理） |
    | 哈尔滨 | water_extraction | v1/v2 | month | 2025-04 ~ 2025-10（实时推理） |
    | 海淀 | building_extraction | v1 | month | 2025-12 ~ 2026-05 |
    | 海淀 | road_extraction | v1 | month（任意） | month 为 2025-12 ~ 2026-05 |
    | 海淀 | construction | v1 | month（任意） | month 为 2025-12 ~ 2026-05 |
    | 海淀 | land_use_classification | v1 | month | 2025-12 ~ 2026-05 |
    | 海淀 | land_cover_classification | v1 | month | 2025-12 ~ 2026-05 |
    | 海淀 | water_extraction | v1 | month | 2025-12 ~ 2026-05 |

    海淀 land_cover_classification V1 图例（整个结果集共 7 色，单个 Patch
    可能只出现其中一部分）：

    | 类别值 | 颜色 | 含义 |
    |----------|------|------|
    | 1 | #006400 | 树木覆盖 |
    | 2 | #B4D250 | 灌木地 |
    | 3 | #F5DC5A | 草地 |
    | 4 | #D23C3C | 耕地 |
    | 5 | #BEAA82 | 建成区 |
    | 6 | #A0DCDC | 裸地/稀疏植被 |
    | 8 | #1E64DC | 永久性水体 |

    注意：这是海淀项目当前 PNG 结果的实际调色板，不是 PCA 嵌入图
    的颜色，也不是 ESA WorldCover 原始 11 类的完整调色板。

    注意：Swagger UI 对二进制响应支持有限，建议在浏览器或 `<img>` 标签中查看图片。
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
                return FileResponse(path, media_type="image/png")
            # Fallback: per-patch tile image (results/.../tiles/patch_*.png)
            path = DataService.get_task_result_path(
                region_id, patch_id, task_type, "tile", version, effective_period
            )
            if path and path.lower().endswith(".png"):
                return FileResponse(path, media_type="image/png")

            # Last resort: run system model inference. For Haidian this uses the
            # latest v1 task heads from models/haidian/v1/task_heads.
            if class_month and is_system_task(task_type):
                tile_path = _resolve_classification_tile(
                    region_id, task_type, patch_id, class_month, version
                )
                if tile_path:
                    return FileResponse(tile_path, media_type="image/png")
    except DataValidationError as e:
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
    version: str = Query(
        "v1",
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
    version: str = Query(
        "v1",
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
    version: str = Query(
        "v1",
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

    raw_tiles = await TileService.list_available_tiles(region_id, task_type, version, period)

    # Include Haidian GT-overridden task tiles (construction, road_extraction)
    # whose PNGs live outside the standard results/tiles directory.
    raw_tiles.extend(_list_haidian_gt_tiles(region_id, task_type, version))

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
    version: str = Query(
        "v1",
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

    raise HTTPException(
        status_code=404,
        detail=f"Tile '{filename}' not found for task '{task_type}'",
    )


@router.get(
    "/regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png",
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
