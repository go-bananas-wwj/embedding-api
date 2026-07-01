"""Downstream task router."""

from pathlib import Path
from typing import Literal, Optional
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
    "v1": {"summary": "V4-based results", "value": "v1"},
    "v2": {"summary": "V5-based results", "value": "v2"},
}

# Classification tasks whose visualizations are stored in xuannv_show static seg_tiles.
_XUANNV_SHOW_SEG_TILE_DIR = Path("/workspace/xuannv_show/static_assets/data/seg_tiles")
_CLASS_TASK_TO_XUANNV_HEAD = {
    "building_extraction": "building_extraction",
    "land_use_classification": "dynamic_world",
    "land_cover_classification": "worldcover",
    "water_extraction": "jrc_water",
}


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
    """
    head = _CLASS_TASK_TO_XUANNV_HEAD.get(task_type)
    if not head:
        return None

    static_path = _XUANNV_SHOW_SEG_TILE_DIR / head / month / f"{patch_id}.png"
    if static_path.exists():
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
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    task_type: str = PathParam(
        ...,
        description="Downstream task type.",
        examples=["change_detection"],
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
        description="Comparison period for time-series tasks, e.g. 2025-04_vs_2025-06.",
        examples=["2025-04_vs_2025-06"],
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
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    patch_id: str = PathParam(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    task_type: str = PathParam(
        ...,
        description="Downstream task type.",
        examples=["change_detection"],
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
        description="对比周期，变化检测任务使用，例如 2025-04_vs_2025-06。",
        examples=["2025-04_vs_2025-06"],
    ),
    month: Optional[str] = Query(
        None,
        description="单期任务（建筑物提取、土地利用分类等）的月份，例如 2025-04。",
        examples=["2025-04"],
    ),
    before_month: Optional[str] = Query(
        None,
        description="变化检测任务的起始月份，例如 2025-04。",
        examples=["2025-04"],
    ),
    after_month: Optional[str] = Query(
        None,
        description="变化检测任务的结束月份，例如 2025-06。",
        examples=["2025-06"],
    ),
):
    """获取某个 Patch 在指定任务下的结果。

    用于在地图上点击 Patch 后展示变化检测、建筑物提取等监测结果图片。
    支持 `png` 和 `npy` 两种格式，返回 PNG 图片或二进制 NumPy 数组。
    单期任务请传 `month`；变化检测任务请传 `before_month` 和 `after_month`。
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
            if task_type in _CLASS_TASK_TO_XUANNV_HEAD and class_month:
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

            # Last resort: run system model inference (harbin land cover / water)
            if task_type in _CLASS_TASK_TO_XUANNV_HEAD and class_month and is_system_task(task_type):
                tile_path = _resolve_classification_tile(
                    region_id, task_type, patch_id, class_month, version
                )
                if tile_path:
                    return FileResponse(tile_path, media_type="image/png")
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(
        status_code=404,
        detail=f"Result not found for patch '{patch_id}', task '{task_type}'",
    )


@router.get("/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction")
async def get_task_prediction(
    region_id: str = PathParam(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    patch_id: str = PathParam(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    task_type: str = PathParam(
        ...,
        description="Downstream task type.",
        examples=["change_detection"],
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
        description="Comparison period for time-series tasks, e.g. 2025-04_vs_2025-06.",
        examples=["2025-04_vs_2025-06"],
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
        detail=f"Prediction not found for patch '{patch_id}', task '{task_type}'",
    )


@router.get("/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label")
async def get_task_label(
    region_id: str = PathParam(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    patch_id: str = PathParam(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    task_type: str = PathParam(
        ...,
        description="Downstream task type.",
        examples=["change_detection"],
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
        description="Comparison period for time-series tasks, e.g. 2025-04_vs_2025-06.",
        examples=["2025-04_vs_2025-06"],
    ),
):
    """获取某个 Patch 的标签数据（.npy 或 .json）。

    用于对比模型预测与真值、计算精度或制作训练样本。
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
        detail=f"Label not found for patch '{patch_id}', task '{task_type}'",
    )


@router.get("/regions/{region_id}/tasks/{task_type}/tiles", response_model=TilesResponse)
async def list_tiles(
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
    """列出某任务下可用的瓦片文件。

    用于前端构建瓦片图层索引，如结果大图叠加在地图上。
    返回瓦片文件名、所属 Patch 及对比期信息的 JSON 列表。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    raw_tiles = await TileService.list_available_tiles(region_id, task_type, version, period)
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
