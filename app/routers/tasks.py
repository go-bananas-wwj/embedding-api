"""Downstream task router."""

from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Query, Path as PathParam
from fastapi.responses import FileResponse

from app.config import get_config
from app.schemas.models import (
    TasksResponse, TaskInfo, TaskSummary, TilesResponse, TileInfo, ErrorResponse,
)
from app.services.data_service import DataService, DataServiceError, DataValidationError
from app.services.tile_service import TileService

router = APIRouter()
TASK_FORMATS = Literal["png", "npy"]

_TASK_OPENAPI_EXAMPLES = {
    "change_detection": {"summary": "Change detection", "value": "change_detection"},
    "building_extraction": {"summary": "Building extraction", "value": "building_extraction"},
    "land_use_classification": {"summary": "Land use classification", "value": "land_use_classification"},
    "land_cover_classification": {"summary": "Land cover classification", "value": "land_cover_classification"},
    "water_extraction": {"summary": "Water extraction", "value": "water_extraction"},
}

_VERSION_OPENAPI_EXAMPLES = {
    "v1": {"summary": "V4-based results", "value": "v1"},
    "v2": {"summary": "V5-based results", "value": "v2"},
}


@router.get("/regions/{region_id}/tasks", response_model=TasksResponse)
async def list_tasks(
    region_id: str = PathParam(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
):
    """List available downstream tasks for a region."""
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
    """Get task summary statistics."""
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
        description="Comparison period for time-series tasks, e.g. 2025-04_vs_2025-06.",
        examples=["2025-04_vs_2025-06"],
    ),
):
    """Get task result for a specific patch."""
    if format not in ("png", "npy"):
        raise HTTPException(
            status_code=422, detail=f"Invalid format '{format}'. Use: png, npy"
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
                region_id, patch_id, task_type, "npy", version, period
            )
            if path:
                return FileResponse(
                    path,
                    media_type="application/octet-stream",
                    filename=f"{patch_id}_{task_type}_prediction.npy",
                )
        else:
            # Try summary image first, then fallback to per-patch tile
            path = DataService.get_task_result_path(
                region_id, patch_id, task_type, "png", version, period
            )
            if path and path.lower().endswith(".png"):
                return FileResponse(path, media_type="image/png")
            # Fallback: per-patch tile image (results/.../tiles/patch_*.png)
            path = DataService.get_task_result_path(
                region_id, patch_id, task_type, "tile", version, period
            )
            if path and path.lower().endswith(".png"):
                return FileResponse(path, media_type="image/png")
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
    """Get raw prediction array (.npy) for a patch."""
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
    """Get label array (.npy) or metadata for a patch."""
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
    """List available tiles for a task."""
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
    """Serve a map tile image."""
    raise HTTPException(
        status_code=501,
        detail="Tile serving is not yet implemented. Use /regions/{region_id}/tasks/{task_type}/tiles for available patch tiles.",
    )
