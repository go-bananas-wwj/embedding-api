"""Downstream task router."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from app.config import get_config
from app.schemas.models import TasksResponse, TaskInfo, TaskSummary
from app.services.data_service import DataService
from app.services.tile_service import TileService

router = APIRouter()


@router.get("/regions/{region_id}/tasks", response_model=TasksResponse)
async def list_tasks(region_id: str):
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


@router.get("/regions/{region_id}/tasks/{task_type}/summary")
async def get_task_summary(
    region_id: str,
    task_type: str,
    version: str = Query("v1"),
    period: Optional[str] = Query(None),
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
    summary = DataService.load_task_summary(region_id, task_type, version, period)

    if not summary:
        raise HTTPException(
            status_code=404, detail=f"Summary not found for task '{task_type}'"
        )

    return TaskSummary(
        task=summary.get("task", task_type),
        name=task.get("name", task_type),
        version=version,
        period=summary.get("period") or period,
        grid_size=summary.get("grid_size"),
        total_polygons=summary.get("total_polygons"),
        total_patches=summary.get("total_patches"),
        positive_patches=summary.get("positive_patches"),
        negative_patches=summary.get("negative_patches"),
    )


@router.get("/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result")
async def get_task_result(
    region_id: str,
    patch_id: str,
    task_type: str,
    format: str = Query("png", description="Format: png, npy"),
    version: str = Query("v1"),
    period: Optional[str] = Query(None),
):
    """Get task result for a specific patch."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    patch = DataService.get_patch(region_id, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

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
        # PNG result
        path = DataService.get_task_result_path(
            region_id, patch_id, task_type, "png", version, period
        )
        if path and path.endswith(".png"):
            return FileResponse(path, media_type="image/png")

    raise HTTPException(
        status_code=404,
        detail=f"Result not found for patch '{patch_id}', task '{task_type}'",
    )


@router.get("/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction")
async def get_task_prediction(
    region_id: str,
    patch_id: str,
    task_type: str,
    version: str = Query("v1"),
    period: Optional[str] = Query(None),
):
    """Get raw prediction array (.npy) for a patch."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    patch = DataService.get_patch(region_id, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    path = DataService.get_task_result_path(
        region_id, patch_id, task_type, "npy", version, period
    )
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
    region_id: str,
    patch_id: str,
    task_type: str,
    version: str = Query("v1"),
    period: Optional[str] = Query(None),
):
    """Get label array (.npy) or metadata for a patch."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    patch = DataService.get_patch(region_id, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    path = DataService.get_task_result_path(
        region_id, patch_id, task_type, "label", version, period
    )
    if path:
        if path.endswith(".npy"):
            return FileResponse(
                path,
                media_type="application/octet-stream",
                filename=f"{patch_id}_{task_type}_label.npy",
            )
        elif path.endswith(".json"):
            return FileResponse(path, media_type="application/json")

    raise HTTPException(
        status_code=404,
        detail=f"Label not found for patch '{patch_id}', task '{task_type}'",
    )


@router.get("/regions/{region_id}/tasks/{task_type}/tiles")
async def list_tiles(
    region_id: str,
    task_type: str,
    version: str = Query("v1"),
    period: Optional[str] = Query(None),
):
    """List available tiles for a task."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    tiles = TileService.list_available_tiles(region_id, task_type, version, period)
    return {"tiles": tiles, "total": len(tiles)}


@router.get("/regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png")
async def get_tile(
    region_id: str,
    task_type: str,
    z: int,
    x: int,
    y: int,
    version: str = Query("v1"),
    period: Optional[str] = Query(None),
):
    """Serve a map tile image."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    path = TileService.get_tile_path(region_id, task_type, z, x, y, version, period)
    if path:
        return FileResponse(path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Tile not found")
