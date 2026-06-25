"""System pre-trained model inference routes.
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.schemas.models import ErrorResponse
from app.services.auth_service import get_current_user
from app.services.system_model_service import (
    get_system_model_classes,
    infer_system_model,
    list_system_models,
)

router = APIRouter(prefix="/system-models", tags=["system-models"])


@router.get("")
async def get_system_models(region_id: str, user: dict = Depends(get_current_user)) -> List[dict]:
    """List system pre-trained models available for a region."""
    return list_system_models(region_id)


@router.get("/{task_id}/classes")
async def get_classes(
    task_id: str,
    region_id: str,
    version: str = "v2",
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """Get class definitions for a system model."""
    try:
        return get_system_model_classes(region_id, task_id, version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{task_id}/infer")
async def infer(
    task_id: str,
    region_id: str,
    patch_id: str,
    month: str,
    version: str = "v2",
    user: dict = Depends(get_current_user),
) -> dict:
    """Run a system pre-trained model on a single patch."""
    results_dir = Path(f"users/{user['user_id']}/system_model_results")
    try:
        result_path = infer_system_model(
            region_id, task_id, patch_id, month, version, results_dir
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = result_path.name
    return {"result_url": f"/system-models/results/{filename}"}


@router.get("/results/{filename}")
async def get_result(
    filename: str, user: dict = Depends(get_current_user)
) -> FileResponse:
    """Serve a system model inference result PNG."""
    results_dir = Path(f"users/{user['user_id']}/system_model_results")
    file_path = results_dir / filename

    try:
        resolved = file_path.resolve()
        base_resolved = results_dir.resolve()
        resolved.relative_to(base_resolved)
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(file_path, media_type="image/png")
