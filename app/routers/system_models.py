"""System pre-trained model inference routes.
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Path as PathParam
from fastapi.responses import FileResponse

from app.schemas.models import ErrorResponse
from app.services.auth_service import get_current_user
from app.services.data_service import DataValidationError
from app.services.system_model_service import (
    get_system_model_classes,
    infer_system_model,
    list_system_models,
)

router = APIRouter(prefix="/system-models", tags=["system-models"])

_SYSTEM_TASK_OPENAPI_EXAMPLES = {
    "building_extraction": {"summary": "Building extraction", "value": "building_extraction"},
    "land_cover_classification": {"summary": "Land cover classification", "value": "land_cover_classification"},
    "water_extraction": {"summary": "Water extraction", "value": "water_extraction"},
}


@router.get("")
async def get_system_models(
    region_id: str = Query(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """List system pre-trained models available for a region."""
    return list_system_models(region_id)


@router.get("/{task_id}/classes")
async def get_classes(
    task_id: str = PathParam(
        ...,
        description="System model task identifier.",
        examples=["building_extraction"],
        openapi_examples=_SYSTEM_TASK_OPENAPI_EXAMPLES,
    ),
    region_id: str = Query(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    version: str = Query(
        "v2",
        description="Model checkpoint version. Allowed values: v1, v2.",
        examples=["v2"],
        openapi_examples={
            "v1": {"summary": "V4-based checkpoint", "value": "v1"},
            "v2": {"summary": "V5-based checkpoint", "value": "v2"},
        },
    ),
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
    task_id: str = PathParam(
        ...,
        description="System model task identifier.",
        examples=["building_extraction"],
        openapi_examples=_SYSTEM_TASK_OPENAPI_EXAMPLES,
    ),
    region_id: str = Query(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    patch_id: str = Query(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    month: str = Query(
        ...,
        description="Month for the source embedding, e.g. 2025-04.",
        examples=["2025-04"],
    ),
    version: str = Query(
        "v2",
        description="Model checkpoint version. Allowed values: v1, v2.",
        examples=["v2"],
        openapi_examples={
            "v1": {"summary": "V4-based checkpoint", "value": "v1"},
            "v2": {"summary": "V5-based checkpoint", "value": "v2"},
        },
    ),
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
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = result_path.name
    return {"result_url": f"/system-models/results/{filename}"}


@router.get("/results/{filename}")
async def get_result(
    filename: str = PathParam(
        ...,
        description="System model inference result filename returned by POST /system-models/{task_id}/infer.",
        examples=["building_extraction_harbin_patch_000000_2025-04.png"],
    ),
    user: dict = Depends(get_current_user),
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
