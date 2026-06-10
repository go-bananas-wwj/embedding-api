"""Patch management router."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.config import load_patches_meta
from app.schemas.models import PaginatedPatchesResponse, PatchDetail
from app.services.data_service import DataService

router = APIRouter()


@router.get("/regions/{region_id}/patches", response_model=PaginatedPatchesResponse)
async def list_patches(
    region_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    bbox: Optional[str] = Query(None, description="Bounding box: minx,miny,maxx,maxy"),
):
    """List patches with pagination and optional bbox filtering."""
    from app.config import get_config

    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    patches, total = DataService.list_patches(region_id, page, page_size, bbox)

    patch_details = []
    for p in patches:
        patch_id = p.get("patch_id", "")
        patch_details.append(
            PatchDetail(
                patch_id=patch_id,
                bounds_wgs84=p.get("bounds_wgs84", []),
                bounds=p.get("bounds"),
                crs=p.get("crs"),
                sources=p.get("sources", {}),
                time_range=p.get("time_range", []),
                has_embedding=DataService.has_embedding(region_id, patch_id),
                available_tasks=DataService.get_available_tasks(region_id, patch_id),
            )
        )

    return PaginatedPatchesResponse(
        total=total, page=page, page_size=page_size, patches=patch_details
    )


@router.get("/regions/{region_id}/patches/{patch_id}")
async def get_patch(region_id: str, patch_id: str):
    """Get detailed information for a single patch."""
    from app.config import get_config

    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    patch = DataService.get_patch(region_id, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    return PatchDetail(
        patch_id=patch_id,
        bounds_wgs84=patch.get("bounds_wgs84", []),
        bounds=patch.get("bounds"),
        crs=patch.get("crs"),
        sources=patch.get("sources", {}),
        time_range=patch.get("time_range", []),
        has_embedding=DataService.has_embedding(region_id, patch_id),
        available_tasks=DataService.get_available_tasks(region_id, patch_id),
    )
