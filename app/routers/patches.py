"""Patch management router."""

import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Path
from app.config import get_config
from app.schemas.models import PaginatedPatchesResponse, PatchDetail
from app.services.data_service import DataService, DataServiceError

router = APIRouter()


@router.get("/regions/{region_id}/patches", response_model=PaginatedPatchesResponse)
async def list_patches(
    region_id: str = Path(
        ...,
        description="Region identifier. Use 'harbin' (Harbin New Area) or 'haidian' (Haidian District).",
        examples=["harbin"],
        openapi_examples={
            "harbin": {"summary": "Harbin New Area", "value": "harbin"},
            "haidian": {"summary": "Haidian District", "value": "haidian"},
        },
    ),
    page: int = Query(1, ge=1, description="Page number (1-based).", examples=[1]),
    page_size: int = Query(
        20,
        ge=1,
        le=100,
        description="Number of patches per page (1-100).",
        examples=[20],
    ),
    bbox: Optional[str] = Query(
        None,
        description="Optional bounding box filter in WGS84: minx,miny,maxx,maxy.",
        examples=["126.5,45.74,126.55,45.76"],
    ),
):
    """List patches with pagination and optional bbox filtering."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    try:
        patches, total = DataService.list_patches(region_id, page, page_size, bbox)
    except DataServiceError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Offload batch enrichment to thread pool to avoid blocking the event loop
    enriched = await asyncio.to_thread(DataService.enrich_patches, region_id, patches)
    patch_details = [PatchDetail(**p) for p in enriched]

    has_next = total > page * page_size
    return PaginatedPatchesResponse(
        total=total, page=page, page_size=page_size, has_next=has_next, patches=patch_details
    )


@router.get("/regions/{region_id}/patches/{patch_id}", response_model=PatchDetail)
async def get_patch(
    region_id: str = Path(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    patch_id: str = Path(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
):
    """Get detailed information for a single patch."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    try:
        patch = DataService.get_patch(region_id, patch_id)
    except DataServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    # Offload enrichment to thread pool
    enriched = await asyncio.to_thread(
        DataService.enrich_patches, region_id, [patch]
    )
    return PatchDetail(**enriched[0])
