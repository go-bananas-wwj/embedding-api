"""Embedding service router."""

import asyncio
from typing import Literal
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import numpy as np

from app.config import get_config
from app.schemas.models import EmbeddingStats
from app.services.data_service import DataService, DataServiceError

router = APIRouter()

EMB_FORMATS = Literal["png", "npy", "json"]


@router.get("/regions/{region_id}/patches/{patch_id}/embedding")
async def get_embedding(
    region_id: str,
    patch_id: str,
    format: str = Query("png", description="Output format: png, npy, json"),
):
    """Get embedding data for a patch.

    - `png`: Returns visualization image
    - `npy`: Returns raw embedding array (application/octet-stream)
    - `json`: Returns embedding statistics
    """
    if format not in ("png", "npy", "json", "cache"):
        raise HTTPException(
            status_code=422, detail=f"Invalid format '{format}'. Use: png, npy, json"
        )

    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    try:
        patch = DataService.get_patch(region_id, patch_id)
    except DataServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    # Resolve embedding path
    try:
        emb_path = DataService.get_embedding_path(region_id, patch_id, format)
        if not emb_path:
            # Try alternative formats for fallback
            for alt_fmt in ("png", "npy", "cache"):
                emb_path = DataService.get_embedding_path(region_id, patch_id, alt_fmt)
                if emb_path:
                    break
    except DataServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not emb_path:
        raise HTTPException(
            status_code=404, detail=f"Embedding not found for patch '{patch_id}'"
        )

    if format == "json":
        if emb_path.endswith(".npy"):
            try:
                arr = await asyncio.to_thread(np.load, emb_path, allow_pickle=False)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404, detail=f"Embedding file no longer exists"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load embedding: {e}"
                )
            return EmbeddingStats(
                patch_id=patch_id,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
                min=float(arr.min()),
                max=float(arr.max()),
                mean=float(arr.mean()),
            )
        else:
            from PIL import Image
            try:
                img = await asyncio.to_thread(Image.open, emb_path)
                img_arr = np.array(img)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404, detail=f"Image file no longer exists"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load image: {e}"
                )
            return EmbeddingStats(
                patch_id=patch_id,
                shape=list(img_arr.shape),
                dtype=str(img_arr.dtype),
                min=float(img_arr.min()),
                max=float(img_arr.max()),
                mean=float(img_arr.mean()),
            )
    elif format == "npy":
        if emb_path.endswith(".npy"):
            return FileResponse(
                emb_path,
                media_type="application/octet-stream",
                filename=f"{patch_id}_embedding.npy",
            )
        raise HTTPException(
            status_code=404,
            detail=f"NPY embedding not available for patch '{patch_id}' (found {emb_path})",
        )
    else:
        # png or cache or any format - serve as image if possible
        if emb_path.endswith((".png", ".jpg", ".jpeg")):
            media_type = "image/png" if emb_path.endswith(".png") else "image/jpeg"
            return FileResponse(emb_path, media_type=media_type)
        raise HTTPException(
            status_code=404,
            detail=f"Image embedding not available for patch '{patch_id}' (found {emb_path})",
        )
