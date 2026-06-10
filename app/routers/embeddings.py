"""Embedding service router."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import numpy as np

from app.config import get_config
from app.schemas.models import EmbeddingStats
from app.services.data_service import DataService

router = APIRouter()


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
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    patch = DataService.get_patch(region_id, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail=f"Patch '{patch_id}' not found")

    # Resolve embedding path
    emb_path = DataService.get_embedding_path(region_id, patch_id, format)
    if not emb_path:
        # Try alternative formats
        for alt_format in ["png", "npy", "cache"]:
            emb_path = DataService.get_embedding_path(region_id, patch_id, alt_format)
            if emb_path:
                break

    if not emb_path:
        raise HTTPException(
            status_code=404, detail=f"Embedding not found for patch '{patch_id}'"
        )

    if format == "json":
        # Return statistics
        if emb_path.endswith(".npy"):
            arr = np.load(emb_path)
            return EmbeddingStats(
                patch_id=patch_id,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
                min=float(arr.min()),
                max=float(arr.max()),
                mean=float(arr.mean()),
            )
        else:
            # For PNG, return basic info
            from PIL import Image

            img = Image.open(emb_path)
            return EmbeddingStats(
                patch_id=patch_id,
                shape=[img.height, img.width, len(img.getbands())],
                dtype="uint8",
                min=0.0,
                max=255.0,
                mean=128.0,
            )
    elif format == "npy":
        # Return .npy file
        npy_path = DataService.get_embedding_path(region_id, patch_id, "npy")
        if npy_path and npy_path.endswith(".npy"):
            return FileResponse(
                npy_path,
                media_type="application/octet-stream",
                filename=f"{patch_id}_embedding.npy",
            )
        raise HTTPException(
            status_code=404, detail=f"NPY embedding not available for patch '{patch_id}'"
        )
    else:
        # Return PNG (default)
        png_path = DataService.get_embedding_path(region_id, patch_id, "png")
        if png_path and png_path.endswith(".png"):
            return FileResponse(png_path, media_type="image/png")
        raise HTTPException(
            status_code=404, detail=f"PNG embedding not available for patch '{patch_id}'"
        )
