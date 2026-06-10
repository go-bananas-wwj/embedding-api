"""Embedding service router."""

import asyncio
import os
from typing import Literal
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import numpy as np

from app.config import get_config
from app.schemas.models import EmbeddingStats, ErrorResponse
from app.services.data_service import DataService, DataServiceError, DataValidationError, _check_file_size

router = APIRouter()

EMB_FORMATS = Literal["png", "npy", "json"]


def _load_image_array(path: str):
    """Load image and convert to numpy array (for use in thread pool)."""
    from PIL import Image
    img = Image.open(path)
    return np.array(img)


def _load_npy_array(path: str):
    """Load numpy array (for use in thread pool)."""
    return np.load(path, allow_pickle=False)


@router.get(
    "/regions/{region_id}/patches/{patch_id}/embedding",
    responses={
        200: {
            "description": "Embedding data",
            "content": {
                "image/png": {},
                "application/octet-stream": {},
                "application/json": {"model": EmbeddingStats},
            },
        },
        404: {"model": ErrorResponse},
        406: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_embedding(
    region_id: str,
    patch_id: str,
    format: str = Query("png", description="Output format: png, npy, json, cache"),
):
    """Get embedding data for a patch.

    - `png`: Returns visualization image
    - `npy`: Returns raw embedding array (application/octet-stream)
    - `json`: Returns embedding statistics
    - `cache`: Falls back to available format (PNG preferred)
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
    except DataValidationError as e:
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
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not emb_path:
        raise HTTPException(
            status_code=404, detail=f"Embedding not found for patch '{patch_id}'"
        )

    # Check file size before loading
    try:
        _check_file_size(emb_path)
    except DataServiceError as e:
        raise HTTPException(status_code=413, detail=str(e))

    if format == "json":
        if emb_path.endswith(".npy"):
            try:
                arr = await asyncio.to_thread(_load_npy_array, emb_path)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404, detail="Embedding file no longer exists"
                )
            except (OSError, ValueError) as e:
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
            try:
                img_arr = await asyncio.to_thread(_load_image_array, emb_path)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404, detail="Image file no longer exists"
                )
            except (OSError, ValueError) as e:
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
        # Format mismatch: requested NPY but only image available
        raise HTTPException(
            status_code=406,
            detail=f"NPY format not available for this patch. Available: {os.path.basename(emb_path)}",
        )
    else:
        # png or cache or any format - serve as image if possible
        if emb_path.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")):
            media_type = "image/png" if emb_path.lower().endswith(".png") else "image/jpeg"
            return FileResponse(emb_path, media_type=media_type)
        # Format mismatch: requested image but only NPY available
        raise HTTPException(
            status_code=406,
            detail=f"Image format not available for this patch. Available: {os.path.basename(emb_path)}",
        )
