"""Embedding service router."""

import asyncio
import os
import re
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import FileResponse
import numpy as np

from app.config import get_config
from app.schemas.models import EmbeddingStats, ErrorResponse
from app.services.data_service import (
    DataService, DataServiceError, DataValidationError, _check_file_size,
)

router = APIRouter()

EMB_FORMATS = Literal["png", "npy", "json"]

# Maximum array elements to prevent .npy header-based memory bomb attacks.
# A malicious .npy can declare a huge shape in its header while the file
# itself is tiny, causing np.load() to allocate PBs of RAM.
MAX_NPY_ELEMENTS = 500_000_000  # ~4GB for float32, ~2GB for float64

# Maximum decompressed image pixels to prevent PIL decompression bombs.
MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP


def _load_image_array(path: str):
    """Load image and convert to numpy array (for use in thread pool).

    Protects against decompression bombs by limiting MAX_IMAGE_PIXELS.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    with Image.open(path) as img:
        width, height = img.size
        pixels = width * height
        if pixels > MAX_IMAGE_PIXELS:
            raise DataServiceError(
                f"Image too large: {width}x{height} = {pixels} pixels "
                f"(max {MAX_IMAGE_PIXELS})"
            )
        return np.array(img)


def _load_npy_array(path: str):
    """Load numpy array with memory-bomb protection.

    A malicious .npy can have a tiny file size but declare a huge shape
    in its header. np.load() allocates memory based on the header, not
    the file size. We validate the actual element count after loading.
    """
    arr = np.load(path, allow_pickle=False)
    elements = arr.size
    if elements > MAX_NPY_ELEMENTS:
        raise DataServiceError(
            f"Array too large: {elements} elements (max {MAX_NPY_ELEMENTS})"
        )
    return arr


def _load_npz_embedding(path: str):
    """Load embedding array from .npz archive (used by haidian v1).

    Expects the archive to contain an 'embedding' key with a numpy array.
    """
    data = np.load(path, allow_pickle=False)
    if "embedding" not in data:
        raise DataServiceError(f"NPZ file missing 'embedding' key: {path}")
    arr = data["embedding"]
    elements = arr.size
    if elements > MAX_NPY_ELEMENTS:
        raise DataServiceError(
            f"Array too large: {elements} elements (max {MAX_NPY_ELEMENTS})"
        )
    return arr


@router.get(
    "/regions/{region_id}/patches/{patch_id}/embedding",
    responses={
        200: {
            "description": "Embedding data",
            "content": {
                "image/png": {},
                "application/octet-stream": {},
                "application/json": {"schema": EmbeddingStats.model_json_schema()},
            },
        },
        404: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_embedding(
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
    format: str = Query(
        "png",
        description="Output format. Allowed values: png, npy, json, cache.",
        examples=["png"],
        openapi_examples={
            "png": {"summary": "PNG visualization", "value": "png"},
            "npy": {"summary": "Raw NumPy array", "value": "npy"},
            "json": {"summary": "Array statistics", "value": "json"},
            "cache": {"summary": "Cache fallback", "value": "cache"},
        },
    ),
    version: Optional[str] = Query(
        None,
        description="Embedding version. Allowed values: v1 (V4 model), v2 (V5 model).",
        examples=["v1"],
        openapi_examples={
            "v1": {"summary": "V4 embedding model", "value": "v1"},
            "v2": {"summary": "V5 embedding model", "value": "v2"},
        },
    ),
    month: Optional[str] = Query(
        None,
        description="Month for time-series embeddings, e.g. 2025-04. Falls back to the first available month if omitted.",
        examples=["2025-04"],
    ),
):
    """获取指定 Patch 的嵌入数据。

    用于模型可视化、特征分析或作为下游任务的输入。
    支持 `png`、`npy`、`json`、`cache` 四种格式，分别返回图片、原始数组、统计信息或缓存回退结果。
    注意：`png`/`npy` 为二进制响应，Swagger UI 可能无法直接渲染，建议使用 curl、浏览器或 `<img>` 标签访问。
    """
    if format not in ("png", "npy", "json", "cache"):
        raise HTTPException(
            status_code=422, detail=f"Invalid format '{format}'. Use: png, npy, json, cache"
        )

    if month is not None and not re.match(r"^[\w\-]{1,32}$", month):
        raise HTTPException(
            status_code=422, detail=f"Invalid month format: '{month}'"
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

    # If month not provided, try first available month (backward compat)
    effective_month = month
    if not effective_month:
        available_months = DataService.get_available_months(region_id, patch_id)
        if available_months:
            effective_month = available_months[0]

    # Resolve embedding path
    try:
        emb_path = DataService.get_embedding_path(
            region_id, patch_id, format, version=version, month=effective_month
        )
        if not emb_path:
            # Try alternative formats for fallback
            for alt_fmt in ("png", "npy", "cache"):
                if alt_fmt == format:
                    continue
                alt_path = DataService.get_embedding_path(
                    region_id, patch_id, alt_fmt, version=version, month=effective_month
                )
                if alt_path:
                    emb_path = alt_path
                    break
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not emb_path:
        detail_msg = f"Embedding not found for patch '{patch_id}'"
        if month:
            detail_msg += f", month '{month}'"
        if version:
            detail_msg += f", version '{version}'"
        raise HTTPException(status_code=404, detail=detail_msg)

    # Check file size before loading
    try:
        _check_file_size(emb_path)
    except DataServiceError as e:
        raise HTTPException(status_code=413, detail=str(e))

    if format == "json":
        try:
            if emb_path.endswith(".npy"):
                arr = await asyncio.to_thread(_load_npy_array, emb_path)
            elif emb_path.endswith(".npz"):
                arr = await asyncio.to_thread(_load_npz_embedding, emb_path)
            else:
                img_arr = await asyncio.to_thread(_load_image_array, emb_path)
                return EmbeddingStats(
                    patch_id=patch_id,
                    shape=list(img_arr.shape),
                    dtype=str(img_arr.dtype),
                    min=float(img_arr.min()),
                    max=float(img_arr.max()),
                    mean=float(img_arr.mean()),
                )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail="Embedding file no longer exists"
            )
        except (OSError, ValueError, EOFError, DataServiceError):
            logger.exception("Failed to load embedding")
            raise HTTPException(
                status_code=500, detail="Failed to load embedding"
            )
        return EmbeddingStats(
            patch_id=patch_id,
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            min=float(arr.min()),
            max=float(arr.max()),
            mean=float(arr.mean()),
        )
    elif format == "npy":
        if emb_path.endswith(".npy"):
            return FileResponse(
                emb_path,
                media_type="application/octet-stream",
                filename=f"{patch_id}_embedding.npy",
            )
        elif emb_path.endswith(".npz"):
            # Extract embedding from NPZ and return as NPY stream
            try:
                arr = await asyncio.to_thread(_load_npz_embedding, emb_path)
            except (OSError, ValueError, DataServiceError) as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to load NPZ: {e}"
                )
            import io
            buf = io.BytesIO()
            np.save(buf, arr)
            buf.seek(0)
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                buf,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{patch_id}_embedding.npy"'},
            )
        # Format not available for this patch — return 404 with hint
        raise HTTPException(
            status_code=404,
            detail=f"NPY format not available for this patch",
            headers={"X-Available-Format": "png"},
        )
    else:
        # png or cache or any format - serve as image if possible
        if emb_path.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")):
            media_type = "image/png" if emb_path.lower().endswith(".png") else "image/jpeg"
            return FileResponse(emb_path, media_type=media_type)
        # For NPZ files, PNG is not pre-generated — return 404 with hint
        if emb_path.endswith(".npz"):
            raise HTTPException(
                status_code=404,
                detail=f"PNG format not pre-generated for this patch",
                headers={"X-Available-Format": "npy"},
            )
        # Format not available — return 404 with hint
        raise HTTPException(
            status_code=404,
            detail=f"Image format not available for this patch",
            headers={"X-Available-Format": "npy"},
        )
