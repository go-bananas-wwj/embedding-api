"""Embedding service router."""

import asyncio
import json
import logging
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
from app.services.aef_pca_service import (
    AEF_YEAR,
    PCA_VERSION,
    AefEmbeddingNotFound,
    AefPcaError,
    InvalidPatchId,
    get_or_create_pca_png,
    response_etag,
)
from app.services.time_utils import is_valid_month_or_date

router = APIRouter()
logger = logging.getLogger(__name__)

EMB_FORMATS = Literal["png", "npy", "json"]

# Maximum array elements to prevent .npy header-based memory bomb attacks.
# A malicious .npy can declare a huge shape in its header while the file
# itself is tiny, causing np.load() to allocate PBs of RAM.
MAX_NPY_ELEMENTS = 500_000_000  # ~4GB for float32, ~2GB for float64

# Maximum decompressed image pixels to prevent PIL decompression bombs.
MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP


@router.get(
    "/regions/haidian/patches/{patch_id}/embeddings/aef/pca",
    summary="获取海淀 AEF 2025 PCA 可视化",
    responses={
        200: {
            "description": "AEF 2025 年度 embedding 的全域统一 PCA 彩色图。",
            "content": {"image/png": {}},
        },
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_haidian_aef_pca(
    patch_id: str = Path(
        ...,
        description=(
            "海淀 Patch ID。格式必须为 `patch_` 加六位数字；"
            "默认联调示例填写 `patch_000106`。"
        ),
        examples=["patch_000106"],
    ),
):
    """返回海淀 AEF 2025 年度 embedding 的 PCA 彩色 PNG。

    仅支持海淀区，数据固定为本地 AEF 2025 年年度 embedding。
    调用方只需填写 `patch_id`，不需要月份、年份、版本或格式参数。

    全部 Patch 使用同一个海淀全域 PCA 模型和统一的 2%~98% 显示范围，
    因此不同 Patch 的颜色可以直接比较。响应为 `image/png`，浏览器和
    前端 `<img>` 标签可以直接显示。
    """
    try:
        png_path = await asyncio.to_thread(get_or_create_pca_png, patch_id)
    except InvalidPatchId as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except AefEmbeddingNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (OSError, ValueError, AefPcaError):
        logger.exception("Failed to render Haidian AEF PCA")
        raise HTTPException(
            status_code=500,
            detail="AEF embedding exists but could not be visualized",
        )
    return FileResponse(
        png_path,
        media_type="image/png",
        filename=f"haidian_{patch_id}_aef_{AEF_YEAR}_pca.png",
        content_disposition_type="inline",
        headers={
            "Cache-Control": "public, max-age=86400",
            "ETag": response_etag(png_path),
            "X-Embedding-Source": "AEF",
            "X-Embedding-Year": AEF_YEAR,
            "X-Patch-Id": patch_id,
            "X-PCA-Version": PCA_VERSION,
        },
    )


def _default_embedding_version(region_id: str) -> str:
    """Return the single frontend default embedding for each region."""
    return "v1" if region_id == "haidian" else "v2"


def _latest_available_month(months):
    """Return the latest month from DataService's chronological list."""
    return months[-1] if months else None


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


def _load_embedding_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        description=(
            "Embedding 版本，通常不需要填写。省略时海淀自动使用 P10C（API `v1`），"
            "哈尔滨自动使用 V5（API `v2`）。"
        ),
        openapi_examples={
            "haidian_p10c": {"summary": "海淀 P10C（API v1）", "value": "v1"},
            "harbin_v5": {"summary": "哈尔滨 V5（API v2）", "value": "v2"},
        },
    ),
    month: Optional[str] = Query(
        None,
        description=(
            "Embedding 月份，可省略；省略时自动选择该区域最新可用月份。"
            "哈尔滨 V5：`2025-04` 至 `2026-05`，其中 2025 年支持 04、06、08、09、10，"
            "2026 年支持 01~05。海淀 P10C：`2025-12` 至 `2026-05`。"
            "两个区域均兼容 `YYYYMM` 和 `YYYY-MM`。"
        ),
        openapi_examples={
            "harbin_v5": {"summary": "哈尔滨 V5", "value": "202510"},
            "haidian_p10c": {"summary": "海淀 P10C", "value": "202512"},
            "hyphen": {"summary": "带横杠写法", "value": "2026-05"},
        },
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

    if month is not None and not is_valid_month_or_date(month):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid month '{month}'. Use a real calendar month/date in "
                "YYYYMM, YYYY-MM, or YYYYMMDD format."
            ),
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

    effective_version = version or _default_embedding_version(region_id)

    # If omitted, use the latest month available for this region and patch.
    effective_month = month
    if not effective_month:
        available_months = DataService.get_available_months(region_id, patch_id)
        effective_month = _latest_available_month(available_months)

    # Resolve embedding path
    try:
        emb_path = DataService.get_embedding_path(
            region_id, patch_id, format, version=effective_version, month=effective_month
        )
        if not emb_path:
            # Try alternative formats for fallback
            fallback_formats = (
                ("npy", "png", "cache")
                if format == "json"
                else ("png", "npy", "cache")
            )
            for alt_fmt in fallback_formats:
                if alt_fmt == format:
                    continue
                alt_path = DataService.get_embedding_path(
                    region_id,
                    patch_id,
                    alt_fmt,
                    version=effective_version,
                    month=effective_month,
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
        detail_msg += f", version '{effective_version}'"
        raise HTTPException(status_code=404, detail=detail_msg)

    # Check file size before loading
    try:
        _check_file_size(emb_path)
    except DataServiceError as e:
        raise HTTPException(status_code=413, detail=str(e))

    if format == "json":
        try:
            if emb_path.endswith(".json"):
                meta = await asyncio.to_thread(_load_embedding_json, emb_path)
                return EmbeddingStats(
                    patch_id=meta.get("patch_id", patch_id),
                    shape=meta.get("shape", []),
                    dtype=str(meta.get("dtype", "")),
                    min=float(meta.get("min", 0.0)),
                    max=float(meta.get("max", 0.0)),
                    mean=float(meta.get("mean", 0.0)),
                )
            elif emb_path.endswith(".npy"):
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
