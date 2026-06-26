"""SAM3 interactive segmentation router."""

import logging

from fastapi import APIRouter, Body, HTTPException, Path

from app.config import get_config
from app.schemas.sam3 import (
    EmbedRequest,
    EmbedResponse,
    SegmentRequest,
    SegmentResponse,
    StatusResponse,
)
from app.services.sam3_service import SAM3Service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/regions/{region_id}/sam3/embed", response_model=EmbedResponse)
async def sam3_embed(
    region_id: str = Path(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    req: EmbedRequest = Body(...),
):
    """为指定 Patch 预计算 SAM3 图像嵌入。

    用于交互式分割前加载 Sentinel-2 影像并缓存图像嵌入。
    返回嵌入 ID 及缓存键，供后续 `/sam3/segment` 调用。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    try:
        result = await svc.embed(region_id, req.patch_id, req.month)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("SAM3 embed failed for %s/%s/%s", region_id, req.patch_id, req.month)
        raise HTTPException(status_code=503, detail="Model inference temporarily unavailable")


@router.post("/regions/{region_id}/sam3/segment", response_model=SegmentResponse)
async def sam3_segment(
    region_id: str = Path(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    req: SegmentRequest = Body(...),
):
    """基于缓存的嵌入和点提示进行 SAM3 实例分割。

    用于用户在前端点击影像上的点后获取对应地物的分割掩膜。
    返回一个或多个掩膜数组，可用于叠加显示或导出标注。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    try:
        masks = await svc.segment(
            req.embedding_id,
            req.point_coords,
            req.point_labels,
            req.multimask_output,
        )
        return {"masks": masks}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("SAM3 segment failed for embedding_id=%s", req.embedding_id)
        raise HTTPException(status_code=503, detail="Segmentation temporarily unavailable")


@router.get("/regions/{region_id}/sam3/status", response_model=StatusResponse)
async def sam3_status(
    region_id: str = Path(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
):
    """获取 SAM3 模型加载状态及缓存信息。

    用于页面初始化时检查模型是否就绪、当前缓存命中情况。
    返回模型状态、缓存大小和设备信息的 JSON。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    return svc.get_status()
