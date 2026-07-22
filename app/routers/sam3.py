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
from app.services.data_service import DataNotFoundError, DataService, DataValidationError
from app.services.sam3_service import SAM3Service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/regions/{region_id}/sam3/embed", response_model=EmbedResponse)
async def sam3_embed(
    region_id: str = Path(
        ...,
        description=(
            "区域 ID。必填。可选值：harbin 或 haidian。"
            "用于指定从哪个区域的数据中查找 patch 和影像。"
        ),
        examples=["harbin"],
    ),
    req: EmbedRequest = Body(...),
):
    """预计算并缓存指定 Patch 的 SAM3 图像嵌入，减少后续点选分割等待时间。

    返回实际使用的影像日期；分割接口会自动复用缓存，无需前端传 `embedding_id`。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    if DataService.get_patch(region_id, req.patch_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patch '{req.patch_id}' not found in region '{region_id}'",
        )

    svc = SAM3Service()
    try:
        result = await svc.embed(region_id, req.patch_id, req.month, req.sensor_type)
        return result
    except (DataNotFoundError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (DataValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("SAM3 embed failed for %s/%s/%s", region_id, req.patch_id, req.month)
        raise HTTPException(status_code=503, detail="Model inference temporarily unavailable")


@router.post("/regions/{region_id}/sam3/segment", response_model=SegmentResponse)
async def sam3_segment(
    region_id: str = Path(
        ...,
        description=(
            "区域 ID。必填。可选值：harbin 或 haidian。"
            "后端会在该区域内根据点位自动定位 patch。"
        ),
        examples=["harbin"],
    ),
    req: SegmentRequest = Body(
        ...,
        openapi_examples={
            "frontend_default": {
                "summary": "前端推荐默认写法",
                "description": "只传日期、传感器和 WGS84 点击点；后端默认所有点都是前景点。",
                "value": {
                    "date": "202512",
                    "sensor_type": "s2",
                    "point_coords": [[116.0954, 40.0628]],
                    "multimask_output": False,
                    "include_masks": False,
                },
            },
            "with_masks": {
                "summary": "需要返回 mask PNG",
                "description": "include_masks=true 会额外返回 base64 PNG mask，响应体更大。",
                "value": {
                    "date": "202512",
                    "sensor_type": "s2",
                    "point_coords": [[116.0954, 40.0628]],
                    "include_masks": True,
                },
            },
            "multi_point": {
                "summary": "多个前景点",
                "description": "多个点会共同约束同一个目标；不传 point_labels 时全部默认为 1。",
                "value": {
                    "date": "202512",
                    "sensor_type": "s2",
                    "point_coords": [
                        [116.0954, 40.0628],
                        [116.0956, 40.0627],
                    ],
                },
            },
        },
    ),
):
    """根据 WGS84 点击点完成实例分割，返回 WGS84 GeoJSON 多边形。

    后端会自动定位 Patch、选择当月最新影像并复用缓存；日期字段只使用 `date`，
    不再接受 `month`。`highres` 影像必须带 CRS 和仿射变换。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    try:
        point_labels = req.point_labels or [1] * len(req.point_coords)
        return await svc.segment_geojson(
            region_id,
            req.date,
            req.sensor_type,
            req.point_coords,
            point_labels,
            req.multimask_output,
            include_masks=req.include_masks,
        )
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (DataValidationError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("SAM3 segment failed for region_id=%s", region_id)
        raise HTTPException(status_code=503, detail="Segmentation temporarily unavailable")


@router.get("/regions/{region_id}/sam3/status", response_model=StatusResponse)
async def sam3_status(
    region_id: str = Path(
        ...,
        description=(
            "区域 ID。必填。可选值：harbin 或 haidian。"
            "用于查看该服务实例当前 SAM3 模型和缓存状态。"
        ),
        examples=["harbin"],
    )
):
    """获取 SAM3 模型加载状态及缓存信息。

    用于页面初始化时检查模型是否就绪、当前缓存命中情况。
    返回模型状态、缓存大小和设备信息的 JSON。

    字段说明：

    | 字段 | 含义 |
    | --- | --- |
    | `model_loaded` | SAM3 模型是否已经加载 |
    | `device` | 当前推理设备，通常是 `cuda` 或 `cpu` |
    | `gpu_memory` | GPU 显存占用，单位 MB |
    | `cache.size` | 当前缓存的 embedding 数量 |
    | `cache.max_size` | 缓存上限 |
    | `cache.entries` | 已缓存的 embedding ID 列表 |
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    return svc.get_status()
