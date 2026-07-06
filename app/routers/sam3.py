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
    """为指定 Patch 预计算 SAM3 图像嵌入。

    适用场景：前端希望先展示某个 patch 的遥感影像，并提前把 SAM3
    embedding 缓存在后端，减少后续点选分割等待时间。

    参数填写说明：

    | 参数 | 默认值/范围 | 怎么填 |
    | --- | --- | --- |
    | `region_id` | `harbin` / `haidian` | 路径参数，选择区域 |
    | `patch_id` | 格式 `patch_000000` | 要预加载的 patch 编号 |
    | `month` | `YYYY-MM` / `YYYYMM` / `YYYYMMDD` | 要加载的影像日期或月份 |
    | `sensor_type` | 默认 `s2`；可选 `s2`、`s1`、`landsat` | 前端普通光学预览建议填 `s2` |

    返回值包含 `embedding_id` 和 base64 PNG 影像；`/sam3/segment` 会自动
    复用缓存，不要求前端手动传 `embedding_id`。
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
    """基于 WGS84 点提示进行 SAM3 实例分割。

    前端只需要传用户点击的 WGS84 经纬度点和影像日期；后端会自动定位
    patch、加载对应传感器影像、计算或复用 SAM3 embedding，并返回 WGS84
    GeoJSON 标注框。

    参数填写说明：

    | 参数 | 必填 | 默认值/范围 | 怎么填 |
    | --- | --- | --- | --- |
    | `region_id` | 是 | `harbin` / `haidian` | 路径参数，选择在哪个区域内分割 |
    | `date` | 是 | `YYYY-MM` / `YYYYMM` / `YYYYMMDD` | 影像日期或月份，如 `202512` |
    | `sensor_type` | 否 | 默认 `s2`；可选 `s2`、`s1`、`landsat` | 普通光学影像点选建议填 `s2` |
    | `point_coords` | 是 | 至少 1 个点；经度 `[-180,180]`，纬度 `[-90,90]` | WGS84 经纬度，格式 `[[经度, 纬度]]` |
    | `point_labels` | 否 | 默认全部为 `1`；可选 `0` 或 `1` | 前端当前不用传；`1`=目标点，`0`=背景排除点 |
    | `multimask_output` | 否 | 默认 `false` | `false` 返回最优候选；`true` 返回多个候选 |
    | `include_masks` | 否 | 默认 `false` | `false` 只返回 GeoJSON 框；`true` 额外返回 base64 mask |

    推荐请求体：

    ```json
    {
      "date": "202512",
      "sensor_type": "s2",
      "point_coords": [[116.0954, 40.0628]],
      "multimask_output": false,
      "include_masks": false
    }
    ```

    注意：本接口不再使用 `month` 字段；如果请求体里传入 `month`，会返回
    `422` 参数校验错误。
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
