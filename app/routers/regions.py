"""Region management router."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response

from app.config import get_config
from app.schemas.models import (
    ErrorResponse, RegionsResponse, RegionInfo, HealthResponse, RegionDetail, RegionTaskMeta,
)
from app.services.data_service import DataNotFoundError, DataValidationError
from app.services.mosaic_service import build_mosaic
from app.services.time_utils import is_valid_month_or_date

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点。

    用于前端页面初始化或服务心跳监控确认 API 是否可用。
    返回服务状态、版本号以及当前已配置的区域列表。
    """
    config = get_config()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        regions=config.list_regions(),
    )


@router.get("/regions", response_model=RegionsResponse)
async def list_regions():
    """获取所有可用区域列表。

    用于前端页面初始化时加载区域选择下拉框。
    返回每个区域的 ID、名称、Patch 数量以及支持的下游任务列表。
    """
    config = get_config()
    regions = []
    for rid, rinfo in config.regions.items():
        patches = config.get_patches(rid)
        tasks = list(rinfo.get("tasks", {}).keys())
        regions.append(
            RegionInfo(
                id=rid,
                name=rinfo.get("name", rid),
                patch_count=len(patches),
                tasks=tasks,
            )
        )
    return RegionsResponse(regions=regions)


@router.get("/regions/{region_id}", response_model=RegionDetail)
async def get_region(
    region_id: str = Path(
        ...,
        description="Region identifier. Use 'harbin' (Harbin New Area) or 'haidian' (Haidian District).",
        examples=["harbin"],
        openapi_examples={
            "harbin": {"summary": "Harbin New Area", "value": "harbin"},
            "haidian": {"summary": "Haidian District", "value": "haidian"},
        },
    )
):
    """获取指定区域的详细信息。

    用于用户选择某个区域后展示该区域的任务列表、嵌入版本及 Patch 数量。
    返回区域元数据的 JSON 详情。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    region = config.get_region(region_id)
    patches = config.get_patches(region_id)
    tasks = {
        tid: RegionTaskMeta(
            name=tinfo.get("name", tid),
            description=tinfo.get("description", ""),
            versions=list(tinfo.get("versions", {}).keys()),
        )
        for tid, tinfo in region.get("tasks", {}).items()
    }

    return RegionDetail(
        id=region_id,
        name=region.get("name", region_id),
        patch_count=len(patches),
        tasks=tasks,
        embeddings=list(region.get("embeddings", {}).keys()),
    )


@router.get(
    "/regions/{region_id}/mosaic",
    response_class=Response,
    responses={
        200: {
            "description": "区域影像或 Embedding 大图",
            "content": {"image/png": {}, "image/tiff": {}},
        },
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_region_mosaic(
    region_id: str = Path(
        ...,
        description="区域 ID。当前可用：'harbin'（哈尔滨新区）、'haidian'（海淀区）。",
        examples=["harbin"],
        openapi_examples={
            "harbin": {"summary": "哈尔滨新区", "value": "harbin"},
            "haidian": {"summary": "北京海淀区", "value": "haidian"},
        },
    ),
    date: str = Query(
        ...,
        description=(
            "影像月份或精确日期。两个区域统一支持 `YYYYMM`、`YYYY-MM`；"
            "需要精确到天时可使用 `YYYYMMDD`。月度请求会选择当月最新影像。"
        ),
        examples=["202512"],
        openapi_examples={
            "haidian": {"summary": "海淀月份", "value": "202512"},
            "harbin": {"summary": "哈尔滨月份", "value": "202510"},
            "hyphen": {"summary": "带横杠写法", "value": "2025-10"},
        },
    ),
    sensor_type: str = Query(
        "s2",
        description=(
            "大图数据源。可选 `s2`、`s1`、`landsat`、`highres` 或 `embedding`；"
            "`embedding` 返回按 Patch 空间位置拼接的 PCA 色彩可视化。"
        ),
        examples=["s2"],
        openapi_examples={
            "s2": {"summary": "Sentinel-2 真彩色", "value": "s2"},
            "s1": {"summary": "Sentinel-1 SAR 伪彩色", "value": "s1"},
            "landsat": {"summary": "Landsat 真彩色", "value": "landsat"},
            "highres": {"summary": "高分辨率光学影像", "value": "highres"},
            "embedding": {"summary": "Embedding PCA 色彩图", "value": "embedding"},
        },
    ),
    version: Optional[str] = Query(
        None,
        description=(
            "仅 `sensor_type=embedding` 时使用，通常留空。"
            "海淀默认 P10C（API `v1`），哈尔滨默认 V5（API `v2`）。"
        ),
    ),
    format: str = Query(
        "png",
        description="输出格式。'png'（默认，可视化 RGB）或 'tif'（GeoTIFF，保留原始多波段与坐标）。",
        examples=["png"],
        openapi_examples={
            "png": {"summary": "PNG 可视化", "value": "png"},
            "tif": {"summary": "GeoTIFF 原始数据", "value": "tif"},
        },
    ),
    patch_ids: Optional[List[str]] = Query(
        None,
        description="可选，只拼接指定的 Patch ID 列表（用于快速预览或局部大图）。不传则拼全区域。",
        examples=[["patch_000000", "patch_000001"]],
        openapi_examples={
            "two_patches": {
                "summary": "只拼前两个 patch",
                "value": ["patch_000000", "patch_000001"],
            },
            "empty": {
                "summary": "拼全区域（不传）",
                "value": [],
            },
        },
    ),
):
    """获取指定日期、区域的整区域马赛克大图。

    将区域内 Patch 按真实空间位置拼接成一张大图。支持原始光学/SAR 影像，
    也支持 `sensor_type=embedding` 的 PCA 色彩可视化。
    首次生成后会缓存到 users/default/mosaic/，后续直接读取。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")
    if not is_valid_month_or_date(date):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid date '{date}'. Use a real calendar month/date in "
                "YYYYMM, YYYY-MM, or YYYYMMDD format."
            ),
        )
    if patch_ids:
        configured = {
            patch.get("patch_id")
            for patch in config.get_patches(region_id)
            if isinstance(patch, dict)
        }
        missing = [patch_id for patch_id in patch_ids if patch_id not in configured]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Patch '{missing[0]}' not found in region '{region_id}'",
            )
    try:
        data, mime = build_mosaic(
            region_id=region_id,
            date=date,
            sensor_type=sensor_type,
            version=version,
            fmt=format,
            patch_ids=patch_ids,
        )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build mosaic: {e}")

    return Response(content=data, media_type=mime)
