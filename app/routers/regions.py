"""Region management router."""

from fastapi import APIRouter, HTTPException, Path

from app.config import get_config
from app.schemas.models import (
    HealthResponse,
    RegionDetail,
    RegionInfo,
    RegionsResponse,
    RegionTaskMeta,
)
from app.services.region_mosaic_catalog import get_region_mosaic_info


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
    """获取所有可用区域及静态大图信息。

    返回区域、任务、Patch 数量，以及静态 PNG 压缩包使用的统一 WGS84
    四至、覆盖范围、传感器和月份清单。大图不再由在线 API 临时生成。
    """
    config = get_config()
    regions = []
    for region_id, region in config.regions.items():
        patches = config.get_patches(region_id)
        regions.append(
            RegionInfo(
                id=region_id,
                name=region.get("name", region_id),
                patch_count=len(patches),
                tasks=list(region.get("tasks", {}).keys()),
                mosaic=get_region_mosaic_info(region_id),
            )
        )
    return RegionsResponse(regions=regions)


@router.get("/regions/{region_id}", response_model=RegionDetail)
async def get_region(
    region_id: str = Path(
        ...,
        description="区域 ID。可用值：`harbin`（哈尔滨新区）、`haidian`（海淀区）。",
        examples=["haidian"],
        openapi_examples={
            "haidian": {"summary": "北京海淀区", "value": "haidian"},
            "harbin": {"summary": "哈尔滨新区", "value": "harbin"},
        },
    )
):
    """获取指定区域详情及静态大图信息。

    返回任务、Embedding 版本和静态 PNG 压缩包的区域级 WGS84 定位信息。
    PNG 路径按 `{regionId}/{sensor}/{date}/mosaic.png` 组织。
    """
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    region = config.get_region(region_id)
    patches = config.get_patches(region_id)
    tasks = {
        task_id: RegionTaskMeta(
            name=task.get("name", task_id),
            description=task.get("description", ""),
            versions=list(task.get("versions", {}).keys()),
        )
        for task_id, task in region.get("tasks", {}).items()
    }
    return RegionDetail(
        id=region_id,
        name=region.get("name", region_id),
        patch_count=len(patches),
        tasks=tasks,
        embeddings=list(region.get("embeddings", {}).keys()),
        mosaic=get_region_mosaic_info(region_id),
    )
