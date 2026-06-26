"""Region management router."""

from fastapi import APIRouter, HTTPException, Path
from app.config import get_config
from app.schemas.models import (
    RegionsResponse, RegionInfo, HealthResponse, RegionDetail, RegionTaskMeta,
)

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
