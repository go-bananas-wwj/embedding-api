"""Region management router."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, Response

from app.config import get_config
from app.schemas.models import (
    ErrorResponse,
    HealthResponse,
    MosaicMetadataResponse,
    RegionDetail,
    RegionInfo,
    RegionsResponse,
    RegionTaskMeta,
)
from app.services.data_service import DataNotFoundError, DataValidationError
from app.services.mosaic_service import build_mosaic_artifact
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
            "description": "区域影像、GeoTIFF 或带 WGS84 空间信息的 JSON",
            "model": MosaicMetadataResponse,
            "content": {
                "image/png": {},
                "image/tiff": {},
            },
        },
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_region_mosaic(
    request: Request,
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
            "具体可用月份取决于区域和 sensor_type，请查看本接口正文中的"
            "“区域、数据源与月份”表格。"
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
            "大图数据源。参数名称与前端数据库保持一致。常用值："
            "`s2`、`s1`、`landsat`、`highres`、`highres_sar`、`s1_hr`、"
            "`s2_hr`。不同区域支持的数据源不同，请查看本接口正文表格；"
            "`embedding` 返回按 Patch 空间位置拼接的 PCA 色彩可视化。"
        ),
        examples=["s2"],
        openapi_examples={
            "s2": {"summary": "Sentinel-2 真彩色", "value": "s2"},
            "s1": {"summary": "Sentinel-1 SAR 伪彩色", "value": "s1"},
            "landsat": {"summary": "Landsat 真彩色", "value": "landsat"},
            "highres": {"summary": "高分辨率光学影像", "value": "highres"},
            "highres_sar": {"summary": "高分辨率 SAR", "value": "highres_sar"},
            "s1_hr": {"summary": "天仪高分辨率 SAR", "value": "s1_hr"},
            "s2_hr": {"summary": "天仪高分辨率光学", "value": "s2_hr"},
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
        description=(
            "输出格式。`png` 使用各传感器固定量程转换为 8 位图，不做逐图增强；"
            "`tif` 原样保留多波段数值和地理坐标；`json` 返回图片 URL、"
            "WGS84 四至、整体 footprint_wgs84 和逐 Patch footprint_wgs84。"
        ),
        examples=["png"],
        openapi_examples={
            "png": {"summary": "PNG 可视化", "value": "png"},
            "tif": {"summary": "GeoTIFF 原始数据", "value": "tif"},
            "json": {"summary": "前端地图叠加元数据", "value": "json"},
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
    """获取指定区域、月份和数据源的 Patch 拼接大图。

    将区域内 Patch 按真实空间位置拼接成一张大图。支持原始光学/SAR 影像，
    也支持 `sensor_type=embedding` 的 PCA 色彩可视化。
    PNG 不进行按图百分位增强；需要完全保留原始数值时请选择 GeoTIFF。
    首次生成后会缓存到 users/default/mosaic/，后续直接读取。

    ### 区域、数据源与月份

    #### 哈尔滨新区（`region_id=harbin`）

    | `sensor_type` | 数据内容 | 可用月份 |
    |---|---|---|
    | `s1` | Sentinel-1 SAR | `202301`～`202506`、`202508`～`202605`；没有 `202507` |
    | `s2` | Sentinel-2 光学 | `202301`～`202310`、`202401`～`202411`、`202501`～`202510`、`202601`～`202605` |
    | `landsat` | Landsat 光学 | `202301`～`202303`、`202305`、`202308`～`202405`、`202407`～`202605` |
    | `s1_hr` | 天仪高分辨率 SAR | `202506`、`202508`、`202509`、`202510`；**不支持 `202504`** |
    | `s2_hr` | 天仪高分辨率光学 | `202504`、`202506`、`202508`、`202509`、`202510` |
    | `embedding` | Embedding PCA 色彩图 | `v1`：`202504/202506/202508/202509/202510`；`v2`：上述月份及 `202601`～`202605` |

    哈尔滨不提供 `highres` 和 `highres_sar`，高分辨率数据请分别使用
    `s2_hr` 和 `s1_hr`。

    #### 海淀区（`region_id=haidian`）

    | `sensor_type` | 数据内容 | 可用月份 |
    |---|---|---|
    | `s1` | Sentinel-1 SAR | `202512`～`202605` |
    | `s2` | Sentinel-2 光学 | `202512`～`202605` |
    | `landsat` | Landsat 光学 | `202512`～`202605` |
    | `highres` | 高分辨率光学 | `202512`～`202604`；没有 `202605` |
    | `highres_sar` | 高分辨率 SAR | `202512`～`202605` |
    | `embedding` | P10C Embedding PCA 色彩图 | `202512`～`202605`，使用 `version=v1` 或留空 |

    海淀不提供 `s1_hr` 和 `s2_hr`，高分辨率数据请分别使用
    `highres_sar` 和 `highres`。

    ### 参数组合示例

    - 哈尔滨 2025 年 4 月高分辨率光学：
      `region_id=harbin&date=202504&sensor_type=s2_hr`
    - 哈尔滨 2025 年 6 月高分辨率 SAR：
      `region_id=harbin&date=202506&sensor_type=s1_hr`
    - 海淀 2026 年 3 月 Sentinel-1：
      `region_id=haidian&date=202603&sensor_type=s1`
    - 海淀 2026 年 4 月高分辨率光学：
      `region_id=haidian&date=202604&sensor_type=highres`

    > 月份可以写成 `YYYYMM` 或 `YYYY-MM`。如果同月存在多景影像，
    > 接口按采集日期倒序选择当月最新的一景。
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
        requested_format = format.lower()
        data, mime, metadata = build_mosaic_artifact(
            region_id=region_id,
            date=date,
            sensor_type=sensor_type,
            version=version,
            fmt="png" if requested_format == "json" else requested_format,
            patch_ids=patch_ids,
        )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build mosaic: {e}")

    if requested_format == "json":
        metadata["image_url"] = str(
            request.url.include_query_params(format="png")
        )
        return JSONResponse(metadata)

    headers = {}
    if metadata.get("bounds_wgs84"):
        headers = {
            "X-Mosaic-CRS": str(metadata.get("crs", "")),
            "X-Mosaic-Bounds-WGS84": ",".join(
                str(value) for value in metadata["bounds_wgs84"]
            ),
            "X-Mosaic-Width": str(metadata.get("width", "")),
            "X-Mosaic-Height": str(metadata.get("height", "")),
        }
    return Response(content=data, media_type=mime, headers=headers)
