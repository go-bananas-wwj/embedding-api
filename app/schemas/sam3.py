"""Pydantic schemas for SAM3 endpoints."""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageData(BaseModel):
    width: int = Field(..., description="Image width in pixels.")
    height: int = Field(..., description="Image height in pixels.")
    format: str = Field("png", description="Image format.")
    data: str = Field(..., description="Base64-encoded PNG image.")


class EmbedRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patch_id": "patch_000212",
                "month": "202512",
                "sensor_type": "s2",
            }
        }
    )

    patch_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "Patch 编号。必填。格式固定为 patch_ 后接 6 位数字，例如 patch_000212。"
            "用于明确预加载哪一个 patch 的影像。"
        ),
        examples=["patch_000000"],
    )
    month: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description=(
            "影像日期或月份。必填。支持 YYYY-MM、YYYYMM 或 YYYYMMDD，"
            "例如 2025-10、202510、20251214。YYYYMMDD 表示精确日期；"
            "YYYY-MM/YYYYMM 表示月度请求，若同月有多景日级影像，会按日期"
            "倒序选择当月最新的一景。"
        ),
        examples=["202512"],
    )
    sensor_type: Literal["s2", "s1", "landsat", "highres"] = Field(
        "s2",
        description=(
            "传感器类型。可选值：s2、s1、landsat、highres。默认 s2。"
            "highres=带 CRS/仿射变换的高分辨率 RGB 光学 GeoTIFF，波段顺序为 R/G/B。"
        ),
        examples=["s2"],
    )

    @field_validator("patch_id")
    @classmethod
    def validate_patch_id(cls, v: str) -> str:
        import re
        if not re.fullmatch(r"patch_\d{6}", v):
            raise ValueError("patch_id must match the form patch_000000")
        return v

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: str) -> str:
        import re
        if not re.match(r"^[\w\-]{1,32}$", v):
            raise ValueError("Invalid month format")
        return v


class EmbedResponse(BaseModel):
    embedding_id: str = Field(..., description="Identifier for the cached embedding.")
    status: str = Field("ready", description="Embedding status.")
    source_scene: Optional[str] = Field(
        None,
        description="实际加载的原始影像文件 stem，例如 20251214。",
    )
    selected_image_date: Optional[str] = Field(
        None,
        description="实际选中的影像日期。日级影像通常为 YYYYMMDD。",
    )
    image: ImageData


class SegmentRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "date": "202512",
                "sensor_type": "s2",
                "point_coords": [[116.0954, 40.0628]],
                "multimask_output": False,
                "include_masks": False,
            }
        },
    )

    date: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description=(
            "影像日期或月份。用于选择要分割的遥感影像。"
            "支持 YYYY-MM、YYYYMM 或 YYYYMMDD，例如 2025-10、202510、20251001。"
            "YYYYMMDD 表示精确日期，不会自动改用其它日期；YYYY-MM/YYYYMM "
            "表示月度请求，若同月有多景日级影像，会按日期倒序选择当月最新的"
            "一景，并在返回 properties.selected_image_date 中说明。"
        ),
        examples=["2025-10"],
    )
    sensor_type: Literal["s2", "s1", "landsat", "highres"] = Field(
        "s2",
        description=(
            "传感器类型。s2=Sentinel-2 光学影像；s1=Sentinel-1 SAR；"
            "landsat=Landsat 光学影像；highres=高分辨率 RGB 光学 GeoTIFF。"
            "高分辨率影像必须包含 CRS 和仿射变换，前三个波段依次为 R/G/B。"
        ),
        examples=["s2"],
    )
    point_coords: List[List[float]] = Field(
        ...,
        min_length=1,
        description=(
            "用户点击的 WGS84 经纬度点列表。每个点格式为 [longitude, latitude]，"
            "即 [经度, 纬度]。后端会自动根据点位定位 patch。"
        ),
        examples=[[[116.30, 39.98]]],
    )
    point_labels: Optional[List[int]] = Field(
        None,
        min_length=1,
        description=(
            "可选点标签，长度必须与 point_coords 一致。1 表示前景目标点，"
            "0 表示背景排除点。当前前端不需要传；不传时后端默认所有点都是 1。"
        ),
        examples=[[1]],
    )
    multimask_output: bool = Field(
        False,
        description=(
            "是否返回多个候选分割结果。false=只返回一个最优候选，适合常规前端交互；"
            "true=返回多个候选，适合让用户二次选择。"
        ),
        examples=[False],
    )
    include_masks: bool = Field(
        False,
        description=(
            "是否在 GeoJSON 多边形之外额外返回 base64 PNG mask。false=只返回 WGS84 "
            "GeoJSON Polygon/MultiPolygon，响应更小；true=同时返回 mask，响应体更大。"
        ),
        examples=[False],
    )

    @field_validator("point_coords")
    @classmethod
    def validate_coords(cls, v: List[List[float]]) -> List[List[float]]:
        for coord in v:
            if len(coord) != 2:
                raise ValueError("Each point must have exactly 2 coordinates [lon, lat]")
            if not (-180.0 <= coord[0] <= 180.0 and -90.0 <= coord[1] <= 90.0):
                raise ValueError("Coordinates must be valid WGS84 lon/lat values")
        return v

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        import re
        if not re.match(r"^[\w\-]{1,32}$", v):
            raise ValueError("Invalid date format")
        return v

    @field_validator("point_labels")
    @classmethod
    def validate_labels(cls, v: Optional[List[int]], info) -> Optional[List[int]]:
        if v is None:
            return v
        point_coords = info.data.get("point_coords", [])
        if len(v) != len(point_coords):
            raise ValueError("point_labels length must match point_coords length")
        for label in v:
            if label not in (0, 1):
                raise ValueError("point_labels must be 0 (negative) or 1 (positive)")
        return v


class SAM3BBoxProperties(BaseModel):
    score: float = Field(..., description="Mask confidence score.")
    bbox: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Mask bounding box on the SAM input image [x, y, width, height].",
    )
    bbox_wgs84: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="WGS84 bbox [min_lon, min_lat, max_lon, max_lat].",
    )
    patch_id: str = Field(..., description="Patch containing the prompt points.")
    sensor_type: Literal["s2", "s1", "landsat", "highres"] = Field(
        ...,
        description="Remote-sensing source used for segmentation.",
    )
    date: str = Field(..., description="Image date/month used for segmentation.")
    source_scene: Optional[str] = Field(
        None,
        description="实际参与 SAM3 推理的原始影像文件 stem，例如 20251214。",
    )
    selected_image_date: Optional[str] = Field(
        None,
        description="实际选中的影像日期。用于解释月度请求最终使用了哪一景。",
    )
    candidate_index: int = Field(..., description="Candidate mask index.")
    geometry_kind: Literal["mask_polygon", "bbox"] = Field(
        "mask_polygon",
        description=(
            "Geometry type. `mask_polygon` means the geometry follows the SAM mask "
            "outline; `bbox` is only used as a defensive fallback."
        ),
    )


class SAM3PolygonGeometry(BaseModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: List[List[List[float]]] = Field(
        ...,
        description="GeoJSON polygon coordinates in WGS84.",
    )


class SAM3MultiPolygonGeometry(BaseModel):
    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: List[List[List[List[float]]]] = Field(
        ...,
        description="GeoJSON multipolygon coordinates in WGS84.",
    )


class SAM3BBoxFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: Union[SAM3PolygonGeometry, SAM3MultiPolygonGeometry]
    properties: SAM3BBoxProperties


class MaskData(BaseModel):
    data: str = Field(..., description="Base64-encoded PNG mask.")
    score: float = Field(..., description="Mask confidence score.")
    bbox: List[int] = Field(..., description="Bounding box [x, y, width, height].")
    bbox_wgs84: List[float] = Field(
        ...,
        description="Bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat].",
    )


class SegmentResponse(BaseModel):
    type: Literal["FeatureCollection"] = Field(
        "FeatureCollection",
        description="GeoJSON FeatureCollection type.",
    )
    features: List[SAM3BBoxFeature] = Field(
        ...,
        description="GeoJSON polygon features for SAM3 candidate boxes in WGS84.",
    )
    masks: Optional[List[MaskData]] = Field(
        None,
        description="Optional base64 PNG masks when include_masks=true.",
    )


class CacheInfo(BaseModel):
    size: int = Field(..., description="Current number of cached embeddings.")
    max_size: int = Field(..., description="Maximum cache size.")
    entries: List[str] = Field(..., description="List of cached embedding IDs.")


class GpuMemory(BaseModel):
    allocated_mb: int = Field(..., description="Allocated GPU memory in MB.")
    reserved_mb: int = Field(..., description="Reserved GPU memory in MB.")


class StatusResponse(BaseModel):
    model_loaded: bool = Field(..., description="Whether the SAM3 model is loaded.")
    device: str = Field(..., description="Device used for inference (cuda or cpu).")
    gpu_memory: GpuMemory
    cache: CacheInfo
