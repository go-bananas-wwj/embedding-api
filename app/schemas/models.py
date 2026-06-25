"""Pydantic models for API request/response schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    regions: List[str]


class RegionInfo(BaseModel):
    id: str
    name: str
    patch_count: int
    tasks: List[str]


class RegionTaskMeta(BaseModel):
    name: str
    description: str
    versions: List[str]


class RegionDetail(BaseModel):
    id: str
    name: str
    patch_count: int
    tasks: Dict[str, RegionTaskMeta]
    embeddings: List[str]


class RegionsResponse(BaseModel):
    regions: List[RegionInfo]


class PatchBase(BaseModel):
    patch_id: str
    bounds_wgs84: List[float]
    sources: Dict[str, int]
    time_range: List[str]


class PatchDetail(PatchBase):
    bounds: Optional[List[float]] = None
    crs: Optional[str] = None
    has_embedding: bool = False
    available_months: List[str] = []
    available_tasks: List[str] = []


class PaginatedPatchesResponse(BaseModel):
    total: int
    page: int
    page_size: int
    has_next: bool
    patches: List[PatchDetail]


class TaskInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    versions: List[str]


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]


class TaskSummary(BaseModel):
    task: str
    name: str
    version: str
    period: Optional[str] = None
    grid_size: Optional[int] = None
    total_polygons: Optional[int] = None
    total_patches: Optional[int] = None
    positive_patches: Optional[int] = None
    negative_patches: Optional[int] = None


class EmbeddingStats(BaseModel):
    patch_id: str
    shape: List[int]
    dtype: str
    min: float
    max: float
    mean: float


class TileInfo(BaseModel):
    patch_id: str
    period: Optional[str] = None
    filename: str


class TilesResponse(BaseModel):
    tiles: List[TileInfo]
    total: int



# Annotation / Custom Training schemas

class ClassCreate(BaseModel):
    name: str
    color: str


class ClassOut(BaseModel):
    id: str
    name: str
    color: str


class ClassRenameRequest(BaseModel):
    name: str


class GeometryMask(BaseModel):
    type: str
    mask_b64: str


class GeometryPolygon(BaseModel):
    type: str
    points: List[List[float]]


class GeometryPolyline(BaseModel):
    type: str
    points: List[List[float]]


class AnnotationCreate(BaseModel):
    region_id: str
    patch_id: str
    month: str
    class_id: str
    geometry: Dict[str, Any]
    task_type: Optional[str] = None
    score: float = 1.0
    before_month: Optional[str] = None
    after_month: Optional[str] = None


class AnnotationOut(BaseModel):
    id: str
    region_id: str
    patch_id: str
    month: str
    class_id: str
    task_type: Optional[str] = None
    score: float
    geometry: Dict[str, Any]
    before_month: Optional[str] = None
    after_month: Optional[str] = None
    created_at: str


class ErrorResponse(BaseModel):
    detail: str
