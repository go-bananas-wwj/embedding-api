"""Pydantic models for API request/response schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status.")
    version: str = Field("0.1.0", description="API version.")
    regions: List[str] = Field(..., description="List of available region identifiers.")


class RegionInfo(BaseModel):
    id: str = Field(..., description="Region identifier.")
    name: str = Field(..., description="Human-readable region name.")
    patch_count: int = Field(..., description="Number of patches in the region.")
    tasks: List[str] = Field(..., description="Task types available for the region.")


class RegionTaskMeta(BaseModel):
    name: str = Field(..., description="Task display name.")
    description: str = Field(..., description="Short task description.")
    versions: List[str] = Field(..., description="Available result versions for the task.")


class RegionDetail(BaseModel):
    id: str = Field(..., description="Region identifier.")
    name: str = Field(..., description="Human-readable region name.")
    patch_count: int = Field(..., description="Number of patches in the region.")
    tasks: Dict[str, RegionTaskMeta] = Field(..., description="Task metadata keyed by task type.")
    embeddings: List[str] = Field(..., description="Available embedding versions.")


class RegionsResponse(BaseModel):
    regions: List[RegionInfo]


class PatchBase(BaseModel):
    patch_id: str = Field(..., description="Patch identifier.")
    bounds_wgs84: List[float] = Field(..., description="Bounding box in WGS84 [minx, miny, maxx, maxy].")
    sources: Dict[str, int] = Field(..., description="Source data counts for the patch.")
    time_range: List[str] = Field(..., description="Available time range [start, end].")


class PatchDetail(PatchBase):
    bounds: Optional[List[float]] = Field(None, description="Native CRS bounding box.")
    crs: Optional[str] = Field(None, description="Coordinate reference system.")
    has_embedding: bool = Field(False, description="Whether an embedding is available.")
    available_months: List[str] = Field([], description="Months with available embeddings.")
    available_tasks: List[str] = Field([], description="Tasks with available results.")


class PaginatedPatchesResponse(BaseModel):
    total: int = Field(..., description="Total number of patches matching the query.")
    page: int = Field(..., description="Current page number.")
    page_size: int = Field(..., description="Number of patches per page.")
    has_next: bool = Field(..., description="Whether another page exists.")
    patches: List[PatchDetail]


class TaskInfo(BaseModel):
    id: str = Field(..., description="Task identifier.")
    name: str = Field(..., description="Task display name.")
    description: Optional[str] = Field(None, description="Short task description.")
    versions: List[str] = Field(..., description="Available result versions.")


class TasksResponse(BaseModel):
    tasks: List[TaskInfo]


class TaskSummary(BaseModel):
    task: str = Field(..., description="Task identifier.")
    name: str = Field(..., description="Task display name.")
    version: str = Field(..., description="Result version.")
    period: Optional[str] = Field(None, description="Comparison period, if applicable.")
    grid_size: Optional[int] = Field(None, description="Task grid size.")
    total_polygons: Optional[int] = Field(None, description="Total polygon count.")
    total_patches: Optional[int] = Field(None, description="Total patch count.")
    positive_patches: Optional[int] = Field(None, description="Patches with positive samples.")
    negative_patches: Optional[int] = Field(None, description="Patches with negative samples.")


class EmbeddingStats(BaseModel):
    patch_id: str = Field(..., description="Patch identifier.")
    shape: List[int] = Field(..., description="Array shape.")
    dtype: str = Field(..., description="NumPy data type.")
    min: float = Field(..., description="Minimum value.")
    max: float = Field(..., description="Maximum value.")
    mean: float = Field(..., description="Mean value.")


class TileInfo(BaseModel):
    patch_id: str = Field(..., description="Patch identifier.")
    period: Optional[str] = Field(None, description="Comparison period, if applicable.")
    filename: str = Field(..., description="Tile filename.")


class TilesResponse(BaseModel):
    tiles: List[TileInfo]
    total: int



# Annotation / Custom Training schemas

class ClassCreate(BaseModel):
    name: str = Field(
        ...,
        description="Display name for the class.",
        examples=["Building"],
    )
    color: str = Field(
        ...,
        description="CSS color string (hex or named color) used to render the class.",
        examples=["#FF5733"],
    )


class ClassOut(BaseModel):
    id: str = Field(..., description="Class identifier.")
    name: str = Field(..., description="Class display name.")
    color: str = Field(..., description="Class color.")


class ClassRenameRequest(BaseModel):
    name: str = Field(
        ...,
        description="New display name for the class.",
        examples=["Building (renamed)"],
    )


class StatusOut(BaseModel):
    status: str


class GeometryMask(BaseModel):
    type: str = Field(..., description="Geometry type. Use 'mask'.")
    mask_b64: str = Field(..., description="Base64-encoded PNG mask.")


class GeometryPolygon(BaseModel):
    type: str = Field(..., description="Geometry type. Use 'polygon'.")
    points: List[List[float]] = Field(
        ...,
        description="Polygon vertices as normalized [x, y] coordinates (0-1).",
    )


class GeometryPolyline(BaseModel):
    type: str = Field(..., description="Geometry type. Use 'polyline'.")
    points: List[List[float]] = Field(
        ...,
        description="Polyline vertices as normalized [x, y] coordinates (0-1).",
    )


class AnnotationCreate(BaseModel):
    region_id: str = Field(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
    patch_id: str = Field(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    )
    month: str = Field(
        ...,
        description="Month for the source data, e.g. 2025-04.",
        examples=["2025-04"],
    )
    class_id: str = Field(
        ...,
        description="Class ID returned by POST /annotations/classes. Replace with the real ID from the create response.",
        examples=["cls_abc123"],
    )
    geometry: Dict[str, Any] = Field(
        ...,
        description="Geometry object. Supported types: 'mask' (mask_b64), 'polygon' (points), 'polyline' (points).",
        examples=[{"type": "polygon", "points": [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]]}],
    )
    task_type: Optional[str] = Field(
        None,
        description="Optional downstream task type, e.g. building_extraction.",
        examples=["building_extraction"],
    )
    score: float = Field(
        1.0,
        description="Confidence score (0-1).",
        examples=[1.0],
    )
    before_month: Optional[str] = Field(
        None,
        description="For change-detection annotations, the 'before' month.",
        examples=["2025-04"],
    )
    after_month: Optional[str] = Field(
        None,
        description="For change-detection annotations, the 'after' month.",
        examples=["2025-06"],
    )


class AnnotationOut(BaseModel):
    id: str = Field(..., description="Annotation identifier.")
    region_id: str = Field(..., description="Region identifier.")
    patch_id: str = Field(..., description="Patch identifier.")
    month: str = Field(..., description="Month for the source data.")
    class_id: str = Field(..., description="Class identifier.")
    task_type: Optional[str] = Field(None, description="Task type, if provided.")
    score: float = Field(..., description="Confidence score.")
    geometry: Dict[str, Any] = Field(..., description="Geometry object.")
    before_month: Optional[str] = Field(None, description="'Before' month for change detection.")
    after_month: Optional[str] = Field(None, description="'After' month for change detection.")
    created_at: str = Field(..., description="ISO timestamp when the annotation was created.")


class ModelCreate(BaseModel):
    """Request body for creating a custom model.

    Example:
    ```bash
    curl -X POST http://localhost:9061/models \
      -H 'Content-Type: application/json' \
      -d '{
        "name": "my-building-head",
        "model_type": "classification",
        "task_type": "building_extraction",
        "region_id": "harbin",
        "embedding_version": "v2"
      }'
    ```
    """
    model_config = {"protected_namespaces": ()}

    name: str = Field(
        ...,
        description="Human-readable model name.",
        examples=["my-building-head"],
    )
    model_type: str = Field(
        ...,
        description="Model head type. Allowed values: 'classification' or 'change_detection'.",
        examples=["classification"],
    )
    task_type: str = Field(
        ...,
        description="Downstream task type. Allowed values: change_detection, building_extraction, land_use_classification, land_cover_classification, water_extraction.",
        examples=["building_extraction"],
    )
    region_id: str = Field(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
    embedding_version: str = Field(
        "v2",
        description="Embedding version used for training. Allowed values: v1, v2.",
        examples=["v2"],
    )


class ModelOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str = Field(..., description="Model identifier.")
    name: str = Field(..., description="Model name.")
    type: str = Field(..., description="Model type ('classification' or 'change_detection').")
    task_type: Optional[str] = Field(None, description="Downstream task type.")
    status: str = Field(..., description="Training status: running, completed, or failed.")
    created_at: str = Field(..., description="ISO timestamp when the model was created.")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when training completed.")
    classes: List[Dict[str, Any]] = Field(..., description="Classes used to train the model.")
    accuracy: Optional[float] = Field(None, description="Training accuracy, if available.")
    n_samples: Optional[int] = Field(None, description="Number of training samples.")
    model_path: Optional[str] = Field(None, description="Path to the saved model artifact.")
    message: Optional[str] = Field(None, description="Status or error message.")
    job_id: Optional[str] = Field(None, description="Training job identifier.")


class ModelRenameRequest(BaseModel):
    name: str = Field(
        ...,
        description="New model name.",
        examples=["my-model-renamed"],
    )


class InferRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    region_id: str = Field(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
    patch_id: str = Field(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    )
    month: str = Field(
        ...,
        description="Month for the source embedding, e.g. 2025-04.",
        examples=["2025-04"],
    )


class BatchInferRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    region_id: str = Field(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
    patch_ids: List[str] = Field(
        ...,
        description="List of patch identifiers to infer (max 100).",
        examples=[["patch_000000", "patch_000001"]],
    )
    month: str = Field(
        ...,
        description="Month for the source embedding, e.g. 2025-04.",
        examples=["2025-04"],
    )


class BatchInferResult(BaseModel):
    patch_id: str = Field(..., description="Patch identifier.")
    status: str = Field(..., description="Inference status for this patch.")
    result_url: Optional[str] = Field(None, description="URL to the result PNG, if successful.")
    error: Optional[str] = Field(None, description="Error message, if failed.")


class JobStatusOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    job_id: str = Field(..., description="Training job identifier.")
    status: str = Field(..., description="Job status: running, completed, or failed.")
    model_id: str = Field(..., description="Associated model identifier.")
    accuracy: Optional[float] = Field(None, description="Training accuracy, if available.")
    n_samples: Optional[int] = Field(None, description="Number of training samples.")
    model_path: Optional[str] = Field(None, description="Path to the saved model artifact.")
    message: Optional[str] = Field(None, description="Status or error message.")


class ErrorResponse(BaseModel):
    detail: str
