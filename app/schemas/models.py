"""Pydantic models for API request/response schemas."""

import re
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


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


# ── GeoJSON annotation schemas for custom training ──

class ModelClass(BaseModel):
    """Class definition submitted by the frontend for training."""

    id: str = Field(..., description="Class identifier.", examples=["cls_001"])
    name: str = Field(..., description="Class display name.", examples=["建筑用地"])
    color: str = Field(..., description="Class color (hex).", examples=["#FF0000"])


class GeoJSONProperties(BaseModel):
    """Properties attached to each GeoJSON annotation feature."""

    patch_id: str = Field(..., description="Patch identifier.", examples=["patch_000000"])
    region_id: str = Field(..., description="Region identifier.", examples=["harbin"])
    class_id: str = Field(..., description="Class identifier.", examples=["cls_001"])
    class_name: Optional[str] = Field(None, description="Human-readable class name.", examples=["建筑用地"])
    color: Optional[str] = Field(None, description="Class color.", examples=["#FF0000"])
    task_type: Optional[str] = Field(
        None,
        description=(
            "Downstream task type. Optional for custom training; when omitted, "
            "single_time_detection defaults to building_extraction and change_detection "
            "defaults to change_detection."
        ),
        examples=["building_extraction"],
    )
    month: Optional[str] = Field(None, description="Month for single-time tasks.", examples=["2025-04"])
    before_month: Optional[str] = Field(None, description="Before month for change detection.", examples=["2025-04"])
    after_month: Optional[str] = Field(None, description="After month for change detection.", examples=["2025-06"])


class GeoJSONFeature(BaseModel):
    """A single annotation feature in GeoJSON format."""

    type: Literal["Feature"] = "Feature"
    properties: GeoJSONProperties
    geometry: Dict[str, Any] = Field(
        ...,
        description="GeoJSON geometry object. Supported types: Polygon, MultiPolygon. Coordinates must be WGS84 [lon, lat].",
    )

    @model_validator(mode="after")
    def validate_geometry_type(self):
        geom_type = self.geometry.get("type")
        if geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Unsupported geometry type: {geom_type}. Only Polygon and MultiPolygon are allowed.")
        return self


class GeoJSONFeatureCollection(BaseModel):
    """Annotation package submitted by the frontend when creating a model."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature] = Field(
        ...,
        min_length=1,
        description="List of GeoJSON annotation features.",
    )


# ── Custom training schemas ──

class ModelCreate(BaseModel):
    """Request body for creating a custom model.

    The frontend submits a complete annotation package (`annotations` as GeoJSON
    FeatureCollection plus `classes`). The backend parses the package, extracts
    training samples, and trains a downstream task head.

    All core fields are required in production. Empty or malformed request
    bodies must fail validation instead of silently creating a demo model.
    """

    model_config = {"protected_namespaces": ()}

    name: str = Field(
        ...,
        description="User-defined model name.",
        examples=["我的建筑提取模型"],
    )
    model_type: str = Field(
        ...,
        description=(
            "训练类型。可选 'single_time_detection'（单时间检测）或 "
            "'change_detection'（双时相变化检测）。兼容旧值 'classification'，"
            "会自动按 single_time_detection 处理。"
        ),
        examples=["single_time_detection"],
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
    epochs: int = Field(
        100,
        ge=1,
        le=1000,
        description="Number of training iterations (mapped to LogisticRegression max_iter).",
        examples=[100],
    )
    class_ids: Optional[List[str]] = Field(
        None,
        description="Optional subset of class IDs to use for training. If empty, all classes in annotations are used.",
        examples=[["cls_001"]],
    )
    description: Optional[str] = Field(
        None,
        description="Optional model description.",
        examples=["基于用户标注的建筑提取模型"],
    )
    annotations: GeoJSONFeatureCollection = Field(
        ...,
        description="GeoJSON FeatureCollection containing user annotations. Coordinates must be WGS84.",
    )
    classes: List[ModelClass] = Field(
        ...,
        min_length=1,
        description="Class definitions referenced by the annotations.",
    )

    @model_validator(mode="after")
    def validate_model_type_consistency(self):
        valid_classification_tasks = {
            "building_extraction",
            "road_extraction",
            "construction",
            "land_use_classification",
            "land_cover_classification",
            "water_extraction",
        }
        task_type = self.resolved_task_type()
        for feature in self.annotations.features:
            if feature.properties.task_type is None:
                feature.properties.task_type = task_type

        if self.model_type == "single_time_detection" and task_type not in valid_classification_tasks:
            raise ValueError(
                f"single_time_detection model does not support task_type '{task_type}'"
            )
        if self.model_type == "change_detection" and task_type != "change_detection":
            raise ValueError(
                "change_detection model requires task_type 'change_detection'"
            )

        class_ids = {c.id for c in self.classes}
        if self.class_ids:
            for cid in self.class_ids:
                if cid not in class_ids:
                    raise ValueError(f"class_id '{cid}' is not defined in classes")

        total_vertices = 0
        max_features = 10000
        max_vertices = 100_000
        if len(self.annotations.features) > max_features:
            raise ValueError(f"annotations exceed maximum of {max_features} features")

        for feature in self.annotations.features:
            props = feature.properties
            if props.region_id != self.region_id:
                raise ValueError(
                    f"feature region_id '{props.region_id}' does not match top-level region_id '{self.region_id}'"
                )
            if props.class_id not in class_ids:
                raise ValueError(
                    f"feature class_id '{props.class_id}' is not defined in classes"
                )
            if props.task_type != task_type:
                raise ValueError(
                    f"feature task_type '{props.task_type}' does not match inferred model task_type '{task_type}'"
                )

            if self.model_type == "single_time_detection":
                if not props.month:
                    raise ValueError(f"single_time_detection model requires 'month' for patch {props.patch_id}")
            elif self.model_type == "change_detection":
                if not props.before_month or not props.after_month:
                    raise ValueError(
                        f"change_detection model requires 'before_month' and 'after_month' for patch {props.patch_id}"
                    )
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            coords = feature.geometry.get("coordinates", [])
            total_vertices += self._count_vertices(coords)

        if total_vertices > max_vertices:
            raise ValueError(f"annotations exceed maximum of {max_vertices} total vertices")

        return self

    @field_validator("model_type")
    @classmethod
    def normalize_model_type(cls, value: str) -> str:
        if value == "classification":
            return "single_time_detection"
        if value not in ("single_time_detection", "change_detection"):
            raise ValueError(
                "model_type must be 'single_time_detection' or 'change_detection'"
            )
        return value

    def resolved_task_type(self) -> str:
        """Infer the training task type without requiring a top-level field."""
        if self.model_type == "change_detection":
            return "change_detection"

        task_types = {
            feature.properties.task_type
            for feature in self.annotations.features
            if feature.properties.task_type is not None
        }
        if not task_types:
            return "building_extraction"
        if len(task_types) != 1:
            raise ValueError(
                "single_time_detection annotations must use exactly one feature task_type"
            )
        return next(iter(task_types))

    @staticmethod
    def _count_vertices(coords):
        """Recursively count coordinate pairs in a GeoJSON coordinate array."""
        count = 0
        if isinstance(coords, list):
            if coords and isinstance(coords[0], (int, float)):
                return 1
            for item in coords:
                count += ModelCreate._count_vertices(item)
        return count


class ModelOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str = Field(..., description="Model identifier.")
    name: str = Field(..., description="Model name.")
    type: str = Field(..., description="Model type ('single_time_detection' or 'change_detection').")
    task_type: Optional[str] = Field(None, description="Downstream task type.")
    status: str = Field(..., description="Training status: running, completed, failed, or ready (system models).")
    created_at: str = Field(..., description="ISO timestamp when the model was created.")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when training completed.")
    classes: List[Dict[str, Any]] = Field(..., description="Classes used by the model.")
    accuracy: Optional[float] = Field(None, description="Training accuracy, if available.")
    n_samples: Optional[int] = Field(None, description="Number of training samples.")
    model_path: Optional[str] = Field(None, description="Path to the saved model artifact.")
    description: Optional[str] = Field(None, description="Model description.")
    message: Optional[str] = Field(None, description="Status or error message.")
    job_id: Optional[str] = Field(None, description="Training job identifier.")
    source: Optional[Literal["custom", "system"]] = Field(
        "custom",
        description="Model source: 'custom' (user-trained) or 'system' (pre-trained).",
    )
    versions: Optional[List[str]] = Field(
        None,
        description="Available checkpoint versions (system models only).",
    )


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
    month: Optional[str] = Field(
        None,
        description="Month for the source embedding, e.g. 2025-04. Required for single_time_detection models.",
        examples=["2025-04"],
    )
    before_month: Optional[str] = Field(
        None,
        description="Before month for change-detection inference, e.g. 2025-04.",
        examples=["2025-04"],
    )
    after_month: Optional[str] = Field(
        None,
        description="After month for change-detection inference, e.g. 2025-06.",
        examples=["2025-06"],
    )
    version: Optional[str] = Field(
        "v2",
        description="Checkpoint version for system pre-trained models. Allowed values: v1, v2. Ignored for custom models.",
        examples=["v2"],
    )

    @model_validator(mode="after")
    def validate_infer_months(self):
        has_single = bool(self.month)
        has_pair = bool(self.before_month) and bool(self.after_month)
        if has_single and has_pair:
            raise ValueError("请勿同时传入 'month' 和 'before_month'/'after_month'；单时间检测传 month，变化检测传 before_month+after_month")
        if not has_single and not has_pair:
            raise ValueError("单时间检测请传入 'month'，变化检测请同时传入 'before_month' 和 'after_month'")
        return self


class BatchInferRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    region_id: str = Field(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    )
    patch_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of patch identifiers to infer (max 100).",
        examples=[["patch_000000", "patch_000001"]],
    )
    month: Optional[str] = Field(
        None,
        description="Month for the source embedding, e.g. 2025-04. Required for single_time_detection models.",
        examples=["2025-04"],
    )
    before_month: Optional[str] = Field(
        None,
        description="Before month for change-detection inference, e.g. 2025-04.",
        examples=["2025-04"],
    )
    after_month: Optional[str] = Field(
        None,
        description="After month for change-detection inference, e.g. 2025-06.",
        examples=["2025-06"],
    )
    version: Optional[str] = Field(
        "v2",
        description="Checkpoint version for system pre-trained models. Allowed values: v1, v2. Ignored for custom models.",
        examples=["v2"],
    )

    @field_validator("patch_ids")
    @classmethod
    def validate_patch_ids(cls, v: List[str]) -> List[str]:
        for patch_id in v:
            if not re.fullmatch(r"patch_\d{6}", patch_id):
                raise ValueError(
                    "Each patch_id must match the form patch_000000"
                )
        return v

    @model_validator(mode="after")
    def validate_batch_infer_months(self):
        has_single = bool(self.month)
        has_pair = bool(self.before_month) and bool(self.after_month)
        if has_single and has_pair:
            raise ValueError("请勿同时传入 'month' 和 'before_month'/'after_month'；单时间检测传 month，变化检测传 before_month+after_month")
        if not has_single and not has_pair:
            raise ValueError("单时间检测请传入 'month'，变化检测请同时传入 'before_month' 和 'after_month'")
        return self


class InferResult(BaseModel):
    result_url: str = Field(..., description="URL to the generated result PNG.")


class BatchInferResult(BaseModel):
    patch_id: str = Field(..., description="Patch identifier.")
    status: str = Field(..., description="Inference status for this patch.")
    result_url: Optional[str] = Field(None, description="URL to the result PNG, if successful.")
    error: Optional[str] = Field(None, description="Error message, if failed.")


class BatchInferResponse(BaseModel):
    total: int = Field(..., description="Total number of requested patches.")
    success_count: int = Field(..., description="Number of successful inferences.")
    error_count: int = Field(..., description="Number of failed inferences.")
    results: List[BatchInferResult] = Field(
        ...,
        description="Per-patch inference results.",
    )


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
