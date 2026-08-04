"""Pydantic models for API request/response schemas."""

import re
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status.")
    version: str = Field("0.1.0", description="API version.")
    regions: List[str] = Field(..., description="List of available region identifiers.")


class RegionMosaicCornerCoordinates(BaseModel):
    top_left: List[float] = Field(..., description="区域大图左上角 `[经度, 纬度]`。")
    top_right: List[float] = Field(..., description="区域大图右上角 `[经度, 纬度]`。")
    bottom_right: List[float] = Field(..., description="区域大图右下角 `[经度, 纬度]`。")
    bottom_left: List[float] = Field(..., description="区域大图左下角 `[经度, 纬度]`。")


class RegionMosaicAsset(BaseModel):
    sensor_type: str = Field(
        ...,
        description="静态 PNG 数据源目录名，例如 `s2`、`highres` 或 `embedding-v1`。",
    )
    start_date: str = Field(..., description="该传感器最早可用月份，格式 `YYYYMM`。")
    end_date: str = Field(..., description="该传感器最晚可用月份，格式 `YYYYMM`。")
    date_count: int = Field(..., description="压缩包中该传感器实际包含的大图数量。")
    available_dates: List[str] = Field(
        ...,
        description="实际可用月份列表；用于识别起止范围中间缺失的月份。",
    )
    path_template: str = Field(
        ...,
        description="ZIP 内 PNG 相对路径模板：`{regionId}/{sensor}/{date}/mosaic.png`。",
    )


class RegionMosaicInfo(BaseModel):
    crs: str = Field("EPSG:4326", description="全部静态 PNG 使用的坐标系。")
    bounds_wgs84: List[float] = Field(
        ...,
        description="区域大图统一四至 `[西, 南, 东, 北]`。",
    )
    footprint_wgs84: Dict[str, Any] = Field(
        ...,
        description="所有 Patch 合并后的 WGS84 GeoJSON Polygon/MultiPolygon。",
    )
    corner_coordinates_wgs84: RegionMosaicCornerCoordinates = Field(
        ...,
        description="区域大图外接矩形四个顶点。",
    )
    image_format: Literal["png"] = Field("png", description="静态大图统一使用 PNG。")
    transparent_background: bool = Field(
        True,
        description="没有 Patch 或没有影像的区域使用透明背景。",
    )
    package_filename: str = Field(
        "regional-mosaics.zip",
        description="交付给前端的静态大图压缩包文件名。",
    )
    assets: List[RegionMosaicAsset] = Field(
        default_factory=list,
        description="该区域可用传感器、起止月份、实际月份和 PNG 路径模板。",
    )


class RegionInfo(BaseModel):
    id: str = Field(..., description="Region identifier.")
    name: str = Field(..., description="Human-readable region name.")
    patch_count: int = Field(..., description="Number of patches in the region.")
    tasks: List[str] = Field(..., description="Task types available for the region.")
    mosaic: RegionMosaicInfo = Field(..., description="该区域静态 PNG 大图的统一空间信息。")


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
    mosaic: RegionMosaicInfo = Field(..., description="该区域静态 PNG 大图的统一空间信息。")


class RegionsResponse(BaseModel):
    regions: List[RegionInfo]


class PatchBase(BaseModel):
    patch_id: str = Field(..., description="Patch identifier.")
    bounds_wgs84: List[float] = Field(..., description="Bounding box in WGS84 [minx, miny, maxx, maxy].")
    footprint_wgs84: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Exact patch footprint polygon in WGS84 GeoJSON. Prefer this over "
            "bounds_wgs84 when drawing patch borders on a map."
        ),
    )
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
    schema_version: str = Field("2.0", description="摘要结构版本。")
    task: str = Field(..., description="Task identifier.")
    name: str = Field(..., description="Task display name.")
    region_id: Optional[str] = Field(None, description="摘要所属区域。")
    version: str = Field(..., description="Result version.")
    period: Optional[str] = Field(None, description="Comparison period, if applicable.")
    status: Optional[str] = Field(None, description="任务资产状态：ready、partial 或 unavailable。")
    analysis_scope: Dict[str, Any] = Field(
        default_factory=dict,
        description="本次分析的月份、Patch 范围和汇总方式。多个 Patch 会分别推理后汇总。",
    )
    summary_text: Optional[str] = Field(None, description="根据真实统计生成的中文综合分析。")
    analysis_notes: List[str] = Field(default_factory=list, description="供智能体快速读取的分析要点。")
    model: Dict[str, Any] = Field(default_factory=dict, description="基座模型、特征和下游头信息。")
    temporal_coverage: Dict[str, Any] = Field(default_factory=dict, description="可用月份及时间覆盖范围。")
    data_coverage: Dict[str, Any] = Field(default_factory=dict, description="预测、结果、标签和缺失 Patch 统计。")
    prediction_statistics: Dict[str, Any] = Field(default_factory=dict, description="预测值、类别或目标密度统计。")
    color_legend: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="结果图颜色说明；包含颜色、类别名称、中文含义及占比。",
    )
    image_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="结果图片尺寸、总像素数、目标像素数和目标占比。",
    )
    result_images: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="逐 Patch 公网临时图片完整 URL；图片目录每两小时自动清理。",
    )
    insights: List[Dict[str, Any]] = Field(default_factory=list, description="带证据的结构化分析结论。")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="数据缺失、覆盖不足或不可评估警告。")
    generated_at: Optional[str] = Field(None, description="摘要生成时间，ISO 8601。")
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
    training_method: Literal[
        "xuannv_earth", "traditional_ml", "aef", "dinov3_sat493m"
    ] = Field(
        "xuannv_earth",
        description=(
            "训练方式。默认 xuannv_earth：使用玄女 embedding，并按类别分别统计；"
            "海淀某类少于 10 个有效 Polygon 时采用 ExtraTrees Sparse Region，否则采用 Binary Conv 3x3；traditional_ml "
            "读取 Sentinel-2 六波段和四个光谱指数，为每个有标注类别分别训练 Random Forest；aef 使用年度 "
            "embedding（当前固定回退到 2025 年），dinov3_sat493m 使用月度光学影像，"
            "并为每个有标注类别分别训练两层像素 MLP。多类别最终仍返回一个 model_id。"
        ),
        examples=["xuannv_earth"],
    )
    epochs: int = Field(
        100,
        ge=1,
        le=1000,
        description=(
            "训练迭代次数。按类别统计：某类有效 Polygon 大于等于 10 个时使用 "
            "Binary Conv 3x3，服务端最多执行 100 轮；少于 10 个时使用免迭代的 "
            "ExtraTrees Sparse Region 少样本检测。"
        ),
        examples=[100],
    )
    class_ids: Optional[List[str]] = Field(
        None,
        description=(
            "候选目标类别 ID，可省略。后端以 GeoJSON 中实际出现的 class_id 为准，"
            "为每个有 Polygon 的类别分别训练二分类头；没有 Polygon 的类别自动跳过。"
        ),
        examples=[["cls_001"]],
    )
    description: Optional[str] = Field(
        None,
        description="Optional model description.",
        examples=["基于用户标注的建筑提取模型"],
    )
    annotations: GeoJSONFeatureCollection = Field(
        ...,
        description=(
            "GeoJSON FeatureCollection 用户标注包。坐标必须是 WGS84。"
            "Polygon 内部作为目标正样本，Polygon 外是未标注样本而不是直接负样本。"
            "后端按 class_id 独立训练；海淀某类有效 Polygon 少于 10 个时使用 ExtraTrees Sparse Region，"
            "达到 10 个时使用 Binary Conv 3x3。没有 Polygon 标注的类别自动跳过。"
        ),
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
        if self.training_method == "traditional_ml" and self.model_type != "single_time_detection":
            raise ValueError(
                "traditional_ml currently supports single_time_detection only"
            )
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
        annotated_class_ids = {
            feature.properties.class_id for feature in self.annotations.features
        }
        unknown_annotation_ids = annotated_class_ids - class_ids
        if unknown_annotation_ids:
            unknown = sorted(unknown_annotation_ids)[0]
            raise ValueError(f"feature class_id '{unknown}' is not defined in classes")
        # GeoJSON annotations are authoritative. Frontend class selectors may
        # include categories that were never annotated; those must not create
        # empty heads or block valid training.
        self.class_ids = sorted(annotated_class_ids)

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
    requested_training_method: Optional[str] = Field(
        None, description="前端请求的训练方式；旧模型和系统模型可能为空。"
    )
    resolved_training_method: Optional[str] = Field(
        None, description="后端实际执行的训练算法或模型族。"
    )
    feature_source: Optional[str] = Field(
        None, description="训练输入来源，例如 xuannv_embedding 或 sentinel2_l2a。"
    )
    foundation_model_id: Optional[str] = Field(
        None, description="与下游头绑定的基座模型 ID，例如 xuannv_earth、aef 或 dinov3_sat493m。"
    )
    foundation_model_version: Optional[str] = Field(
        None, description="训练和推理必须一致的基座模型或 embedding 版本。"
    )
    feature_dimension: Optional[int] = Field(
        None, description="下游头期望的输入特征维度。"
    )
    preprocessing_version: Optional[str] = Field(
        None, description="生成输入特征时使用的预处理契约版本。"
    )
    head_type: Optional[str] = Field(
        None, description="下游头类型，例如 sparse_region_model、binary_conv3x3、pu_query_retrieval（历史模型）或 pixel_mlp。"
    )
    class_heads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "按实际标注类别训练的下游头摘要。每项包含 class_id、训练策略和 Polygon 数量；"
            "无标注类别不会出现在此列表。"
        ),
    )
    checkpoint_format: Optional[str] = Field(
        None, description="用于后端自动分派推理流程的 checkpoint 格式。"
    )
    compatible_regions: List[str] = Field(
        default_factory=list, description="该模型允许推理的区域 ID。"
    )
    status: str = Field(..., description="Training status: running, completed, failed, or ready (system models).")
    created_at: str = Field(..., description="ISO timestamp when the model was created.")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when training completed.")
    classes: List[Dict[str, Any]] = Field(..., description="Classes used by the model.")
    accuracy: Optional[float] = Field(None, description="Training accuracy, if available.")
    metric_name: Optional[str] = Field(
        None, description="指标名称；例如 training_f1。训练集指标不等于泛化精度。"
    )
    n_samples: Optional[int] = Field(
        None, description="实际参与训练的有效 Polygon 数量；MultiPolygon 按独立 Polygon 分别计数。"
    )
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
        None,
        description="Checkpoint version for system pre-trained models. Omit to use the best available version for the region. Ignored for custom models.",
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
        description=(
            "需要批量推理的 Patch ID 列表，最多 100 个。自定义多类别模型会对每个 "
            "Patch 自动运行 model_id 中绑定的全部类别头，不需要再次传类别或训练方式。"
        ),
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
        None,
        description="Checkpoint version for system pre-trained models. Omit to use the best available version for the region. Ignored for custom models.",
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
    metric_name: Optional[str] = None
    n_samples: Optional[int] = Field(
        None, description="实际参与训练的有效 Polygon 数量；MultiPolygon 按独立 Polygon 分别计数。"
    )
    model_path: Optional[str] = Field(None, description="Path to the saved model artifact.")
    message: Optional[str] = Field(None, description="Status or error message.")
    requested_training_method: Optional[str] = None
    resolved_training_method: Optional[str] = None
    feature_source: Optional[str] = None
    foundation_model_id: Optional[str] = None
    foundation_model_version: Optional[str] = None
    feature_dimension: Optional[int] = None
    preprocessing_version: Optional[str] = None
    head_type: Optional[str] = None
    checkpoint_format: Optional[str] = None
    compatible_regions: List[str] = Field(default_factory=list)


class TrainingMethodCapability(BaseModel):
    id: Literal["xuannv_earth", "traditional_ml", "aef", "dinov3_sat493m"]
    name: str
    available: bool
    feature_source: str
    supported_model_types: List[str]
    trainer: Optional[str] = None
    selection_rule: Optional[str] = None
    required_sensor: Optional[str] = None
    unavailable_reason: Optional[str] = None


class TrainingCapabilitiesResponse(BaseModel):
    schema_version: int
    default_training_method: str
    regions: List[str]
    methods: List[TrainingMethodCapability]
    task_contracts: Dict[str, Dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
