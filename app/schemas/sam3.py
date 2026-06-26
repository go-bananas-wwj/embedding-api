"""Pydantic schemas for SAM3 endpoints."""

from typing import List
from pydantic import BaseModel, Field, field_validator


class ImageData(BaseModel):
    width: int = Field(..., description="Image width in pixels.")
    height: int = Field(..., description="Image height in pixels.")
    format: str = Field("png", description="Image format.")
    data: str = Field(..., description="Base64-encoded PNG image.")


class EmbedRequest(BaseModel):
    patch_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    )
    month: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Sentinel-2 month used to load imagery, e.g. 2025-10.",
        examples=["2025-10"],
    )

    @field_validator("patch_id")
    @classmethod
    def validate_patch_id(cls, v: str) -> str:
        import re
        if not re.match(r"^[\w\-]+$", v):
            raise ValueError("Invalid patch_id format")
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
    image: ImageData


class SegmentRequest(BaseModel):
    embedding_id: str = Field(
        ...,
        min_length=1,
        description="Embedding ID returned by /regions/{region_id}/sam3/embed.",
        examples=["harbin_patch_000000_2025-10"],
    )
    point_coords: List[List[float]] = Field(
        ...,
        min_length=1,
        description="List of normalized point coordinates [x, y] in [0, 1].",
        examples=[[[0.5, 0.5]]],
    )
    point_labels: List[int] = Field(
        ...,
        min_length=1,
        description="Point labels: 1 for positive (foreground), 0 for negative (background). Must match point_coords length.",
        examples=[[1]],
    )
    multimask_output: bool = Field(
        True,
        description="If true, return multiple candidate masks.",
        examples=[True],
    )

    @field_validator("point_coords")
    @classmethod
    def validate_coords(cls, v: List[List[float]]) -> List[List[float]]:
        for coord in v:
            if len(coord) != 2:
                raise ValueError("Each point must have exactly 2 coordinates [x, y]")
            if not (0.0 <= coord[0] <= 1.0 and 0.0 <= coord[1] <= 1.0):
                raise ValueError("Coordinates must be in [0, 1]")
        return v

    @field_validator("point_labels")
    @classmethod
    def validate_labels(cls, v: List[int], info) -> List[int]:
        point_coords = info.data.get("point_coords", [])
        if len(v) != len(point_coords):
            raise ValueError("point_labels length must match point_coords length")
        for label in v:
            if label not in (0, 1):
                raise ValueError("point_labels must be 0 (negative) or 1 (positive)")
        return v


class MaskData(BaseModel):
    data: str = Field(..., description="Base64-encoded PNG mask.")
    score: float = Field(..., description="Mask confidence score.")
    bbox: List[int] = Field(..., description="Bounding box [x, y, width, height].")


class SegmentResponse(BaseModel):
    masks: List[MaskData]


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
