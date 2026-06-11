"""Pydantic schemas for SAM3 endpoints."""

from typing import List
from pydantic import BaseModel, Field, field_validator


class ImageData(BaseModel):
    width: int
    height: int
    format: str = "png"
    data: str  # base64 encoded PNG


class EmbedRequest(BaseModel):
    patch_id: str = Field(..., min_length=1, max_length=64)
    month: str = Field(..., min_length=1, max_length=32)

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
    embedding_id: str
    status: str = "ready"
    image: ImageData


class SegmentRequest(BaseModel):
    embedding_id: str = Field(..., min_length=1)
    point_coords: List[List[float]] = Field(..., min_length=1)
    point_labels: List[int] = Field(..., min_length=1)
    multimask_output: bool = True

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
    data: str  # base64 encoded PNG
    score: float
    bbox: List[int]  # [x, y, width, height]


class SegmentResponse(BaseModel):
    masks: List[MaskData]


class CacheInfo(BaseModel):
    size: int
    max_size: int
    entries: List[str]


class GpuMemory(BaseModel):
    allocated_mb: int
    reserved_mb: int


class StatusResponse(BaseModel):
    model_loaded: bool
    device: str
    gpu_memory: GpuMemory
    cache: CacheInfo
