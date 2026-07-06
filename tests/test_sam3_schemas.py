"""Tests for SAM3 schemas."""

import pytest
from pydantic import ValidationError
from app.schemas.sam3 import EmbedRequest, SegmentRequest


def test_embed_request_valid():
    req = EmbedRequest(patch_id="patch_000000", month="2025-10")
    assert req.patch_id == "patch_000000"


def test_embed_request_invalid_patch_id():
    with pytest.raises(ValidationError):
        EmbedRequest(patch_id="../../etc/passwd", month="2025-10")


def test_segment_request_valid():
    req = SegmentRequest(
        date="2025-10",
        sensor_type="s2",
        point_coords=[[126.52, 45.75]],
    )
    assert req.point_labels is None
    assert req.multimask_output is False


def test_segment_request_rejects_legacy_month():
    with pytest.raises(ValidationError):
        SegmentRequest(
            date="2025-10",
            month="2025-10",
            sensor_type="s2",
            point_coords=[[126.52, 45.75]],
        )


def test_segment_request_coords_out_of_range():
    with pytest.raises(ValidationError):
        SegmentRequest(
            date="2025-10",
            sensor_type="s2",
            point_coords=[[181.0, 45.75]],
            point_labels=[1],
        )


def test_segment_request_mismatched_lengths():
    with pytest.raises(ValidationError):
        SegmentRequest(
            date="2025-10",
            sensor_type="s2",
            point_coords=[[126.52, 45.75], [126.521, 45.751]],
            point_labels=[1],
        )
