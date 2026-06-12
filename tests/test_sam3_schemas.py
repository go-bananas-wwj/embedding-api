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
        embedding_id="harbin_patch_000000_2025-10",
        point_coords=[[0.5, 0.5]],
        point_labels=[1],
    )
    assert req.multimask_output is True


def test_segment_request_coords_out_of_range():
    with pytest.raises(ValidationError):
        SegmentRequest(
            embedding_id="test",
            point_coords=[[1.5, 0.5]],
            point_labels=[1],
        )


def test_segment_request_mismatched_lengths():
    with pytest.raises(ValidationError):
        SegmentRequest(
            embedding_id="test",
            point_coords=[[0.5, 0.5], [0.3, 0.3]],
            point_labels=[1],
        )
