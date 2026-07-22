"""Immutable foundation-model binding tests for custom downstream heads."""

import pytest

from app.services.model_binding import build_model_binding, validate_model_binding


def test_build_external_embedding_binding():
    binding = build_model_binding(
        {
            "__format__": "external_embedding_mlp_v1",
            "training_method": "dinov3_sat493m",
            "head_type": "pixel_mlp",
            "embed_dim": 1024,
            "region_id": "haidian",
        }
    )

    assert binding == {
        "foundation_model_id": "dinov3_sat493m",
        "foundation_model_version": "sat493m_token14_v2",
        "feature_source": "dinov3_sat493m",
        "feature_dimension": 1024,
        "preprocessing_version": "dinov3_sat493m_token14_v2",
        "head_type": "pixel_mlp",
        "checkpoint_format": "external_embedding_mlp_v1",
        "compatible_regions": ["haidian"],
    }


def test_validate_binding_rejects_request_for_another_region():
    checkpoint = {
        "__format__": "torch_fewshot_head",
        "head_type": "binary_conv3x3",
        "embed_dim": 64,
        "embedding_version": "v1",
        "region_id": "haidian",
    }

    with pytest.raises(ValueError, match="trained for region 'haidian'"):
        validate_model_binding({}, checkpoint, requested_region="harbin")


def test_validate_binding_rejects_registry_checkpoint_mismatch():
    checkpoint = {
        "__format__": "external_embedding_mlp_v1",
        "training_method": "aef",
        "head_type": "pixel_mlp",
        "embed_dim": 64,
        "region_id": "haidian",
    }
    registry_record = {"foundation_model_id": "dinov3_sat493m"}

    with pytest.raises(ValueError, match="foundation_model_id"):
        validate_model_binding(
            registry_record, checkpoint, requested_region="haidian"
        )
