"""Immutable foundation-model bindings for custom downstream heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import torch


_FOUNDATION_DEFAULTS = {
    "xuannv_earth": ("xuannv_embedding", "p10c_embedding"),
    "aef": ("aef", "aef_precomputed_v1"),
    "dinov3_sat493m": ("dinov3_sat493m", "dinov3_sat493m_token14_v2"),
    "traditional_ml": ("sentinel2_l2a", "s2_6band_indices_v1"),
}


def build_model_binding(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a stable runtime manifest from a saved model artifact."""
    checkpoint_format = str(checkpoint.get("__format__", "legacy_sklearn"))
    method = checkpoint.get("training_method")
    if checkpoint_format == "s2_random_forest_v1":
        method = "traditional_ml"
    elif checkpoint_format in {
        "torch_fewshot_head",
        "pu_query_retrieval_v1",
        "sparse_region_model_v1",
    }:
        method = "xuannv_earth"
    elif checkpoint_format == "external_embedding_mlp_v1":
        method = method or checkpoint.get("feature_source")
    else:
        method = method or "xuannv_earth"

    feature_source, preprocessing = _FOUNDATION_DEFAULTS.get(
        str(method), (str(method), f"{method}_preprocessing_v1")
    )
    if method == "xuannv_earth":
        foundation_version = str(checkpoint.get("embedding_version", "legacy"))
        preprocessing = f"{preprocessing}_{foundation_version}"
    elif method == "dinov3_sat493m":
        foundation_version = "sat493m_token14_v2"
    elif method == "aef":
        foundation_version = str(checkpoint.get("foundation_model_version", "aef_embedding_v1"))
    else:
        foundation_version = str(
            checkpoint.get("foundation_model_version", "sentinel2_latest_scene_v1")
        )

    dimension = checkpoint.get("embed_dim")
    if dimension is None and checkpoint_format == "s2_random_forest_v1":
        dimension = len(checkpoint.get("feature_names", [])) or None
    region = checkpoint.get("region_id")
    return {
        "foundation_model_id": str(method),
        "foundation_model_version": foundation_version,
        "feature_source": checkpoint.get("feature_source", feature_source),
        "feature_dimension": int(dimension) if dimension is not None else None,
        "preprocessing_version": checkpoint.get(
            "preprocessing_version", preprocessing
        ),
        "head_type": checkpoint.get("head_type", "random_forest" if method == "traditional_ml" else "legacy_head"),
        "checkpoint_format": checkpoint_format,
        "compatible_regions": [str(region)] if region else [],
    }


def load_model_binding(model_path: Path) -> Dict[str, Any]:
    """Load an artifact and return its runtime binding manifest."""
    try:
        checkpoint = joblib.load(model_path)
    except Exception:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Custom model checkpoint must contain a metadata dictionary")
    return build_model_binding(checkpoint)


def validate_model_binding(
    registry_record: Dict[str, Any],
    checkpoint: Dict[str, Any],
    requested_region: str,
) -> Dict[str, Any]:
    """Reject region or foundation-model mismatches before feature loading."""
    binding = build_model_binding(checkpoint)
    trained_region = checkpoint.get("region_id") or registry_record.get("region_id")
    if trained_region and requested_region != trained_region:
        raise ValueError(
            f"Model is trained for region '{trained_region}', not '{requested_region}'"
        )

    for field in (
        "foundation_model_id",
        "foundation_model_version",
        "feature_dimension",
        "preprocessing_version",
        "head_type",
        "checkpoint_format",
    ):
        recorded = registry_record.get(field)
        actual = binding.get(field)
        if recorded is not None and actual is not None and recorded != actual:
            raise ValueError(
                f"Model binding mismatch for {field}: registry={recorded!r}, checkpoint={actual!r}"
            )
    return binding
