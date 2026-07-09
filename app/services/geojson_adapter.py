"""GeoJSON annotation adapter for custom model training.

Converts a GeoJSON FeatureCollection (WGS84 coordinates) submitted by the
frontend into per-patch 256x256 binary masks and training samples.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from rasterio import features as rasterio_features
from rasterio import Affine
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

from app.schemas.models import GeoJSONFeature, GeoJSONFeatureCollection, ModelClass
from app.services.data_service import DataService

logger = logging.getLogger(__name__)

DEFAULT_MASK_SIZE = (128, 128)


def _get_patch_bbox(region_id: str, patch_id: str) -> Tuple[float, float, float, float]:
    """Fetch WGS84 bbox [minx, miny, maxx, maxy] for a patch."""
    patch = DataService.get_patch(region_id, patch_id)
    if patch is None:
        raise ValueError(f"Patch not found: {patch_id} in region {region_id}")
    bbox = patch.get("bounds_wgs84")
    if not bbox or len(bbox) != 4:
        raise ValueError(f"Patch {patch_id} has no valid bounds_wgs84")
    return tuple(bbox)


def _get_patch_spatial_ref(region_id: str, patch_id: str) -> Dict[str, Any]:
    """Fetch patch spatial metadata for CRS-aware rasterization."""
    patch = DataService.get_patch(region_id, patch_id)
    if patch is None:
        raise ValueError(f"Patch not found: {patch_id} in region {region_id}")
    return {
        "bounds_wgs84": patch.get("bounds_wgs84"),
        "bounds": patch.get("bounds"),
        "crs": patch.get("crs"),
    }


def _build_transform(bbox: Tuple[float, float, float, float], size: Tuple[int, int]) -> Affine:
    """Build an Affine transform from pixel coords to WGS84 for the patch.

    Pixel (0, 0) corresponds to the upper-left corner (minx, maxy).
    """
    minx, miny, maxx, maxy = bbox
    x_res = (maxx - minx) / size[1]
    y_res = (miny - maxy) / size[0]  # negative
    return Affine.translation(minx, maxy) * Affine.scale(x_res, y_res)


def rasterize_geometry(
    geometry: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    size: Tuple[int, int] = DEFAULT_MASK_SIZE,
) -> np.ndarray:
    """Rasterize a GeoJSON Polygon/MultiPolygon onto a 256x256 mask.

    Args:
        geometry: GeoJSON geometry dict with WGS84 coordinates.
        bbox: Patch bounds [minx, miny, maxx, maxy] in WGS84.
        size: Output mask size (height, width).

    Returns:
        Binary uint8 mask with shape (height, width).
    """
    geom = shape(geometry)
    transform = _build_transform(bbox, size)

    mask = rasterio_features.rasterize(
        [(geom, 1)],
        out_shape=size,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    return mask


def rasterize_patch_geometry(
    geometry: Dict[str, Any],
    spatial_ref: Dict[str, Any],
    size: Tuple[int, int] = DEFAULT_MASK_SIZE,
) -> np.ndarray:
    """Rasterize WGS84 geometry using the patch's native grid when available."""
    native_bounds = spatial_ref.get("bounds")
    native_crs = spatial_ref.get("crs")
    if native_bounds and len(native_bounds) == 4 and native_crs:
        geom = shape(geometry)
        transformer = Transformer.from_crs(
            "EPSG:4326", native_crs, always_xy=True
        )
        native_geom = shapely_transform(transformer.transform, geom)
        transform = _build_transform(tuple(native_bounds), size)
        return rasterio_features.rasterize(
            [(native_geom, 1)],
            out_shape=size,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )

    bbox = spatial_ref.get("bounds_wgs84")
    if not bbox or len(bbox) != 4:
        raise ValueError("Patch has no valid spatial bounds for rasterization")
    return rasterize_geometry(geometry, tuple(bbox), size=size)


def group_features_by_patch(features: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group GeoJSON features by patch_id."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for f in features:
        patch_id = f["properties"]["patch_id"]
        groups.setdefault(patch_id, []).append(f)
    return groups


def build_class_map(classes: List[ModelClass]) -> Dict[str, int]:
    """Map class_id to a numeric label index.

    Background is implicitly label 0; user classes start from 1.
    """
    sorted_ids = sorted(c.id for c in classes)
    return {cls_id: idx + 1 for idx, cls_id in enumerate(sorted_ids)}


def parse_annotations_for_training(
    annotations: GeoJSONFeatureCollection,
    classes: List[ModelClass],
    class_ids: List[str],
    model_type: str,
) -> List[Dict[str, Any]]:
    """Parse a GeoJSON annotation package into per-patch training records.

    Each record contains:
        - region_id, patch_id, month (or before/after_month)
        - mask: 256x256 binary mask merged across all features for this patch/class
        - class_id, label_index

    Returns:
        A tuple (records, class_map) where class_map maps active class_id to
        numeric label index (background is 0, user classes start at 1).
    """
    all_class_ids = {c.id for c in classes}
    active_class_ids = set(class_ids) if class_ids else all_class_ids

    class_map = build_class_map([c for c in classes if c.id in active_class_ids])

    raw_features = [f.model_dump() for f in annotations.features]
    groups = group_features_by_patch(raw_features)

    records = []
    for patch_id, features in groups.items():
        region_id = features[0]["properties"]["region_id"]
        spatial_ref = _get_patch_spatial_ref(region_id, patch_id)

        # Merge masks per (class_id, time)
        masks: Dict[Tuple[str, str], np.ndarray] = {}
        for f in features:
            props = f["properties"]
            cls_id = props["class_id"]
            if cls_id not in active_class_ids:
                continue

            if model_type == "classification":
                time_key = props.get("month")
            else:  # change_detection
                time_key = f"{props.get('before_month')}_vs_{props.get('after_month')}"

            mask = rasterize_patch_geometry(f["geometry"], spatial_ref)
            key = (cls_id, time_key)
            if key in masks:
                masks[key] = np.maximum(masks[key], mask)
            else:
                masks[key] = mask

        for (cls_id, time_key), mask in masks.items():
            if model_type == "classification":
                month = time_key
                record = {
                    "region_id": region_id,
                    "patch_id": patch_id,
                    "month": month,
                    "class_id": cls_id,
                    "label_index": class_map[cls_id],
                    "mask": mask,
                }
            else:
                before_month, after_month = time_key.split("_vs_")
                record = {
                    "region_id": region_id,
                    "patch_id": patch_id,
                    "before_month": before_month,
                    "after_month": after_month,
                    "class_id": cls_id,
                    "label_index": class_map[cls_id],
                    "mask": mask,
                }
            records.append(record)

    if not records:
        raise ValueError("No valid training records after parsing annotations")

    return records, class_map
