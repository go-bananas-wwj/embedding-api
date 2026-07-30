"""Static region mosaic geometry and packaged asset inventory."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

from app.config import get_config
from app.services.data_service import DataService, DataNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = PROJECT_ROOT / "static_mosaic_inventory.json"


@lru_cache(maxsize=1)
def load_static_mosaic_inventory() -> Dict[str, Any]:
    """Load the small inventory committed alongside the frontend package."""
    with INVENTORY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=4)
def get_region_mosaic_info(region_id: str) -> Dict[str, Any]:
    """Return canonical WGS84 geometry and static PNG inventory for a region."""
    config = get_config()
    patches = config.get_patches(region_id)
    polygons = []
    for patch in patches:
        footprint = DataService._build_patch_footprint_wgs84(patch)
        if footprint and footprint.get("type") == "Polygon":
            polygons.append(Polygon(footprint["coordinates"][0]))
    if not polygons:
        raise DataNotFoundError(
            f"No WGS84 Patch footprints found for region '{region_id}'"
        )

    geometry = unary_union(polygons)
    west, south, east, north = geometry.bounds
    inventory = load_static_mosaic_inventory()
    region_assets = inventory.get("regions", {}).get(region_id, {}).get("assets", {})
    assets: List[Dict[str, Any]] = [
        {
            "sensor_type": sensor,
            "start_date": sorted(set(dates))[0],
            "end_date": sorted(set(dates))[-1],
            "date_count": len(set(dates)),
            "available_dates": sorted(set(dates)),
            "path_template": f"{region_id}/{sensor}/{{date}}/mosaic.png",
        }
        for sensor, dates in sorted(region_assets.items())
        if dates
    ]
    return {
        "crs": "EPSG:4326",
        "bounds_wgs84": [west, south, east, north],
        "footprint_wgs84": mapping(geometry),
        "corner_coordinates_wgs84": {
            "top_left": [west, north],
            "top_right": [east, north],
            "bottom_right": [east, south],
            "bottom_left": [west, south],
        },
        "image_format": "png",
        "transparent_background": True,
        "package_filename": inventory.get(
            "package_filename", "regional-mosaics.zip"
        ),
        "assets": assets,
    }
