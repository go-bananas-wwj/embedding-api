#!/usr/bin/env python3
"""Generate Harbin road extraction tiles from OpenStreetMap highway data.

Inspired by pipelines/harbin/shp_to_patch_masks.py and
xuannv_show/scripts/extract_osm_buildings.py.

Steps:
  1. Download OSM highway ways for the whole Harbin region bbox.
  2. Convert ways to LineString geometries.
  3. For each patch: reproject roads to patch CRS, clip, buffer by ~5 m,
     rasterize to a binary mask, and save as a 128×128 PNG tile.

Output layout mirrors the existing Harbin v1 result tiles:
  data/harbin/tasks/road_extraction/v1/results/tiles/patch_{id}_2025-10.png

The month suffix "2025-10" is chosen to align with Harbin building_extraction v1,
so the same front-end month picker works for both tasks.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Harbin road extraction tiles from OSM")
    parser.add_argument(
        "--patches-meta",
        type=Path,
        default=Path("data/harbin/patches_meta.json"),
        help="Path to harbin patches_meta.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/harbin/tasks/road_extraction/v1/results/tiles"),
        help="Output directory for 128x128 PNG tiles",
    )
    parser.add_argument(
        "--month",
        type=str,
        default="2025-10",
        help="Month suffix for tile filenames (e.g. 2025-10)",
    )
    parser.add_argument(
        "--buffer-meters",
        type=float,
        default=5.0,
        help="Road line buffer radius in meters",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Output tile size in pixels",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tiles",
    )
    return parser.parse_args()


def download_osm_highways(bounds_wgs84: tuple[float, float, float, float]) -> dict:
    """Download OSM highway ways for a WGS84 bbox via Overpass API."""
    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    query = (
        f'[out:json];way["highway"]'
        f"({min_lat},{min_lon},{max_lat},{max_lon});"
        "(._;>;);out body;"
    )
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers={"User-Agent": "embedding-api-osm-roads/1.0"},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def overpass_to_lines(overpass_data: dict):
    """Yield (geometry, value) tuples suitable for rasterio.features.rasterize."""
    nodes = {}
    ways = []
    for element in overpass_data.get("elements", []):
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element["type"] == "way":
            ways.append(element)

    for way in ways:
        node_ids = way.get("nodes", [])
        coords = [nodes.get(nid) for nid in node_ids if nid in nodes]
        # Filter duplicate consecutive coords
        deduped = []
        for c in coords:
            if not deduped or c != deduped[-1]:
                deduped.append(c)
        if len(deduped) >= 2:
            try:
                line = LineString(deduped)
                if line.is_valid and line.length > 0:
                    yield line
            except Exception:
                pass


def rasterize_roads_for_patch(
    patch: dict,
    roads_gdf,
    buffer_meters: float,
    tile_size: int,
):
    """Return a uint8 binary mask [H, W] for roads inside the patch."""
    bounds_utm = patch.get("bounds")
    crs = patch.get("crs", "EPSG:32652")
    if not bounds_utm or len(bounds_utm) != 4:
        raise ValueError(f"Invalid bounds for patch {patch.get('patch_id')}")

    min_x, min_y, max_x, max_y = bounds_utm
    transform = from_bounds(min_x, min_y, max_x, max_y, tile_size, tile_size)

    # Reproject and clip roads to patch bbox
    patch_box = box(min_x, min_y, max_x, max_y)
    roads_utm = roads_gdf.to_crs(crs)
    clipped = roads_utm[roads_utm.intersects(patch_box)]
    if clipped.empty:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    # Buffer lines by a few meters so thin roads are visible at 128x128
    buffered = clipped.buffer(buffer_meters, cap_style=2, join_style=2)
    shapes = ((geom, 1) for geom in buffered.geometry if geom.is_valid and not geom.is_empty)

    mask = features.rasterize(
        shapes=shapes,
        out_shape=(tile_size, tile_size),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    return mask


def mask_to_png(mask: np.ndarray) -> Image.Image:
    """Convert binary mask to RGB: road=red, background=white."""
    h, w = mask.shape
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 255
    rgb[mask > 0] = [255, 0, 0]
    return Image.fromarray(rgb)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.patches_meta) as f:
        patches = json.load(f)

    all_bounds = [p["bounds_wgs84"] for p in patches if p.get("bounds_wgs84")]
    if not all_bounds:
        raise ValueError("No bounds found in patches_meta.json")

    region_bbox = (
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    )
    print(f"Region bbox (WGS84): {region_bbox}")

    print("Downloading OSM highways from Overpass API...")
    overpass_data = download_osm_highways(region_bbox)
    lines = list(overpass_to_lines(overpass_data))
    print(f"Downloaded {len(lines)} highway ways")

    if not lines:
        print("No highways found; aborting.")
        return

    import geopandas as gpd
    roads_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

    generated = 0
    skipped = 0
    for patch in tqdm(patches, desc="Rasterizing road tiles"):
        patch_id = patch["patch_id"]
        out_path = args.output_dir / f"{patch_id}_{args.month}.png"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            mask = rasterize_roads_for_patch(
                patch, roads_gdf, args.buffer_meters, args.tile_size
            )
            img = mask_to_png(mask)
            img.save(out_path)
            generated += 1
        except Exception as e:
            print(f"[{patch_id}] Failed: {e}")

    print(f"Done. Generated: {generated}, Skipped: {skipped}, Total: {len(patches)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
