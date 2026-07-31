"""Build reproducible Haidian athletics-playground labels from OpenStreetMap."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as transform_geometry


ROOT = Path(__file__).resolve().parents[1]
PATCHES_META_PATH = ROOT / "data/haidian/patches_meta_v2.json"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_USER_AGENT = "embedding-api-haidian-osm-labels/1.0"
MIN_PIXEL_COVERAGE = 4
MIN_PLAYGROUNDS = 3
OVERPASS_QUERY = """
[out:json][timeout:120];
(
  way["leisure"="track"]["sport"="athletics"]({south},{west},{north},{east});
  way["leisure"="pitch"]["sport"="athletics"]({south},{west},{north},{east});
);
out tags geom;
"""


@dataclass(frozen=True)
class PlaygroundFeature:
    osm_id: int
    name: str
    geometry: Polygon
    tags: dict[str, str]


def fetch_overpass(bounds: tuple[float, float, float, float], cache_path: Path) -> dict:
    """Fetch an Overpass response once, retaining the raw payload as a local cache."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    south, west, north, east = bounds
    query = OVERPASS_QUERY.format(south=south, west=west, north=north, east=east)
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={"User-Agent": OVERPASS_USER_AGENT},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def extract_playgrounds(payload: dict) -> list[PlaygroundFeature]:
    """Keep only valid OSM athletics-track or athletics-pitch polygons."""
    result = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        if tags.get("sport") != "athletics":
            continue
        if tags.get("leisure") not in {"track", "pitch"}:
            continue
        coordinates = [
            (point["lon"], point["lat"])
            for point in element.get("geometry", [])
            if "lon" in point and "lat" in point
        ]
        if len(coordinates) < 4:
            continue
        polygon = Polygon(coordinates).buffer(0)
        if polygon.is_empty or not polygon.is_valid or polygon.geom_type != "Polygon":
            continue
        result.append(
            PlaygroundFeature(
                osm_id=int(element["id"]),
                name=tags.get("name:zh") or tags.get("name") or f"OSM way {element['id']}",
                geometry=polygon,
                tags={str(key): str(value) for key, value in tags.items()},
            )
        )
    return sorted(result, key=lambda feature: feature.osm_id)


def rasterize_feature(
    feature: PlaygroundFeature,
    patch: dict,
    shape: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Return a boolean mask for an OSM polygon in one projected embedding patch."""
    height, width = shape
    transformer = Transformer.from_crs("EPSG:4326", patch["crs"], always_xy=True)
    geometry = transform_geometry(transformer.transform, feature.geometry)
    if not geometry.intersects(box(*patch["bounds"])):
        return np.zeros(shape, dtype=bool)
    transform = from_bounds(*patch["bounds"], width=width, height=height)
    return rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def _load_patches() -> list[dict[str, Any]]:
    payload = json.loads(PATCHES_META_PATH.read_text(encoding="utf-8"))
    patches = payload.get("patches") if isinstance(payload, dict) else payload
    if not isinstance(patches, list):
        raise ValueError(f"Expected a patch list or patches envelope in {PATCHES_META_PATH}")
    return patches


def _patch_bounds_wgs84(patches: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    bounds = [patch["bounds_wgs84"] for patch in patches]
    return (
        min(item[1] for item in bounds),
        min(item[0] for item in bounds),
        max(item[3] for item in bounds),
        max(item[2] for item in bounds),
    )


def build_dataset(output_root: Path) -> dict:
    """Fetch, filter, rasterize, and write the reviewable OSM playground dataset."""
    patches = _load_patches()
    bounds = _patch_bounds_wgs84(patches)
    raw_path = output_root / "osm_raw.json"
    payload = fetch_overpass(bounds, raw_path)
    features = extract_playgrounds(payload)

    manifest_items = []
    geojson_features = []
    for feature in features:
        matches = []
        for patch in patches:
            mask = rasterize_feature(feature, patch)
            pixel_count = int(mask.sum())
            if pixel_count >= MIN_PIXEL_COVERAGE:
                matches.append({"patch_id": patch["patch_id"], "pixel_count": pixel_count})
        if not matches:
            continue
        manifest_items.append(
            {
                "osm_id": feature.osm_id,
                "name": feature.name,
                "tags": feature.tags,
                "patches": matches,
            }
        )
        geojson_features.append(
            {
                "type": "Feature",
                "id": feature.osm_id,
                "properties": {
                    "osm_id": feature.osm_id,
                    "name": feature.name,
                    "tags": feature.tags,
                    "patch_count": len(matches),
                },
                "geometry": mapping(feature.geometry),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    geojson = {"type": "FeatureCollection", "features": geojson_features}
    manifest = {
        "source": {
            "overpass_url": OVERPASS_URL,
            "bounds_wgs84": list(bounds),
            "query": OVERPASS_QUERY.strip(),
            "raw_response": raw_path.name,
        },
        "constraints": {
            "tags": {"leisure": ["track", "pitch"], "sport": "athletics"},
            "minimum_patch_pixels": MIN_PIXEL_COVERAGE,
            "minimum_playgrounds": MIN_PLAYGROUNDS,
        },
        "playgrounds": manifest_items,
    }
    (output_root / "playgrounds.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if len(manifest_items) < MIN_PLAYGROUNDS:
        raise RuntimeError(
            "Coverage constraint failed: expected at least "
            f"{MIN_PLAYGROUNDS} athletics polygons covering at least "
            f"{MIN_PIXEL_COVERAGE} embedding pixels in a Haidian patch; found "
            f"{len(manifest_items)}."
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/haidian/labels/osm_playgrounds",
        help="Directory for the cached response and filtered label artifacts.",
    )
    args = parser.parse_args()
    try:
        manifest = build_dataset(args.output)
    except (OSError, ValueError, requests.RequestException, RuntimeError) as error:
        print(error)
        return 1
    print(f"Wrote {len(manifest['playgrounds'])} athletics playground polygons to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
