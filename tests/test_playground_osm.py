import json

import numpy as np
import pytest
from shapely.geometry import Polygon

import scripts.playground_osm as playground_osm
from scripts.playground_osm import PlaygroundFeature, extract_playgrounds, rasterize_feature


PATCH_FIXTURE = {
    "patch_id": "patch_fixture",
    "bounds": [116.299, 39.949, 116.311, 39.961],
    "crs": "EPSG:4326",
}


def test_extract_playgrounds_keeps_only_athletics_areas():
    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"leisure": "track", "sport": "athletics"},
         "geometry": [{"lon": 116.30, "lat": 39.95}, {"lon": 116.31, "lat": 39.95},
                      {"lon": 116.31, "lat": 39.96}, {"lon": 116.30, "lat": 39.95}]},
        {"type": "way", "id": 2, "tags": {"leisure": "pitch", "sport": "basketball"},
         "geometry": [{"lon": 116.30, "lat": 39.95}] * 4},
    ]}

    assert [item.osm_id for item in extract_playgrounds(payload)] == [1]


def test_rasterized_playground_intersects_only_matching_patch():
    feature = PlaygroundFeature(
        osm_id=1,
        name="测试田径场",
        geometry=Polygon([(116.30, 39.95), (116.31, 39.95),
                          (116.31, 39.96), (116.30, 39.96)]),
        tags={"leisure": "track", "sport": "athletics"},
    )

    mask = rasterize_feature(feature, PATCH_FIXTURE)

    assert mask.dtype == np.bool_
    assert 0 < mask.sum() < mask.size


def test_load_patches_reads_the_metadata_envelope(tmp_path, monkeypatch):
    metadata_path = tmp_path / "patches_meta_v2.json"
    metadata_path.write_text(
        '{"city": "haidian", "patches": [{"patch_id": "patch_fixture"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(playground_osm, "PATCHES_META_PATH", metadata_path)

    assert playground_osm._load_patches() == [{"patch_id": "patch_fixture"}]


def test_fetch_overpass_identifies_the_dataset_builder(tmp_path, monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"elements": []}

    def fake_post(url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return Response()

    monkeypatch.setattr(playground_osm.requests, "post", fake_post)

    payload = playground_osm.fetch_overpass((39.9, 116.2, 40.0, 116.3), tmp_path / "raw.json")

    assert payload == {"elements": []}
    assert request["headers"]["User-Agent"] == "embedding-api-haidian-osm-labels/1.0"


def test_build_dataset_writes_partial_review_artifacts_before_coverage_error(tmp_path, monkeypatch):
    metadata_path = tmp_path / "patches_meta_v2.json"
    metadata_path.write_text(
        '{"city": "haidian", "patches": ['
        '{"patch_id": "patch_fixture", "bounds": [116.299, 39.949, 116.311, 39.961], '
        '"bounds_wgs84": [116.299, 39.949, 116.311, 39.961], "crs": "EPSG:4326"}]}',
        encoding="utf-8",
    )
    output_root = tmp_path / "labels"
    output_root.mkdir()
    (output_root / "osm_raw.json").write_text(
        '{"elements": [{"type": "way", "id": 1, '
        '"tags": {"leisure": "track", "sport": "athletics"}, '
        '"geometry": [{"lon": 116.30, "lat": 39.95}, {"lon": 116.31, "lat": 39.95}, '
        '{"lon": 116.31, "lat": 39.96}, {"lon": 116.30, "lat": 39.95}]}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(playground_osm, "PATCHES_META_PATH", metadata_path)

    with pytest.raises(RuntimeError, match="Coverage constraint failed"):
        playground_osm.build_dataset(output_root)

    assert len(json.loads((output_root / "playgrounds.geojson").read_text())["features"]) == 1
    assert len(json.loads((output_root / "manifest.json").read_text())["playgrounds"]) == 1
