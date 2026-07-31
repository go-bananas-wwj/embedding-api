import hashlib
import json

import numpy as np
import pytest
from pyproj import Transformer
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


def test_extract_playgrounds_excludes_specialised_athletics_subfacilities():
    ring = [
        {"lon": 116.30, "lat": 39.95},
        {"lon": 116.31, "lat": 39.95},
        {"lon": 116.31, "lat": 39.96},
        {"lon": 116.30, "lat": 39.95},
    ]
    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"leisure": "track", "sport": "athletics"}, "geometry": ring},
        {"type": "way", "id": 2, "tags": {"leisure": "pitch", "sport": "athletics", "athletics": "shot_put"}, "geometry": ring},
        {"type": "way", "id": 3, "tags": {"leisure": "pitch", "sport": "athletics", "athletics": "long_jump"}, "geometry": ring},
        {"type": "way", "id": 4, "tags": {"leisure": "pitch", "sport": "athletics", "athletics": "high_jump"}, "geometry": ring},
    ]}

    assert [item.osm_id for item in extract_playgrounds(payload)] == [1]


def test_extract_playgrounds_rejects_open_or_invalid_way_geometry():
    payload = {"elements": [
        {"type": "way", "id": 1, "tags": {"leisure": "track", "sport": "athletics"},
         "geometry": [{"lon": 116.30, "lat": 39.95}, {"lon": 116.31, "lat": 39.95},
                      {"lon": 116.31, "lat": 39.96}, {"lon": 116.30, "lat": 39.96}]},
        {"type": "way", "id": 2, "tags": {"leisure": "track", "sport": "athletics"},
         "geometry": [{"lon": 116.30, "lat": 39.95}, {"lon": 116.31, "lat": 39.96},
                      {"lon": 116.31, "lat": 39.95}, {"lon": 116.30, "lat": 39.96},
                      {"lon": 116.30, "lat": 39.95}]},
    ]}

    assert extract_playgrounds(payload) == []


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


def test_rasterized_playground_transforms_wgs84_into_utm_patch():
    patch = {
        "patch_id": "patch_utm",
        "bounds": [435000.0, 4415000.0, 436000.0, 4416000.0],
        "crs": "EPSG:32650",
    }
    transformer = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
    coordinates = [
        transformer.transform(x, y)
        for x, y in [(435200.0, 4415200.0), (435800.0, 4415200.0),
                     (435800.0, 4415800.0), (435200.0, 4415200.0)]
    ]
    feature = PlaygroundFeature(1, "UTM athletics ground", Polygon(coordinates), {})

    mask = rasterize_feature(feature, patch)

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
        headers = {"Content-Type": "application/json"}

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


def test_fetch_overpass_retries_and_atomically_writes_validated_cache(tmp_path, monkeypatch):
    attempts = []
    pauses = []

    class Response:
        headers = {"Content-Type": "application/json; charset=utf-8"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"elements": [], "osm3s": {"timestamp_osm_base": "2026-07-31T00:00:00Z"}}

    def fake_post(url, **kwargs):
        attempts.append((url, kwargs))
        if len(attempts) == 1:
            raise playground_osm.requests.ConnectionError("temporary failure")
        return Response()

    monkeypatch.setattr(playground_osm.requests, "post", fake_post)
    monkeypatch.setattr(playground_osm.time, "sleep", pauses.append)
    cache_path = tmp_path / "raw.json"

    payload = playground_osm.fetch_overpass((39.9, 116.2, 40.0, 116.3), cache_path)

    assert payload["elements"] == []
    assert len(attempts) == 2
    assert pauses == [1.0]
    assert json.loads(cache_path.read_text()) == payload
    assert not list(tmp_path.glob(".raw.json.*.tmp"))


def test_fetch_overpass_rejects_an_invalid_cached_response(tmp_path):
    cache_path = tmp_path / "raw.json"
    cache_path.write_text('{"elements": "not-a-list"}', encoding="utf-8")

    with pytest.raises(ValueError, match="elements"):
        playground_osm.fetch_overpass((39.9, 116.2, 40.0, 116.3), cache_path)


def test_build_dataset_records_reference_target_metadata(tmp_path, monkeypatch):
    metadata_path = tmp_path / "patches_meta_v2.json"
    metadata_path.write_text(
        '{"city": "haidian", "patches": ['
        '{"patch_id": "patch_fixture", "bounds": [116.299, 39.949, 116.311, 39.961], '
        '"bounds_wgs84": [116.299, 39.949, 116.311, 39.961], "crs": "EPSG:4326"}]}',
        encoding="utf-8",
    )
    output_root = tmp_path / "labels"
    output_root.mkdir()
    raw_path = output_root / "osm_raw.json"
    raw_path.write_text(
        '{"osm3s": {"timestamp_osm_base": "2026-07-31T00:00:00Z"}, "elements": [{"type": "way", "id": 1, '
        '"tags": {"leisure": "track", "sport": "athletics"}, '
        '"geometry": [{"lon": 116.30, "lat": 39.95}, {"lon": 116.31, "lat": 39.95}, '
        '{"lon": 116.31, "lat": 39.96}, {"lon": 116.30, "lat": 39.95}]}]}',
        encoding="utf-8",
    )
    embeddings_root = tmp_path / "embeddings"
    embedding_path = embeddings_root / "v1" / "202604" / "patch_fixture.npy"
    embedding_path.parent.mkdir(parents=True)
    embedding_path.write_bytes(b"fixture")
    monkeypatch.setattr(playground_osm, "PATCHES_META_PATH", metadata_path)
    monkeypatch.setattr(playground_osm, "EMBEDDINGS_ROOT", embeddings_root)

    manifest = playground_osm.build_dataset(output_root)

    assert len(json.loads((output_root / "playgrounds.geojson").read_text())["features"]) == 1
    assert len(json.loads((output_root / "manifest.json").read_text())["playgrounds"]) == 1
    assert manifest["purpose"]["reference_role"] == "independent locator for the existing playground_xuannv head"
    assert manifest["target"] == {
        "embedding_version": "v1",
        "embedding_month": "202604",
        "patch_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    assert manifest["source"]["timestamp_osm_base"] == "2026-07-31T00:00:00Z"
    assert manifest["source"]["raw_response_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert manifest["source"]["attribution"] == "OpenStreetMap contributors, ODbL 1.0"


def test_build_dataset_requires_each_referenced_target_embedding(tmp_path, monkeypatch):
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
    monkeypatch.setattr(playground_osm, "EMBEDDINGS_ROOT", tmp_path / "embeddings")

    with pytest.raises(FileNotFoundError, match="patch_fixture.npy"):
        playground_osm.build_dataset(output_root)
