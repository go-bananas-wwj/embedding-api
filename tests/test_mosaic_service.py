"""Tests for raw TIFF date resolution used by mosaic and SAM3."""

from pathlib import Path

from app.services.mosaic_service import (
    _candidate_period_prefixes,
    _available_sensor_months,
    _configured_sensor_roots,
    _get_raw_tiff_path,
    _sensor_storage_key,
)


def test_configured_sensor_roots_use_the_requested_sensor_directory():
    roots = _configured_sensor_roots(
        {
            "s2_dir": "/data/s2",
            "highres_dir": "/data/highres_optical",
        },
        "highres",
        raw_root="/data/raw",
    )

    assert roots == ["/data/raw", "/data/highres_optical"]


def test_configured_sensor_roots_cover_frontend_optical_and_sar_values():
    region = {
        "s1_dir": "/data/s1",
        "s2_dir": "/data/s2",
        "landsat_dir": "/data/landsat",
        "highres_dir": "/data/highres_optical",
    }

    assert _configured_sensor_roots(region, "s1", raw_root="/raw") == [
        "/raw",
        "/data/s1",
    ]
    assert _configured_sensor_roots(region, "s2", raw_root="/raw") == [
        "/raw",
        "/data/s2",
    ]
    assert _configured_sensor_roots(region, "landsat", raw_root="/raw") == [
        "/raw",
        "/data/landsat",
    ]
    assert _configured_sensor_roots(region, "highres", raw_root="/raw") == [
        "/raw",
        "/data/highres_optical",
    ]


def test_frontend_sensor_aliases_use_the_expected_storage_prefix():
    assert _sensor_storage_key("highres") == "highres_optical"
    assert _sensor_storage_key("highres_sar") == "highres_sar"
    assert _sensor_storage_key("s1_hr") == "s1_hr"
    assert _sensor_storage_key("s2_hr") == "s2_hr"


def test_available_sensor_months_reads_patch_layout(tmp_path: Path):
    sensor_root = tmp_path / "s1_hr"
    patch = sensor_root / "patch_000000"
    patch.mkdir(parents=True)
    (patch / "20250627.tif").touch()
    (patch / "20250810.tif").touch()

    months = _available_sensor_months(
        [str(sensor_root)], "harbin", "s1_hr"
    )

    assert months == ["202506", "202508"]


def test_candidate_period_prefixes_do_not_use_bare_year():
    prefixes = _candidate_period_prefixes(["2025-10", "2025Q4", "202510"])

    assert "2025" not in prefixes
    assert "202510" in prefixes
    assert "202511" not in prefixes
    assert "202512" not in prefixes


def test_candidate_period_prefixes_do_not_expose_quarter_as_months():
    prefixes = _candidate_period_prefixes(["2025Q4"])

    assert prefixes == []


def test_get_raw_tiff_path_prefers_requested_month_over_same_year(tmp_path: Path):
    layout = tmp_path / "harbin" / "s2" / "patch_000000"
    layout.mkdir(parents=True)
    january = layout / "20250101.tif"
    october = layout / "20251003.tif"
    january.write_text("jan")
    october.write_text("oct")

    path = _get_raw_tiff_path(
        "harbin",
        "patch_000000",
        "s2",
        ["202510", "2025Q4"],
        roots=[str(tmp_path)],
    )

    assert path == str(october)


def test_get_raw_tiff_path_uses_latest_scene_inside_month(tmp_path: Path):
    layout = tmp_path / "harbin" / "s2" / "patch_000000"
    layout.mkdir(parents=True)
    later = layout / "20251020.tif"
    earlier = layout / "20251003.tif"
    latest = layout / "20251027.tif"
    later.write_text("later")
    earlier.write_text("earlier")
    latest.write_text("latest")

    path = _get_raw_tiff_path(
        "harbin",
        "patch_000000",
        "s2",
        ["202510"],
        roots=[str(tmp_path)],
    )

    assert path == str(latest)


def test_get_raw_tiff_path_exact_day_does_not_fall_back_to_month(tmp_path: Path):
    layout = tmp_path / "harbin" / "s2" / "patch_000000"
    layout.mkdir(parents=True)
    nearby = layout / "20251004.tif"
    nearby.write_text("nearby")

    path = _get_raw_tiff_path(
        "harbin",
        "patch_000000",
        "s2",
        ["20251003"],
        roots=[str(tmp_path)],
    )

    assert path is None


def test_get_raw_tiff_path_exact_day_prefers_exact_match(tmp_path: Path):
    layout = tmp_path / "harbin" / "s2" / "patch_000000"
    layout.mkdir(parents=True)
    exact = layout / "20251003.tif"
    later = layout / "20251004.tif"
    exact.write_text("exact")
    later.write_text("later")

    path = _get_raw_tiff_path(
        "harbin",
        "patch_000000",
        "s2",
        ["20251003"],
        roots=[str(tmp_path)],
    )

    assert path == str(exact)


def test_get_raw_tiff_path_month_daily_beats_legacy_quarter(tmp_path: Path):
    layout = tmp_path / "harbin" / "s2" / "patch_000000"
    layout.mkdir(parents=True)
    legacy_quarter = layout / "2025Q4.tif"
    monthly_scene = layout / "20251020.tif"
    legacy_quarter.write_text("quarter")
    monthly_scene.write_text("month")

    path = _get_raw_tiff_path(
        "harbin",
        "patch_000000",
        "s2",
        ["2025-10", "2025Q4", "202510"],
        roots=[str(tmp_path)],
    )

    assert path == str(monthly_scene)


def test_get_raw_tiff_path_month_can_fall_back_to_legacy_quarter(tmp_path: Path):
    layout = tmp_path / "harbin" / "s2" / "patch_000000"
    layout.mkdir(parents=True)
    legacy_quarter = layout / "2025Q4.tif"
    legacy_quarter.write_text("quarter")

    path = _get_raw_tiff_path(
        "harbin",
        "patch_000000",
        "s2",
        ["2025-10", "2025Q4", "202510"],
        roots=[str(tmp_path)],
    )

    assert path == str(legacy_quarter)


def test_get_raw_tiff_path_supports_flat_extracted_haidian_scenes(tmp_path: Path):
    scene_dir = tmp_path / "s2"
    scene_dir.mkdir()
    older = scene_dir / "s2_20260302_patch_000212.tif"
    latest = scene_dir / "s2_20260327_patch_000212.tif"
    older.touch()
    latest.touch()

    path = _get_raw_tiff_path(
        "haidian",
        "patch_000212",
        "s2",
        ["202603", "2026-03"],
        roots=[str(scene_dir)],
    )

    assert path == str(latest)
