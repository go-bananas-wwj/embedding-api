"""Tests for raw TIFF date resolution used by mosaic and SAM3."""

from pathlib import Path

from app.services.mosaic_service import _candidate_period_prefixes, _get_raw_tiff_path


def test_candidate_period_prefixes_do_not_use_bare_year():
    prefixes = _candidate_period_prefixes(["2025-10", "2025Q4", "202510"])

    assert "2025" not in prefixes
    assert "202510" in prefixes
    assert "202511" not in prefixes
    assert "202512" not in prefixes


def test_candidate_period_prefixes_expand_quarter_only_for_quarter():
    prefixes = _candidate_period_prefixes(["2025Q4"])

    assert prefixes == ["2025Q4", "202510", "202511", "202512"]


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
