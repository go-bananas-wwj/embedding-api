"""Tests for raw TIFF date resolution used by mosaic and SAM3."""

from pathlib import Path

from app.services.mosaic_service import _candidate_period_prefixes, _get_raw_tiff_path


def test_candidate_period_prefixes_do_not_use_bare_year():
    prefixes = _candidate_period_prefixes(["2025-10", "2025Q4", "202510"])

    assert "2025" not in prefixes
    assert "202510" in prefixes
    assert "202511" in prefixes
    assert "202512" in prefixes


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


def test_get_raw_tiff_path_uses_first_sorted_image_inside_month(tmp_path: Path):
    layout = tmp_path / "harbin" / "s2" / "patch_000000"
    layout.mkdir(parents=True)
    later = layout / "20251020.tif"
    earlier = layout / "20251003.tif"
    later.write_text("later")
    earlier.write_text("earlier")

    path = _get_raw_tiff_path(
        "harbin",
        "patch_000000",
        "s2",
        ["202510"],
        roots=[str(tmp_path)],
    )

    assert path == str(earlier)
