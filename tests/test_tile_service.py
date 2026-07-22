"""Patch tile filename and period filtering regressions."""

from app.services.tile_service import TileService


def test_parse_change_detection_tile_preserves_full_period():
    assert TileService._parse_patch_tile(
        "patch_000404_2025-04_vs_2025-06.png"
    ) == ("patch_000404", "2025-04_vs_2025-06")


def test_parse_monthly_and_periodless_tiles():
    assert TileService._parse_patch_tile("patch_000010_202604.png") == (
        "patch_000010",
        "202604",
    )
    assert TileService._parse_patch_tile("patch_000010.png") == (
        "patch_000010",
        None,
    )
