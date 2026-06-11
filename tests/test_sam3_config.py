"""Tests for SAM3 config integration."""

from app.config import get_config


def test_sam3_config_exists():
    config = get_config()
    sam3_cfg = config.get_sam3_config()
    assert sam3_cfg is not None
    assert "model_path" in sam3_cfg
    assert "max_cache_size" in sam3_cfg


def test_region_s2_dir():
    config = get_config()
    harbin = config.get_region("harbin")
    assert harbin is not None
    assert "s2_dir" in harbin
    assert "/workspace/raw/harbin_scenes/s2" in harbin["s2_dir"]
