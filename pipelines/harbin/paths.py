"""Path helpers for Harbin ModelScope asset packaging and download.

The Harbin API assets (embeddings, task results, system-model checkpoints,
SAM3 weights, and raw satellite scenes) are uploaded to ModelScope so that
deployments can pull them on demand instead of keeping them in the Git repo.
"""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

HARBIN_DIR: Path = DATA_DIR / "harbin"
HARBIN_MODELS_DIR: Path = MODELS_DIR / "harbin"
HARBIN_PATCHES_META: Path = HARBIN_DIR / "patches_meta.json"

SAM3_MODELS_DIR: Path = MODELS_DIR / "sam3"

RAW_HARBIN_DIR: Path = Path("/workspace/raw/harbin")
RAW_HARBIN_SCENES_DIR: Path = Path("/workspace/raw/harbin_scenes")

DEFAULT_MODELSCOPE_REPO = "WeijieWu/xuannv_embdding_api"
DEFAULT_MODELSCOPE_PREFIX = "harbin/v1/api_ready"
