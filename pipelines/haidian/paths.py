"""Path helpers for Haidian V1 P2A deployment assets.

The Haidian API V1 version is backed by the xuannv P2A embedding model.  Large
data and checkpoints are downloaded from ModelScope into the standard
``data/haidian`` and ``models/haidian`` folders, matching the Harbin layout.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

HAIDIAN_DIR: Path = DATA_DIR / "haidian"
HAIDIAN_MODELS_DIR: Path = MODELS_DIR / "haidian"
HAIDIAN_PATCHES_META: Path = HAIDIAN_DIR / "patches_meta_v1.json"

HAIDIAN_V1_DATA_DIR: Path = HAIDIAN_DIR
HAIDIAN_V1_MODELS_DIR: Path = HAIDIAN_MODELS_DIR / "v1"
HAIDIAN_V1_EMBEDDINGS_DIR: Path = HAIDIAN_V1_DATA_DIR / "embeddings" / "v1"
HAIDIAN_V1_TASKS_DIR: Path = HAIDIAN_V1_DATA_DIR / "tasks"

DEFAULT_MODELSCOPE_REPO = "WeijieWu/xuannv_haidian_embdding"
DEFAULT_MODELSCOPE_PREFIX = "artifacts/haidian-embedding-v1"
DEFAULT_EMBEDDING_ARTIFACT = (
    "embeddings/haidian_202512_202605_p10c_epoch800/haidian"
)

# Source locations used when preparing a local upload package from the training
# machine.  These can be overridden without editing the script.
P2A_EMBEDDING_ROOT: Path = Path(
    os.environ.get(
        "P2A_EMBEDDING_ROOT",
        "/data/xuannv_embedding/embeddings/v2_202512_202605/"
        "20260627_v2_p2a_semantic_probe_full_20260627_best_"
        "p2a_semantic_probe_full_best_20260627",
    )
)
P2A_OUTPUT_ROOT: Path = Path(
    os.environ.get(
        "P2A_OUTPUT_ROOT",
        "/data/xuannv_embedding/outputs/v2_p2a_semantic_probe_full_20260627",
    )
)
P2A_DOWNSTREAM_ROOT: Path = Path(
    os.environ.get(
        "P2A_DOWNSTREAM_ROOT",
        "/data/xuannv_embedding/experiments/v2_202512_202605/"
        "expanded_downstream/p2a_semantic_probe_full_quick_20260627_185700",
    )
)
P2A_BENCHMARK_ROOT: Path = Path(
    os.environ.get(
        "P2A_BENCHMARK_ROOT",
        "/data/xuannv_embedding/experiments/v2_202512_202605/"
        "benchmarks/p2a_semantic_probe_full_quick_20260627_185700",
    )
)
XUANNV_REPO_ROOT: Path = Path(os.environ.get("XUANNV_REPO_ROOT", "/root/workspace/xuannv"))
XUANNV_DATA_ROOT: Path = Path(os.environ.get("XUANNV_DATA_ROOT", "/data/xuannv_embedding"))

MONTHS = ("202512", "202601", "202602", "202603", "202604", "202605")


def task_dir(task: str, version: str = "v1") -> Path:
    return HAIDIAN_V1_TASKS_DIR / task / version


def task_predictions_dir(task: str, version: str = "v1") -> Path:
    return task_dir(task, version) / "predictions"


def task_results_dir(task: str, version: str = "v1") -> Path:
    return task_dir(task, version) / "results"


def task_labels_dir(task: str, version: str = "v1") -> Path:
    return task_dir(task, version) / "labels"
