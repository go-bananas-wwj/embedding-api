"""统一路径管理 — embedding-api 版本.

供 pipelines/harbin/ 下所有脚本使用，路径基于 embedding-api 项目根目录。
"""
from __future__ import annotations

import os
from pathlib import Path


# 项目根目录: pipelines/harbin/ -> pipelines/ -> embedding-api/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
PIPELINES_DIR: Path = PROJECT_ROOT / "pipelines"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Harbin 数据根目录
HARBIN_DIR: Path = DATA_DIR / "harbin"
HARBIN_MODELS_DIR: Path = MODELS_DIR / "harbin"
HARBIN_PATCHES_META: Path = HARBIN_DIR / "patches_meta.json"
HARBIN_EMBEDDINGS_DIR: Path = HARBIN_DIR / "embeddings"

# 原始 embedding 源数据（外部大文件，非 API 内部数据）
# 用于 inference / training，默认指向 xuannv_modelscope_upload 路径
RAW_EMBEDDINGS_DIR: Path = Path(os.environ.get(
    "RAW_EMBEDDINGS_DIR",
    "/workspace/raw/xuannv_modelscope_upload/embeddings/v5_mixed_scale/monthly_embeddings_2025",
))

# S2 原始影像（外部，未复制到 embedding-api）
S2_DIR: Path = Path(os.environ.get(
    "S2_DIR",
    "/workspace/raw/harbin_scenes/s2",
))

# Shapefile / Excel 源数据（外部）
SHP_DIR: Path = Path(os.environ.get(
    "SHP_DIR",
    "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测shp文件",
))
EXCEL_DIR: Path = Path(os.environ.get(
    "EXCEL_DIR",
    "/workspace/哈尔滨松北新区变化检测汇总文件/变化检测清单",
))


def get_task_dir(task: str, version: str = "v1") -> Path:
    """获取任务数据目录，例如 construction/v1."""
    return HARBIN_DIR / "tasks" / task / version


def get_task_results_dir(task: str, version: str = "v1") -> Path:
    return get_task_dir(task, version) / "results"


def get_task_predictions_dir(task: str, version: str = "v1") -> Path:
    return get_task_dir(task, version) / "predictions"


def get_task_labels_dir(task: str, version: str = "v1") -> Path:
    return get_task_dir(task, version) / "labels"


def get_task_label_vis_dir(task: str, version: str = "v1") -> Path:
    return get_task_dir(task, version) / "label_vis"


def get_models_dir(version: str = "v1") -> Path:
    return HARBIN_MODELS_DIR / version
