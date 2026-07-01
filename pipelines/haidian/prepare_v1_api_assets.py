#!/usr/bin/env python3
"""Prepare Haidian API V1 assets from xuannv P2A experiment outputs.

This script is intended to run on the training machine.  It creates an
``api_ready`` directory that can be uploaded to ModelScope and later unpacked
directly into the ``embedding-api`` project root.

Outputs:
  api_ready/data/haidian/...
  api_ready/models/haidian/...
  manifest.json
  checksums.sha256
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pipelines.haidian.paths import (  # noqa: E402
    MONTHS,
    P2A_BENCHMARK_ROOT,
    P2A_DOWNSTREAM_ROOT,
    P2A_EMBEDDING_ROOT,
    P2A_OUTPUT_ROOT,
    XUANNV_DATA_ROOT,
    XUANNV_REPO_ROOT,
)


TASK_SOURCES = {
    "building_extraction": {
        "prediction_dir": P2A_DOWNSTREAM_ROOT
        / "unet"
        / "haidian_building_osm"
        / "fold_0"
        / "predictions",
        "label_dir": XUANNV_DATA_ROOT / "processed/haidian/labels/building_osm",
        "head_dirs": {
            "linear": P2A_DOWNSTREAM_ROOT / "linear/haidian_building_osm/fold_0",
            "mlp": P2A_DOWNSTREAM_ROOT / "mlp/haidian_building_osm/fold_0",
            "unet": P2A_DOWNSTREAM_ROOT / "unet/haidian_building_osm/fold_0",
        },
    },
    "road_extraction": {
        "prediction_dir": P2A_DOWNSTREAM_ROOT
        / "unet"
        / "haidian_road_osm"
        / "fold_0"
        / "predictions",
        "label_dir": XUANNV_DATA_ROOT / "processed/haidian/labels/road_osm",
        "head_dirs": {
            "linear": P2A_DOWNSTREAM_ROOT / "linear/haidian_road_osm/fold_0",
            "mlp": P2A_DOWNSTREAM_ROOT / "mlp/haidian_road_osm/fold_0",
            "unet": P2A_DOWNSTREAM_ROOT / "unet/haidian_road_osm/fold_0",
        },
    },
    "construction": {
        "prediction_dir": P2A_BENCHMARK_ROOT / "construction/fold_0/predictions",
        "label_dir": XUANNV_DATA_ROOT / "processed/haidian/labels/construction",
        "head_dirs": {"unet": P2A_BENCHMARK_ROOT / "construction/fold_0"},
    },
    "construction_joint": {
        "prediction_dir": P2A_BENCHMARK_ROOT / "construction_joint/fold_0/predictions",
        "label_dir": XUANNV_DATA_ROOT / "processed/construction_joint_v2",
        "head_dirs": {"unet": P2A_BENCHMARK_ROOT / "construction_joint/fold_0"},
        "haidian_prefix_only": True,
    },
}

MONTH_START = dt.date(2025, 12, 1)
MONTH_END = dt.date(2026, 5, 31)
STATIC_RAW_DIRS = {"highres_optical", "highres_sar", "labels", "esri_lulc_2023"}
STATIC_PROCESSED_DIRS = {"labels"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/modelscope_upload/haidian/v1"),
    )
    parser.add_argument("--max-patches", type=int, default=0, help="Debug limit.")
    parser.add_argument(
        "--skip-raw-training-data",
        action="store_true",
        help="Skip raw/processed training data archive links/copies.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Use symlink for a fast local dry run; use copy for upload packages.",
    )
    return parser.parse_args()


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def _copytree_or_link(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _month_overlaps_window(start: dt.date, end: dt.date) -> bool:
    return start <= MONTH_END and end >= MONTH_START


def _parse_date_token(token: str) -> dt.date | None:
    if len(token) != 8 or not token.isdigit():
        return None
    try:
        return dt.date(int(token[:4]), int(token[4:6]), int(token[6:8]))
    except ValueError:
        return None


def _path_has_target_month(path: Path) -> bool:
    text = path.as_posix()
    return any(month in text for month in MONTHS)


def _path_date_range_overlaps(path: Path) -> bool:
    tokens = [
        _parse_date_token(part)
        for part in path.stem.replace("-", "_").split("_")
    ]
    dates = [token for token in tokens if token is not None]
    if len(dates) >= 2:
        return _month_overlaps_window(min(dates), max(dates))
    if len(dates) == 1:
        return _month_overlaps_window(dates[0], dates[0])
    return False


def _copy_filtered_files(
    src_root: Path,
    dst_root: Path,
    include_file,
    mode: str,
    desc: str,
) -> dict[str, object]:
    if not src_root.exists():
        return {"source": str(src_root), "files": 0, "bytes": 0}
    copied_files = 0
    copied_bytes = 0
    for src in tqdm(sorted(src_root.rglob("*")), desc=desc):
        if src.is_dir() or not include_file(src):
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        _copy_or_link(src, dst, mode)
        copied_files += 1
        try:
            copied_bytes += src.stat().st_size
        except OSError:
            pass
    return {"source": str(src_root), "files": copied_files, "bytes": copied_bytes}


def _tar_filtered_files(
    src_root: Path,
    dst_tar: Path,
    include_file,
    desc: str,
) -> dict[str, object]:
    if not src_root.exists():
        return {"source": str(src_root), "files": 0, "bytes": 0}
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    byte_count = 0
    if dst_tar.exists():
        dst_tar.unlink()
    with tarfile.open(dst_tar, "w") as tar:
        for src in tqdm(sorted(src_root.rglob("*")), desc=desc):
            if src.is_dir() or not include_file(src):
                continue
            rel = src.relative_to(src_root)
            tar.add(src, arcname=rel.as_posix(), recursive=False)
            file_count += 1
            try:
                byte_count += src.stat().st_size
            except OSError:
                pass
    return {
        "source": str(src_root),
        "archive": str(dst_tar),
        "files": file_count,
        "bytes": byte_count,
        "format": "tar",
    }


def copy_training_archive(archive: Path, mode: str) -> dict[str, object]:
    """Copy only the Haidian inputs relevant to 2025-12 through 2026-05."""

    def include_raw(path: Path) -> bool:
        rel = path.relative_to(XUANNV_DATA_ROOT / "raw/haidian")
        top = rel.parts[0] if rel.parts else ""
        if top in STATIC_RAW_DIRS:
            return True
        return _path_has_target_month(rel) or _path_date_range_overlaps(rel)

    def include_processed(path: Path) -> bool:
        rel = path.relative_to(XUANNV_DATA_ROOT / "processed/haidian")
        top = rel.parts[0] if rel.parts else ""
        if path.name in {"manifest.json"} or top in STATIC_PROCESSED_DIRS:
            return True
        return _path_has_target_month(rel) or _path_date_range_overlaps(rel)

    summary = {
        "date_window": {
            "start": MONTH_START.isoformat(),
            "end": MONTH_END.isoformat(),
            "months": list(MONTHS),
        },
        "raw_training_data": _copy_filtered_files(
            XUANNV_DATA_ROOT / "raw/haidian",
            archive / "raw_training_data/haidian_202512_202605",
            include_raw,
            mode,
            "Haidian raw training archive",
        ),
        "processed_training_data": _tar_filtered_files(
            XUANNV_DATA_ROOT / "processed/haidian",
            archive / "processed_training_data/haidian_202512_202605.tar",
            include_processed,
            "Haidian processed training archive",
        ),
    }
    _copytree_or_link(
        XUANNV_DATA_ROOT / "statistics/haidian",
        archive / "statistics/haidian",
        mode,
    )
    summary["statistics"] = {"source": str(XUANNV_DATA_ROOT / "statistics/haidian")}
    readme = archive / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Haidian V1 Archive",
                "",
                "This archive contains the P2A Haidian V1 embedding model assets,",
                "downstream evaluation outputs, and the training inputs used for the",
                "December 2025 through May 2026 time window.",
                "",
                "The API-ready subset is under `../api_ready` and is what the",
                "`embedding-api` service downloads for deployment.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (archive / "training_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def normalize_rgb(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = np.percentile(arr, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(arr, 98, axis=(0, 1), keepdims=True)
    arr = (arr - lo) / (hi - lo + 1e-6)
    return np.clip(arr, 0, 1)


def fit_embedding_pca(files: list[Path], sample_pixels: int = 300_000) -> PCA:
    samples: list[np.ndarray] = []
    rng = np.random.default_rng(42)
    for path in files:
        emb = torch.load(path, map_location="cpu").numpy().astype(np.float32)
        d, h, w = emb.shape
        flat = emb.reshape(d, h * w).T
        n = min(max(1, sample_pixels // max(1, len(files))), flat.shape[0])
        idx = rng.choice(flat.shape[0], size=n, replace=False)
        samples.append(flat[idx])
    pca = PCA(n_components=3, random_state=42)
    pca.fit(np.concatenate(samples, axis=0))
    return pca


def write_embedding_assets(api_root: Path, max_patches: int) -> dict[str, object]:
    src_root = P2A_EMBEDDING_ROOT / "haidian"
    patch_dirs = sorted(src_root.glob("haidian_patch_*"))
    if max_patches:
        patch_dirs = patch_dirs[:max_patches]
    pca_files = []
    for patch_dir in patch_dirs[: min(40, len(patch_dirs))]:
        for month in MONTHS:
            path = patch_dir / f"{month}_embedding_map.pt"
            if path.exists():
                pca_files.append(path)
                break
    pca = fit_embedding_pca(pca_files) if pca_files else None

    count = 0
    for patch_dir in tqdm(patch_dirs, desc="Haidian V1 embeddings"):
        patch_id = patch_dir.name.replace("haidian_", "")
        for month in MONTHS:
            src = patch_dir / f"{month}_embedding_map.pt"
            if not src.exists():
                continue
            emb = torch.load(src, map_location="cpu").numpy().astype(np.float32)
            month_dir = api_root / "data/haidian/embeddings/v1" / month
            month_dir.mkdir(parents=True, exist_ok=True)
            np.save(month_dir / f"{patch_id}.npy", emb)
            stats = {
                "patch_id": patch_id,
                "month": month,
                "shape": list(emb.shape),
                "dtype": str(emb.dtype),
                "min": float(emb.min()),
                "max": float(emb.max()),
                "mean": float(emb.mean()),
                "std": float(emb.std()),
            }
            (month_dir / f"{patch_id}.json").write_text(
                json.dumps(stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if pca is not None:
                d, h, w = emb.shape
                rgb = pca.transform(emb.reshape(d, h * w).T).reshape(h, w, 3)
                rgb = normalize_rgb(rgb)
                Image.fromarray((rgb * 255).astype(np.uint8)).save(
                    month_dir / f"{patch_id}.png"
                )
            count += 1
    return {"patch_count": len(patch_dirs), "embedding_files": count}


def build_patches_meta(api_root: Path) -> None:
    src = XUANNV_REPO_ROOT / "configs/regions/haidian_patches.json"
    patches = json.loads(src.read_text(encoding="utf-8"))
    for patch in patches:
        patch["has_embedding"] = True
        patch["available_months"] = list(MONTHS)
        patch["available_tasks"] = []
    out = api_root / "data/haidian/patches_meta_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(patches, ensure_ascii=False, indent=2), encoding="utf-8")


def prob_to_red_png(prob: np.ndarray, threshold: float) -> np.ndarray:
    mask = prob.astype(np.float32) >= threshold
    rgb = np.full((prob.shape[0], prob.shape[1], 3), 255, dtype=np.uint8)
    rgb[mask] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def task_visual_threshold(info: dict[str, object]) -> float:
    head_dirs = info.get("head_dirs", {})
    if isinstance(head_dirs, dict):
        unet_dir = head_dirs.get("unet")
        if isinstance(unet_dir, Path):
            metrics = unet_dir / "metrics.json"
            if metrics.exists():
                data = json.loads(metrics.read_text(encoding="utf-8"))
                for key in ("val_threshold", "threshold", "best_threshold"):
                    if key in data:
                        return float(data[key])
    return 0.5


def write_task_assets(api_root: Path, models_root: Path, mode: str) -> dict[str, object]:
    task_counts: dict[str, int] = {}
    for task, info in TASK_SOURCES.items():
        pred_dir: Path = info["prediction_dir"]
        threshold = task_visual_threshold(info)
        out_task = api_root / "data/haidian/tasks" / task / "v1"
        pred_out = out_task / "predictions"
        tiles_out = out_task / "results" / "tiles"
        labels_out = out_task / "labels"
        pred_out.mkdir(parents=True, exist_ok=True)
        tiles_out.mkdir(parents=True, exist_ok=True)

        count = 0
        positive_pixels = 0
        for src in sorted(pred_dir.glob("*_prob.tif")):
            patch_id = src.name[: -len("_prob.tif")]
            if info.get("haidian_prefix_only"):
                if not patch_id.startswith("haidian_"):
                    continue
                patch_id = patch_id.replace("haidian_", "")
            with rasterio.open(src) as ds:
                prob = ds.read(1).astype(np.float32)
            np.save(pred_out / f"{patch_id}.npy", prob)
            Image.fromarray(prob_to_red_png(prob, threshold)).save(
                tiles_out / f"{patch_id}.png"
            )
            positive_pixels += int((prob >= threshold).sum())
            count += 1

        visualization_threshold = threshold
        threshold_note = "metrics_threshold"
        if count > 0 and positive_pixels == 0 and threshold > 0.5:
            visualization_threshold = 0.5
            threshold_note = "fallback_0.5_metrics_threshold_was_empty"
            for pred in sorted(pred_out.glob("*.npy")):
                prob = np.load(pred).astype(np.float32)
                Image.fromarray(prob_to_red_png(prob, visualization_threshold)).save(
                    tiles_out / f"{pred.stem}.png"
                )

        summary = {
            "task": task,
            "version": "v1",
            "total_patches": count,
            "positive_patches": None,
            "negative_patches": None,
            "visualization_threshold": visualization_threshold,
            "visualization_threshold_source": threshold_note,
            "source_prediction_dir": str(pred_dir),
        }
        (out_task / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        label_src: Path = info["label_dir"]
        if label_src.exists():
            write_label_arrays(label_src, labels_out, bool(info.get("haidian_prefix_only")))
        for head_name, head_dir in info["head_dirs"].items():
            ckpt = head_dir / "checkpoints/best.pt"
            if ckpt.exists():
                _copy_or_link(
                    ckpt,
                    models_root
                    / "models/haidian/v1/task_heads"
                    / task
                    / head_name
                    / "best.pt",
                    mode,
                )
            for meta_name in ("metrics.json",):
                meta = head_dir / meta_name
                if meta.exists():
                    _copy_or_link(
                        meta,
                        models_root
                        / "models/haidian/v1/task_heads"
                        / task
                        / head_name
                        / meta_name,
                        mode,
                    )
        task_counts[task] = count
    return task_counts


def write_label_arrays(label_src: Path, labels_out: Path, haidian_prefix_only: bool) -> None:
    mask_dir = label_src / "masks"
    if not mask_dir.exists():
        return
    labels_out.mkdir(parents=True, exist_ok=True)
    for src in sorted(mask_dir.glob("*.tif")):
        patch_id = src.stem
        if haidian_prefix_only:
            if not patch_id.startswith("haidian_"):
                continue
            patch_id = patch_id.replace("haidian_", "")
        with rasterio.open(src) as ds:
            arr = ds.read(1)
        np.save(labels_out / f"{patch_id}.npy", arr.astype(np.uint8))
    meta = {
        "source_label_dir": str(label_src),
        "converted_from": "GeoTIFF masks",
        "format": "uint8 numpy arrays",
    }
    (labels_out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            yield path


def write_checksums(root: Path) -> None:
    lines = []
    for path in iter_files(root):
        if path.name == "checksums.sha256":
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(root: Path, summary: dict[str, object]) -> None:
    manifest = {
        "region": "haidian",
        "api_version": "v1",
        "model_family": "xuannv P2A",
        "months": list(MONTHS),
        "created_by": "pipelines/haidian/prepare_v1_api_assets.py",
        "summary": summary,
        "api_ready_prefix": "haidian/v1/api_ready",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_modelscope_readme(root: Path) -> None:
    readme = """# 玄女 Embedding API 资产库

这个 ModelScope 数据集用于存放 `go-bananas-wwj/embedding-api` 项目的部署资产，包括模型权重、嵌入结果、下游任务头、预测结果、可视化结果和复现实验所需的数据归档。

当前已经发布的是 **海淀区 V1 版本**，底座模型为 xuannv P2A embedding 模型，覆盖时间范围为 **2025 年 12 月到 2026 年 5 月**。

## 当前可用版本

| 区域 | API 版本 | 模型 | 时间范围 | ModelScope 路径 | 状态 |
|---|---|---|---|---|---|
| 海淀区 | `v1` | xuannv P2A | 2025-12 到 2026-05 | `haidian/v1` | 可用 |

## 海淀区 V1 包含哪些内容

```text
haidian/v1/
  README.md
  manifest.json
  checksums.sha256
  api_ready/
    data/haidian/
      patches_meta_v1.json
      embeddings/v1/{月份}/{patch_id}.npy
      embeddings/v1/{月份}/{patch_id}.png
      embeddings/v1/{月份}/{patch_id}.json
      tasks/{任务}/v1/predictions/{patch_id}.npy
      tasks/{任务}/v1/results/tiles/{patch_id}.png
      tasks/{任务}/v1/labels/{patch_id}.npy
    models/haidian/v1/
      embedding/best.pt
      embedding/config.yaml
      task_heads/{任务}/{下游头}/best.pt
      task_heads/{任务}/{下游头}/metrics.json
  archive/
    raw_training_data/haidian_202512_202605/
    processed_training_data/haidian_202512_202605.tar
    training_output/
    downstream_osm_eval/
    benchmark_eval/
    training_data_summary.json
```

其中：

- `api_ready/`：部署 API 时真正需要下载的内容。
- `archive/`：训练数据、训练输出、测评输出等归档内容，主要用于复现和审计。
- `manifest.json`：资产包的基本信息，例如区域、版本、月份、任务数量等。
- `checksums.sha256`：文件校验表，用于检查文件是否完整。

## 容量说明

下面是当前海淀区 V1 资产的大致大小，实际显示会因为文件系统和压缩方式略有差异：

| 内容 | 路径 | 大小 |
|---|---|---:|
| 完整海淀区 V1 包 | `haidian/v1/` | 约 95G |
| API 部署必需内容 | `haidian/v1/api_ready/` | 约 9.5G |
| API 数据部分 | `haidian/v1/api_ready/data/haidian/` | 约 7.8G |
| 6 个月 embedding 总量 | `haidian/v1/api_ready/data/haidian/embeddings/` | 约 7.6G |
| 每个月 embedding | `haidian/v1/api_ready/data/haidian/embeddings/v1/{月份}/` | 约 1.3G/月 |
| 下游任务预测和可视化总量 | `haidian/v1/api_ready/data/haidian/tasks/` | 约 117M |
| 建筑物提取结果 | `haidian/v1/api_ready/data/haidian/tasks/building_extraction/` | 约 33M |
| 道路提取结果 | `haidian/v1/api_ready/data/haidian/tasks/road_extraction/` | 约 33M |
| 施工变化检测结果 | `haidian/v1/api_ready/data/haidian/tasks/construction/` | 约 26M |
| 联合施工变化检测结果 | `haidian/v1/api_ready/data/haidian/tasks/construction_joint/` | 约 27M |
| API 模型权重 | `haidian/v1/api_ready/models/haidian/` | 约 1.8G |
| 复现归档总量 | `haidian/v1/archive/` | 约 86G |
| 原始训练数据 | `haidian/v1/archive/raw_training_data/` | 约 64G |
| 预处理训练数据归档 | `haidian/v1/archive/processed_training_data/` | 约 15G |
| P2A 训练输出 | `haidian/v1/archive/training_output/` | 约 7.2G |
| OSM 下游测评输出 | `haidian/v1/archive/downstream_osm_eval/` | 约 134M |
| 变化检测基准测评输出 | `haidian/v1/archive/benchmark_eval/` | 约 42M |

如果只是部署 API，通常只需要下载 `haidian/v1/api_ready/`，大约 `9.5G`；不需要下载 `archive/`。

## 覆盖月份

海淀区 V1 包含以下 6 个月份的 embedding：

- `202512`
- `202601`
- `202602`
- `202603`
- `202604`
- `202605`

## 支持的下游任务

| 任务名 | 说明 |
|---|---|
| `building_extraction` | 建筑物提取，标签主要来自 OSM 派生结果 |
| `road_extraction` | 道路/路网提取，标签主要来自 OSM 派生结果 |
| `construction` | 海淀区施工变化检测 |
| `construction_joint` | 联合施工变化检测基准中的海淀区子集 |

每个任务都包含：

- `.npy` 概率图：用于程序读取和后续分析。
- `.png` 可视化图：红色表示预测前景，白色表示背景，方便人工查看。
- 下游任务头权重：保存在 `models/haidian/v1/task_heads/` 下。

## 快速部署方式

先克隆 API 项目：

```bash
git clone git@github.com:go-bananas-wwj/embedding-api.git
cd embedding-api
```

安装 ModelScope 命令行工具：

```bash
pip install modelscope
```

下载海淀区 V1 的 API 部署资产：

```bash
export MODELSCOPE_TOKEN="你的 ModelScope Token"  # 如果数据集是公开的，可以不设置
python pipelines/haidian/download_modelscope_assets.py \
  --repo WeijieWu/xuannv_embdding_api \
  --prefix haidian/v1/api_ready \
  --target .
```

下载完成后，项目根目录下会出现：

```text
data/haidian/...
models/haidian/...
```

然后启动 API：

```bash
DOCS_URL=/docs uvicorn app.main:app --host 0.0.0.0 --port 9061
```

启动后打开：

```text
http://localhost:9061/docs
```

## 快速检查 API 是否可用

可以用下面几条命令做最小化检查：

```bash
curl -s "http://localhost:9061/regions/haidian" | python -m json.tool
curl -s "http://localhost:9061/regions/haidian/patches?page=1&page_size=2" | python -m json.tool
curl -s "http://localhost:9061/regions/haidian/patches/patch_000000/embedding?format=json&version=v1&month=202512" | python -m json.tool
curl -s "http://localhost:9061/regions/haidian/patches/patch_000000/tasks/road_extraction/result?format=png&version=v1" -o /tmp/haidian_road.png
```

如果这些接口能正常返回，就说明海淀区 V1 资产已经部署成功。

## 直接使用 ModelScope CLI 下载

如果不使用项目里的下载脚本，也可以直接下载 `api_ready` 部分：

```bash
modelscope download \
  --dataset WeijieWu/xuannv_embdding_api \
  --include "haidian/v1/api_ready/**" "haidian/v1/manifest.json" "haidian/v1/README.md" \
  --local_dir ./modelscope_cache
```

然后把下面这个目录里的内容复制到 `embedding-api` 项目根目录：

```text
modelscope_cache/haidian/v1/api_ready/
```

## 文件格式说明

- Embedding 文件：`.npy`，形状为 `[C, H, W]`。
- Embedding 预览图：`.png`，由 PCA 降到 RGB 后生成。
- 任务预测结果：`.npy`，表示每个像素属于目标类别的概率。
- 任务可视化图：`.png`，红色为预测前景，白色为背景。
- 标签文件：`.npy`，uint8 mask。
- 模型权重：`.pt`，PyTorch checkpoint。

## 普通部署只需要哪些内容

如果只是部署 API，只需要下载：

```text
haidian/v1/api_ready/
```

`archive/` 目录很大，里面主要是原始训练数据、预处理数据、训练日志和测评结果，普通部署不需要下载。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.output_root
    api_root = root / "api_ready"
    archive = root / "archive"
    api_root.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)

    build_patches_meta(api_root)
    emb_summary = write_embedding_assets(api_root, args.max_patches)

    # Embedding checkpoint and config are part of both API-ready model metadata
    # and the archival copy.
    _copy_or_link(P2A_OUTPUT_ROOT / "best.pt", api_root / "models/haidian/v1/embedding/best.pt", args.copy_mode)
    _copy_or_link(
        XUANNV_REPO_ROOT / "configs/v2_p2a_semantic_probe_full_20260627.yaml",
        api_root / "models/haidian/v1/embedding/config.yaml",
        args.copy_mode,
    )
    _copytree_or_link(P2A_OUTPUT_ROOT, archive / "training_output", args.copy_mode)
    _copytree_or_link(P2A_DOWNSTREAM_ROOT, archive / "downstream_osm_eval", args.copy_mode)
    _copytree_or_link(P2A_BENCHMARK_ROOT, archive / "benchmark_eval", args.copy_mode)
    task_summary = write_task_assets(api_root, api_root, args.copy_mode)

    if not args.skip_raw_training_data:
        copy_training_archive(archive, args.copy_mode)

    summary = {"embeddings": emb_summary, "tasks": task_summary}
    write_manifest(root, summary)
    write_modelscope_readme(root)
    write_checksums(api_root)
    write_checksums(root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Prepared Haidian V1 package at {root}")


if __name__ == "__main__":
    main()
