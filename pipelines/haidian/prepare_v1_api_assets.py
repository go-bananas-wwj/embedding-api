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


def prob_to_red_png(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob.astype(np.float32), 0, 1)
    rgb = np.full((prob.shape[0], prob.shape[1], 3), 255, dtype=np.uint8)
    rgb[..., 1] = ((1.0 - prob) * 255).astype(np.uint8)
    rgb[..., 2] = ((1.0 - prob) * 255).astype(np.uint8)
    return rgb


def write_task_assets(api_root: Path, models_root: Path, mode: str) -> dict[str, object]:
    task_counts: dict[str, int] = {}
    for task, info in TASK_SOURCES.items():
        pred_dir: Path = info["prediction_dir"]
        out_task = api_root / "data/haidian/tasks" / task / "v1"
        pred_out = out_task / "predictions"
        tiles_out = out_task / "results" / "tiles"
        labels_out = out_task / "labels"
        pred_out.mkdir(parents=True, exist_ok=True)
        tiles_out.mkdir(parents=True, exist_ok=True)

        count = 0
        for src in sorted(pred_dir.glob("*_prob.tif")):
            patch_id = src.name[: -len("_prob.tif")]
            if info.get("haidian_prefix_only"):
                if not patch_id.startswith("haidian_"):
                    continue
                patch_id = patch_id.replace("haidian_", "")
            with rasterio.open(src) as ds:
                prob = ds.read(1).astype(np.float32)
            np.save(pred_out / f"{patch_id}.npy", prob)
            Image.fromarray(prob_to_red_png(prob)).save(tiles_out / f"{patch_id}.png")
            count += 1

        summary = {
            "task": task,
            "version": "v1",
            "total_patches": count,
            "positive_patches": None,
            "negative_patches": None,
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
    write_checksums(api_root)
    write_checksums(root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Prepared Haidian V1 package at {root}")


if __name__ == "__main__":
    main()
