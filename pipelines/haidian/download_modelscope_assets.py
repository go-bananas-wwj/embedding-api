#!/usr/bin/env python3
"""Download latest Haidian assets from ModelScope into embedding-api.

The current source is the ModelScope dataset
``WeijieWu/xuannv_haidian_embdding`` under
``artifacts/haidian-embedding-v1``. The dataset stores raw ``*.pt`` embedding
maps by patch; this installer converts them into the API layout:

``data/haidian/embeddings/v1/{month}/{patch_id}.npy|png|json`` and
``models/haidian/v1/...``.

Credentials stay outside the repo: pass ``MODELSCOPE_TOKEN`` when needed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paths import (
    DEFAULT_EMBEDDING_ARTIFACT,
    DEFAULT_MODELSCOPE_PREFIX,
    DEFAULT_MODELSCOPE_REPO,
    PROJECT_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_MODELSCOPE_REPO)
    parser.add_argument("--prefix", default=DEFAULT_MODELSCOPE_PREFIX)
    parser.add_argument("--embedding-artifact", default=DEFAULT_EMBEDDING_ARTIFACT)
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".modelscope_cache/haidian_v1"),
        help="Temporary dataset download cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files under the target directory.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Install from cache-dir without invoking the ModelScope downloader.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers passed to `modelscope download`.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "ModelScope include glob. May be repeated. Defaults to API-serving "
            "assets only: embeddings, checkpoints, and downstream_heads."
        ),
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    printable = list(cmd)
    for idx, item in enumerate(printable[:-1]):
        if item == "--token":
            printable[idx + 1] = "***"
    print("+", " ".join(printable))
    subprocess.run(cmd, check=True)


def _default_includes(prefix: str) -> list[str]:
    prefix = prefix.rstrip("/")
    return [
        f"{prefix}/embeddings/**",
        f"{prefix}/checkpoints/**",
        f"{prefix}/downstream_heads/**",
    ]


def download_with_cli(
    repo: str,
    cache_dir: Path,
    includes: list[str],
    max_workers: int,
) -> Path:
    token = os.environ.get("MODELSCOPE_TOKEN")
    cmd = [
        "modelscope",
        "download",
        "--dataset",
        repo,
        "--local_dir",
        str(cache_dir),
        "--max-workers",
        str(max_workers),
    ]
    for pattern in includes:
        cmd.extend(["--include", pattern])
    if token:
        cmd.extend(["--token", token])
    run(cmd)
    return cache_dir


def _as_numpy(obj) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for key in ("embedding", "embedding_map", "features", "x"):
            if key in obj:
                return _as_numpy(obj[key])
        for value in obj.values():
            try:
                return _as_numpy(value)
            except TypeError:
                continue
    raise TypeError(f"Unsupported embedding object type: {type(obj)!r}")


def _write_embedding_outputs(src_pt: Path, dst_dir: Path, patch_id: str, month: str) -> None:
    arr = _as_numpy(torch.load(src_pt, map_location="cpu"))
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4) and arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))

    dst_dir.mkdir(parents=True, exist_ok=True)
    npy_path = dst_dir / f"{patch_id}.npy"
    np.save(npy_path, arr.astype(np.float32, copy=False))

    stats = {
        "patch_id": patch_id,
        "month": month,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "source": "WeijieWu/xuannv_haidian_embdding",
    }
    (dst_dir / f"{patch_id}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    vis = arr
    if vis.ndim == 3:
        channels = min(3, vis.shape[0])
        vis = vis[:channels]
        if channels == 1:
            vis = np.repeat(vis, 3, axis=0)
        elif channels == 2:
            vis = np.concatenate([vis, vis[:1]], axis=0)
        vis = np.transpose(vis, (1, 2, 0))
    elif vis.ndim == 2:
        vis = np.repeat(vis[:, :, None], 3, axis=2)
    else:
        return
    lo, hi = np.nanpercentile(vis, [2, 98])
    if hi <= lo:
        img = np.zeros(vis.shape, dtype=np.uint8)
    else:
        img = np.clip((vis - lo) / (hi - lo), 0, 1)
        img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(dst_dir / f"{patch_id}.png")


def install_artifacts(src: Path, target: Path, embedding_artifact: str, force: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"ModelScope prefix not found after download: {src}")

    emb_root = src / embedding_artifact
    if not emb_root.exists():
        raise FileNotFoundError(f"Embedding artifact not found: {emb_root}")

    data_root = target / "data" / "haidian"
    emb_out = data_root / "embeddings" / "v1"
    models_root = target / "models" / "haidian" / "v1"

    if force and emb_out.exists():
        for old_dir in emb_out.glob("patch_*"):
            if old_dir.is_dir():
                shutil.rmtree(old_dir)

    converted = 0
    for pt_path in sorted(emb_root.glob("patch_*/[0-9]*_embedding_map.pt")):
        patch_id = pt_path.parent.name
        month = pt_path.name.split("_", 1)[0]
        out_dir = emb_out / month
        out_npy = out_dir / f"{patch_id}.npy"
        if out_npy.exists() and not force:
            continue
        _write_embedding_outputs(pt_path, out_dir, patch_id, month)
        converted += 1

    checkpoint_dir = src / "checkpoints"
    if checkpoint_dir.exists():
        dst = models_root / "embedding"
        dst.mkdir(parents=True, exist_ok=True)
        if force:
            for item in dst.glob("*"):
                if item.is_file() and item.name != ".gitkeep":
                    item.unlink()
        for item in checkpoint_dir.glob("*"):
            if item.is_file():
                shutil.copy2(item, dst / item.name)

    heads_dir = src / "downstream_heads"
    if heads_dir.exists():
        dst = models_root / "task_heads"
        dst.mkdir(parents=True, exist_ok=True)
        if force:
            for item in dst.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file() and item.name != ".gitkeep":
                    item.unlink()
        for item in heads_dir.glob("*"):
            if item.is_file():
                shutil.copy2(item, dst / item.name)

    print(f"Converted {converted} embedding maps into {emb_out}")
    print(f"Installed checkpoints/heads into {models_root}")


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_dir
    if not args.skip_download:
        cache_root = download_with_cli(
            args.repo,
            args.cache_dir,
            args.include or _default_includes(args.prefix),
            args.max_workers,
        )
    src = cache_root / args.prefix
    install_artifacts(src, args.target, args.embedding_artifact, args.force)
    print(f"Haidian V1 assets installed into {args.target}")


if __name__ == "__main__":
    main()
