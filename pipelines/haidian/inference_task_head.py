#!/usr/bin/env python3
"""Run Haidian V1 downstream task-head inference.

The API serves precomputed task results, but this script allows operators to
regenerate them from downloaded V1 embeddings and task-head checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

from paths import (
    HAIDIAN_PATCHES_META,
    HAIDIAN_V1_EMBEDDINGS_DIR,
    HAIDIAN_V1_MODELS_DIR,
    task_predictions_dir,
    task_results_dir,
)


TASK_TO_HEAD = {
    "building_extraction": "unet",
    "road_extraction": "unet",
    "construction": "unet",
    "construction_joint": "unet",
}


class LinearProbeHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MLPProbeHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim + embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim + embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.up1(x)
        x1 = self.conv1(torch.cat([x1, F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)], dim=1))
        x2 = self.up2(x1)
        x2 = self.conv2(torch.cat([x2, F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)], dim=1))
        return F.interpolate(self.final(x2), size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_head(head_type: str, embed_dim: int = 192) -> nn.Module:
    if head_type == "linear":
        return LinearProbeHead(embed_dim, 2)
    if head_type == "mlp":
        return MLPProbeHead(embed_dim, 2)
    if head_type == "unet":
        return UNetHead(embed_dim, 2)
    raise ValueError(f"Unsupported head type: {head_type}")


def prob_to_red_png(prob: np.ndarray, threshold: float) -> np.ndarray:
    mask = prob.astype(np.float32) >= threshold
    rgb = np.full((prob.shape[0], prob.shape[1], 3), 255, dtype=np.uint8)
    rgb[mask] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def load_threshold(models_dir: Path, task: str, head_type: str, fallback: float = 0.5) -> float:
    metrics = models_dir / "task_heads" / task / head_type / "metrics.json"
    if not metrics.exists():
        return fallback
    data = json.loads(metrics.read_text(encoding="utf-8"))
    for key in ("val_threshold", "threshold", "best_threshold"):
        if key in data:
            return float(data[key])
    return fallback


def load_embedding(embeddings_dir: Path, month: str, patch_id: str) -> np.ndarray:
    path = embeddings_dir / month / f"{patch_id}.npy"
    return np.load(path).astype(np.float32)


def make_concat_diff(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.concatenate([before, after, after - before], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_TO_HEAD))
    parser.add_argument("--before-month", default="202512")
    parser.add_argument("--after-month", default="202605")
    parser.add_argument("--head", default=None, choices=["linear", "mlp", "unet"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patches-meta", type=Path, default=HAIDIAN_PATCHES_META)
    parser.add_argument("--embeddings-dir", type=Path, default=HAIDIAN_V1_EMBEDDINGS_DIR)
    parser.add_argument("--models-dir", type=Path, default=HAIDIAN_V1_MODELS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Visualization threshold. Defaults to the task head metrics threshold.",
    )
    parser.add_argument("--max-patches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    head_type = args.head or TASK_TO_HEAD[args.task]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = args.models_dir / "task_heads" / args.task / head_type / "best.pt"
    model = build_head(head_type).to(device)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    threshold = (
        args.threshold
        if args.threshold is not None
        else load_threshold(args.models_dir, args.task, head_type)
    )

    patches = json.loads(args.patches_meta.read_text(encoding="utf-8"))
    patch_ids = [p["patch_id"] for p in patches]
    if args.max_patches > 0:
        patch_ids = patch_ids[: args.max_patches]
    pred_dir = args.predictions_dir or task_predictions_dir(args.task)
    tile_dir = args.results_dir or (task_results_dir(args.task) / "tiles")
    pred_dir.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)

    batch = []
    batch_ids = []
    with torch.no_grad():
        for patch_id in tqdm(patch_ids, desc=f"Haidian {args.task}"):
            before = load_embedding(args.embeddings_dir, args.before_month, patch_id)
            after = load_embedding(args.embeddings_dir, args.after_month, patch_id)
            batch.append(torch.from_numpy(make_concat_diff(before, after)))
            batch_ids.append(patch_id)
            if len(batch) >= args.batch_size:
                save_batch(model, batch, batch_ids, pred_dir, tile_dir, device, threshold)
                batch, batch_ids = [], []
        if batch:
            save_batch(model, batch, batch_ids, pred_dir, tile_dir, device, threshold)


def save_batch(
    model: nn.Module,
    batch: list[torch.Tensor],
    patch_ids: list[str],
    pred_dir: Path,
    tile_dir: Path,
    device: torch.device,
    threshold: float,
) -> None:
    x = torch.stack(batch).to(device)
    prob = torch.sigmoid(model(x)[:, 1]).cpu().numpy()
    for patch_id, arr in zip(patch_ids, prob):
        np.save(pred_dir / f"{patch_id}.npy", arr.astype(np.float32))
        Image.fromarray(prob_to_red_png(arr, threshold)).save(tile_dir / f"{patch_id}.png")


if __name__ == "__main__":
    main()
