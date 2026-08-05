#!/usr/bin/env python3
"""Generate monthly Haidian system-head predictions and display tiles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.fewshot_heads import BinaryConv3x3ProbeHead

MONTHS = ("202512", "202601", "202602", "202603", "202604", "202605")
TASK_FILES = {
    "building_extraction": "building_conv3x3_best.pt",
    "road_extraction": "road_conv3x3_best.pt",
    "water_extraction": "water_conv3x3_best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_FILES))
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def render(prediction: np.ndarray) -> Image.Image:
    rgb = np.full((*prediction.shape, 3), 255, dtype=np.uint8)
    rgb[prediction] = (230, 0, 0)
    return Image.fromarray(rgb).resize((128, 128), Image.Resampling.NEAREST)


def main() -> None:
    args = parse_args()
    checkpoint_path = (
        ROOT / "models/haidian/v1/task_heads" / TASK_FILES[args.task]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("__format__") != "embedding-api.system-head.v1":
        raise ValueError(f"Unsupported checkpoint: {checkpoint_path}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BinaryConv3x3ProbeHead(
        int(checkpoint["embed_dim"]),
        int(checkpoint.get("hidden_dim", 128)),
        float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    patches = json.loads((ROOT / "data/haidian/patches_meta_v1.json").read_text())
    patch_ids = [item["patch_id"] for item in patches]
    task_root = ROOT / "data/haidian/tasks" / args.task / "v1"

    with torch.no_grad():
        for month in args.months:
            prediction_dir = task_root / "predictions" / month
            tile_dir = task_root / "results" / month / "tiles"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            tile_dir.mkdir(parents=True, exist_ok=True)
            for patch_id in tqdm(patch_ids, desc=f"{args.task} {month}"):
                prediction_path = prediction_dir / f"{patch_id}.npy"
                tile_path = tile_dir / f"{patch_id}.png"
                if not args.force and prediction_path.exists() and tile_path.exists():
                    continue
                embedding_path = (
                    ROOT / "data/haidian/embeddings/v1" / month / f"{patch_id}.npy"
                )
                embedding = np.load(embedding_path).astype(np.float32, copy=False)
                tensor = torch.from_numpy(embedding).unsqueeze(0).to(device)
                probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                prediction = probability >= threshold
                np.save(prediction_path, probability.astype(np.float32))
                render(prediction).save(tile_path)


if __name__ == "__main__":
    main()
