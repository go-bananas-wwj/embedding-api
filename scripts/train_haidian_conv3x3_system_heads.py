#!/usr/bin/env python3
"""Train production Binary Conv 3x3 heads on Haidian P10C embeddings."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.fewshot_heads import BinaryConv3x3ProbeHead


EMBEDDING_ROOT = ROOT / "data/haidian/embeddings/v1"
TASK_ROOT = ROOT / "data/haidian/tasks"
ARCHIVE_LABEL_ROOT = (
    ROOT / "data/haidian/archive/processed_training_data/extracted/labels"
)
MONTHS = ("202512", "202601", "202602", "202603", "202604", "202605")
TASKS = ("building_extraction", "road_extraction", "water_extraction")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_label(task: str, patch_id: str) -> np.ndarray:
    if task in {"building_extraction", "road_extraction"}:
        path = TASK_ROOT / task / "v1/labels" / f"{patch_id}.npy"
        label = np.load(path) > 0
    elif task == "water_extraction":
        matches = sorted(
            (ARCHIVE_LABEL_ROOT / "worldcover").glob(
                f"worldcover_*_{patch_id}.tif"
            )
        )
        if not matches:
            raise FileNotFoundError(f"WorldCover label not found: {patch_id}")
        with rasterio.open(matches[-1]) as dataset:
            # This archive uses normalized project classes; class 1 is water.
            label = dataset.read(1) == 1
    else:
        raise KeyError(task)
    return label.astype(np.uint8)


def resize_label(label: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(label).float()[None, None]
    resized = F.interpolate(tensor, size=shape, mode="nearest")
    return resized[0, 0].numpy().astype(np.float32)


def available_patch_ids(task: str) -> list[str]:
    embedding_ids = {
        path.stem for path in (EMBEDDING_ROOT / "202604").glob("patch_*.npy")
    }
    if task in {"building_extraction", "road_extraction"}:
        label_ids = {
            path.stem for path in (TASK_ROOT / task / "v1/labels").glob("patch_*.npy")
        }
    else:
        label_ids = {
            "patch_" + path.stem.rsplit("patch_", 1)[-1]
            for path in (ARCHIVE_LABEL_ROOT / "worldcover").glob(
                "worldcover_*_patch_*.tif"
            )
        }
    return sorted(embedding_ids & label_ids)


def load_split(task: str, patch_ids: list[str], fold: int = 0) -> dict[str, list[str]]:
    source_name = "building_osm" if task == "building_extraction" else "road_osm"
    source = ARCHIVE_LABEL_ROOT / source_name / "split_5fold.json"
    if task != "water_extraction" and source.exists():
        data = json.loads(source.read_text(encoding="utf-8"))
        selected = next(item for item in data["folds"] if item["fold"] == fold)
        allowed = set(patch_ids)
        return {
            key: [pid for pid in selected[key] if pid in allowed]
            for key in ("train", "val", "test")
        }

    # Water has no dedicated split. Build a deterministic, ratio-stratified fold.
    ranked = sorted(
        patch_ids,
        key=lambda pid: (float(load_label(task, pid).mean()), pid),
    )
    buckets = [[] for _ in range(5)]
    for index, patch_id in enumerate(ranked):
        buckets[index % 5].append(patch_id)
    test = set(buckets[fold])
    remaining = [pid for pid in ranked if pid not in test]
    val = set(remaining[::10])
    train = [pid for pid in remaining if pid not in val]
    return {"train": train, "val": sorted(val), "test": sorted(test)}


class EmbeddingLabelDataset(Dataset):
    def __init__(
        self,
        task: str,
        patch_ids: Iterable[str],
        months: Iterable[str],
        augment: bool = False,
    ) -> None:
        self.task = task
        self.items = [
            (patch_id, month) for patch_id in patch_ids for month in months
        ]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        patch_id, month = self.items[index]
        feature = np.load(
            EMBEDDING_ROOT / month / f"{patch_id}.npy", mmap_mode="r"
        )
        feature = np.nan_to_num(
            np.asarray(feature, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        label = resize_label(load_label(self.task, patch_id), feature.shape[-2:])
        x = torch.from_numpy(feature.copy())
        y = torch.from_numpy(label).unsqueeze(0)
        if self.augment:
            if torch.rand(()) > 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(()) > 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y, patch_id, month


def bce_dice_tversky(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    smooth = 1.0
    tp = (probs * target).sum(dims)
    fp = (probs * (1.0 - target)).sum(dims)
    fn = ((1.0 - probs) * target).sum(dims)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    tversky = (tp + smooth) / (tp + 0.3 * fp + 0.7 * fn + smooth)
    return bce + (1.0 - dice.mean()) + (1.0 - tversky.mean())


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    iou: float


def binary_metrics(tp: int, fp: int, fn: int) -> Metrics:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return Metrics(
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(1e-12, precision + recall),
        iou=tp / max(1, tp + fp + fn),
    )


def evaluate_thresholds(model, loader, device, thresholds) -> dict[float, Metrics]:
    counts = {float(t): [0, 0, 0] for t in thresholds}
    model.eval()
    with torch.no_grad():
        for feature, label, _, _ in loader:
            probs = torch.sigmoid(model(feature.to(device))).cpu()
            truth = label.bool()
            for threshold, count in counts.items():
                pred = probs >= threshold
                count[0] += int(torch.logical_and(pred, truth).sum())
                count[1] += int(torch.logical_and(pred, ~truth).sum())
                count[2] += int(torch.logical_and(~pred, truth).sum())
    return {threshold: binary_metrics(*count) for threshold, count in counts.items()}


def evaluate_mlp(model_path: Path, loader) -> Metrics:
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
    )
    model.load_state_dict(
        {key.replace("net.", "", 1): value for key, value in state.items()}
    )
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for feature, label, _, _ in loader:
            b, d, h, w = feature.shape
            logits = model(feature.permute(0, 2, 3, 1).reshape(-1, d))
            pred = (torch.sigmoid(logits).reshape(b, 1, h, w) >= 0.5)
            truth = label.bool()
            tp += int(torch.logical_and(pred, truth).sum())
            fp += int(torch.logical_and(pred, ~truth).sum())
            fn += int(torch.logical_and(~pred, truth).sum())
    return binary_metrics(tp, fp, fn)


def train(args) -> Path:
    set_seed(args.seed)
    task = args.task
    if task == "building_extraction" and not args.allow_incomplete_osm_labels:
        raise ValueError(
            "Building labels are sparse OSM coverage, not complete ground truth. "
            "Install and validate a complete footprint source before production "
            "training, or pass --allow-incomplete-osm-labels for diagnostics only."
        )
    output = args.output / task
    output.mkdir(parents=True, exist_ok=True)
    patch_ids = available_patch_ids(task)
    split = load_split(task, patch_ids, args.fold)
    (output / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_loader = DataLoader(
        EmbeddingLabelDataset(task, split["train"], args.months, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        EmbeddingLabelDataset(task, split["val"], args.months),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        EmbeddingLabelDataset(task, split["test"], args.months),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device(args.device)
    model = BinaryConv3x3ProbeHead(64, hidden_dim=128, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    thresholds = np.linspace(0.05, 0.95, 19)
    best = {"f1": -1.0, "epoch": 0, "threshold": 0.5, "state_dict": None}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for feature, label, _, _ in train_loader:
            feature = feature.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_dice_tversky(model(feature), label)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            scores = evaluate_thresholds(model, val_loader, device, thresholds)
            threshold, metrics = max(scores.items(), key=lambda item: item[1].f1)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "threshold": threshold,
                **metrics.__dict__,
            }
            history.append(row)
            print(json.dumps({"task": task, **row}, ensure_ascii=False), flush=True)
            if metrics.f1 > best["f1"]:
                best = {
                    "f1": metrics.f1,
                    "epoch": epoch,
                    "threshold": threshold,
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }

    model.load_state_dict(best["state_dict"])
    test_scores = evaluate_thresholds(
        model, test_loader, device, [best["threshold"]]
    )
    test_metrics = test_scores[best["threshold"]]
    old_mlp_path = (
        ROOT / "models/haidian/v1/task_heads"
        / f"{task.removesuffix('_extraction')}_mlp_fold0_best.pt"
    )
    old_metrics = evaluate_mlp(old_mlp_path, test_loader)
    checkpoint = {
        "__format__": "embedding-api.system-head.v1",
        "head_type": "binary_conv3x3",
        "state_dict": best["state_dict"],
        "embed_dim": 64,
        "hidden_dim": 128,
        "dropout": 0.1,
        "threshold": float(best["threshold"]),
        "task_type": task,
        "region_id": "haidian",
        "embedding_version": "v1",
        "embedding_name": "P10C epoch800 semantic 64D",
        "months": list(args.months),
        "fold": args.fold,
        "best_epoch": best["epoch"],
        "validation_f1": best["f1"],
        "test_metrics": test_metrics.__dict__,
        "baseline_mlp_metrics": old_metrics.__dict__,
        "loss": "bce_dice_tversky",
        "tversky_beta": 0.7,
        "seed": args.seed,
    }
    checkpoint_path = output / f"{task}_conv3x3_best.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "task": task,
        "checkpoint": str(checkpoint_path),
        "split_sizes": {key: len(value) for key, value in split.items()},
        "months": list(args.months),
        "best_epoch": best["epoch"],
        "threshold": best["threshold"],
        "validation_f1": best["f1"],
        "conv3x3_test": test_metrics.__dict__,
        "mlp_test": old_metrics.__dict__,
        "history": history,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-incomplete-osm-labels",
        action="store_true",
        help="Allow diagnostic building training on sparse OSM labels (not production ground truth).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
