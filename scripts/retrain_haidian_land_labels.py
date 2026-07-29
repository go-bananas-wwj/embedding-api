#!/usr/bin/env python3
"""Train independent Haidian land-cover and land-use Conv 3x3 heads."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "data/haidian/embeddings/v1"
WORLDCOVER = ROOT / "data/haidian/archive/processed_training_data/extracted/labels/worldcover"
BUILDINGS = ROOT / "data/haidian/tasks/building_extraction/v1/labels"
ROADS = ROOT / "data/haidian/tasks/road_extraction/v1/labels"
MODELS = ROOT / "models/haidian/v1/task_heads"
MONTHS = ("202512", "202601", "202602", "202603", "202604", "202605")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    checkpoint_name: str
    metrics_name: str
    output_values: tuple[int, ...]
    colors: tuple[tuple[int, int, int], ...]
    label_source: str
    label_loader: Callable[[str], np.ndarray]

    @property
    def source_to_index(self) -> dict[int, int]:
        return {value: index for index, value in enumerate(self.output_values)}


LAND_COVER_VALUES = (1, 2, 3, 4, 5, 6, 8)
LAND_USE_VALUES = (0, 1, 2, 4, 5, 6, 7)
WORLDCOVER_TO_LAND_USE = {1: 0, 8: 1, 3: 2, 4: 4, 2: 5, 5: 6, 6: 7}


def load_worldcover(patch_id: str) -> np.ndarray:
    matches = sorted(WORLDCOVER.glob(f"worldcover_*_{patch_id}.tif"))
    if not matches:
        raise FileNotFoundError(patch_id)
    with rasterio.open(matches[-1]) as dataset:
        return dataset.read(1)


def load_optional_mask(root: Path, patch_id: str) -> np.ndarray:
    path = root / f"{patch_id}.npy"
    if not path.exists():
        return np.zeros_like(load_worldcover(patch_id), dtype=bool)
    return np.load(path) > 0


def _encode(source: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    label = np.full(source.shape, -1, dtype=np.int64)
    for source_value, output_value in mapping.items():
        label[source == source_value] = output_value
    return label


def load_land_cover_label(patch_id: str) -> np.ndarray:
    """Physical surface cover from the local WorldCover snapshot."""
    source_to_index = {value: index for index, value in enumerate(LAND_COVER_VALUES)}
    return _encode(load_worldcover(patch_id), source_to_index)


def load_land_use_label(patch_id: str) -> np.ndarray:
    """Use-oriented labels with OSM human-activity masks as independent supervision."""
    value_to_index = {value: index for index, value in enumerate(LAND_USE_VALUES)}
    mapping = {source: value_to_index[target] for source, target in WORLDCOVER_TO_LAND_USE.items()}
    label = _encode(load_worldcover(patch_id), mapping)
    built_index = value_to_index[6]
    label[load_optional_mask(BUILDINGS, patch_id)] = built_index
    label[load_optional_mask(ROADS, patch_id)] = built_index
    return label


LAND_COVER_SPEC = TaskSpec(
    task_id="land_cover_classification",
    checkpoint_name="land_cover_conv3x3_best.pt",
    metrics_name="land_cover_conv3x3_metrics.json",
    output_values=LAND_COVER_VALUES,
    colors=((30, 100, 220), (180, 210, 80), (245, 220, 90), (210, 60, 60),
            (190, 170, 130), (160, 220, 220), (0, 100, 0)),
    label_source="WorldCover local 2023-01-01 snapshot",
    label_loader=load_land_cover_label,
)
LAND_USE_SPEC = TaskSpec(
    task_id="land_use_classification",
    checkpoint_name="land_use_conv3x3_best.pt",
    metrics_name="land_use_conv3x3_metrics.json",
    output_values=LAND_USE_VALUES,
    colors=((40, 110, 230), (70, 180, 80), (245, 220, 90), (255, 180, 150),
            (230, 60, 40), (110, 110, 110), (150, 90, 70)),
    label_source="use-oriented pseudo labels: WorldCover + OSM buildings + OSM roads",
    label_loader=load_land_use_label,
)
SPECS = (LAND_COVER_SPEC, LAND_USE_SPEC)


class LandDataset(Dataset):
    def __init__(self, patch_ids: list[str], spec: TaskSpec, augment: bool):
        self.patch_ids = patch_ids
        self.spec = spec
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, index: int):
        patch_id = self.patch_ids[index]
        feature = np.load(EMBEDDINGS / "202604" / f"{patch_id}.npy").astype(np.float32)
        x = torch.from_numpy(np.nan_to_num(feature))
        y = torch.from_numpy(self.spec.label_loader(patch_id).astype(np.int64))
        if self.augment:
            if torch.rand(()) > 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(()) > 0.5:
                x, y = x.flip(-2), y.flip(-2)
        return x, y


class MulticlassConv3x3(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, class_count, 1),
        )

    def forward(self, x):
        return self.net(x)


def remove_tiny_water(
    prediction: np.ndarray,
    minimum_pixels: int = 4,
    water_index: int = 0,
) -> np.ndarray:
    result = prediction.copy()
    components, count = ndimage.label(result == water_index)
    for component in range(1, count + 1):
        mask = components == component
        if int(mask.sum()) >= minimum_pixels:
            continue
        ring = ndimage.binary_dilation(mask, iterations=1) & ~mask
        neighbors = result[ring & (result != water_index)]
        if neighbors.size:
            result[mask] = np.bincount(neighbors).argmax()
    return result


def _patch_ids() -> list[str]:
    return sorted(
        "patch_" + path.stem.rsplit("patch_", 1)[-1]
        for path in WORLDCOVER.glob("worldcover_*_patch_*.tif")
        if (EMBEDDINGS / "202604" / ("patch_" + path.stem.rsplit("patch_", 1)[-1] + ".npy")).exists()
    )


def train(spec: TaskSpec, device: torch.device, epochs: int) -> tuple[MulticlassConv3x3, dict]:
    patch_ids = _patch_ids()
    random.Random(42).shuffle(patch_ids)
    split = int(len(patch_ids) * 0.8)
    train_ids, val_ids = patch_ids[:split], patch_ids[split:]
    train_data = LandDataset(train_ids, spec, True)
    val_data = LandDataset(val_ids, spec, False)
    counts = np.zeros(len(spec.output_values), dtype=np.int64)
    for patch_id in train_ids:
        label = spec.label_loader(patch_id)
        for index in range(len(spec.output_values)):
            counts[index] += int((label == index).sum())
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights = np.clip(weights / weights.mean(), 0.35, 5.0)
    model = MulticlassConv3x3(len(spec.output_values)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        ignore_index=-1,
    )
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=4)
    best_accuracy, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for feature, label in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(feature.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for feature, label in val_loader:
                target = label.to(device)
                valid = target >= 0
                prediction = model(feature.to(device)).argmax(1)
                correct += int((prediction[valid] == target[valid]).sum())
                total += int(valid.sum())
        accuracy = correct / max(total, 1)
        print(f"task={spec.task_id} epoch={epoch + 1} val_accuracy={accuracy:.4f}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {
        "task_id": spec.task_id,
        "validation_accuracy": best_accuracy,
        "train_patches": len(train_ids),
        "validation_patches": len(val_ids),
        "class_pixel_counts": counts.tolist(),
        "output_values": list(spec.output_values),
        "label_source": spec.label_source,
    }


def generate(spec: TaskSpec, model: MulticlassConv3x3, device: torch.device) -> None:
    palette = np.asarray(spec.colors, dtype=np.uint8)
    model.eval()
    root = ROOT / f"data/haidian/tasks/{spec.task_id}/v1"
    for month in MONTHS:
        for embedding_path in sorted((EMBEDDINGS / month).glob("patch_*.npy")):
            feature = np.load(embedding_path).astype(np.float32)
            tensor = torch.from_numpy(np.nan_to_num(feature)).unsqueeze(0).to(device)
            with torch.inference_mode():
                prediction = model(tensor).argmax(1).squeeze(0).cpu().numpy()
            prediction = remove_tiny_water(prediction)
            tile = root / f"results/{month}/tiles/{embedding_path.stem}.png"
            array = root / f"predictions/{month}/{embedding_path.stem}.npy"
            tile.parent.mkdir(parents=True, exist_ok=True)
            array.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(palette[prediction]).save(tile)
            np.save(array, np.asarray(spec.output_values, dtype=np.uint8)[prediction])
            if month == MONTHS[-1]:
                latest = root / f"results/tiles/{embedding_path.stem}.png"
                latest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(palette[prediction]).save(latest)


def save(spec: TaskSpec, model: MulticlassConv3x3, metrics: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "__format__": "haidian_independent_land_conv3x3_v2",
            "task_id": spec.task_id,
            "state_dict": model.state_dict(),
            "output_values": spec.output_values,
            "feature_source": "P10C 64D embedding v1",
            **metrics,
        },
        MODELS / spec.checkpoint_name,
    )
    (MODELS / spec.metrics_name).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--task", choices=("all", "cover", "use"), default="all")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = SPECS if args.task == "all" else (LAND_COVER_SPEC if args.task == "cover" else LAND_USE_SPEC,)
    for spec in selected:
        model, metrics = train(spec, device, args.epochs)
        save(spec, model, metrics)
        generate(spec, model, device)
        print(MODELS / spec.checkpoint_name, flush=True)


if __name__ == "__main__":
    main()
