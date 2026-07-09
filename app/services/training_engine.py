"""Training engines for user-defined classification and change-detection heads.

Training data is supplied by the frontend as a GeoJSON FeatureCollection; the
backend parses it into pixel masks, extracts embeddings, and trains a lightweight
downstream task head.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from app.schemas.models import GeoJSONFeatureCollection, ModelClass
from app.services.data_service import DataService
from app.services.fewshot_heads import BinaryConv3x3ProbeHead
from app.services.geojson_adapter import parse_annotations_for_training
from app.services.model_registry import get_model_registry
from app.services.user_paths import get_user_dir

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "torch_fewshot_head"
DEFAULT_HEAD_TYPE = "binary_conv3x3"
IGNORE_LABEL = -1.0


def _load_embedding_for_training(
    region_id: str, patch_id: str, month: str, version: str = "v2"
) -> Optional[np.ndarray]:
    """Load raw embedding array for a patch/month used during training.

    Supports both Harbin (month/patch_id.npy) and Haidian
    (patch_id/patch_id_month.npz) layouts via DataService.
    """
    npz_path = DataService.get_embedding_path(
        region_id, patch_id, fmt="npz", version=version, month=month
    )
    if npz_path and npz_path.endswith(".npz"):
        data = np.load(npz_path)
        if "embedding" in data:
            return data["embedding"]
        return data[data.files[0]]

    npy_path = DataService.get_embedding_path(
        region_id, patch_id, fmt="npy", version=version, month=month
    )
    if npy_path and npy_path.endswith(".npy"):
        return np.load(npy_path)

    return None


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(np.uint8)
    return np.array(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (width, height), Image.Resampling.NEAREST
        )
    )


def _normalize_feature_map(feature: np.ndarray) -> np.ndarray:
    """Normalize channels per patch while keeping spatial structure."""
    feature = feature.astype(np.float32, copy=False)
    mean = feature.mean(axis=(1, 2), keepdims=True)
    std = feature.std(axis=(1, 2), keepdims=True)
    return (feature - mean) / np.maximum(std, 1e-6)


def _build_binary_target(feature: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    """Create a few-shot target mask.

    Foreground polygons are positive. Unlabeled pixels are ignored by default.
    When no explicit negative labels are provided by the frontend, a small set
    of low-similarity pixels is used as weak negatives so the binary head has a
    stable contrast without treating every outside pixel as background.
    """
    _, height, width = feature.shape
    target = np.full((height, width), IGNORE_LABEL, dtype=np.float32)
    pos = positive_mask > 0
    if not np.any(pos):
        return target

    target[pos] = 1.0
    flat_feature = feature.reshape(feature.shape[0], -1).T
    pos_indices = np.where(pos.reshape(-1))[0]
    unlabeled_indices = np.where(~pos.reshape(-1))[0]
    if len(unlabeled_indices) == 0:
        return target

    proto = flat_feature[pos_indices].mean(axis=0)
    proto_norm = np.linalg.norm(proto) + 1e-6
    feat_norm = np.linalg.norm(flat_feature[unlabeled_indices], axis=1) + 1e-6
    similarity = flat_feature[unlabeled_indices].dot(proto) / (feat_norm * proto_norm)
    n_weak_neg = min(len(unlabeled_indices), max(16, len(pos_indices) * 2))
    weak_neg_local = np.argsort(similarity)[:n_weak_neg]
    weak_neg_indices = unlabeled_indices[weak_neg_local]
    target.reshape(-1)[weak_neg_indices] = 0.0
    return target


def _bce_dice_tversky_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    valid_logits = logits[valid]
    valid_target = target[valid]
    bce = F.binary_cross_entropy_with_logits(valid_logits, valid_target)

    probs = torch.sigmoid(valid_logits)
    smooth = 1.0
    tp = (probs * valid_target).sum()
    fp = (probs * (1.0 - valid_target)).sum()
    fn = ((1.0 - probs) * valid_target).sum()
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    tversky = (tp + smooth) / (tp + 0.3 * fp + 0.7 * fn + smooth)
    return bce + (1.0 - dice) + (1.0 - tversky)


def _train_binary_conv_head(
    samples: List[Tuple[np.ndarray, np.ndarray]],
    epochs: int,
) -> Tuple[BinaryConv3x3ProbeHead, float, float, int, int]:
    if not samples:
        raise ValueError("No valid training samples after filtering")

    embed_dim = samples[0][0].shape[0]
    device = _select_training_device()
    model = BinaryConv3x3ProbeHead(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    effective_epochs = max(1, min(int(epochs), 100))

    tensors = []
    valid_count = 0
    positive_count = 0
    for feature, target in samples:
        x = torch.from_numpy(_normalize_feature_map(feature)).float().unsqueeze(0)
        y = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
        valid_count += int((target >= 0).sum())
        positive_count += int((target == 1).sum())
        tensors.append((x.to(device), y.to(device)))

    if positive_count == 0 or valid_count <= positive_count:
        raise ValueError(
            "Training requires foreground polygons and at least a few contrast pixels"
        )

    model.train()
    for _ in range(effective_epochs):
        for x, y in tensors:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            valid = y >= 0
            loss = _bce_dice_tversky_loss(logits, y.clamp(min=0.0), valid)
            loss.backward()
            optimizer.step()

    threshold, f1 = _tune_threshold(model, tensors)
    model.cpu().eval()
    return model, threshold, f1, valid_count, effective_epochs


def _select_training_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 512 * 1024 * 1024:
            logger.warning(
                "CUDA free memory is low (%s MB); using CPU for few-shot training",
                int(free_bytes / 1024 / 1024),
            )
            return torch.device("cpu")
    except RuntimeError as exc:
        logger.warning("Unable to inspect CUDA memory; using CPU: %s", exc)
        return torch.device("cpu")
    return torch.device("cuda")


def _tune_threshold(
    model: BinaryConv3x3ProbeHead,
    tensors: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[float, float]:
    device = next(model.parameters()).device
    probs_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for x, y in tensors:
            x = x.to(device)
            y = y.to(device)
            valid = y >= 0
            probs = torch.sigmoid(model(x))[valid].detach().cpu().numpy()
            labels = y[valid].detach().cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels)

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all).astype(np.uint8)
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.linspace(0.1, 0.9, 17):
        pred = probs >= threshold
        tp = np.logical_and(pred, labels == 1).sum()
        fp = np.logical_and(pred, labels == 0).sum()
        fn = np.logical_and(~pred, labels == 1).sum()
        f1 = (2 * tp) / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _save_torch_checkpoint(
    user_id: str,
    model_id: str,
    model: BinaryConv3x3ProbeHead,
    metadata: Dict[str, Any],
) -> Path:
    registry = get_model_registry(user_id)
    record = registry.get_model(model_id)
    if record is None:
        raise ValueError(f"Model {model_id} not found in registry")
    model_path = Path(record["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "__format__": CHECKPOINT_FORMAT,
        "head_type": DEFAULT_HEAD_TYPE,
        "state_dict": model.state_dict(),
        "embed_dim": metadata["embed_dim"],
        "hidden_dim": 128,
        **metadata,
    }
    torch.save(checkpoint, model_path)
    return model_path


class ClassificationTrainingEngine:
    """Train a lightweight classification head from frontend GeoJSON annotations."""

    def __init__(self, user_id: str = "default") -> None:
        self._user_id = user_id
        self._user_dir = get_user_dir(user_id)

    def train(
        self,
        model_id: str,
        region_id: str,
        task_type: str,
        embedding_version: str,
        annotations: GeoJSONFeatureCollection,
        classes: List[ModelClass],
        class_ids: List[str],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """Train a classification head.

        Args:
            model_id: Registry model identifier.
            region_id: Training region.
            task_type: Downstream task type.
            embedding_version: Embedding version (v1/v2).
            annotations: GeoJSON FeatureCollection from the frontend.
            classes: Class definitions from the frontend.
            class_ids: Active class IDs to train on.

        Returns:
            Dict with model_path, accuracy, n_samples.
        """
        records, class_map = parse_annotations_for_training(
            annotations=annotations,
            classes=classes,
            class_ids=class_ids,
            model_type="classification",
        )

        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for record in records:
            patch_id = record["patch_id"]
            month = record["month"]

            emb = _load_embedding_for_training(
                region_id, patch_id, month, version=embedding_version
            )
            if emb is None:
                logger.warning(f"Embedding not found for {patch_id} {month}; skipping")
                continue

            D, H, W = emb.shape
            mask = _resize_mask(record["mask"], H, W)
            if not np.any(mask > 0):
                continue
            target = _build_binary_target(emb, mask)
            samples.append((emb.astype(np.float32, copy=False), target))

        model, threshold, f1, n_samples, effective_epochs = _train_binary_conv_head(
            samples, epochs
        )

        class_records = [c.model_dump() for c in classes if c.id in class_ids]
        metadata = {
            "classes": class_records,
            "class_ids": class_ids,
            "class_map": class_map,
            "positive_class_id": class_ids[0] if class_ids else None,
            "feature_type": "embedding",
            "task_type": task_type,
            "region_id": region_id,
            "embedding_version": embedding_version,
            "embed_dim": samples[0][0].shape[0],
            "threshold": threshold,
            "epochs": effective_epochs,
            "trained_at": datetime.now().isoformat(),
        }
        model_path = _save_torch_checkpoint(self._user_id, model_id, model, metadata)
        return {
            "model_id": model_id,
            "model_path": str(model_path),
            "accuracy": f1,
            "n_samples": n_samples,
        }


class ChangeDetectionTrainingEngine:
    """Train a lightweight change-detection head from frontend GeoJSON annotations."""

    def __init__(self, user_id: str = "default") -> None:
        self._user_id = user_id
        self._user_dir = get_user_dir(user_id)

    def train(
        self,
        model_id: str,
        region_id: str,
        task_type: str,
        embedding_version: str,
        annotations: GeoJSONFeatureCollection,
        classes: List[ModelClass],
        class_ids: List[str],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """Train a change-detection head.

        Args:
            model_id: Registry model identifier.
            region_id: Training region.
            task_type: Downstream task type (change_detection).
            embedding_version: Embedding version (v1/v2).
            annotations: GeoJSON FeatureCollection from the frontend.
            classes: Class definitions from the frontend.
            class_ids: Active class IDs to train on.

        Returns:
            Dict with model_path, accuracy, n_samples.
        """
        records, class_map = parse_annotations_for_training(
            annotations=annotations,
            classes=classes,
            class_ids=class_ids,
            model_type="change_detection",
        )

        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        used_months: set = set()
        for record in records:
            patch_id = record["patch_id"]
            before_month = record["before_month"]
            after_month = record["after_month"]

            emb_before = _load_embedding_for_training(
                region_id, patch_id, before_month, version=embedding_version
            )
            emb_after = _load_embedding_for_training(
                region_id, patch_id, after_month, version=embedding_version
            )
            if emb_before is None or emb_after is None:
                logger.warning(
                    f"Embedding not found for {patch_id} {before_month}/{after_month}"
                )
                continue

            diff = emb_after - emb_before
            D, H, W = diff.shape
            mask = _resize_mask(record["mask"], H, W)
            if not np.any(mask > 0):
                continue
            target = _build_binary_target(diff, mask)
            samples.append((diff.astype(np.float32, copy=False), target))
            used_months.add((before_month, after_month))

        model, threshold, f1, n_samples, effective_epochs = _train_binary_conv_head(
            samples, epochs
        )

        class_records = [c.model_dump() for c in classes if c.id in class_ids]
        # Determine the before/after months that were actually used.
        before_month, after_month = next(iter(used_months))
        metadata = {
            "classes": class_records,
            "class_ids": class_ids,
            "class_map": class_map,
            "positive_class_id": class_ids[0] if class_ids else None,
            "feature_type": "diff",
            "task_type": task_type,
            "region_id": region_id,
            "embedding_version": embedding_version,
            "before_month": before_month,
            "after_month": after_month,
            "embed_dim": samples[0][0].shape[0],
            "threshold": threshold,
            "epochs": effective_epochs,
            "trained_at": datetime.now().isoformat(),
        }
        model_path = _save_torch_checkpoint(self._user_id, model_id, model, metadata)
        return {
            "model_id": model_id,
            "model_path": str(model_path),
            "accuracy": f1,
            "n_samples": n_samples,
        }
