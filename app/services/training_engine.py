"""Training engines for user-defined classification and change-detection heads.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.services.annotation_service import (
    get_annotation_store,
    get_class_manager,
    _get_user_dir,
)
from app.services.data_service import DataService
from app.services.model_registry import get_model_registry

logger = logging.getLogger(__name__)


# Shared helpers for mask decoding. These mirror annotation_service but operate
# on the already-rasterized .npz masks rather than geometry payloads.


def _load_embedding_for_training(
    region_id: str, patch_id: str, month: str, version: str = "v2"
) -> Optional[np.ndarray]:
    """Load raw embedding array for a patch/month used during training.

    Supports both Harbin (month/patch_id.npy) and Haidian
    (patch_id/patch_id_month.npz) layouts via DataService.
    """
    fmt = "npy"
    npz_path = DataService.get_embedding_path(region_id, patch_id, fmt="npz", version=version, month=month)
    if npz_path and npz_path.endswith(".npz"):
        data = np.load(npz_path)
        # Haidian stores embedding under key 'embedding' or as the sole array
        if "embedding" in data:
            return data["embedding"]
        return data[data.files[0]]

    npy_path = DataService.get_embedding_path(region_id, patch_id, fmt=fmt, version=version, month=month)
    if npy_path and npy_path.endswith(".npy"):
        return np.load(npy_path)

    return None


class ClassificationTrainingEngine:
    """Train a lightweight classification head from user annotations."""

    def __init__(self, user_id: str = "default") -> None:
        self._user_id = user_id
        self._user_dir = _get_user_dir(user_id)

    def train(
        self,
        model_id: str,
        region_id: str,
        task_type: str,
        embedding_version: str = "v2",
    ) -> Dict[str, Any]:
        store = get_annotation_store(self._user_id)
        mgr = get_class_manager(self._user_id)
        classes = {c["id"]: c for c in mgr.list_classes()}

        annotations = store.list_annotations(
            region_id=region_id, task_type=task_type
        )
        if not annotations:
            raise ValueError("No annotations available for training")

        X_train, y_train = [], []
        for ann in annotations:
            month = ann.get("month")
            patch_id = ann.get("patch_id")
            if not month or not patch_id:
                continue

            emb = _load_embedding_for_training(
                region_id, patch_id, month, version=embedding_version
            )
            if emb is None:
                logger.warning(
                    f"Embedding not found for {patch_id} {month}; skipping annotation {ann['id']}"
                )
                continue

            mask_path = self._user_dir / "masks" / f"{ann['id']}.npz"
            if not mask_path.exists():
                continue

            mask = np.load(mask_path)["mask"]

            # Resize mask to match embedding spatial size if needed
            D, H, W = emb.shape
            if mask.shape != (H, W):
                mask_pil = Image.fromarray(mask.astype(np.uint8))
                mask = np.array(
                    mask_pil.resize((W, H), Image.Resampling.NEAREST)
                )

            emb_flat = emb.reshape(D, -1).T  # [H*W, D]
            mask_flat = mask.flatten()

            pos_indices = np.where(mask_flat > 0)[0]
            neg_indices = np.where(mask_flat == 0)[0]

            if len(pos_indices) == 0:
                continue

            class_idx = list(classes.keys()).index(ann["class_id"])
            X_train.append(emb_flat[pos_indices])
            y_train.append(np.full(len(pos_indices), class_idx))

            n_neg = min(len(neg_indices), len(pos_indices) * 2)
            if n_neg > 0:
                neg_sample = np.random.choice(
                    neg_indices, n_neg, replace=False
                )
                X_train.append(emb_flat[neg_sample])
                y_train.append(np.full(n_neg, -1))

        if not X_train:
            raise ValueError("No valid training samples after filtering")

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_scaled, y_train)

        model_data = {
            "scaler": scaler,
            "model": clf,
            "classes": list(classes.values()),
            "class_ids": list(classes.keys()),
            "task_type": task_type,
            "region_id": region_id,
            "embedding_version": embedding_version,
            "trained_at": datetime.now().isoformat(),
        }

        registry = get_model_registry(self._user_id)
        record = registry.get_model(model_id)
        if record is None:
            raise ValueError(f"Model {model_id} not found in registry")
        model_path = Path(record["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)

        accuracy = float(clf.score(X_scaled, y_train))
        return {
            "model_id": model_id,
            "model_path": str(model_path),
            "accuracy": accuracy,
            "n_samples": len(y_train),
        }


class ChangeDetectionTrainingEngine:
    """Train a lightweight change-detection head from user annotations."""

    def __init__(self, user_id: str = "default") -> None:
        self._user_id = user_id
        self._user_dir = _get_user_dir(user_id)

    def train(
        self,
        model_id: str,
        region_id: str,
        embedding_version: str = "v2",
    ) -> Dict[str, Any]:
        store = get_annotation_store(self._user_id)
        annotations = store.list_annotations(
            region_id=region_id, task_type="change_detection"
        )

        cd_annotations = [
            ann
            for ann in annotations
            if ann.get("before_month") and ann.get("after_month")
        ]
        if not cd_annotations:
            raise ValueError(
                "No change-detection annotations available. "
                "Annotate with before/after months first."
            )

        X_train, y_train = [], []
        for ann in cd_annotations:
            before_month = ann["before_month"]
            after_month = ann["after_month"]
            patch_id = ann.get("patch_id")
            if not patch_id:
                continue

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

            mask_path = self._user_dir / "masks" / f"{ann['id']}.npz"
            if not mask_path.exists():
                continue

            mask = np.load(mask_path)["mask"]
            diff = emb_after - emb_before
            D, H, W = diff.shape
            if mask.shape != (H, W):
                mask_pil = Image.fromarray(mask.astype(np.uint8))
                mask = np.array(
                    mask_pil.resize((W, H), Image.Resampling.NEAREST)
                )

            diff_flat = diff.reshape(D, -1).T
            mask_flat = mask.flatten()

            pos_indices = np.where(mask_flat > 0)[0]
            neg_indices = np.where(mask_flat == 0)[0]

            if len(pos_indices) == 0:
                continue

            X_train.append(diff_flat[pos_indices])
            y_train.append(np.ones(len(pos_indices), dtype=np.int32))

            n_neg = min(len(neg_indices), len(pos_indices) * 3)
            if n_neg > 0:
                neg_sample = np.random.choice(
                    neg_indices, n_neg, replace=False
                )
                X_train.append(diff_flat[neg_sample])
                y_train.append(np.zeros(n_neg, dtype=np.int32))

        if not X_train:
            raise ValueError("No valid training samples after filtering")

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_scaled, y_train)

        model_data = {
            "scaler": scaler,
            "model": clf,
            "feature_type": "diff",
            "task_type": "change_detection",
            "region_id": region_id,
            "embedding_version": embedding_version,
            "trained_at": datetime.now().isoformat(),
        }

        registry = get_model_registry(self._user_id)
        record = registry.get_model(model_id)
        if record is None:
            raise ValueError(f"Model {model_id} not found in registry")
        model_path = Path(record["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)

        accuracy = float(clf.score(X_scaled, y_train))
        return {
            "model_id": model_id,
            "model_path": str(model_path),
            "accuracy": accuracy,
            "n_samples": len(y_train),
        }
