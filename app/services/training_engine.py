"""Training engines for user-defined classification and change-detection heads.

Training data is supplied by the frontend as a GeoJSON FeatureCollection; the
backend parses it into pixel masks, extracts embeddings, and trains a lightweight
downstream task head.
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

from app.schemas.models import GeoJSONFeatureCollection, ModelClass
from app.services.data_service import DataService
from app.services.geojson_adapter import parse_annotations_for_training
from app.services.model_registry import get_model_registry
from app.services.user_paths import get_user_dir

logger = logging.getLogger(__name__)


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
        records = parse_annotations_for_training(
            annotations=annotations,
            classes=classes,
            class_ids=class_ids,
            model_type="classification",
        )

        X_train, y_train = [], []
        for record in records:
            patch_id = record["patch_id"]
            month = record["month"]

            emb = _load_embedding_for_training(
                region_id, patch_id, month, version=embedding_version
            )
            if emb is None:
                logger.warning(f"Embedding not found for {patch_id} {month}; skipping")
                continue

            mask = record["mask"]
            D, H, W = emb.shape
            if mask.shape != (H, W):
                mask_pil = Image.fromarray(mask.astype(np.uint8))
                mask = np.array(mask_pil.resize((W, H), Image.Resampling.NEAREST))

            emb_flat = emb.reshape(D, -1).T  # [H*W, D]
            mask_flat = mask.flatten()

            pos_indices = np.where(mask_flat > 0)[0]
            neg_indices = np.where(mask_flat == 0)[0]

            if len(pos_indices) == 0:
                continue

            label = record["label_index"]
            X_train.append(emb_flat[pos_indices])
            y_train.append(np.full(len(pos_indices), label))

            n_neg = min(len(neg_indices), len(pos_indices) * 2)
            if n_neg > 0:
                neg_sample = np.random.choice(neg_indices, n_neg, replace=False)
                X_train.append(emb_flat[neg_sample])
                y_train.append(np.full(n_neg, 0))  # background

        if not X_train:
            raise ValueError("No valid training samples after filtering")

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_scaled, y_train)

        class_records = [c.model_dump() for c in classes if c.id in class_ids]
        model_data = {
            "scaler": scaler,
            "model": clf,
            "classes": class_records,
            "class_ids": class_ids,
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
        records = parse_annotations_for_training(
            annotations=annotations,
            classes=classes,
            class_ids=class_ids,
            model_type="change_detection",
        )

        X_train, y_train = [], []
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

            mask = record["mask"]
            diff = emb_after - emb_before
            D, H, W = diff.shape
            if mask.shape != (H, W):
                mask_pil = Image.fromarray(mask.astype(np.uint8))
                mask = np.array(mask_pil.resize((W, H), Image.Resampling.NEAREST))

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
                neg_sample = np.random.choice(neg_indices, n_neg, replace=False)
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

        class_records = [c.model_dump() for c in classes if c.id in class_ids]
        model_data = {
            "scaler": scaler,
            "model": clf,
            "classes": class_records,
            "class_ids": class_ids,
            "feature_type": "diff",
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
