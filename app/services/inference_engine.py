"""Inference engine for user-trained and system downstream-task heads.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image

from app.services.data_service import DataService
from app.services.model_registry import get_model_registry
from app.services.user_paths import get_user_dir

logger = logging.getLogger(__name__)


def _load_embedding_for_inference(
    region_id: str, patch_id: str, month: str, version: str = "v2"
) -> Optional[np.ndarray]:
    """Load raw embedding array for inference."""
    npz_path = DataService.get_embedding_path(
        region_id, patch_id, fmt="npz", version=version, month=month
    )
    if npz_path and npz_path.endswith(".npz"):
        with np.load(npz_path) as data:
            if "embedding" in data:
                return data["embedding"]
            return data[data.files[0]]

    npy_path = DataService.get_embedding_path(
        region_id, patch_id, fmt="npy", version=version, month=month
    )
    if npy_path and npy_path.endswith(".npy"):
        return np.load(npy_path)

    return None


class InferenceEngine:
    """Run inference with a user-trained or system model."""

    def __init__(self, user_id: str = "default") -> None:
        self._user_id = user_id
        self._user_dir = get_user_dir(user_id)
        self.results_dir = self._user_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_model(self, model_id: str) -> Dict[str, Any]:
        if model_id in self._cache:
            return self._cache[model_id]

        registry = get_model_registry(self._user_id)
        record = registry.get_model(model_id)
        if record is None:
            raise ValueError(f"Model {model_id} not found")

        model_path = Path(record["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_data = joblib.load(model_path)
        self._cache[model_id] = model_data
        return model_data

    def infer(
        self,
        model_id: str,
        region_id: str,
        patch_id: str,
        month: Optional[str] = None,
        before_month: Optional[str] = None,
        after_month: Optional[str] = None,
    ) -> str:
        """Run single-patch inference. Returns path to result PNG."""
        model_data = self._load_model(model_id)
        scaler = model_data["scaler"]
        clf = model_data["model"]
        task_type = model_data.get("task_type", "classification")
        embedding_version = model_data.get("embedding_version", "v2")
        feature_type = model_data.get("feature_type", "embedding")

        if feature_type == "diff" or task_type == "change_detection":
            # Change-detection head: compute embedding difference.
            if not before_month or not after_month:
                # Fall back to months stored during training if available.
                before_month = before_month or model_data.get("before_month")
                after_month = after_month or model_data.get("after_month")
                if not before_month or not after_month:
                    raise ValueError(
                        "Change-detection inference requires before_month and after_month"
                    )
            emb_before = _load_embedding_for_inference(
                region_id, patch_id, before_month, version=embedding_version
            )
            emb_after = _load_embedding_for_inference(
                region_id, patch_id, after_month, version=embedding_version
            )
            if emb_before is None:
                raise FileNotFoundError(
                    f"Embedding not found for {patch_id} {before_month}"
                )
            if emb_after is None:
                raise FileNotFoundError(
                    f"Embedding not found for {patch_id} {after_month}"
                )
            emb = emb_after - emb_before
            result_filename = (
                f"infer_{model_id}_{region_id}_{patch_id}_"
                f"{before_month}_vs_{after_month}.png"
            )
        else:
            if not month:
                raise ValueError("Classification inference requires month")
            emb = _load_embedding_for_inference(
                region_id, patch_id, month, version=embedding_version
            )
            if emb is None:
                raise FileNotFoundError(
                    f"Embedding not found for {patch_id} {month}"
                )
            result_filename = (
                f"infer_{model_id}_{region_id}_{patch_id}_{month}.png"
            )

        D, H, W = emb.shape
        flat = emb.reshape(D, -1).T
        flat_s = scaler.transform(flat)
        pred = clf.predict(flat_s).reshape(H, W)

        if feature_type == "diff" or task_type == "change_detection":
            rgb = self._color_encode_cd(pred)
        else:
            classes = model_data.get("classes", [])
            class_map = model_data.get("class_map", {})
            rgb = self._color_encode_classes(pred, classes, class_map)

        img = Image.fromarray(rgb).resize(
            (128, 128), Image.Resampling.NEAREST
        )
        result_path = self.results_dir / result_filename
        img.save(result_path)
        return str(result_path)

    def infer_batch(
        self,
        model_id: str,
        region_id: str,
        patch_ids: List[str],
        month: Optional[str] = None,
        before_month: Optional[str] = None,
        after_month: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference for a list of patches."""
        results = []
        for patch_id in patch_ids:
            try:
                path = self.infer(
                    model_id,
                    region_id,
                    patch_id,
                    month=month,
                    before_month=before_month,
                    after_month=after_month,
                )
                results.append(
                    {
                        "patch_id": patch_id,
                        "status": "success",
                        "result_path": path,
                    }
                )
            except Exception as e:
                logger.warning(f"Batch inference failed for {patch_id}: {e}")
                results.append(
                    {
                        "patch_id": patch_id,
                        "status": "error",
                        "error": str(e),
                        "result_path": None,
                    }
                )
        return results

    @staticmethod
    def _color_encode_classes(
        pred: np.ndarray,
        classes: List[Dict[str, Any]],
        class_map: Dict[str, int],
    ) -> np.ndarray:
        """Color-encode a multi-class prediction.

        Background is label 0. User classes have labels from class_map.
        """
        H, W = pred.shape
        rgb = np.full((H, W, 3), 200, dtype=np.uint8)

        # Build reverse map: label_index -> class dict
        label_to_class: Dict[int, Dict[str, Any]] = {}
        for cls in classes:
            label = class_map.get(cls["id"])
            if label is not None:
                label_to_class[label] = cls

        for label, cls in label_to_class.items():
            color = cls.get("color", "#cccccc")
            rgb[pred == label] = InferenceEngine._hex_to_rgb(color)

        rgb[pred == -1] = (200, 200, 200)
        return rgb

    @staticmethod
    def _color_encode_cd(pred: np.ndarray) -> np.ndarray:
        """Color-encode a binary change-detection prediction with alpha."""
        H, W = pred.shape
        rgba = np.full((H, W, 4), 0, dtype=np.uint8)
        rgba[pred == 1] = [239, 68, 68, 180]
        rgba[pred == 0] = [0, 0, 0, 0]
        return rgba

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
