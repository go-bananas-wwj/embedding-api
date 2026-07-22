"""Inference engine for user-trained and system downstream-task heads.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
import torch

from app.services.data_service import DataService
from app.services.fewshot_heads import BinaryConv3x3ProbeHead, PixelMLPHead
from app.services.external_embeddings import EXTERNAL_MLP_CHECKPOINT_FORMAT, load_external_embedding
from app.services.model_binding import validate_model_binding
from app.services.model_registry import get_model_registry
from app.services.pu_query import CHECKPOINT_FORMAT as PU_QUERY_CHECKPOINT_FORMAT
from app.services.pu_query import score_pu_query
from app.services.user_paths import get_user_dir
from app.services.s2_ml import CHECKPOINT_FORMAT as S2_RF_CHECKPOINT_FORMAT, load_s2_features

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

        try:
            model_data = joblib.load(model_path)
        except Exception:
            model_data = torch.load(model_path, map_location="cpu", weights_only=False)
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
        record = get_model_registry(self._user_id).get_model(model_id) or {}
        validate_model_binding(record, model_data, requested_region=region_id)
        if model_data.get("__format__") == S2_RF_CHECKPOINT_FORMAT:
            return self._infer_s2_random_forest(
                model_data, model_id, region_id, patch_id, month=month
            )
        if model_data.get("__format__") == EXTERNAL_MLP_CHECKPOINT_FORMAT:
            return self._infer_external_mlp(
                model_data,
                model_id,
                region_id,
                patch_id,
                month=month,
                before_month=before_month,
                after_month=after_month,
            )
        if model_data.get("__format__") == PU_QUERY_CHECKPOINT_FORMAT:
            return self._infer_pu_query(
                model_data,
                model_id,
                region_id,
                patch_id,
                month=month,
                before_month=before_month,
                after_month=after_month,
            )
        if model_data.get("__format__") == "torch_fewshot_head":
            return self._infer_torch_fewshot(
                model_data,
                model_id,
                region_id,
                patch_id,
                month=month,
                before_month=before_month,
                after_month=after_month,
            )

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

    def _infer_s2_random_forest(
        self,
        model_data: Dict[str, Any],
        model_id: str,
        region_id: str,
        patch_id: str,
        month: Optional[str],
    ) -> str:
        if not month:
            raise ValueError("Traditional Sentinel-2 inference requires month")
        trained_region = model_data.get("region_id")
        if trained_region and trained_region != region_id:
            raise ValueError(
                f"Model was trained for region '{trained_region}', not '{region_id}'"
            )
        feature, valid, _source_path = load_s2_features(region_id, patch_id, month)
        flat = feature.reshape(feature.shape[0], -1).T
        probability = model_data["model"].predict_proba(flat)[:, 1]
        prediction = (
            probability.reshape(valid.shape) >= float(model_data.get("threshold", 0.5))
        ).astype(np.uint8)
        prediction[~valid] = 0
        rendered = self._color_encode_binary_fewshot(prediction, model_data)
        result_path = self.results_dir / (
            f"infer_{model_id}_{region_id}_{patch_id}_{month}.png"
        )
        Image.fromarray(rendered).resize(
            (128, 128), Image.Resampling.NEAREST
        ).save(result_path)
        return str(result_path)

    def _infer_external_mlp(
        self,
        model_data: Dict[str, Any],
        model_id: str,
        region_id: str,
        patch_id: str,
        month: Optional[str],
        before_month: Optional[str],
        after_month: Optional[str],
    ) -> str:
        trained_region = model_data.get("region_id")
        if trained_region and trained_region != region_id:
            raise ValueError(
                f"Model was trained for region '{trained_region}', not '{region_id}'"
            )
        method = model_data["training_method"]
        is_change = model_data.get("model_type") == "change_detection"
        if is_change:
            if not before_month or not after_month:
                raise ValueError("Change-detection inference requires before_month and after_month")
            before = load_external_embedding(method, region_id, patch_id, before_month)
            after = load_external_embedding(method, region_id, patch_id, after_month)
            if before.shape != after.shape:
                raise ValueError(
                    f"External embedding grids do not match: before={before.shape}, after={after.shape}"
                )
            feature = np.concatenate(
                [before, after, np.abs(after - before), before * after], axis=0
            )
            suffix = f"{before_month}_vs_{after_month}"
        else:
            if not month:
                raise ValueError("Single-time inference requires month")
            feature = load_external_embedding(method, region_id, patch_id, month)
            suffix = month
        head = PixelMLPHead(
            int(model_data["embed_dim"]), int(model_data.get("hidden_dim", 128)), dropout=0.0
        )
        head.load_state_dict(model_data["state_dict"])
        head.eval()
        mean = np.asarray(model_data["normalization_mean"], dtype=np.float32)
        std = np.asarray(model_data["normalization_std"], dtype=np.float32)
        if feature.shape[0] != len(mean) or len(mean) != len(std):
            raise ValueError("External embedding does not match the trained MLP normalization")
        normalized = (feature - mean[:, None, None]) / std[:, None, None]
        with torch.inference_mode():
            probability = torch.sigmoid(
                head(torch.from_numpy(normalized).float().unsqueeze(0))
            ).squeeze().numpy()
        prediction = (probability >= float(model_data.get("threshold", 0.5))).astype(np.uint8)
        rendered = (
            self._color_encode_cd(prediction)
            if is_change
            else self._color_encode_binary_fewshot(prediction, model_data)
        )
        result_path = self.results_dir / f"infer_{model_id}_{region_id}_{patch_id}_{suffix}.png"
        Image.fromarray(rendered).resize((128, 128), Image.Resampling.NEAREST).save(result_path)
        return str(result_path)

    def _infer_pu_query(
        self,
        model_data: Dict[str, Any],
        model_id: str,
        region_id: str,
        patch_id: str,
        month: Optional[str] = None,
        before_month: Optional[str] = None,
        after_month: Optional[str] = None,
    ) -> str:
        """Run sparse positive-unlabeled retrieval with guarded query adaptation."""
        task_type = model_data.get("task_type", "single_time_detection")
        embedding_version = model_data.get("embedding_version", "v2")
        feature_type = model_data.get("feature_type", "embedding")
        is_change = feature_type == "diff" or task_type == "change_detection"

        if is_change:
            before_month = before_month or model_data.get("before_month")
            after_month = after_month or model_data.get("after_month")
            if not before_month or not after_month:
                raise ValueError(
                    "Change-detection inference requires before_month and after_month"
                )
            before = _load_embedding_for_inference(
                region_id, patch_id, before_month, version=embedding_version
            )
            after = _load_embedding_for_inference(
                region_id, patch_id, after_month, version=embedding_version
            )
            if before is None:
                raise FileNotFoundError(
                    f"Embedding not found for {patch_id} {before_month}"
                )
            if after is None:
                raise FileNotFoundError(
                    f"Embedding not found for {patch_id} {after_month}"
                )
            feature = after - before
            result_filename = (
                f"infer_{model_id}_{region_id}_{patch_id}_"
                f"{before_month}_vs_{after_month}.png"
            )
        else:
            if not month:
                raise ValueError("Single-time inference requires month")
            feature = _load_embedding_for_inference(
                region_id, patch_id, month, version=embedding_version
            )
            if feature is None:
                raise FileNotFoundError(f"Embedding not found for {patch_id} {month}")
            result_filename = (
                f"infer_{model_id}_{region_id}_{patch_id}_{month}.png"
            )

        score, query_adapted = score_pu_query(feature, model_data)
        logger.debug(
            "PU query inference model=%s patch=%s query_adapted=%s",
            model_id,
            patch_id,
            query_adapted,
        )
        prediction = (score >= float(model_data["threshold"])).astype(np.uint8)
        if is_change:
            rendered = self._color_encode_cd(prediction)
        else:
            rendered = self._color_encode_binary_fewshot(prediction, model_data)
        image = Image.fromarray(rendered).resize(
            (128, 128), Image.Resampling.NEAREST
        )
        result_path = self.results_dir / result_filename
        image.save(result_path)
        return str(result_path)

    def _infer_torch_fewshot(
        self,
        model_data: Dict[str, Any],
        model_id: str,
        region_id: str,
        patch_id: str,
        month: Optional[str] = None,
        before_month: Optional[str] = None,
        after_month: Optional[str] = None,
    ) -> str:
        task_type = model_data.get("task_type", "single_time_detection")
        embedding_version = model_data.get("embedding_version", "v2")
        feature_type = model_data.get("feature_type", "embedding")

        if feature_type == "diff" or task_type == "change_detection":
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
                raise ValueError("Single-time inference requires month")
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

        feature = self._normalize_feature_map(emb)
        model = BinaryConv3x3ProbeHead(
            embed_dim=int(model_data["embed_dim"]),
            hidden_dim=int(model_data.get("hidden_dim", 128)),
            dropout=0.0,
        )
        model.load_state_dict(model_data["state_dict"])
        model.eval()

        with torch.no_grad():
            x = torch.from_numpy(feature).float().unsqueeze(0)
            prob = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        pred = (prob >= float(model_data.get("threshold", 0.5))).astype(np.uint8)

        if feature_type == "diff" or task_type == "change_detection":
            rgb = self._color_encode_cd(pred)
        else:
            rgb = self._color_encode_binary_fewshot(pred, model_data)

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
        model_data = self._load_model(model_id)
        record = get_model_registry(self._user_id).get_model(model_id) or {}
        validate_model_binding(record, model_data, requested_region=region_id)
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
    def _color_encode_binary_fewshot(
        pred: np.ndarray, model_data: Dict[str, Any]
    ) -> np.ndarray:
        """Color-encode the binary few-shot head output."""
        H, W = pred.shape
        rgb = np.full((H, W, 3), 200, dtype=np.uint8)
        classes = model_data.get("classes", [])
        positive_class_id = model_data.get("positive_class_id")
        positive_class = None
        for cls in classes:
            if cls.get("id") == positive_class_id:
                positive_class = cls
                break
        if positive_class is None and classes:
            positive_class = classes[0]
        color = (255, 0, 0)
        if positive_class:
            color = InferenceEngine._hex_to_rgb(positive_class.get("color", "#ff0000"))
        rgb[pred == 1] = color
        return rgb

    @staticmethod
    def _normalize_feature_map(feature: np.ndarray) -> np.ndarray:
        feature = feature.astype(np.float32, copy=False)
        mean = feature.mean(axis=(1, 2), keepdims=True)
        std = feature.std(axis=(1, 2), keepdims=True)
        return (feature - mean) / np.maximum(std, 1e-6)

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
