"""Persistent registry for user-trained downstream-task heads.
"""

import json
import uuid
import fcntl
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.user_paths import get_users_dir


class ModelRegistry:
    """Persistent registry for user-trained classification and change-detection heads.

    Stores an index JSON per user and model artifacts (.pkl) under
    ``users/{user_id}/models/``.
    """

    def __init__(self, index_path: Path, models_dir: Path) -> None:
        self._path = index_path
        self._models_dir = models_dir
        self._lock_path = index_path.with_suffix(".lock")
        self._data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                if not isinstance(self._data, list):
                    self._data = []
            except Exception:
                self._data = []

    @contextmanager
    def _locked(self):
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save(self) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)
        finally:
            # Clean up the unique temp file if replace() did not move it.
            if tmp.exists():
                tmp.unlink()

    def list_models(self) -> List[Dict[str, Any]]:
        with self._locked():
            self._load()
            return sorted(
                [dict(m) for m in self._data],
                key=lambda m: m.get("created_at", ""),
                reverse=True,
            )

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._locked():
            self._load()
            for m in self._data:
                if m.get("id") == model_id:
                    return dict(m)
        return None

    def create_model(
        self,
        name: str,
        model_type: str,
        classes: List[Dict[str, Any]],
        task_type: Optional[str] = None,
        region_id: Optional[str] = None,
        description: Optional[str] = None,
        requested_training_method: Optional[str] = None,
        feature_source: Optional[str] = None,
    ) -> str:
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        record = {
            "id": model_id,
            "name": name,
            "type": model_type,
            "task_type": task_type,
            "region_id": region_id,
            "description": description,
            "requested_training_method": requested_training_method,
            "resolved_training_method": None,
            "feature_source": feature_source,
            "foundation_model_id": None,
            "foundation_model_version": None,
            "feature_dimension": None,
            "preprocessing_version": None,
            "head_type": None,
            "checkpoint_format": None,
            "compatible_regions": [region_id] if region_id else [],
            "status": "training",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "classes": classes,
            "accuracy": None,
            "metric_name": None,
            "n_samples": None,
            "model_path": str(self._models_dir / f"{model_id}.pkl"),
            "message": None,
        }
        with self._locked():
            self._load()
            self._data.append(record)
            self._save()
        return model_id

    def update_model(self, model_id: str, **kwargs: Any) -> bool:
        with self._locked():
            self._load()
            for m in self._data:
                if m.get("id") == model_id:
                    m.update(kwargs)
                    self._save()
                    return True
        return False

    def rename_model(self, model_id: str, name: str) -> bool:
        return self.update_model(model_id, name=name)

    def delete_model(self, model_id: str) -> bool:
        with self._locked():
            self._load()
            record = next((m for m in self._data if m.get("id") == model_id), None)
            if record is None:
                return False
            pkl_path = Path(record.get("model_path", ""))
            if pkl_path.exists():
                pkl_path.unlink()
            self._data = [m for m in self._data if m.get("id") != model_id]
            self._save()
            return True


# Per-user singleton
_registries: Dict[str, ModelRegistry] = {}


def get_model_registry(user_id: str = "default") -> ModelRegistry:
    if user_id not in _registries:
        user_dir = get_users_dir() / user_id
        _registries[user_id] = ModelRegistry(
            user_dir / "models_index.json", user_dir / "models"
        )
    return _registries[user_id]
