"""Annotation storage and class management for user-generated training labels.

Each user has an isolated workspace under ``users/{user_id}/``:

  users/{user_id}/
  ├── classes.json
  ├── annotations.json
  └── masks/
      └── {ann_id}.npz
"""

import base64
import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from app.config import get_config


# Base directory for per-user annotation data. This is intentionally outside
# the repository data dirs so user-generated artifacts can be excluded from Git.
DEFAULT_USERS_DIR = Path("users")


def _get_users_dir() -> Path:
    """Return the root users directory from config or default."""
    users_dir = get_config().get("users_dir", default=None)
    path = Path(users_dir) if users_dir else DEFAULT_USERS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_user_dir(user_id: str) -> Path:
    """Return (and create) the annotation directory for a user."""
    d = _get_users_dir() / user_id / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _base64_to_mask(b64_str: str) -> np.ndarray:
    """Decode a base64 PNG to a binary mask."""
    img = Image.open(BytesIO(base64.b64decode(b64_str)))
    img = img.convert("L")
    return np.array(img) > 128


def _mask_to_base64_png(mask: np.ndarray) -> str:
    """Encode a binary mask to a base64 RGBA PNG string."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask > 0] = [255, 255, 255, 255]
    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ClassManager:
    """Manage user-defined classification classes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> List[Dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def list_classes(self) -> List[Dict[str, Any]]:
        return self._load()

    def create_class(self, name: str, color: str) -> Dict[str, Any]:
        classes = self._load()
        cls = {"id": f"cls_{uuid.uuid4().hex[:8]}", "name": name, "color": color}
        classes.append(cls)
        self._save(classes)
        return cls

    def get_class(self, class_id: str) -> Optional[Dict[str, Any]]:
        for c in self._load():
            if c.get("id") == class_id:
                return c
        return None

    def delete_class(self, class_id: str) -> bool:
        classes = self._load()
        classes = [c for c in classes if c.get("id") != class_id]
        self._save(classes)
        return True

    def rename_class(self, class_id: str, new_name: str) -> bool:
        classes = self._load()
        for c in classes:
            if c.get("id") == class_id:
                c["name"] = new_name
                self._save(classes)
                return True
        return False


class AnnotationStore:
    """Store user annotations and their rasterized masks."""

    def __init__(self, index_path: Path, masks_dir: Path) -> None:
        self.index_path = index_path
        self.masks_dir = masks_dir
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        if not self.index_path.exists():
            self.index_path.write_text("[]", encoding="utf-8")
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[Dict[str, Any]]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, data: List[Dict[str, Any]]) -> None:
        tmp = self.index_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.index_path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def list_annotations(
        self,
        region_id: Optional[str] = None,
        patch_id: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        data = self._load()
        if region_id is not None:
            data = [a for a in data if a.get("region_id") == region_id]
        if patch_id is not None:
            data = [a for a in data if a.get("patch_id") == patch_id]
        if task_type is not None:
            data = [a for a in data if a.get("task_type") == task_type]
        return data

    def create_annotation(
        self,
        region_id: str,
        patch_id: str,
        month: str,
        class_id: str,
        geometry: Dict[str, Any],
        task_type: Optional[str] = None,
        score: float = 1.0,
        before_month: Optional[str] = None,
        after_month: Optional[str] = None,
    ) -> Dict[str, Any]:
        ann_id = f"ann_{uuid.uuid4().hex[:8]}"
        mask_path = self.masks_dir / f"{ann_id}.npz"

        geom_type = geometry.get("type", "mask")
        if geom_type == "mask":
            mask = _base64_to_mask(geometry["mask_b64"])
        elif geom_type == "polygon":
            mask = self._rasterize_polygon(geometry["points"])
        elif geom_type == "polyline":
            mask = self._rasterize_polyline(geometry["points"])
        else:
            raise ValueError(f"Unknown geometry type: {geom_type}")

        np.savez_compressed(mask_path, mask=mask)

        ann = {
            "id": ann_id,
            "region_id": region_id,
            "patch_id": patch_id,
            "month": month,
            "class_id": class_id,
            "score": score,
            "geometry": geometry,
            "created_at": datetime.now().isoformat(),
        }
        if task_type is not None:
            ann["task_type"] = task_type
        if before_month is not None:
            ann["before_month"] = before_month
        if after_month is not None:
            ann["after_month"] = after_month

        data = self._load()
        data.append(ann)
        self._save(data)
        return ann

    def get_annotation(self, ann_id: str) -> Optional[Dict[str, Any]]:
        for a in self._load():
            if a.get("id") == ann_id:
                return a
        return None

    def delete_annotation(self, ann_id: str) -> bool:
        data = self._load()
        filtered = [a for a in data if a.get("id") != ann_id]
        if len(filtered) == len(data):
            return False
        self._save(filtered)
        mask_path = (self.masks_dir / f"{ann_id}.npz").resolve()
        masks_dir_resolved = self.masks_dir.resolve()
        try:
            mask_path.relative_to(masks_dir_resolved)
        except ValueError:
            return True
        if mask_path.exists() and mask_path.is_file():
            mask_path.unlink()
        return True

    @staticmethod
    def _rasterize_polygon(points: List[List[float]], size: int = 256) -> np.ndarray:
        """Rasterize normalized polygon points to a binary mask."""
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        coords = [(x * size, y * size) for x, y in points]
        if len(coords) >= 3:
            draw.polygon(coords, fill=255)
        return np.array(img) > 0

    @staticmethod
    def _rasterize_polyline(
        points: List[List[float]], size: int = 256, width: int = 3
    ) -> np.ndarray:
        """Rasterize normalized polyline points to a binary mask."""
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        coords = [(x * size, y * size) for x, y in points]
        if len(coords) >= 2:
            draw.line(coords, fill=255, width=width)
        return np.array(img) > 0


# Per-user singletons
_class_managers: Dict[str, ClassManager] = {}
_annotation_stores: Dict[str, AnnotationStore] = {}


def get_class_manager(user_id: str = "default") -> ClassManager:
    if user_id not in _class_managers:
        user_dir = _get_user_dir(user_id)
        _class_managers[user_id] = ClassManager(user_dir / "classes.json")
    return _class_managers[user_id]


def get_annotation_store(user_id: str = "default") -> AnnotationStore:
    if user_id not in _annotation_stores:
        user_dir = _get_user_dir(user_id)
        _annotation_stores[user_id] = AnnotationStore(
            user_dir / "annotations.json", user_dir / "masks"
        )
    return _annotation_stores[user_id]
