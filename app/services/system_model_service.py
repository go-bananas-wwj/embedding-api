"""System pre-trained classification-head service.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image

from app.config import get_config
from app.services.data_service import DataService

logger = logging.getLogger(__name__)


def _load_embedding(
    region_id: str, patch_id: str, month: str, version: str = "v2"
) -> Optional[np.ndarray]:
    """Load raw embedding for system model inference."""
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


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def list_system_models(region_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available system pre-trained classification heads.

    If ``region_id`` is provided, only models with a configured checkpoint for
    that region are returned.
    """
    cfg = get_config()
    models_cfg = cfg.get("models", default={})
    region_cfg = models_cfg.get(region_id, {}) if region_id else {}

    result = []
    # Unified thematic task IDs -> checkpoint keys in config
    task_aliases = {
        "land_cover_classification": ["worldcover", "dynamic_world"],
        "land_use_classification": ["dynamic_world"],
        "water_extraction": ["jrc_water"],
        "building_extraction": ["osm_buildings"],
    }

    for task_id, aliases in task_aliases.items():
        available_versions = []
        for version in ("v1", "v2"):
            version_cfg = region_cfg.get(version, {})
            classification = version_cfg.get("classification", {})
            tasks = classification.get("tasks", {})
            convhead = version_cfg.get("classification_convhead", {}).get("tasks", {})
            for alias in aliases:
                if alias in tasks or alias in convhead:
                    available_versions.append(version)
                    break

        result.append(
            {
                "id": task_id,
                "name": _task_display_name(task_id),
                "description": _task_description(task_id),
                "versions": list(set(available_versions)),
            }
        )

    return result


def _task_display_name(task_id: str) -> str:
    return {
        "land_cover_classification": "土地覆盖分类",
        "land_use_classification": "土地利用分类",
        "water_extraction": "水体提取",
        "building_extraction": "建筑物提取",
    }.get(task_id, task_id)


def _task_description(task_id: str) -> str:
    return {
        "land_cover_classification": "WorldCover / Dynamic World 土地覆盖分类",
        "land_use_classification": "Dynamic World 土地利用分类",
        "water_extraction": "JRC Global Surface Water 水体提取",
        "building_extraction": "OSM 建筑物提取",
    }.get(task_id, "")


SYSTEM_TASK_IDS = {"land_cover_classification", "land_use_classification", "water_extraction", "building_extraction"}


def is_system_task(task_id: str) -> bool:
    """Return True if task_id identifies a system pre-trained model."""
    return task_id in SYSTEM_TASK_IDS


def get_system_model_versions(region_id: str, task_id: str) -> List[str]:
    """Return available checkpoint versions for a system model in a region."""
    for m in list_system_models(region_id):
        if m["id"] == task_id:
            return m.get("versions", [])
    return []


def get_system_model_info(region_id: str, task_id: str, version: str = "v2") -> Dict[str, Any]:
    """Return ModelOut-compatible metadata for a system pre-trained model.

    Raises FileNotFoundError if the model is not available for the region/version.
    """
    if not is_system_task(task_id):
        raise ValueError(f"Not a system task: {task_id}")

    versions = get_system_model_versions(region_id, task_id)
    if version not in versions:
        version = versions[0] if versions else "v2"

    # Ensure model file exists (will raise FileNotFoundError if not)
    _resolve_model_path(region_id, task_id, version)

    classes = get_system_model_classes(region_id, task_id, version)
    return {
        "id": task_id,
        "name": _task_display_name(task_id),
        "type": "classification",
        "task_type": task_id,
        "status": "ready",
        "created_at": "1970-01-01T00:00:00",
        "completed_at": "1970-01-01T00:00:00",
        "classes": classes,
        "accuracy": None,
        "n_samples": None,
        "model_path": None,
        "description": _task_description(task_id),
        "message": None,
        "job_id": None,
        "source": "system",
        "versions": versions,
    }


def _resolve_model_path(
    region_id: str, task_id: str, version: str = "v2"
) -> Optional[Path]:
    """Resolve the .pkl checkpoint path for a system model task."""
    cfg = get_config()
    models_cfg = cfg.get("models", default={})
    region_cfg = models_cfg.get(region_id, {})
    version_cfg = region_cfg.get(version, {})

    alias_map = {
        "land_cover_classification": ["worldcover", "dynamic_world"],
        "land_use_classification": ["dynamic_world"],
        "water_extraction": ["jrc_water"],
        "building_extraction": ["osm_buildings"],
    }

    for alias in alias_map.get(task_id, []):
        # Prefer linear_probe
        linear_probe = version_cfg.get("classification", {})
        if linear_probe:
            tasks = linear_probe.get("tasks", {})
            if alias in tasks:
                base = Path(linear_probe["path"])
                return base / tasks[alias]["file"]

        # Fallback to convhead
        convhead = version_cfg.get("classification_convhead", {})
        if convhead:
            tasks = convhead.get("tasks", {})
            if alias in tasks:
                base = Path(convhead["path"])
                return base / tasks[alias]["file"]

    return None


def get_system_model_classes(
    region_id: str, task_id: str, version: str = "v2"
) -> List[Dict[str, Any]]:
    """Return class definitions for a system model."""
    model_path = _resolve_model_path(region_id, task_id, version)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"System model not found: {task_id} ({version})")

    model_data = joblib.load(model_path)
    class_names = model_data.get("class_names", [])
    colors = model_data.get("colors", [])

    classes = []
    for idx, name in enumerate(class_names):
        color = colors[idx] if idx < len(colors) else (200, 200, 200)
        if isinstance(color, tuple):
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        else:
            hex_color = str(color)
        classes.append({"id": f"sys_{task_id}_{idx}", "name": name, "color": hex_color})
    return classes


def infer_system_model(
    region_id: str,
    task_id: str,
    patch_id: str,
    month: str,
    version: str = "v2",
    results_dir: Optional[Path] = None,
) -> Path:
    """Run a system pre-trained model on a single patch.

    Returns the path to the generated result PNG.
    """
    model_path = _resolve_model_path(region_id, task_id, version)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"System model not found: {task_id} ({version})")

    emb = _load_embedding(region_id, patch_id, month, version=version)
    if emb is None:
        raise FileNotFoundError(
            f"Embedding not found for {patch_id} {month}"
        )

    model_data = joblib.load(model_path)
    scaler = model_data["scaler"]
    clf = model_data["model"]
    class_names = model_data.get("class_names", [])
    colors = model_data.get("colors", [])

    D, H, W = emb.shape
    flat = emb.reshape(D, -1).T
    flat_s = scaler.transform(flat)
    pred = clf.predict(flat_s).reshape(H, W)

    rgb = np.full((H, W, 3), 200, dtype=np.uint8)
    for idx, color in enumerate(colors):
        if isinstance(color, tuple):
            rgb[pred == idx] = color
        else:
            rgb[pred == idx] = _hex_to_rgb(str(color))
    rgb[pred == -1] = (200, 200, 200)

    img = Image.fromarray(rgb).resize((128, 128), Image.Resampling.NEAREST)

    if results_dir is None:
        results_dir = Path("users/default/system_model_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{task_id}_{region_id}_{patch_id}_{month}.png"
    img.save(result_path)
    return result_path
