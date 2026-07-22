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

SYSTEM_HEAD_CHECKPOINT_FORMAT = "embedding-api.system-head.v1"


HAIDIAN_LAND_COVER_CLASSES = [
    {"id": "sys_land_cover_classification_1", "name": "树木覆盖", "color": "#006400"},
    {"id": "sys_land_cover_classification_2", "name": "灌木地", "color": "#B4D250"},
    {"id": "sys_land_cover_classification_3", "name": "草地", "color": "#F5DC5A"},
    {"id": "sys_land_cover_classification_4", "name": "耕地", "color": "#D23C3C"},
    {"id": "sys_land_cover_classification_5", "name": "建成区", "color": "#BEAA82"},
    {"id": "sys_land_cover_classification_6", "name": "裸地/稀疏植被", "color": "#A0DCDC"},
    {"id": "sys_land_cover_classification_8", "name": "永久性水体", "color": "#1E64DC"},
]

HAIDIAN_LAND_USE_CLASSES = [
    {"id": "sys_land_use_classification_0", "name": "水体", "color": "#286EE6"},
    {"id": "sys_land_use_classification_1", "name": "树木", "color": "#46B450"},
    {"id": "sys_land_use_classification_2", "name": "草地", "color": "#F5DC5A"},
    {"id": "sys_land_use_classification_3", "name": "淹水植被", "color": "#DC50B4"},
    {"id": "sys_land_use_classification_4", "name": "农作物", "color": "#FFB496"},
    {"id": "sys_land_use_classification_5", "name": "灌木与矮林", "color": "#E63C28"},
    {"id": "sys_land_use_classification_6", "name": "建成区", "color": "#6E6E6E"},
    {"id": "sys_land_use_classification_7", "name": "裸地", "color": "#965A46"},
    {"id": "sys_land_use_classification_8", "name": "冰雪", "color": "#EBEBEB"},
]


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
        "road_extraction": ["road_extraction"],
    }

    for task_id, aliases in task_aliases.items():
        available_versions = []
        for version in ("v1", "v2"):
            version_cfg = region_cfg.get(version, {})
            classification = version_cfg.get("classification", {})
            tasks = classification.get("tasks", {})
            convhead = version_cfg.get("classification_convhead", {}).get("tasks", {})
            task_heads = version_cfg.get("task_heads", {}).get("tasks", {})
            if task_id in task_heads:
                available_versions.append(version)
                continue
            for alias in aliases:
                if alias in tasks or alias in convhead:
                    available_versions.append(version)
                    break

        versions = sorted(set(available_versions))
        if region_id and not versions:
            continue

        result.append(
            {
                "id": task_id,
                "name": _task_display_name(task_id),
                "description": _task_description(task_id),
                "versions": versions,
                **_system_model_runtime_metadata(region_id, task_id, versions),
            }
        )

    return result


def _system_model_runtime_metadata(
    region_id: Optional[str],
    task_id: str,
    versions: List[str],
    selected_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Return additive frontend metadata for configured task heads."""
    if not region_id or not versions:
        return {}
    cfg = get_config().get("models", default={})
    ordered_versions = [selected_version] if selected_version else list(reversed(versions))
    for version in ordered_versions:
        if not version:
            continue
        task_cfg = (
            cfg.get(region_id, {})
            .get(version, {})
            .get("task_heads", {})
            .get("tasks", {})
            .get(task_id)
        )
        if task_cfg:
            head_type = task_cfg.get("head", "pytorch")
            return {
                "head_type": head_type,
                "feature_source": task_cfg.get("feature_source", "embedding"),
                "foundation_model_id": "p10c" if region_id == "haidian" else "xuannv_earth",
                "foundation_model_version": version,
                "feature_dimension": 64 if region_id == "haidian" else 128,
                "preprocessing_version": f"{region_id}_embedding_{version}",
                "checkpoint_format": SYSTEM_HEAD_CHECKPOINT_FORMAT,
                "compatible_regions": [region_id],
            }
        version_cfg = cfg.get(region_id, {}).get(version, {})
        classification = version_cfg.get("classification", {}).get("tasks", {})
        convhead = version_cfg.get("classification_convhead", {}).get("tasks", {})
        aliases = {
            "land_cover_classification": ["worldcover", "dynamic_world"],
            "land_use_classification": ["dynamic_world"],
            "water_extraction": ["jrc_water"],
            "building_extraction": ["osm_buildings"],
            "road_extraction": ["road_extraction"],
        }.get(task_id, [])
        if any(alias in classification or alias in convhead for alias in aliases):
            uses_convhead = any(alias in convhead for alias in aliases)
            return {
                "head_type": "convhead" if uses_convhead else "linear_probe",
                "feature_source": "xuannv_embedding",
                "foundation_model_id": "xuannv_earth",
                "foundation_model_version": version,
                "feature_dimension": 128 if region_id == "harbin" and version == "v2" else 64,
                "preprocessing_version": f"{region_id}_embedding_{version}",
                "checkpoint_format": SYSTEM_HEAD_CHECKPOINT_FORMAT,
                "compatible_regions": [region_id],
            }
    return {}


def _task_display_name(task_id: str) -> str:
    return {
        "land_cover_classification": "土地覆盖分类",
        "land_use_classification": "土地利用分类",
        "water_extraction": "水体提取",
        "building_extraction": "建筑物提取",
        "road_extraction": "道路提取",
    }.get(task_id, task_id)


def _task_description(task_id: str) -> str:
    return {
        "land_cover_classification": "WorldCover / Dynamic World 土地覆盖分类",
        "land_use_classification": "Dynamic World 土地利用分类",
        "water_extraction": "JRC Global Surface Water 水体提取",
        "building_extraction": "OSM 建筑物提取",
        "road_extraction": "OSM 路网提取",
    }.get(task_id, "")


SYSTEM_TASK_IDS = {
    "land_cover_classification",
    "land_use_classification",
    "water_extraction",
    "building_extraction",
    "road_extraction",
}


def is_system_task(task_id: str) -> bool:
    """Return True if task_id identifies a system pre-trained model."""
    return task_id in SYSTEM_TASK_IDS


def get_system_model_versions(region_id: str, task_id: str) -> List[str]:
    """Return available checkpoint versions for a system model in a region."""
    for m in list_system_models(region_id):
        if m["id"] == task_id:
            return m.get("versions", [])
    return []


def resolve_system_model_version(
    region_id: str, task_id: str, requested: Optional[str] = None
) -> str:
    """Resolve a requested system-model version to one available in the region."""
    versions = get_system_model_versions(region_id, task_id)
    if requested is not None:
        if requested in versions:
            return requested
        available = ", ".join(sorted(versions)) or "none"
        raise FileNotFoundError(
            f"System model '{task_id}' version '{requested}' is not available "
            f"for region '{region_id}'. Available versions: {available}"
        )
    if "v2" in versions:
        return "v2"
    if "v1" in versions:
        return "v1"
    if versions:
        return sorted(versions)[0]
    raise FileNotFoundError(f"System model not found: {task_id}")


def get_system_model_info(
    region_id: str, task_id: str, version: Optional[str] = None
) -> Dict[str, Any]:
    """Return ModelOut-compatible metadata for a system pre-trained model.

    Raises FileNotFoundError if the model is not available for the region/version.
    """
    if not is_system_task(task_id):
        raise ValueError(f"Not a system task: {task_id}")

    versions = get_system_model_versions(region_id, task_id)
    version = resolve_system_model_version(region_id, task_id, version)

    # Ensure model file exists (will raise FileNotFoundError if not)
    _resolve_model_path(region_id, task_id, version)

    classes = get_system_model_classes(region_id, task_id, version)
    return {
        "id": task_id,
        "name": _task_display_name(task_id),
        "type": "single_time_detection",
        "task_type": task_id,
        "status": "ready",
        "created_at": "1970-01-01T00:00:00",
        "completed_at": "1970-01-01T00:00:00",
        "classes": classes,
        "accuracy": None,
        "metric_name": None,
        "n_samples": None,
        "model_path": None,
        "description": _task_description(task_id),
        "message": None,
        "job_id": None,
        "source": "system",
        "versions": versions,
        **_system_model_runtime_metadata(region_id, task_id, versions, version),
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
        "road_extraction": ["road_extraction"],
    }

    task_heads = version_cfg.get("task_heads", {})
    if task_heads:
        tasks = task_heads.get("tasks", {})
        if task_id in tasks:
            base = Path(task_heads["path"])
            return base / tasks[task_id]["file"]

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
    # Haidian land-cover output is a pre-generated monthly product rather than
    # an online checkpoint. Its legend must remain queryable independently of
    # whether real-time inference is available.
    if region_id == "haidian" and task_id in {
        "land_cover_classification",
        "land_use_classification",
    }:
        if version != "v1":
            raise FileNotFoundError(
                f"System model classes not found: {task_id} ({version})"
            )
        classes = (
            HAIDIAN_LAND_COVER_CLASSES
            if task_id == "land_cover_classification"
            else HAIDIAN_LAND_USE_CLASSES
        )
        return [dict(item) for item in classes]

    model_path = _resolve_model_path(region_id, task_id, version)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"System model not found: {task_id} ({version})")

    if model_path.suffix == ".pt":
        return _binary_task_classes(task_id)

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


def _binary_task_classes(task_id: str) -> List[Dict[str, Any]]:
    """Return display classes for binary PyTorch segmentation heads."""
    names = {
        "building_extraction": "建筑物",
        "road_extraction": "道路",
        "water_extraction": "水体",
    }
    colors = {
        "building_extraction": "#ef4444",
        "road_extraction": "#f59e0b",
        "water_extraction": "#2563eb",
    }
    return [
        {"id": f"sys_{task_id}_0", "name": "背景", "color": "#000000"},
        {
            "id": f"sys_{task_id}_1",
            "name": names.get(task_id, _task_display_name(task_id)),
            "color": colors.get(task_id, "#22c55e"),
        },
    ]


def _infer_legacy_torch_mlp(state: Dict[str, Any], emb: np.ndarray) -> np.ndarray:
    """Run a small binary MLP state_dict over a [D,H,W] embedding map."""
    import torch

    in_dim = int(state["net.0.weight"].shape[1])
    hidden_dim = int(state["net.0.weight"].shape[0])
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )
    normalized_state = {
        key.replace("net.", "", 1): value
        for key, value in state.items()
    }
    model.load_state_dict(normalized_state)
    model.eval()

    D, H, W = emb.shape
    if D != in_dim:
        raise ValueError(
            f"Embedding channel mismatch: model expects {in_dim}, got {D}"
        )
    flat = torch.from_numpy(emb.reshape(D, -1).T.astype(np.float32, copy=False))
    with torch.no_grad():
        logits = model(flat).squeeze(1)
        pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8)
    return pred.reshape(H, W)


def _infer_torch_head(model_path: Path, emb: np.ndarray) -> np.ndarray:
    """Run a self-describing Conv3x3 head or a legacy MLP checkpoint."""
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if checkpoint.get("__format__") != SYSTEM_HEAD_CHECKPOINT_FORMAT:
        return _infer_legacy_torch_mlp(checkpoint, emb)
    if checkpoint.get("head_type") != "binary_conv3x3":
        raise ValueError(
            f"Unsupported system head type: {checkpoint.get('head_type')!r}"
        )

    from app.services.fewshot_heads import BinaryConv3x3ProbeHead

    embed_dim = int(checkpoint["embed_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    dropout = float(checkpoint.get("dropout", 0.1))
    threshold = float(checkpoint.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid system head threshold: {threshold}")

    channels, height, width = emb.shape
    if channels != embed_dim:
        raise ValueError(
            f"Embedding channel mismatch: model expects {embed_dim}, got {channels}"
        )

    model = BinaryConv3x3ProbeHead(embed_dim, hidden_dim, dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tensor = torch.from_numpy(emb.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor).squeeze(0).squeeze(0)
        prediction = (torch.sigmoid(logits) >= threshold).cpu().numpy()
    return prediction.astype(np.uint8).reshape(height, width)


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

    pred = infer_system_model_array(region_id, task_id, patch_id, month, version)

    if model_path.suffix == ".pt":
        color = _hex_to_rgb(_binary_task_classes(task_id)[1]["color"])
        rgb = np.full((*pred.shape, 3), 0, dtype=np.uint8)
        rgb[pred == 1] = color
    else:
        model_data = joblib.load(model_path)
        colors = model_data.get("colors", [])
        rgb = np.full((*pred.shape, 3), 200, dtype=np.uint8)
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


def infer_system_model_array(
    region_id: str,
    task_id: str,
    patch_id: str,
    month: str,
    version: str = "v2",
) -> np.ndarray:
    """Run a system model and return its raw two-dimensional class array."""
    model_path = _resolve_model_path(region_id, task_id, version)
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"System model not found: {task_id} ({version})")

    emb = _load_embedding(region_id, patch_id, month, version=version)
    if emb is None:
        raise FileNotFoundError(
            f"Embedding not found for {patch_id} {month}"
        )

    if model_path.suffix == ".pt":
        pred = _infer_torch_head(model_path, emb)
    else:
        model_data = joblib.load(model_path)
        scaler = model_data["scaler"]
        clf = model_data["model"]
        D, H, W = emb.shape
        flat = emb.reshape(D, -1).T
        flat_s = scaler.transform(flat)
        pred = clf.predict(flat_s).reshape(H, W)

    return np.asarray(pred)
