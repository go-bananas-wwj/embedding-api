"""Data file path resolution and metadata loading."""

import asyncio
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_config

logger = logging.getLogger(__name__)

# Valid patch_id pattern: patch_000000
_PATCH_ID_PATTERN = re.compile(r"^patch_\d{6}$")

# Valid period pattern: alphanumeric, hyphen, underscore only (no dots)
_PERIOD_PATTERN = re.compile(r"^[\w\-]+$")


class DataServiceError(Exception):
    """Custom exception for data service errors."""
    pass


def _validate_patch_id(patch_id: str) -> bool:
    """Validate patch_id format to prevent path traversal."""
    return bool(_PATCH_ID_PATTERN.match(patch_id))


def _validate_period(period: Optional[str]) -> bool:
    """Validate period format to prevent path traversal."""
    if period is None:
        return True
    return bool(_PERIOD_PATTERN.match(period))


def _resolve_path(base_dir: str, relative: str) -> Optional[str]:
    """Resolve and validate a path is within the base directory.

    Uses Path.relative_to() to prevent path traversal via prefix matching.
    """
    try:
        base = Path(base_dir).resolve()
        target = (base / relative).resolve()
        # Use relative_to to ensure target is actually inside base,
        # not just sharing a common prefix (e.g., /foo vs /foobar)
        target.relative_to(base)
        return str(target) if target.exists() else None
    except (OSError, ValueError):
        return None


class DataService:
    """Service for resolving data file paths."""

    @staticmethod
    def get_patch(region_id: str, patch_id: str) -> Optional[Dict[str, Any]]:
        """Get patch metadata by ID."""
        if not _validate_patch_id(patch_id):
            raise DataServiceError(f"Invalid patch_id format: '{patch_id}'")
        config = get_config()
        patches = config.get_patches(region_id)
        for p in patches:
            if p.get("patch_id") == patch_id:
                return p
        return None

    @staticmethod
    def list_patches(
        region_id: str,
        page: int = 1,
        page_size: int = 20,
        bbox: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List patches with pagination and bbox filtering."""
        config = get_config()
        patches = config.get_patches(region_id)

        # BBox filtering: bbox=minx,miny,maxx,maxy
        if bbox:
            try:
                parts = bbox.split(",")
                if len(parts) != 4:
                    raise ValueError("bbox must have 4 comma-separated values")
                minx, miny, maxx, maxy = map(float, parts)
                # Validate bbox values are finite
                for v, name in [(minx, "minx"), (miny, "miny"), (maxx, "maxx"), (maxy, "maxy")]:
                    if math.isnan(v) or math.isinf(v):
                        raise ValueError(f"bbox {name} must be a finite number")
                # Validate bbox ordering
                if minx >= maxx or miny >= maxy:
                    raise ValueError("bbox must satisfy minx < maxx and miny < maxy")
                filtered = []
                for p in patches:
                    bounds = p.get("bounds_wgs84", [])
                    if len(bounds) == 4:
                        p_minx, p_miny, p_maxx, p_maxy = bounds
                        if not (p_maxx < minx or p_minx > maxx or p_maxy < miny or p_miny > maxy):
                            filtered.append(p)
                patches = filtered
            except ValueError as e:
                raise DataServiceError(f"Invalid bbox format: {e}")

        total = len(patches)
        start = (page - 1) * page_size
        end = start + page_size
        return patches[start:end], total

    @staticmethod
    def get_embedding_path(region_id: str, patch_id: str, fmt: str = "png") -> Optional[str]:
        """Resolve embedding file path using config-driven templates."""
        if not _validate_patch_id(patch_id):
            raise DataServiceError(f"Invalid patch_id format: '{patch_id}'")
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return None

        embeddings = region.get("embeddings", {})

        # Config-driven path resolution - no hardcoded region logic
        for emb_name, emb_config in embeddings.items():
            if isinstance(emb_config, str):
                # Legacy: direct path string
                base = emb_config
                if fmt == "png":
                    path = _resolve_path(base, f"{patch_id}.png")
                    if path:
                        return path
                elif fmt == "npy":
                    path = _resolve_path(base, f"{patch_id}.npy")
                    if path:
                        return path
            elif isinstance(emb_config, dict):
                # New: config with template support
                base = emb_config.get("path")
                if not base:
                    continue
                supported_formats = emb_config.get("formats", ["png"])
                if fmt not in supported_formats and fmt != "cache":
                    continue
                template = emb_config.get("template", "{patch_id}.{fmt}")
                relative = template.format(patch_id=patch_id, fmt=fmt)
                path = _resolve_path(base, relative)
                if path:
                    return path
                # Try alternative templates for backward compat
                alt_templates = emb_config.get("alt_templates", [])
                for alt in alt_templates:
                    relative = alt.format(patch_id=patch_id)
                    path = _resolve_path(base, relative)
                    if path:
                        return path
        return None

    @staticmethod
    def _find_first_file(base_dir: str, pattern: str) -> Optional[str]:
        """Find first file matching pattern in directory."""
        try:
            files = sorted(Path(base_dir).glob(pattern))
            if files:
                return str(files[0])
        except OSError:
            pass
        return None

    @staticmethod
    def get_task_result_path(
        region_id: str,
        patch_id: str,
        task_type: str,
        format_type: str = "png",
        version: str = "v1",
        period: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve task result file path."""
        if not _validate_patch_id(patch_id):
            raise DataServiceError(f"Invalid patch_id format: '{patch_id}'")
        if not _validate_period(period):
            raise DataServiceError(f"Invalid period format: '{period}'")

        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return None

        tasks = region.get("tasks", {})
        task = tasks.get(task_type)
        if not task:
            return None

        versions = task.get("versions", {})
        ver = versions.get(version)
        if not ver:
            return None

        if format_type == "png":
            base = ver.get("results")
            if base:
                if period:
                    path = _resolve_path(base, f"{period}.png")
                    return path
                # Dynamic discovery: find first PNG instead of hardcoded names
                path = DataService._find_first_file(base, "*.png")
                if path:
                    return path
                # Fallback to common names
                for fname in ["result.png"]:
                    path = _resolve_path(base, fname)
                    if path:
                        return path
        elif format_type == "npy":
            base = ver.get("predictions") or ver.get("results")
            if base:
                if period:
                    path = _resolve_path(base, f"{patch_id}_{period}.npy")
                    return path
                # Dynamic discovery
                path = DataService._find_first_file(base, f"{patch_id}_*.npy")
                if path:
                    return path
                # Fallback
                for fname in [f"{patch_id}.npy", "result.npy"]:
                    path = _resolve_path(base, fname)
                    if path:
                        return path
        elif format_type == "label":
            base = ver.get("labels")
            if base:
                if period:
                    period_dir = Path(base) / period
                    path = _resolve_path(str(period_dir), f"{patch_id}.npy")
                    return path
                path = _resolve_path(base, f"{patch_id}.npy")
                if path:
                    return path
                # Try meta.json for summary
                meta_path = _resolve_path(base, "meta.json")
                return meta_path
        elif format_type == "tile":
            base = ver.get("results")
            if base:
                tiles_dir = Path(base) / "tiles"
                if period:
                    path = _resolve_path(str(tiles_dir), f"{patch_id}_{period}.png")
                    return path
                # Dynamic discovery
                path = DataService._find_first_file(str(tiles_dir), f"{patch_id}_*.png")
                if path:
                    return path
        return None

    @staticmethod
    def load_task_summary(
        region_id: str, task_type: str, version: str = "v1", period: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load task summary from meta.json or summary.json."""
        if not _validate_period(period):
            return None

        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return None

        tasks = region.get("tasks", {})
        task = tasks.get(task_type)
        if not task:
            return None

        versions = task.get("versions", {})
        ver = versions.get(version)
        if not ver:
            return None

        # Try labels directory first
        labels_base = ver.get("labels")
        if labels_base:
            if period:
                meta_path = _resolve_path(labels_base, f"{period}/meta.json")
            else:
                meta_path = _resolve_path(labels_base, "meta.json")
            if meta_path:
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load meta.json: {e}")

        # Try summary.json in parent directory
        try:
            parent = Path(labels_base).parent if labels_base else None
            if parent:
                summary_path = parent / "summary.json"
                if summary_path.exists():
                    with open(summary_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if period and period in data:
                            return data[period].get(task_type)
                        if task_type in data:
                            return data[task_type]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load summary.json: {e}")

        return None

    @staticmethod
    def get_available_tasks(region_id: str, patch_id: str) -> List[str]:
        """Get list of tasks that have data for this patch."""
        if not _validate_patch_id(patch_id):
            return []
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return []

        tasks = region.get("tasks", {})
        available = []
        for task_name, task_info in tasks.items():
            versions = task_info.get("versions", {})
            for ver_name, ver_info in versions.items():
                predictions = ver_info.get("predictions")
                if predictions:
                    # Dynamic discovery instead of hardcoded period
                    path = DataService._find_first_file(predictions, f"{patch_id}_*.npy")
                    if path:
                        available.append(task_name)
                        break
                labels = ver_info.get("labels")
                if labels:
                    path = _resolve_path(labels, f"{patch_id}.npy")
                    if path:
                        available.append(task_name)
                        break
        return list(set(available))

    @staticmethod
    def has_embedding(region_id: str, patch_id: str) -> bool:
        """Check if embedding exists for this patch."""
        if not _validate_patch_id(patch_id):
            return False
        return DataService.get_embedding_path(region_id, patch_id) is not None

    @staticmethod
    def list_task_versions(region_id: str, task_type: str) -> List[str]:
        """List available versions for a task."""
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return []

        tasks = region.get("tasks", {})
        task = tasks.get(task_type)
        if not task:
            return []

        return list(task.get("versions", {}).keys())
