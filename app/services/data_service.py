"""Data file path resolution and metadata loading."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from app.config import get_config, load_patches_meta


class DataService:
    """Service for resolving data file paths."""

    @staticmethod
    def get_patch(region_id: str, patch_id: str) -> Optional[Dict[str, Any]]:
        """Get patch metadata by ID."""
        patches = load_patches_meta(region_id)
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
        patches = load_patches_meta(region_id)

        # BBox filtering: bbox=minx,miny,maxx,maxy
        if bbox:
            try:
                minx, miny, maxx, maxy = map(float, bbox.split(","))
                filtered = []
                for p in patches:
                    bounds = p.get("bounds_wgs84", [])
                    if len(bounds) == 4:
                        p_minx, p_miny, p_maxx, p_maxy = bounds
                        # Check overlap
                        if not (p_maxx < minx or p_minx > maxx or p_maxy < miny or p_miny > maxy):
                            filtered.append(p)
                patches = filtered
            except (ValueError, IndexError):
                pass

        total = len(patches)
        start = (page - 1) * page_size
        end = start + page_size
        return patches[start:end], total

    @staticmethod
    def get_embedding_path(region_id: str, patch_id: str, format_type: str = "png") -> Optional[str]:
        """Resolve embedding file path."""
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return None

        embeddings = region.get("embeddings", {})

        if region_id == "harbin":
            # Harbin: PNG visualization
            base = embeddings.get("v2")
            if base:
                path = Path(base) / f"{patch_id}.png"
                if path.exists():
                    return str(path)
        elif region_id == "haidian":
            if format_type == "npy":
                base = embeddings.get("aef")
                if base:
                    path = Path(base) / f"{patch_id}.npy"
                    if path.exists():
                        return str(path)
            elif format_type == "png":
                # Try viz directory for multi-source visualizations
                base = embeddings.get("viz")
                if base:
                    # Look for multisource or timeline images
                    for suffix in ["_multisource_v2.png", "_timeline.png"]:
                        path = Path(base) / f"{patch_id}{suffix}"
                        if path.exists():
                            return str(path)
            elif format_type == "cache":
                base = embeddings.get("cache")
                if base:
                    path = Path(base) / patch_id / "planet_img.pt"
                    if path.exists():
                        return str(path)
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
                    path = Path(base) / f"{period}.png"
                else:
                    # Try common patterns
                    for fname in ["2025-10.png", "result.png"]:
                        path = Path(base) / fname
                        if path.exists():
                            return str(path)
                    # Check if there's a single PNG
                    pngs = list(Path(base).glob("*.png"))
                    if pngs:
                        return str(pngs[0])
        elif format_type == "npy":
            base = ver.get("predictions") or ver.get("results")
            if base:
                if period:
                    path = Path(base) / f"{patch_id}_{period}.npy"
                else:
                    path = Path(base) / f"{patch_id}_2025-10.npy"
                if path.exists():
                    return str(path)
        elif format_type == "label":
            base = ver.get("labels")
            if base:
                if period:
                    # V2 format: labels are organized by period
                    period_dir = Path(base) / period
                    path = period_dir / f"{patch_id}.npy"
                    if path.exists():
                        return str(path)
                else:
                    path = Path(base) / f"{patch_id}.npy"
                    if path.exists():
                        return str(path)
                    # Try meta.json for summary
                    meta_path = Path(base) / "meta.json"
                    if meta_path.exists():
                        return str(meta_path)
        elif format_type == "tile":
            base = ver.get("results")
            if base:
                tiles_dir = Path(base) / "tiles"
                if tiles_dir.exists():
                    if period:
                        path = tiles_dir / f"{patch_id}_{period}.png"
                    else:
                        path = tiles_dir / f"{patch_id}_2025-10.png"
                    if path.exists():
                        return str(path)
        return None

    @staticmethod
    def load_task_summary(
        region_id: str, task_type: str, version: str = "v1", period: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load task summary from meta.json or summary.json."""
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
                meta_path = Path(labels_base) / period / "meta.json"
            else:
                meta_path = Path(labels_base) / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass

        # Try summary.json in parent directory
        parent = Path(labels_base).parent if labels_base else None
        if parent:
            summary_path = parent / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # V2 format: nested by period
                        if period and period in data:
                            return data[period].get(task_type)
                        # V1 format: direct task entry
                        if task_type in data:
                            return data[task_type]
                except Exception:
                    pass

        return None

    @staticmethod
    def get_available_tasks(region_id: str, patch_id: str) -> List[str]:
        """Get list of tasks that have data for this patch."""
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return []

        tasks = region.get("tasks", {})
        available = []
        for task_name, task_info in tasks.items():
            versions = task_info.get("versions", {})
            for ver_name, ver_info in versions.items():
                # Check if any data exists for this patch
                predictions = ver_info.get("predictions")
                if predictions:
                    path = Path(predictions) / f"{patch_id}_2025-10.npy"
                    if path.exists():
                        available.append(task_name)
                        break
                labels = ver_info.get("labels")
                if labels:
                    path = Path(labels) / f"{patch_id}.npy"
                    if path.exists():
                        available.append(task_name)
                        break
        return list(set(available))

    @staticmethod
    def has_embedding(region_id: str, patch_id: str) -> bool:
        """Check if embedding exists for this patch."""
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
