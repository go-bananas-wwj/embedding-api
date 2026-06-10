"""Data file path resolution and metadata loading."""

import json
import logging
import math
import os
import re
import stat
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import get_config

logger = logging.getLogger(__name__)

# Valid patch_id pattern: patch_000000
_PATCH_ID_PATTERN = re.compile(r"^patch_\d{6}$")

# Valid period pattern: alphanumeric, hyphen, underscore only (no dots)
_PERIOD_PATTERN = re.compile(r"^[\w\-]+$")

# Max file size for embeddings (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Callbacks registered to run on config reload
_reload_callbacks: List[Callable[[], None]] = []


class DataServiceError(Exception):
    """Custom exception for data service errors."""
    pass


class DataValidationError(DataServiceError):
    """Raised when input validation fails."""
    pass


class DataNotFoundError(DataServiceError):
    """Raised when requested data is not found."""
    pass


def register_reload_callback(cb: Callable[[], None]) -> None:
    """Register a callback to be invoked when config is reloaded."""
    _reload_callbacks.append(cb)


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

    Uses Path.relative_to() to prevent path traversal via prefix matching,
    and uses os.lstat() (atomic, no-follow) to detect symlinks and
    mitigate TOCTOU race conditions as much as possible.
    """
    try:
        base = Path(base_dir).resolve()
        target = (base / relative).resolve()
        # Use relative_to to ensure target is actually inside base,
        # not just sharing a common prefix (e.g., /foo vs /foobar)
        target.relative_to(base)

        # Atomic symlink check using lstat (doesn't follow symlinks).
        # lstat is a single system call, eliminating the TOCTOU window
        # between exists() and is_symlink().
        try:
            stat_info = os.lstat(str(target))
            if stat.S_ISLNK(stat_info.st_mode):
                logger.warning(f"Symlink target blocked: {relative}")
                return None
        except FileNotFoundError:
            return None

        # Also check parent chain for symlinks using lstat
        for part in target.parents:
            if part == base:
                break
            try:
                part_stat = os.lstat(str(part))
                if stat.S_ISLNK(part_stat.st_mode):
                    logger.warning(f"Symlink in path chain blocked: {relative}")
                    return None
            except (OSError, FileNotFoundError):
                return None

        if target.exists():
            return str(target)
        return None
    except (OSError, ValueError):
        return None


def _check_file_size(path: str) -> None:
    """Check file size is within allowed limit. Raises DataServiceError if too large."""
    try:
        size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            raise DataServiceError(
                f"File too large: {size} bytes (max {MAX_FILE_SIZE})"
            )
    except OSError as e:
        raise DataServiceError(f"Cannot access file: {e}")


class _LRUTTLCache:
    """Thread-safe LRU cache with TTL expiry.

    Prevents unbounded memory growth by enforcing maxsize and
    evicting least-recently-used entries.
    """

    def __init__(self, maxsize: int = 5000, ttl: float = 60.0):
        self._data: OrderedDict[Any, Tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                value, timestamp = self._data[key]
                if time.time() - timestamp < self._ttl:
                    self._data.move_to_end(key)
                    return value
                else:
                    del self._data[key]
            return None

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            now = time.time()
            self._data[key] = (value, now)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class DataService:
    """Service for resolving data file paths."""

    # LRU+TTL cache for available_tasks to avoid N+1 scans and prevent DoS
    _available_tasks_cache = _LRUTTLCache(maxsize=5000, ttl=60.0)

    @staticmethod
    def get_patch(region_id: str, patch_id: str) -> Optional[Dict[str, Any]]:
        """Get patch metadata by ID."""
        if not _validate_patch_id(patch_id):
            raise DataValidationError(f"Invalid patch_id format: '{patch_id}'")
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
                raise DataValidationError(f"Invalid bbox format: {e}")

        total = len(patches)
        start = (page - 1) * page_size
        end = start + page_size
        return patches[start:end], total

    @staticmethod
    def get_embedding_path(
        region_id: str,
        patch_id: str,
        fmt: str = "png",
        version: Optional[str] = None,
        month: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve embedding file path using config-driven templates.

        Args:
            version: Embedding version key from config (e.g., "v1", "v2").
                     If None, searches all configured versions.
            month: Month string for time-series embeddings (e.g., "2025-04").
        """
        if not _validate_patch_id(patch_id):
            raise DataValidationError(f"Invalid patch_id format: '{patch_id}'")
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return None

        embeddings = region.get("embeddings", {})

        # Filter to specific version if requested
        if version and version in embeddings:
            emb_items = [(version, embeddings[version])]
        else:
            emb_items = list(embeddings.items())

        # Config-driven path resolution - no hardcoded region logic
        for emb_name, emb_config in emb_items:
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
                # Build format kwargs - include month if template needs it
                format_kwargs = {"patch_id": patch_id, "fmt": fmt}
                if month:
                    format_kwargs["month"] = month
                path = None
                try:
                    relative = template.format(**format_kwargs)
                    path = _resolve_path(base, relative)
                    if path:
                        return path
                except KeyError:
                    # Template requires month but none provided - try fallback below
                    pass
                # Try alternative templates for backward compat
                alt_templates = emb_config.get("alt_templates", [])
                for alt in alt_templates:
                    try:
                        relative = alt.format(**format_kwargs)
                    except KeyError:
                        continue
                    path = _resolve_path(base, relative)
                    if path:
                        return path
                # Fallback: for patch-subdir structure (haidian), find first file in patch dir
                patch_dir = Path(base) / patch_id
                if patch_dir.is_dir():
                    for ext in (".npz", ".npy", ".png"):
                        first = DataService._find_first_file(str(patch_dir), f"*{ext}")
                        if first:
                            return first
        return None

    @staticmethod
    def _find_first_file(base_dir: str, pattern: str) -> Optional[str]:
        """Find first file matching pattern in directory.

        Validates each candidate via _resolve_path and rejects symlinks
        to prevent directory enumeration info leakage.
        """
        try:
            base_resolved = Path(base_dir).resolve()
            for f in sorted(base_resolved.glob(pattern)):
                if not f.exists():
                    continue
                # Atomic symlink check on the file itself
                try:
                    f_stat = os.lstat(str(f))
                    if stat.S_ISLNK(f_stat.st_mode):
                        continue
                except (OSError, FileNotFoundError):
                    continue
                # Check parent chain for symlinks
                try:
                    valid = True
                    for p in f.parents:
                        if p == base_resolved:
                            break
                        p_stat = os.lstat(str(p))
                        if stat.S_ISLNK(p_stat.st_mode):
                            valid = False
                            break
                    if valid:
                        resolved = _resolve_path(str(f.parent), f.name)
                        if resolved:
                            return resolved
                except (OSError, FileNotFoundError):
                    continue
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
            raise DataValidationError(f"Invalid patch_id format: '{patch_id}'")
        if not _validate_period(period):
            raise DataValidationError(f"Invalid period format: '{period}'")

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
                    # v2 structure: {base}/{period}/{period}.png
                    path = _resolve_path(base, f"{period}/{period}.png")
                    if path:
                        return path
                    # v1 structure: {base}/{period}.png
                    path = _resolve_path(base, f"{period}.png")
                    if path:
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
                    # v2 structure: {base}/{period}/{patch_id}.npy
                    path = _resolve_path(base, f"{period}/{patch_id}.npy")
                    if path:
                        return path
                    # v1 structure: {base}/{patch_id}_{period}.npy
                    path = _resolve_path(base, f"{patch_id}_{period}.npy")
                    if path:
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
                    if path:
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
                if period:
                    # v2 structure: {base}/{period}/tiles/{patch_id}.png
                    path = _resolve_path(base, f"{period}/tiles/{patch_id}.png")
                    if path:
                        return path
                    # v1 structure: {base}/tiles/{patch_id}_{period}.png
                    tiles_dir = Path(base) / "tiles"
                    path = _resolve_path(str(tiles_dir), f"{patch_id}_{period}.png")
                    if path:
                        return path
                # v1 fallback: dynamic discovery in {base}/tiles/
                tiles_dir = Path(base) / "tiles"
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
                # Security: validate parent path is still within allowed data dirs
                summary_path_str = _resolve_path(str(parent), "summary.json")
                if summary_path_str:
                    summary_path = Path(summary_path_str)
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

        cache_key = (region_id, patch_id)
        cached = DataService._available_tasks_cache.get(cache_key)
        if cached is not None:
            return cached

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
        result = list(set(available))

        DataService._available_tasks_cache.set(cache_key, result)
        return result

    @staticmethod
    def has_embedding(region_id: str, patch_id: str) -> bool:
        """Check if embedding exists for this patch."""
        if not _validate_patch_id(patch_id):
            return False
        # Try with explicit month first, then fallback to first available
        for fmt in ("png", "npy", "npz"):
            path = DataService.get_embedding_path(region_id, patch_id, fmt=fmt)
            if path:
                return True
        available_months = DataService.get_available_months(region_id, patch_id)
        if available_months:
            for fmt in ("png", "npy", "npz"):
                path = DataService.get_embedding_path(
                    region_id, patch_id, fmt=fmt, month=available_months[0]
                )
                if path:
                    return True
        return False

    @staticmethod
    def get_available_months(region_id: str, patch_id: str) -> List[str]:
        """Return list of available embedding months/dates for this patch.

        Supports two directory structures:
        - Month subdirs: base/{month}/patch_id.{ext} (harbin)
        - Patch subdirs: base/patch_id/patch_id_{date}.{ext} (haidian)
        """
        if not _validate_patch_id(patch_id):
            return []
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return []

        embeddings = region.get("embeddings", {})
        months = set()
        for emb_name, emb_config in embeddings.items():
            if isinstance(emb_config, dict):
                base = emb_config.get("path")
                if not base:
                    continue
                base_path = Path(base)
                if not base_path.exists():
                    continue
                # Structure 1: month subdirectories (harbin)
                for month_dir in base_path.iterdir():
                    if month_dir.is_dir() and not month_dir.name.startswith("patch_"):
                        for ext in (".npy", ".png", ".npz"):
                            if (month_dir / f"{patch_id}{ext}").exists():
                                months.add(month_dir.name)
                                break
                # Structure 2: patch subdirectory (haidian)
                patch_dir = base_path / patch_id
                if patch_dir.is_dir():
                    for f in patch_dir.iterdir():
                        if f.suffix in (".npy", ".png", ".npz"):
                            # Extract date from filename like patch_000000_20251201.npz
                            import re
                            m = re.search(r'patch_\d+_(\d+)\.\w+$', f.name)
                            if m:
                                months.add(m.group(1))
        return sorted(months)

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


# Register cache-clear callback to break circular import
def _clear_available_tasks_cache() -> None:
    DataService._available_tasks_cache.clear()


register_reload_callback(_clear_available_tasks_cache)
