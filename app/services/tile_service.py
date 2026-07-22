"""Tile serving utilities."""

import asyncio
import os
import re
import stat
from pathlib import Path
from typing import Optional

from app.services.time_utils import normalize_period


class TileService:
    """Service for serving map tiles.

    NOTE: XYZ tile serving is not yet implemented.
    Use list_available_tiles() to get patch-based tile files.
    """

    @staticmethod
    def _validate_period(period: Optional[str]) -> bool:
        """Validate period string to prevent path traversal."""
        if period is None:
            return True
        return bool(re.match(r"^[\w\-]+$", period)) and len(period) <= 64

    @staticmethod
    def _validate_version(version: str) -> bool:
        """Validate version string."""
        return version in ("v1", "v2")

    @staticmethod
    def _parse_patch_tile(filename: str) -> Optional[tuple[str, Optional[str]]]:
        """Parse patch ID and optional period without splitting `_vs_`."""
        match = re.match(r"^(patch_\d+)(?:_(.+))?\.png$", filename)
        if not match:
            return None
        return match.group(1), match.group(2)

    @staticmethod
    def _is_symlink(path: Path) -> bool:
        """Check if path or any parent is a symlink (no-follow)."""
        try:
            st = os.lstat(str(path))
            if stat.S_ISLNK(st.st_mode):
                return True
        except (OSError, FileNotFoundError):
            return True  # Treat inaccessible as unsafe
        for parent in path.parents:
            try:
                st = os.lstat(str(parent))
                if stat.S_ISLNK(st.st_mode):
                    return True
            except (OSError, FileNotFoundError):
                return True
        return False

    @staticmethod
    def get_tile_path(
        region_id: str,
        task_type: str,
        z: int,
        x: int,
        y: int,
        version: str = "v1",
        period: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve tile image path.

        Currently not implemented - always returns None.
        Proper XYZ tiling requires spatial indexing.
        """
        return None

    @staticmethod
    def _list_tiles_sync(
        region_id: str, task_type: str, version: str = "v1", period: Optional[str] = None
    ) -> list:
        """Synchronous implementation of tile listing."""
        from app.config import get_config

        if not TileService._validate_version(version):
            return []
        if not TileService._validate_period(period):
            return []

        config = get_config()
        region = config.get_region(region_id)
        if not region:
            return []

        tasks = region.get("tasks", {})
        task = tasks.get(task_type)
        if not task:
            return []

        versions = task.get("versions", {})
        ver = versions.get(version)
        if not ver:
            return []

        base = ver.get("results")
        if not base:
            return []

        # Build list of tiles directories to scan
        tiles_dirs = []
        period_variants = set(normalize_period(period)) if period else set()
        if period:
            for p in normalize_period(period):
                tiles_dirs.append(Path(base) / p / "tiles")
        tiles_dirs.append(Path(base) / "tiles")

        result = []
        seen = set()
        for tiles_dir in tiles_dirs:
            if TileService._is_symlink(tiles_dir):
                continue
            try:
                tiles = sorted(tiles_dir.glob("*.png"))
            except OSError:
                continue
            for t in tiles:
                if TileService._is_symlink(t):
                    continue
                if t.name in seen:
                    continue
                seen.add(t.name)
                parsed = TileService._parse_patch_tile(t.name)
                if not parsed:
                    continue
                patch_id, filename_period = parsed
                if period_variants and filename_period and filename_period not in period_variants:
                    continue
                result.append(
                    {
                        "patch_id": patch_id,
                        "period": filename_period or period,
                        "filename": t.name,
                    }
                )
        return result

    @staticmethod
    async def list_available_tiles(
        region_id: str, task_type: str, version: str = "v1", period: Optional[str] = None
    ) -> list:
        """List all available tile files for a task (async, non-blocking)."""
        return await asyncio.to_thread(
            TileService._list_tiles_sync, region_id, task_type, version, period
        )
