"""Tile serving utilities."""

from pathlib import Path
from typing import Optional


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
        import re
        return bool(re.match(r"^[\w\-]+$", period)) and len(period) <= 64

    @staticmethod
    def _validate_version(version: str) -> bool:
        """Validate version string."""
        return version in ("v1", "v2")

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
    def list_available_tiles(
        region_id: str, task_type: str, version: str = "v1", period: Optional[str] = None
    ) -> list:
        """List all available tile files for a task."""
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
        # v2: {base}/{period}/tiles/   v1: {base}/tiles/
        tiles_dirs = []
        if period:
            tiles_dirs.append(Path(base) / period / "tiles")
        tiles_dirs.append(Path(base) / "tiles")

        result = []
        seen = set()
        for tiles_dir in tiles_dirs:
            if not tiles_dir.exists():
                continue
            tiles = sorted(tiles_dir.glob("*.png"))
            for t in tiles:
                if t.name in seen:
                    continue
                seen.add(t.name)
                parts = t.stem.split("_")
                if len(parts) >= 3:
                    # v1 format: patch_000000_2025-10.png
                    patch_id = "_".join(parts[:-1])
                    tile_period = parts[-1]
                    result.append(
                        {
                            "patch_id": patch_id,
                            "period": tile_period,
                            "filename": t.name,
                        }
                    )
                elif len(parts) == 2:
                    # v2 format: patch_000000.png
                    patch_id = "_".join(parts)
                    tile_period = period
                    result.append(
                        {
                            "patch_id": patch_id,
                            "period": tile_period,
                            "filename": t.name,
                        }
                    )
        return result
