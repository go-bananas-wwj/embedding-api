"""Tile serving utilities."""

from pathlib import Path
from typing import Optional


class TileService:
    """Service for serving map tiles.

    NOTE: XYZ tile serving is not yet implemented.
    Use list_available_tiles() to get patch-based tile files.
    """

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

        tiles_dir = Path(base) / "tiles"
        if not tiles_dir.exists():
            return []

        tiles = sorted(tiles_dir.glob("*.png"))
        result = []
        for t in tiles:
            parts = t.stem.split("_")
            if len(parts) >= 3:
                patch_id = "_".join(parts[:-1])
                tile_period = parts[-1]
                result.append(
                    {
                        "patch_id": patch_id,
                        "period": tile_period,
                        "filename": t.name,
                    }
                )
        return result
