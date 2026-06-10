"""Tile serving utilities."""

from pathlib import Path
from typing import Optional

from app.services.data_service import DataService


class TileService:
    """Service for serving map tiles."""

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

        For now, tiles are patch-based PNGs stored in results/{task}/tiles/.
        We map tile coordinates to patch IDs based on the patch grid layout.
        """
        # Simple approach: list all tiles and find matching one
        # In production, this should use a proper tile index
        config = DataService._get_config() if hasattr(DataService, '_get_config') else None
        from app.config import get_config

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

        base = ver.get("results")
        if not base:
            return None

        tiles_dir = Path(base) / "tiles"
        if not tiles_dir.exists():
            return None

        # For now, return the first available tile as a placeholder
        # In production, implement proper tile index
        tiles = list(tiles_dir.glob("*.png"))
        if tiles:
            return str(tiles[0])
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
            # Parse filename: patch_XXXXXX_YYYY-MM.png
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
