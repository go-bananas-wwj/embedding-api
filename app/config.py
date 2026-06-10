"""Configuration loading with hot-reload support."""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class ConfigReloadHandler(FileSystemEventHandler):
    """Watchdog handler for config file changes."""

    def __init__(self, config_manager):
        self.config_manager = config_manager

    def on_modified(self, event):
        if event.src_path == str(CONFIG_PATH):
            self.config_manager.reload()


class ConfigManager:
    """Thread-safe configuration manager with hot-reload."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._observer: Optional[Observer] = None
        self.reload()
        self._start_watching()

    def reload(self):
        """Reload configuration from file."""
        with self._lock:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f)
                print(f"[Config] Reloaded from {self.config_path}")
            except Exception as e:
                print(f"[Config] Failed to reload: {e}")

    def _start_watching(self):
        """Start file watcher for hot-reload."""
        handler = ConfigReloadHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.config_path.parent), recursive=False)
        self._observer.start()

    def get(self, *keys, default=None):
        """Get nested config value."""
        with self._lock:
            value = self._config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value

    @property
    def regions(self) -> Dict[str, Any]:
        return self.get("regions", default={})

    def get_region(self, region_id: str) -> Optional[Dict[str, Any]]:
        return self.get("regions", region_id, default=None)

    def region_exists(self, region_id: str) -> bool:
        return region_id in self.regions

    def list_regions(self) -> List[str]:
        return list(self.regions.keys())

    def stop_watching(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_patches_meta(region_id: str) -> List[Dict[str, Any]]:
    """Load patches metadata for a region."""
    config = get_config()
    region = config.get_region(region_id)
    if not region:
        return []

    meta_path = region.get("patches_meta")
    if not meta_path or not os.path.exists(meta_path):
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle both formats: harbin (list) and haidian (dict with "patches" key)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "patches" in data:
            patches = data["patches"]
            # Add city info if present
            if "city" in data:
                for p in patches:
                    p["city"] = data["city"]
            return patches
        return []
    except Exception as e:
        print(f"[Config] Failed to load patches_meta for {region_id}: {e}")
        return []
