"""Configuration loading with hot-reload support."""

import atexit
import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))


class ConfigReloadHandler(FileSystemEventHandler):
    """Watchdog handler for config file changes with debounce."""

    def __init__(self, config_manager, config_path: Path):
        self.config_manager = config_manager
        self.config_path = str(config_path.resolve())
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def on_modified(self, event):
        if event.src_path == self.config_path:
            with self._lock:
                if self._timer:
                    self._timer.cancel()
                self._timer = threading.Timer(0.5, self._do_reload)
                self._timer.start()

    def _do_reload(self):
        self.config_manager.reload()


class ConfigManager:
    """Thread-safe configuration manager with hot-reload."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._observer: Optional[Observer] = None
        self._patches_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.reload()
        self._start_watching()
        # Register cleanup on normal process exit
        atexit.register(self.stop_watching)

    def reload(self):
        """Reload configuration from file."""
        with self._lock:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f)
                # Clear patches cache on config reload
                self._patches_cache.clear()
                # Also clear DataService cache since config changed
                from app.services.data_service import DataService
                with DataService._cache_lock:
                    DataService._available_tasks_cache.clear()
                logger.info(f"Config reloaded from {self.config_path}")
            except (yaml.YAMLError, OSError) as e:
                logger.error(f"Failed to reload config: {e}")

    def _start_watching(self):
        """Start file watcher for hot-reload."""
        handler = ConfigReloadHandler(self, self.config_path)
        self._observer = Observer()
        self._observer.schedule(
            handler, str(self.config_path.parent), recursive=False
        )
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

    def get_patches(self, region_id: str) -> List[Dict[str, Any]]:
        """Load patches metadata with caching.

        Returns a deep copy to prevent callers from mutating the cache.
        """
        with self._lock:
            if region_id in self._patches_cache:
                return copy.deepcopy(self._patches_cache[region_id])

            region = self.get_region(region_id)
            if not region:
                return []

            meta_path = region.get("patches_meta")
            if not meta_path or not os.path.exists(meta_path):
                return []

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    patches = data
                elif isinstance(data, dict) and "patches" in data:
                    patches = data["patches"]
                    if "city" in data:
                        for p in patches:
                            p["city"] = data["city"]
                else:
                    patches = []

                self._patches_cache[region_id] = patches
                return copy.deepcopy(patches)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load patches_meta for {region_id}: {e}")
                return []

    def stop_watching(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()


# Global config instance (lazy initialization with double-checked locking)
_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_config() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        with _config_lock:
            if _config_manager is None:
                _config_manager = ConfigManager()
    return _config_manager
