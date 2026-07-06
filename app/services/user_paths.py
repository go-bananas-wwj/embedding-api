"""Shared helpers for per-user filesystem paths.

Each user has an isolated workspace under ``users/{user_id}/``. This module
provides the canonical paths used by model registry, training engine, and
inference engine.
"""

import re
from pathlib import Path

from app.config import get_config


# Base directory for per-user data. Intentionally outside repository data dirs
# so user-generated artifacts can be excluded from Git.
DEFAULT_USERS_DIR = Path("users")
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def get_users_dir() -> Path:
    """Return the root users directory from config or default."""
    users_dir = get_config().get("users_dir", default=None)
    path = Path(users_dir) if users_dir else DEFAULT_USERS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_dir(user_id: str) -> Path:
    """Return (and create) the root directory for a user."""
    if not _USER_ID_RE.fullmatch(user_id):
        raise ValueError(f"Invalid user_id for filesystem path: {user_id!r}")
    d = get_users_dir() / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d
