"""Small persistent training-job store safe across API worker processes."""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.services.user_paths import get_users_dir


def _job_path(user_id: str, job_id: str) -> Path:
    return get_users_dir() / user_id / "jobs" / f"{job_id}.json"


def save_job(user_id: str, job: Dict[str, Any]) -> None:
    path = _job_path(user_id, job["job_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    try:
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_job(user_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(user_id, job_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def find_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Find a job across users for an authenticated administrator."""
    users_dir = get_users_dir()
    if not users_dir.is_dir():
        return None
    for user_dir in users_dir.iterdir():
        if not user_dir.is_dir():
            continue
        job = load_job(user_dir.name, job_id)
        if job is not None:
            return job
    return None


def update_job(user_id: str, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    job = load_job(user_id, job_id)
    if job is None:
        return None
    job.update(updates)
    save_job(user_id, job)
    return job
