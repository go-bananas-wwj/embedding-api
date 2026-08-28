"""TTL cleanup for user-created model artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from app.services.model_registry import ModelRegistry
from app.services.user_paths import get_users_dir

logger = logging.getLogger(__name__)

DEFAULT_TTL_HOURS = 24
CLEANUP_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def cleanup_enabled() -> bool:
    return os.environ.get("CUSTOM_MODEL_CLEANUP_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _remove_associated_artifacts(user_dir: Path, model_ids: set[str]) -> tuple[int, int, list[str]]:
    removed_files = 0
    freed_bytes = 0
    errors: list[str] = []

    jobs_dir = user_dir / "jobs"
    if jobs_dir.is_dir():
        for path in jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(job, dict) and job.get("model_id") in model_ids:
                    size = path.stat().st_size
                    path.unlink()
                    removed_files += 1
                    freed_bytes += size
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path}: {exc}")

    for directory_name in ("results", "system_model_results"):
        directory = user_dir / directory_name
        if not directory.is_dir():
            continue
        for model_id in model_ids:
            for path in directory.glob(f"*{model_id}*"):
                try:
                    if not path.is_file():
                        continue
                    size = path.stat().st_size
                    path.unlink()
                    removed_files += 1
                    freed_bytes += size
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
    return removed_files, freed_bytes, errors


def _remove_hot_job_cache(model_ids: set[str]) -> int:
    """Drop terminal jobs for deleted models from the process-local cache."""
    try:
        from app.routers.models import _training_jobs
    except ImportError:
        return 0
    stale = [
        job_id
        for job_id, job in _training_jobs.items()
        if job.get("model_id") in model_ids and job.get("status") != "running"
    ]
    for job_id in stale:
        _training_jobs.pop(job_id, None)
    return len(stale)


def cleanup_custom_models(
    ttl_hours: float = DEFAULT_TTL_HOURS,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Clean expired custom models across users; project preset models are untouched."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current.astimezone(timezone.utc) - timedelta(hours=ttl_hours)
    report: Dict[str, Any] = {
        "dry_run": dry_run,
        "ttl_hours": ttl_hours,
        "scanned_users": 0,
        "candidate_model_ids": [],
        "deleted_model_ids": [],
        "removed_associated_files": 0,
        "removed_cached_jobs": 0,
        "freed_bytes": 0,
        "errors": [],
    }

    for user_dir in sorted(get_users_dir().iterdir()):
        if not user_dir.is_dir():
            continue
        index_path = user_dir / "models_index.json"
        if not index_path.exists():
            continue
        report["scanned_users"] += 1
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("registry root must be a list")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            report["errors"].append(f"{index_path}: {exc}")
            continue

        registry = ModelRegistry(index_path, user_dir / "models")
        result = registry.cleanup_expired_models(cutoff, dry_run=dry_run)
        candidate_ids = [str(item.get("id")) for item in result["candidates"]]
        deleted_ids = [str(item.get("id")) for item in result["deleted"]]
        report["candidate_model_ids"].extend(candidate_ids)
        report["deleted_model_ids"].extend(deleted_ids)
        report["freed_bytes"] += result["freed_bytes"]
        report["errors"].extend(result["errors"])

        if deleted_ids:
            report["removed_cached_jobs"] += _remove_hot_job_cache(set(deleted_ids))
            removed, freed, errors = _remove_associated_artifacts(
                user_dir, set(deleted_ids)
            )
            report["removed_associated_files"] += removed
            report["freed_bytes"] += freed
            report["errors"].extend(errors)

    logger.info(
        "Custom model cleanup: dry_run=%s candidates=%d deleted=%d freed_bytes=%d errors=%d",
        dry_run,
        len(report["candidate_model_ids"]),
        len(report["deleted_model_ids"]),
        report["freed_bytes"],
        len(report["errors"]),
    )
    return report


async def custom_model_cleanup_loop() -> None:
    """Run custom-model cleanup at 00:00 Asia/Shanghai every day."""
    ttl = _positive_float_env("CUSTOM_MODEL_TTL_HOURS", DEFAULT_TTL_HOURS)
    while True:
        await asyncio.sleep(_seconds_until_next_midnight())
        if cleanup_enabled():
            await asyncio.to_thread(cleanup_custom_models, ttl, False)


def _seconds_until_next_midnight(now: Optional[datetime] = None) -> float:
    """Return seconds until the next midnight in the service business timezone."""
    current = now or datetime.now(CLEANUP_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=CLEANUP_TIMEZONE)
    else:
        current = current.astimezone(CLEANUP_TIMEZONE)
    tomorrow = current.date() + timedelta(days=1)
    next_midnight = datetime.combine(
        tomorrow, datetime.min.time(), tzinfo=CLEANUP_TIMEZONE
    )
    return (next_midnight - current).total_seconds()
