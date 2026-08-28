"""Daily cleanup tests for user-created model artifacts."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services import custom_model_cleanup, model_registry, user_paths


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


def _write_user(root: Path, records: list[dict]) -> Path:
    user = root / "default"
    (user / "models").mkdir(parents=True)
    (user / "jobs").mkdir()
    (user / "results").mkdir()
    (user / "models_index.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    for record in records:
        Path(record["model_path"]).write_bytes(b"checkpoint")
    return user


def _record(root: Path, model_id: str, age_hours: int, status: str) -> dict:
    return {
        "id": model_id,
        "name": model_id,
        "type": "single_time_detection",
        "status": status,
        "created_at": (NOW - timedelta(hours=age_hours)).isoformat(),
        "classes": [],
        "model_path": str(root / "default" / "models" / f"{model_id}.pkl"),
    }


def _patch_users_dir(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(user_paths, "get_users_dir", lambda: root)
    monkeypatch.setattr(model_registry, "get_users_dir", lambda: root)
    monkeypatch.setattr(custom_model_cleanup, "get_users_dir", lambda: root)
    model_registry._registries.clear()


def test_cleanup_deletes_only_terminal_models_older_than_ttl(tmp_path, monkeypatch):
    _patch_users_dir(monkeypatch, tmp_path)
    old = _record(tmp_path, "model_old", 25, "completed")
    fresh = _record(tmp_path, "model_fresh", 23, "completed")
    running = _record(tmp_path, "model_running", 72, "training")
    user = _write_user(tmp_path, [old, fresh, running])
    (user / "jobs" / "job_old.json").write_text(
        json.dumps({"job_id": "job_old", "model_id": "model_old", "status": "completed"}),
        encoding="utf-8",
    )
    (user / "results" / "infer_model_old_patch.png").write_bytes(b"png")

    report = custom_model_cleanup.cleanup_custom_models(
        ttl_hours=24, dry_run=False, now=NOW
    )

    remaining = json.loads((user / "models_index.json").read_text(encoding="utf-8"))
    assert {item["id"] for item in remaining} == {"model_fresh", "model_running"}
    assert not Path(old["model_path"]).exists()
    assert Path(fresh["model_path"]).exists()
    assert Path(running["model_path"]).exists()
    assert not (user / "jobs" / "job_old.json").exists()
    assert not (user / "results" / "infer_model_old_patch.png").exists()
    assert report["deleted_model_ids"] == ["model_old"]


def test_cleanup_dry_run_reports_without_deleting(tmp_path, monkeypatch):
    _patch_users_dir(monkeypatch, tmp_path)
    old = _record(tmp_path, "model_old", 48, "failed")
    user = _write_user(tmp_path, [old])

    report = custom_model_cleanup.cleanup_custom_models(
        ttl_hours=24, dry_run=True, now=NOW
    )

    assert Path(old["model_path"]).exists()
    assert len(json.loads((user / "models_index.json").read_text())) == 1
    assert report["candidate_model_ids"] == ["model_old"]
    assert report["deleted_model_ids"] == []


def test_cleanup_skips_corrupt_registry(tmp_path, monkeypatch):
    _patch_users_dir(monkeypatch, tmp_path)
    user = tmp_path / "default"
    user.mkdir()
    (user / "models_index.json").write_text("{broken", encoding="utf-8")

    report = custom_model_cleanup.cleanup_custom_models(
        ttl_hours=24, dry_run=False, now=NOW
    )

    assert (user / "models_index.json").read_text(encoding="utf-8") == "{broken"
    assert report["errors"]


def test_seconds_until_next_midnight_uses_shanghai_timezone():
    now = datetime(2026, 8, 28, 15, 30, tzinfo=timezone.utc)

    seconds = custom_model_cleanup._seconds_until_next_midnight(now)

    assert seconds == 30 * 60
