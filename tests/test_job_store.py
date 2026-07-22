"""Persistent training-job store tests."""

from app.services import job_store


def test_saved_job_can_be_loaded_after_memory_state_is_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "get_users_dir", lambda: tmp_path)
    expected = {
        "job_id": "job_persisted",
        "user_id": "default",
        "status": "completed",
    }

    job_store.save_job("default", expected)

    assert job_store.load_job("default", "job_persisted") == expected
    assert job_store.find_job("job_persisted") == expected


def test_corrupt_job_file_is_treated_as_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, "get_users_dir", lambda: tmp_path)
    path = tmp_path / "default" / "jobs" / "job_broken.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    assert job_store.load_job("default", "job_broken") is None
