import os
from pathlib import Path

from app.services import summary_image_service


def test_cleanup_removes_only_images_older_than_two_hours(monkeypatch, tmp_path):
    monkeypatch.setattr(summary_image_service, "SUMMARY_IMAGE_DIR", tmp_path)
    monkeypatch.setattr(summary_image_service, "SUMMARY_IMAGE_TTL_SECONDS", 7200)
    old_image = tmp_path / "old.png"
    fresh_image = tmp_path / "fresh.png"
    old_image.write_bytes(b"old")
    fresh_image.write_bytes(b"fresh")
    os.utime(old_image, (1_000, 1_000))
    os.utime(fresh_image, (8_000, 8_000))

    removed = summary_image_service.cleanup_expired_summary_images(now=9_000)

    assert removed == 1
    assert not old_image.exists()
    assert fresh_image.exists()
