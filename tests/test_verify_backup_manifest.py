import hashlib
import json

from scripts.verify_backup_manifest import verify_release


def test_verify_release_accepts_matching_file(tmp_path):
    payload = tmp_path / "models" / "weights.pt"
    payload.parent.mkdir()
    payload.write_bytes(b"stable-weights")
    manifest = {
        "files": [
            {
                "path": "models/weights.pt",
                "size_bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    assert verify_release(tmp_path) == []


def test_verify_release_reports_missing_and_corrupt_files(tmp_path):
    payload = tmp_path / "data.tar.zst"
    payload.write_bytes(b"corrupt")
    manifest = {
        "files": [
            {
                "path": "data.tar.zst",
                "size_bytes": payload.stat().st_size,
                "sha256": "0" * 64,
            },
            {
                "path": "missing.tar.zst",
                "size_bytes": 1,
                "sha256": "0" * 64,
            },
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    errors = verify_release(tmp_path)

    assert "sha256 mismatch: data.tar.zst" in errors
    assert "missing: missing.tar.zst" in errors
