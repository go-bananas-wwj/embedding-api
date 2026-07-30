#!/usr/bin/env python3
"""Verify files described by a stable backup manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release(release_dir: Path) -> list[str]:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for item in manifest["files"]:
        relative_path = item["path"]
        path = release_dir / relative_path
        if not path.is_file():
            errors.append(f"missing: {relative_path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != item["size_bytes"]:
            errors.append(
                f"size mismatch: {relative_path}: "
                f"{actual_size} != {item['size_bytes']}"
            )
            continue
        actual_sha256 = sha256_file(path)
        if actual_sha256 != item["sha256"]:
            errors.append(f"sha256 mismatch: {relative_path}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="校验 ModelScope 稳定版备份的文件大小和 SHA256。"
    )
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()

    errors = verify_release(args.release_dir.resolve())
    if errors:
        for error in errors:
            print(error)
        return 1
    print("backup verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
