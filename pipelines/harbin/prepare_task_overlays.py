#!/usr/bin/env python3
"""Package existing Harbin task files as small overlays for ModelScope.

The large base archive remains unchanged. Overlay archives are extracted after
it, allowing later task results to be published without uploading 13 GB again.
No inference or result generation is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .paths import PROJECT_ROOT
except ImportError:  # Direct script execution.
    from paths import PROJECT_ROOT


OVERLAYS = {
    "road_extraction": Path("data/harbin/tasks/road_extraction"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/modelscope_upload/harbin/task_overlays"),
    )
    parser.add_argument(
        "--overlays",
        nargs="+",
        choices=sorted(OVERLAYS),
        default=sorted(OVERLAYS),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "region": "harbin",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "existing-files-only",
        "assets": {},
    }
    for name in args.overlays:
        relative_root = OVERLAYS[name]
        source_root = PROJECT_ROOT / relative_root
        files = sorted(item for item in source_root.rglob("*") if item.is_file())
        if not files:
            raise FileNotFoundError(f"No existing files found under {source_root}")
        archive = args.output_root / f"{name}.tar"
        with tarfile.open(archive, "w") as tar:
            for source in files:
                tar.add(source, arcname=source.relative_to(PROJECT_ROOT).as_posix())
        manifest["assets"][name] = {
            "archive": archive.name,
            "files": len(files),
            "bytes": sum(source.stat().st_size for source in files),
            "sha256": _sha256(archive),
            "source": relative_root.as_posix(),
        }
        print(f"Archived {name}: {len(files)} existing files")

    manifest_path = args.output_root / "overlay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums = [
        f"{item['sha256']}  {item['archive']}"
        for item in manifest["assets"].values()
    ]
    checksums.append(f"{_sha256(manifest_path)}  {manifest_path.name}")
    (args.output_root / "overlay_checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(f"Prepared task overlays in {args.output_root}")


if __name__ == "__main__":
    main()
