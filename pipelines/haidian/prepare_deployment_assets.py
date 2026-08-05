#!/usr/bin/env python3
"""Package existing Haidian API assets for ModelScope deployment.

This script never runs inference or rewrites source assets. It archives only
files that already exist in the project and records their counts, sizes, and
checksums so a deployment can reproduce the current API-serving filesystem.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Iterable

try:
    from .paths import DEFAULT_MODELSCOPE_PREFIX, PROJECT_ROOT
except ImportError:  # Direct script execution.
    from paths import DEFAULT_MODELSCOPE_PREFIX, PROJECT_ROOT


ASSET_GROUPS = (
    (
        "task_heads",
        (
            Path("models/haidian/v1/task_heads/building_conv3x3_best.pt"),
            Path("models/haidian/v1/task_heads/road_conv3x3_best.pt"),
            Path("models/haidian/v1/task_heads/water_conv3x3_best.pt"),
        ),
    ),
    (
        "task_results",
        (
            Path("data/haidian/tasks/building_extraction"),
            Path("data/haidian/tasks/road_extraction"),
            Path("data/haidian/tasks/construction"),
            Path("data/haidian/tasks/land_use_classification"),
            Path("data/haidian/tasks/land_cover_classification"),
            Path("data/haidian/tasks/water_extraction"),
        ),
    ),
    (
        "harbin_road_results",
        (Path("data/harbin/tasks/road_extraction/v1/results"),),
    ),
    (
        "external_aef_2025",
        (Path("data/external_embeddings/aef"),),
    ),
    (
        "raw_s1",
        (
            Path(
                "data/haidian/archive/processed_training_data/"
                "extracted/patches/s1"
            ),
        ),
    ),
    (
        "raw_landsat",
        (
            Path(
                "data/haidian/archive/processed_training_data/"
                "extracted/patches/landsat"
            ),
        ),
    ),
    (
        "raw_s2",
        (
            Path(
                "data/haidian/archive/processed_training_data/"
                "extracted/patches/s2"
            ),
        ),
    ),
    (
        "raw_highres_optical",
        (
            Path(
                "data/haidian/archive/processed_training_data/"
                "extracted/patches/highres_optical"
            ),
        ),
    ),
    (
        "raw_highres_sar",
        (
            Path(
                "data/haidian/archive/processed_training_data/"
                "extracted/patches/highres_sar"
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/modelscope_upload/haidian/deployment"),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[name for name, _ in ASSET_GROUPS],
        default=[name for name, _ in ASSET_GROUPS],
    )
    return parser.parse_args()


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from (
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.name.endswith("_mask.tif")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_group(name: str, roots: tuple[Path, ...], output_root: Path) -> dict:
    archive = output_root / f"{name}.tar"
    file_count = 0
    byte_count = 0
    missing = []
    with tarfile.open(archive, "w", dereference=True) as tar:
        for relative_root in roots:
            source_root = PROJECT_ROOT / relative_root
            if not source_root.exists():
                missing.append(relative_root.as_posix())
                continue
            for source in _iter_files(source_root):
                # Deployment archives must be self-contained. Some generated
                # mosaics are symlinks to tiles elsewhere in the tree.
                tar.add(
                    source.resolve(),
                    arcname=source.relative_to(PROJECT_ROOT).as_posix(),
                )
                file_count += 1
                byte_count += source.stat().st_size

    if missing:
        archive.unlink(missing_ok=True)
        raise FileNotFoundError(f"Missing source directories for {name}: {missing}")
    if not file_count:
        archive.unlink(missing_ok=True)
        raise FileNotFoundError(f"No files found for deployment group {name}")
    return {
        "archive": archive.name,
        "files": file_count,
        "bytes": byte_count,
        "sha256": _sha256(archive),
        "sources": [root.as_posix() for root in roots],
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.groups)
    assets = {}
    for name, roots in ASSET_GROUPS:
        if name not in selected:
            continue
        print(f"Archiving existing asset group: {name}")
        assets[name] = _archive_group(name, roots, args.output_root)
        print(
            f"  {assets[name]['files']} files, "
            f"{assets[name]['bytes'] / 1024 / 1024:.1f} MiB"
        )

    manifest = {
        "schema_version": 1,
        "region": "haidian",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "modelscope_path": f"{DEFAULT_MODELSCOPE_PREFIX}/deployment",
        "mode": "existing-files-only",
        "assets": assets,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_lines = [
        f"{info['sha256']}  {info['archive']}" for info in assets.values()
    ]
    checksum_lines.append(
        f"{_sha256(args.output_root / 'manifest.json')}  manifest.json"
    )
    (args.output_root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Prepared deployment assets in {args.output_root}")


if __name__ == "__main__":
    main()
