#!/usr/bin/env python3
"""Build the frontend-ready static region Mosaic PNG package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_INVENTORY = ROOT / "static_mosaic_inventory.json"
DEFAULT_OUTPUT = ROOT / "Tmp" / "static_mosaic_package_20260730"
PACKAGE_NAME = "regional-mosaics.zip"
MAX_RSS_BYTES = 16 * 1024**3
MIN_FREE_DISK_BYTES = 20 * 1024**3
WORKER_TIMEOUT_SECONDS = 20 * 60


def asset_relative_path(region_id: str, sensor: str, date: str) -> PurePosixPath:
    """Return the stable path consumed by the frontend."""
    return PurePosixPath(region_id, sensor, date, "mosaic.png")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_png(path: Path) -> Dict[str, Any]:
    """Validate one packaged PNG and return manifest-ready statistics."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"{path} is not a PNG")
        if image.mode != "RGBA":
            raise ValueError(f"{path} must use RGBA transparency, got {image.mode}")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"{path} has invalid dimensions {image.size}")
        histogram = image.getchannel("A").histogram()
        transparent_pixels = histogram[0]
        opaque_pixels = histogram[255]
    return {
        "width": width,
        "height": height,
        "size_bytes": path.stat().st_size,
        "transparent_pixels": transparent_pixels,
        "opaque_pixels": opaque_pixels,
        "sha256": sha256_file(path),
    }


def iter_assets(inventory: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    for region_id, region in sorted(inventory["regions"].items()):
        for sensor, dates in sorted(region["assets"].items()):
            for date in sorted(set(dates)):
                yield {"region_id": region_id, "sensor": sensor, "date": date}


def build_zip_package(
    staging_dir: Path, output_path: Path, manifest: Dict[str, Any]
) -> None:
    """Create an atomic ZIP64 archive from validated staged PNGs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in sorted(staging_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                archive.write(path, path.relative_to(staging_dir).as_posix())
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    temporary.replace(output_path)


def _read_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return 0


def _run_worker(
    asset: Dict[str, str],
    staging_dir: Path,
    logs_dir: Path,
    *,
    max_rss_bytes: int,
) -> Dict[str, Any]:
    relative = asset_relative_path(
        asset["region_id"], asset["sensor"], asset["date"]
    )
    output_path = staging_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        result = validate_png(output_path)
        return {**asset, "path": relative.as_posix(), **result, "resumed": True}

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / (
        f"{asset['region_id']}_{asset['sensor']}_{asset['date']}.log"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--region",
        asset["region_id"],
        "--sensor",
        asset["sensor"],
        "--date",
        asset["date"],
        "--output",
        str(output_path),
    ]
    started = time.monotonic()
    peak_rss = 0
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            rss = _read_rss_bytes(process.pid)
            peak_rss = max(peak_rss, rss)
            if rss > max_rss_bytes:
                process.kill()
                process.wait()
                output_path.unlink(missing_ok=True)
                raise MemoryError(
                    f"{asset} exceeded RSS limit: {rss / 1024**3:.2f} GiB"
                )
            if time.monotonic() - started > WORKER_TIMEOUT_SECONDS:
                process.kill()
                process.wait()
                output_path.unlink(missing_ok=True)
                raise TimeoutError(f"{asset} exceeded worker timeout")
            time.sleep(0.5)
    if process.returncode != 0:
        output_path.unlink(missing_ok=True)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Worker failed for {asset}:\n{tail}")

    result = validate_png(output_path)
    return {
        **asset,
        "path": relative.as_posix(),
        **result,
        "peak_rss_bytes": peak_rss,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "resumed": False,
    }


def _worker(region_id: str, sensor: str, date: str, output: Path) -> None:
    """Render one asset in an isolated process."""
    from app.services.mosaic_service import build_mosaic_artifact

    if sensor.startswith("embedding-"):
        sensor_type = "embedding"
        version = sensor.split("-", 1)[1]
    else:
        sensor_type = sensor
        version = None
    data, mime, _ = build_mosaic_artifact(
        region_id=region_id,
        date=date,
        sensor_type=sensor_type,
        version=version,
        fmt="png",
        patch_ids=None,
        cache_dir=str(ROOT / "users" / "default" / "mosaic"),
    )
    if mime != "image/png":
        raise RuntimeError(f"Expected image/png, got {mime}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.partial")
    temporary.write_bytes(data)
    validate_png(temporary)
    temporary.replace(output)


def _write_region_metadata(staging_dir: Path, inventory: Dict[str, Any]) -> None:
    from app.services.region_mosaic_catalog import get_region_mosaic_info

    for region_id in sorted(inventory["regions"]):
        destination = staging_dir / region_id / "region.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "region_id": region_id,
                    "mosaic": get_region_mosaic_info(region_id),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def build_all(
    inventory_path: Path,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
    max_rss_bytes: int = MAX_RSS_BYTES,
) -> Dict[str, Any]:
    """Generate, validate, and package all discovered static Mosaic assets."""
    if shutil.disk_usage(output_dir.parent).free < MIN_FREE_DISK_BYTES:
        raise RuntimeError("At least 20 GiB free disk is required")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assets = list(iter_assets(inventory))
    if limit is not None:
        assets = assets[:limit]
    staging_dir = output_dir / "staging"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for index, asset in enumerate(assets, start=1):
        print(
            f"[{index}/{len(assets)}] "
            f"{asset['region_id']}/{asset['sensor']}/{asset['date']}",
            flush=True,
        )
        try:
            results.append(
                _run_worker(
                    asset,
                    staging_dir,
                    logs_dir,
                    max_rss_bytes=max_rss_bytes,
                )
            )
        except Exception as exc:
            failures.append({**asset, "error": str(exc)})
            print(f"  FAILED: {exc}", flush=True)
        checkpoint = {
            "total_requested": len(assets),
            "completed": len(results),
            "failed": len(failures),
            "assets": results,
            "failures": failures,
        }
        (output_dir / "checkpoint.json").write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _write_region_metadata(staging_dir, inventory)
    manifest = {
        "schema_version": "1.0",
        "package_filename": PACKAGE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "path_format": "{regionId}/{sensor}/{date}/mosaic.png",
        "image_format": "png",
        "crs": "EPSG:4326",
        "transparent_background": True,
        "total_assets": len(results),
        "failed_assets": len(failures),
        "regions": inventory["regions"],
        "assets": results,
        "failures": failures,
    }
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "audit-report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} assets failed; inspect audit-report.json before packaging"
        )
    build_zip_package(staging_dir, output_dir / PACKAGE_NAME, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-rss-gib", type=float, default=16.0)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--region")
    parser.add_argument("--sensor")
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        if not all((args.region, args.sensor, args.date, args.output)):
            raise SystemExit("worker requires region, sensor, date, and output")
        _worker(args.region, args.sensor, args.date, args.output)
        return
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    assets = list(iter_assets(inventory))
    if args.inventory_only:
        by_region: Dict[str, Dict[str, int]] = {}
        for asset in assets:
            by_region.setdefault(asset["region_id"], {}).setdefault(
                asset["sensor"], 0
            )
            by_region[asset["region_id"]][asset["sensor"]] += 1
        print(
            json.dumps(
                {"total_assets": len(assets), "counts": by_region},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    manifest = build_all(
        args.inventory,
        args.output_dir,
        limit=args.limit,
        max_rss_bytes=int(args.max_rss_gib * 1024**3),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
