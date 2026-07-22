"""Publish short-lived task-summary images and remove expired copies."""

import asyncio
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence


SUMMARY_IMAGE_DIR = Path(os.environ.get("SUMMARY_IMAGE_DIR", "temp/task_summary_results"))
SUMMARY_IMAGE_TTL_SECONDS = int(os.environ.get("SUMMARY_IMAGE_TTL_SECONDS", "7200"))
PUBLIC_BASE_URL = os.environ.get(
    "PUBLIC_BASE_URL", "http://60.31.21.42:22065"
).rstrip("/")


def publish_summary_images(
    region_id: str,
    task_type: str,
    version: str,
    month: str,
    files: Sequence[Path],
) -> list[dict]:
    SUMMARY_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    published = []
    for source in files:
        match = re.search(r"patch_\d{6}", source.stem)
        if not match or not source.exists():
            continue
        patch_id = match.group(0)
        filename = f"{region_id}_{task_type}_{version}_{patch_id}_{month}.png"
        destination = SUMMARY_IMAGE_DIR / filename
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        published.append({
            "patch_id": patch_id,
            "image_url": f"{PUBLIC_BASE_URL}/task-summary/results/{filename}",
            "cleanup_interval_seconds": SUMMARY_IMAGE_TTL_SECONDS,
        })
    return published


def cleanup_expired_summary_images(now: Optional[float] = None) -> int:
    if not SUMMARY_IMAGE_DIR.exists():
        return 0
    cutoff = (now if now is not None else time.time()) - SUMMARY_IMAGE_TTL_SECONDS
    removed = 0
    for path in SUMMARY_IMAGE_DIR.glob("*.png"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


async def summary_image_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(SUMMARY_IMAGE_TTL_SECONDS)
        if not SUMMARY_IMAGE_DIR.exists():
            continue
        for path in SUMMARY_IMAGE_DIR.glob("*.png"):
            try:
                path.unlink()
            except OSError:
                continue
