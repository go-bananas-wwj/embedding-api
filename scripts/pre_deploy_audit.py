#!/usr/bin/env python3
"""Pre-deployment endpoint audit.

Calls every public endpoint for both regions, saves responses (JSON, PNG, NPY)
to disk, and produces a Markdown report for human review.

Usage:
    python scripts/pre_deploy_audit.py
"""

import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:9061")
RANDOM_SEED = 20260701

# Endpoints that are allowed to return 404 because data is intentionally absent.
def is_optional_404(endpoint: str, params: Dict[str, Any]) -> bool:
    """Return True if a 404 on this endpoint is acceptable (no underlying data)."""
    if endpoint.endswith("/summary"):
        return True
    if "/prediction" in endpoint or "/label" in endpoint:
        return True
    if params.get("format") == "npy":
        return True
    return False


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)[:120]


def save_response(
    output_dir: Path,
    endpoint: str,
    params: Dict[str, Any],
    response: requests.Response,
    index: int,
) -> Tuple[Path, str]:
    """Save response body and return (saved_path, content_type)."""
    content_type = response.headers.get("content-type", "unknown")
    ext = ".bin"
    if "json" in content_type:
        ext = ".json"
    elif "png" in content_type:
        ext = ".png"
    elif "octet-stream" in content_type or "npy" in content_type:
        ext = ".npy"
    elif "tiff" in content_type:
        ext = ".tif"

    name = f"{index:04d}_{sanitize_filename(endpoint)}"
    if params:
        param_str = "_".join(f"{k}={sanitize_filename(str(v))}" for k, v in sorted(params.items()))
        name += f"_{param_str}"
    name += ext
    path = output_dir / name

    path.write_bytes(response.content)
    return path, content_type


def call(
    session: requests.Session,
    method: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Tuple[requests.Response, float]:
    url = f"{BASE_URL}{endpoint}"
    start = time.time()
    if method.upper() == "GET":
        resp = session.get(url, params=params, timeout=120)
    elif method.upper() == "POST":
        resp = session.post(url, params=params, data=data, json=json_body, timeout=120)
    else:
        raise ValueError(f"Unsupported method: {method}")
    elapsed = time.time() - start
    return resp, elapsed


def add_record(
    records: List[Dict[str, Any]],
    endpoint: str,
    params: Dict[str, Any],
    status: int,
    content_type: str,
    saved_path: Path,
    elapsed: float,
    note: str = "",
) -> None:
    records.append(
        {
            "endpoint": endpoint,
            "params": params,
            "status": status,
            "content_type": content_type,
            "saved_path": str(saved_path.relative_to(saved_path.parents[2])),
            "elapsed_seconds": round(elapsed, 3),
            "note": note,
        }
    )


def audit_region(
    session: requests.Session,
    region_id: str,
    output_dir: Path,
    records: List[Dict[str, Any]],
    counter: List[int],
) -> None:
    print(f"\n=== Auditing region: {region_id} ===")

    # Region detail
    endpoint = f"/regions/{region_id}"
    resp, elapsed = call(session, "GET", endpoint)
    path, ct = save_response(output_dir, endpoint, {}, resp, counter[0])
    counter[0] += 1
    add_record(records, endpoint, {}, resp.status_code, ct, path, elapsed)
    region_detail = resp.json() if resp.status_code == 200 else {}
    tasks = list(region_detail.get("tasks", {}).keys())
    task_versions = {
        t: v.get("versions", []) for t, v in region_detail.get("tasks", {}).items()
    }

    # Patches list (page 1, small)
    endpoint = f"/regions/{region_id}/patches"
    params = {"page": 1, "page_size": 5}
    resp, elapsed = call(session, "GET", endpoint, params)
    path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
    counter[0] += 1
    add_record(records, endpoint, params, resp.status_code, ct, path, elapsed)
    patches_list = resp.json() if resp.status_code == 200 else {}
    patches = patches_list.get("patches", patches_list.get("items", []))
    if not patches:
        print(f"  No patches for {region_id}; skipping patch-level calls.")
        return

    first_patch = patches[0]["patch_id"]
    available_months = patches[0].get("available_months", [])
    available_tasks = set(patches[0].get("available_tasks", []))

    # Single patch detail
    endpoint = f"/regions/{region_id}/patches/{first_patch}"
    resp, elapsed = call(session, "GET", endpoint)
    path, ct = save_response(output_dir, endpoint, {}, resp, counter[0])
    counter[0] += 1
    add_record(records, endpoint, {}, resp.status_code, ct, path, elapsed)

    # Tasks list
    endpoint = f"/regions/{region_id}/tasks"
    resp, elapsed = call(session, "GET", endpoint)
    path, ct = save_response(output_dir, endpoint, {}, resp, counter[0])
    counter[0] += 1
    add_record(records, endpoint, {}, resp.status_code, ct, path, elapsed)

    # Embedding formats
    sample_month = available_months[0] if available_months else None
    for emb_fmt in ["png", "npy", "json"]:
        params = {"format": emb_fmt}
        if sample_month:
            params["month"] = sample_month
        endpoint = f"/regions/{region_id}/patches/{first_patch}/embedding"
        resp, elapsed = call(session, "GET", endpoint, params)
        path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
        counter[0] += 1
        note = ""
        if resp.status_code == 404 and emb_fmt == "npy":
            note = "PNG-only embedding fallback expected"
        add_record(records, endpoint, params, resp.status_code, ct, path, elapsed, note)

    # Mosaic with small subset to keep it fast
    if sample_month:
        endpoint = f"/regions/{region_id}/mosaic"
        params = {"date": sample_month, "sensor_type": "s2", "format": "png", "patch_ids": [first_patch]}
        resp, elapsed = call(session, "GET", endpoint, params)
        path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
        counter[0] += 1
        note = ""
        if resp.status_code == 404:
            note = "Expected 404 - raw scene missing for this date"
        add_record(records, endpoint, params, resp.status_code, ct, path, elapsed, note)

    # SAM3 status
    endpoint = f"/regions/{region_id}/sam3/status"
    resp, elapsed = call(session, "GET", endpoint)
    path, ct = save_response(output_dir, endpoint, {}, resp, counter[0])
    counter[0] += 1
    add_record(records, endpoint, {}, resp.status_code, ct, path, elapsed)

    # Per-task calls
    for task in tasks:
        versions = task_versions.get(task, [])
        version = versions[0] if versions else "v1"

        # Summary
        endpoint = f"/regions/{region_id}/tasks/{task}/summary"
        params = {"version": version}
        period = None
        if task == "change_detection":
            if region_id == "harbin":
                period = "2025-04_vs_2025-06"
            else:
                period = "202512_vs_202601"
            params["period"] = period
        resp, elapsed = call(session, "GET", endpoint, params)
        path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
        counter[0] += 1
        note = ""
        if resp.status_code == 404:
            if is_optional_404(endpoint, params):
                note = "Optional data not available (404 acceptable)"
            elif not versions:
                note = "Expected 404 (task has no configured versions)"
        add_record(records, endpoint, params, resp.status_code, ct, path, elapsed, note)

        # Tiles list
        endpoint = f"/regions/{region_id}/tasks/{task}/tiles"
        params = {"version": version}
        if period:
            params["period"] = period
        resp, elapsed = call(session, "GET", endpoint, params)
        path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
        counter[0] += 1
        note = ""
        if resp.status_code == 404:
            if is_optional_404(endpoint, params):
                note = "Optional data not available (404 acceptable)"
            elif not versions:
                note = "Expected 404 (task has no configured versions)"
        add_record(records, endpoint, params, resp.status_code, ct, path, elapsed, note)

        # Pick a patch that has this task available, fallback to first patch
        candidate_patch = first_patch
        for p in patches:
            if task in p.get("available_tasks", []):
                candidate_patch = p["patch_id"]
                break

        # Determine month/period for result call
        result_month = sample_month
        result_period = None
        if task == "change_detection":
            result_period = period
            result_month = None
        elif task in ("building_extraction", "road_extraction", "water_extraction", "construction", "land_use_classification", "land_cover_classification"):
            # Use task-specific known-good month if possible
            if region_id == "harbin":
                result_month = "2025-10" if task in ("building_extraction", "road_extraction") else sample_month
            else:
                result_month = "202512"

        # Task result PNG and NPY
        for fmt in ("png", "npy"):
            endpoint = f"/regions/{region_id}/patches/{candidate_patch}/tasks/{task}/result"
            params = {"format": fmt, "version": version}
            if result_month:
                params["month"] = result_month
            if result_period:
                params["period"] = result_period
            resp, elapsed = call(session, "GET", endpoint, params)
            path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
            counter[0] += 1
            note = ""
            if resp.status_code == 404:
                if is_optional_404(endpoint, params):
                    note = "Optional data not available (404 acceptable)"
                elif not versions:
                    note = "Expected 404 (task has no configured versions)"
                elif task not in available_tasks:
                    note = "Task not available for this patch"
            add_record(records, endpoint, params, resp.status_code, ct, path, elapsed, note)

        # Prediction and label (only if task has configured versions)
        if versions:
            for sub in ("prediction", "label"):
                endpoint = f"/regions/{region_id}/patches/{candidate_patch}/tasks/{task}/{sub}"
                params = {"version": version}
                if result_period:
                    params["period"] = result_period
                resp, elapsed = call(session, "GET", endpoint, params)
                path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
                counter[0] += 1
                note = ""
                if resp.status_code == 404:
                    if is_optional_404(endpoint, params):
                        note = "Optional data not available (404 acceptable)"
                    elif task not in available_tasks:
                        note = "Task not available for this patch"
                add_record(records, endpoint, params, resp.status_code, ct, path, elapsed, note)


def audit_global(session: requests.Session, output_dir: Path, records: List[Dict[str, Any]], counter: List[int]) -> None:
    print("\n=== Auditing global endpoints ===")
    for endpoint in ["/health", "/regions"]:
        resp, elapsed = call(session, "GET", endpoint)
        path, ct = save_response(output_dir, endpoint, {}, resp, counter[0])
        counter[0] += 1
        add_record(records, endpoint, {}, resp.status_code, ct, path, elapsed)

    # Models list (with and without region filter)
    for region_id in (None, "harbin", "haidian"):
        endpoint = "/models"
        params = {"region_id": region_id} if region_id else {}
        resp, elapsed = call(session, "GET", endpoint, params)
        path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
        counter[0] += 1
        add_record(records, endpoint, params, resp.status_code, ct, path, elapsed)

    # System models list (per region)
    for region_id in ("harbin", "haidian"):
        endpoint = "/system-models"
        params = {"region_id": region_id}
        resp, elapsed = call(session, "GET", endpoint, params)
        path, ct = save_response(output_dir, endpoint, params, resp, counter[0])
        counter[0] += 1
        add_record(records, endpoint, params, resp.status_code, ct, path, elapsed)


def _build_table_rows(records: List[Dict[str, Any]], start_index: int = 1) -> List[str]:
    rows = []
    for i, rec in enumerate(records, start_index):
        params = json.dumps(rec["params"], ensure_ascii=False) if rec["params"] else ""
        params = params.replace("|", "\\|")
        note = rec.get("note", "").replace("|", "\\|")
        rows.append(
            f"| {i} | `{rec['endpoint']}` | `{params}` | {rec['status']} | {rec['content_type']} | `{rec['saved_path']}` | {rec['elapsed_seconds']}s | {note} |\n"
        )
    return rows


def _write_single_report(
    output_dir: Path,
    filename: str,
    title: str,
    records: List[Dict[str, Any]],
) -> None:
    md = [f"# {title}\n", f"Generated: {datetime.utcnow().isoformat()}Z\n", f"Base URL: {BASE_URL}\n\n"]
    md.append("| # | Endpoint | Params | Status | Content-Type | Saved | Elapsed | Note |\n")
    md.append("|---|----------|--------|--------|--------------|-------|---------|------|\n")
    md.extend(_build_table_rows(records))

    failures = [
        r for r in records
        if r["status"] >= 400
        and not r.get("note", "").startswith("Expected")
        and not r.get("note", "").startswith("Optional")
    ]
    md.append(f"\n## Summary\n\n")
    md.append(f"Total calls: {len(records)}\n")
    md.append(f"Failures (non-expected 4xx/5xx): {len(failures)}\n")
    if failures:
        md.append("\n### Failures\n\n")
        for rec in failures:
            md.append(f"- `{rec['endpoint']}` params={rec['params']} status={rec['status']}\n")

    (output_dir / filename).write_text("".join(md), encoding="utf-8")


def write_report(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    global_records = [r for r in records if not r["endpoint"].startswith("/regions/")]
    harbin_records = [r for r in records if r["endpoint"].startswith("/regions/harbin")]
    haidian_records = [r for r in records if r["endpoint"].startswith("/regions/haidian")]

    _write_single_report(output_dir, "report_global.md", "Global Endpoints", global_records)
    _write_single_report(output_dir, "report_harbin.md", "Harbin New Area Endpoints", harbin_records)
    _write_single_report(output_dir, "report_haidian.md", "Haidian District Endpoints", haidian_records)
    _write_single_report(output_dir, "report_all.md", "All Endpoints", records)


def main() -> None:
    random.seed(RANDOM_SEED)
    session = requests.Session()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_output") / f"pre_deploy_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving audit artifacts to: {output_dir}")

    records: List[Dict[str, Any]] = []
    counter = [0]

    audit_global(session, output_dir, records, counter)
    for region_id in ("harbin", "haidian"):
        audit_region(session, region_id, output_dir, records, counter)

    write_report(output_dir, records)

    failures = [
        r for r in records
        if r["status"] >= 400
        and not r.get("note", "").startswith("Expected")
        and not r.get("note", "").startswith("Optional")
    ]
    print(f"\nAudit complete: {len(records)} calls, {len(failures)} unexpected failures.")
    print(f"Report: {output_dir / 'report.md'}")
    if failures:
        print("Unexpected failures:")
        for r in failures:
            print(f"  {r['endpoint']} {r['params']} -> {r['status']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
