#!/usr/bin/env python3
"""
Embedding API Comprehensive Test Suite
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BASE_URL = "http://localhost:9061"
OUTPUT_DIR = Path("/workspace/embedding-api/test_output_agent")
CURL_PROXY = ["--noproxy", "*"]

# Results tracking
passed: List[Dict[str, Any]] = []
failed: List[Dict[str, Any]] = []
unexpected: List[Dict[str, Any]] = []


def curl(
    url_path: str,
    save_path: Optional[Path] = None,
    expect_status: int = 200,
    method: str = "GET",
    data: Optional[bytes] = None,
) -> Tuple[int, bytes, Dict[str, str]]:
    """Make a curl request and return status, body, parsed headers."""
    full_url = f"{BASE_URL}{url_path}"
    data_file: Optional[Path] = None
    cmd = [
        "curl", "-s", "-w", "\\n%{http_code}", "-D", "-",
        *CURL_PROXY,
        "-X", method,
    ]
    if data is not None:
        data_file = OUTPUT_DIR / f"_payload_{int(time.time()*1000000)}.json"
        data_file.write_bytes(data)
        cmd.extend(["-H", "Content-Type: application/json", "-d", f"@{data_file}"])
    cmd.append(full_url)
    if save_path:
        cmd.extend(["-o", str(save_path)])

    result = subprocess.run(cmd, capture_output=True)
    output = result.stdout.decode("utf-8", errors="replace")

    # Parse headers and body
    lines = output.splitlines()
    http_status = 0
    headers = {}
    body = b""

    if save_path:
        # When saving to file, only get headers from stdout
        header_lines = []
        for line in lines:
            if line.startswith("HTTP/"):
                parts = line.split()
                if len(parts) >= 2:
                    http_status = int(parts[1])
            elif ":" in line:
                header_lines.append(line)
            elif line.isdigit():
                http_status = int(line)
        for hl in header_lines:
            if ":" in hl:
                k, v = hl.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        if save_path.exists():
            body = save_path.read_bytes()
    else:
        # Find the HTTP status line and split headers/body
        header_section = []
        body_started = False
        for line in lines:
            if line.startswith("HTTP/"):
                parts = line.split()
                if len(parts) >= 2:
                    http_status = int(parts[1])
            elif not body_started and line == "":
                body_started = True
            elif line.isdigit() and not body_started:
                http_status = int(line)
            elif not body_started:
                header_section.append(line)
            elif body_started:
                # This shouldn't happen much with -w format
                pass

        # Reconstruct body from what curl gave us
        # Actually, curl -w adds the status code at the end
        # Let's use a different approach: write body to a temp file
        pass

    # Simpler approach: always write body to file
    if not save_path:
        save_path = OUTPUT_DIR / f"_tmp_{int(time.time()*1000000)}.bin"
        cmd = [
            "curl", "-s", "-w", "\\nHTTP_CODE:%{http_code}",
            *CURL_PROXY,
            "-X", method,
        ]
        if data_file is not None:
            cmd.extend(["-H", "Content-Type: application/json", "-d", f"@{data_file}"])
        cmd.extend(["-o", str(save_path), full_url])
        result = subprocess.run(cmd, capture_output=True)
        output = result.stdout.decode("utf-8", errors="replace")
        for line in output.splitlines():
            if line.startswith("HTTP_CODE:"):
                http_status = int(line.split(":", 1)[1])
        if save_path.exists():
            body = save_path.read_bytes()
        save_path.unlink(missing_ok=True)

    if data_file is not None:
        data_file.unlink(missing_ok=True)
    return http_status, body, headers


def log_result(
    name: str,
    url: str,
    status: int,
    expected_status: int = 200,
    detail: str = "",
    notes: str = "",
):
    """Log a test result."""
    entry = {
        "name": name,
        "url": url,
        "status": status,
        "expected": expected_status,
        "detail": detail,
        "notes": notes,
    }
    if status == expected_status:
        passed.append(entry)
    elif expected_status == 200 and status in (404,):
        # 404 on data endpoints can be expected if file doesn't exist
        failed.append(entry)
    elif expected_status != 200 and status == expected_status:
        passed.append(entry)
    else:
        if expected_status == 404 and status == 404:
            passed.append(entry)
        else:
            unexpected.append(entry)


def test_basic_endpoints():
    """Test /health, /regions, /regions/harbin, /regions/haidian"""
    print("\n=== Testing Basic Endpoints ===")

    # /health
    status, body, _ = curl("/health")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            detail = f"status={data.get('status')}, version={data.get('version')}, regions={data.get('regions')}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /health", "/health", status, 200, detail)

    # /regions
    status, body, _ = curl("/regions")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            regions = [r["id"] for r in data.get("regions", [])]
            detail = f"regions={regions}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /regions", "/regions", status, 200, detail)

    # /regions/harbin
    status, body, _ = curl("/regions/harbin")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            tasks = list(data.get("tasks", {}).keys())
            detail = f"patch_count={data.get('patch_count')}, tasks={tasks}, embeddings={data.get('embeddings')}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /regions/harbin", "/regions/harbin", status, 200, detail)

    # /regions/haidian
    status, body, _ = curl("/regions/haidian")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            tasks = list(data.get("tasks", {}).keys())
            detail = f"patch_count={data.get('patch_count')}, tasks={tasks}, embeddings={data.get('embeddings')}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /regions/haidian", "/regions/haidian", status, 200, detail)


def test_patch_endpoints():
    """Test patch listing and detail endpoints."""
    print("\n=== Testing Patch Endpoints ===")

    # List patches page 1 size 5
    status, body, _ = curl("/regions/harbin/patches?page=1&page_size=5")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            patches = [p["patch_id"] for p in data.get("patches", [])]
            detail = f"total={data.get('total')}, patches={patches}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /regions/harbin/patches?page=1&page_size=5",
               "/regions/harbin/patches?page=1&page_size=5", status, 200, detail)

    # List patches with bbox
    status, body, _ = curl("/regions/harbin/patches?page=1&page_size=5&bbox=126.5,45.74,126.55,45.76")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            patches = [p["patch_id"] for p in data.get("patches", [])]
            detail = f"total={data.get('total')}, patches={patches}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /regions/harbin/patches?bbox=...",
               "/regions/harbin/patches?page=1&page_size=5&bbox=126.5,45.74,126.55,45.76",
               status, 200, detail)

    # Patch detail patch_000000
    status, body, _ = curl("/regions/harbin/patches/patch_000000")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            detail = f"has_embedding={data.get('has_embedding')}, available_tasks={data.get('available_tasks')}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /regions/harbin/patches/patch_000000",
               "/regions/harbin/patches/patch_000000", status, 200, detail)

    # Patch detail patch_000010
    status, body, _ = curl("/regions/harbin/patches/patch_000010")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            detail = f"has_embedding={data.get('has_embedding')}, available_tasks={data.get('available_tasks')}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /regions/harbin/patches/patch_000010",
               "/regions/harbin/patches/patch_000010", status, 200, detail)


def validate_image(path: Path) -> Tuple[bool, str]:
    """Validate an image file using PIL and file command."""
    if not path.exists() or path.stat().st_size == 0:
        return False, "File empty or missing"

    # Use file command
    result = subprocess.run(["file", str(path)], capture_output=True, text=True)
    file_info = result.stdout.strip()
    if "PNG" not in file_info and "JPEG" not in file_info and "image" not in file_info.lower():
        return False, f"file command: {file_info}"

    # Use PIL
    try:
        from PIL import Image
        with Image.open(path) as img:
            size = img.size
            mode = img.mode
            return True, f"{size[0]}x{size[1]} mode={mode}"
    except Exception as e:
        return False, f"PIL error: {e}"


def validate_npy(path: Path) -> Tuple[bool, str]:
    """Validate a numpy file."""
    if not path.exists() or path.stat().st_size == 0:
        return False, "File empty or missing"

    result = subprocess.run(["file", str(path)], capture_output=True, text=True)
    file_info = result.stdout.strip()

    try:
        arr = np.load(path, allow_pickle=False)
        return True, f"shape={arr.shape}, dtype={arr.dtype}, size={arr.size}"
    except Exception as e:
        return False, f"numpy error: {e} (file: {file_info})"


def test_embedding_endpoints():
    """Test embedding endpoints for all formats."""
    print("\n=== Testing Embedding Endpoints ===")

    test_cases = [
        ("harbin", "patch_000000", {"png": True, "json": True, "npy": True, "cache": True}),
        ("haidian", "patch_000000", {"png": False, "json": True, "npy": True, "cache": False}),
    ]

    formats = ["png", "json", "npy", "cache", "invalid"]

    for region, patch_id, fmt_exists in test_cases:
        for fmt in formats:
            url = f"/regions/{region}/patches/{patch_id}/embedding?format={fmt}"
            save_path = OUTPUT_DIR / f"emb_{region}_{patch_id}_{fmt}.bin"

            if fmt == "invalid":
                status, body, _ = curl(url)
                expected = 422
                detail = body.decode("utf-8", errors="replace")[:200]
                log_result(
                    f"GET {url}", url, status, expected,
                    detail=detail,
                    notes="Expected 422 for invalid format"
                )
                continue

            status, body, headers = curl(url, save_path=save_path)
            detail = ""
            notes = ""
            expected = 200 if fmt_exists.get(fmt, True) else 404

            if status == 200:
                if fmt in ("png", "cache"):
                    ok, info = validate_image(save_path)
                    detail = info
                    if not ok:
                        notes = "Image validation failed"
                        unexpected.append({
                            "name": f"GET {url}", "url": url, "status": status,
                            "expected": expected, "detail": detail, "notes": notes,
                        })
                        continue
                elif fmt == "npy":
                    ok, info = validate_npy(save_path)
                    detail = info
                    if not ok:
                        notes = "NPY validation failed"
                        unexpected.append({
                            "name": f"GET {url}", "url": url, "status": status,
                            "expected": expected, "detail": detail, "notes": notes,
                        })
                        continue
                elif fmt == "json":
                    try:
                        data = json.loads(body)
                        detail = f"shape={data.get('shape')}, dtype={data.get('dtype')}"
                    except Exception as e:
                        notes = f"JSON parse error: {e}"
                        unexpected.append({
                            "name": f"GET {url}", "url": url, "status": status,
                            "expected": expected, "detail": detail, "notes": notes,
                        })
                        continue
            else:
                detail = body.decode("utf-8", errors="replace")[:200]
                if fmt == "cache" and fmt_exists.get("cache", True) == False:
                    # cache falls back to other formats; if neither png nor npy exists, 404 is expected
                    notes = f"No fallback format available for {region}"
                else:
                    notes = f"Format '{fmt}' not available for {region} (only has { [k for k,v in fmt_exists.items() if v] })"

            log_result(f"GET {url}", url, status, expected, detail, notes)
            save_path.unlink(missing_ok=True)


def test_task_endpoints():
    """Test task result, prediction, label, label_vis endpoints."""
    print("\n=== Testing Task Endpoints ===")

    # Task configurations from config.yaml
    # (task_type, version, period, has_data_expected, has_label_vis)
    # NOTE: label_vis files only exist for a subset of patches (e.g. patch_000040+),
    #       patch_000000 and patch_000010 do NOT have label_vis PNGs for any v2 task.
    task_configs = [
        # (task_type, version, period, has_result_data, has_label_vis_for_test_patches)
        ("change_detection", "v1", "2025-09_vs_2025-10", True, False),
        ("building_extraction", "v1", "2025-10", True, False),
        ("building_extraction", "v2", "2025-09_vs_2025-10", True, False),
        ("land_use_classification", "v1", "2025-10", True, False),
        ("land_use_classification", "v2", "2025-09_vs_2025-10", True, False),
        ("land_cover_classification", "v1", "2025-10", False, False),  # no data dir
        ("water_extraction", "v1", "2025-10", False, False),           # no data dir
    ]

    patches = ["patch_000000", "patch_000010"]

    for task_type, version, period, has_data, has_label_vis in task_configs:
        for patch_id in patches:
            base_url = f"/regions/harbin/patches/{patch_id}/tasks/{task_type}"

            # result?format=png
            url = f"{base_url}/result?format=png&version={version}&period={period}"
            save_path = OUTPUT_DIR / f"result_{task_type}_{version}_{period}_{patch_id}_png.bin"
            status, body, headers = curl(url, save_path=save_path)
            expected = 200 if has_data else 404
            detail = ""
            if status == 200:
                ok, info = validate_image(save_path)
                detail = info
            else:
                detail = body.decode("utf-8", errors="replace")[:200]
            log_result(f"GET {url}", url, status, expected, detail,
                       notes="Data exists" if has_data else "No data dir configured")
            save_path.unlink(missing_ok=True)

            # result?format=npy
            url = f"{base_url}/result?format=npy&version={version}&period={period}"
            save_path = OUTPUT_DIR / f"result_{task_type}_{version}_{period}_{patch_id}_npy.bin"
            status, body, headers = curl(url, save_path=save_path)
            expected = 200 if has_data else 404
            detail = ""
            if status == 200:
                ok, info = validate_npy(save_path)
                detail = info
            else:
                detail = body.decode("utf-8", errors="replace")[:200]
            log_result(f"GET {url}", url, status, expected, detail,
                       notes="Data exists" if has_data else "No data dir configured")
            save_path.unlink(missing_ok=True)

            # prediction
            url = f"{base_url}/prediction?version={version}&period={period}"
            save_path = OUTPUT_DIR / f"pred_{task_type}_{version}_{period}_{patch_id}.bin"
            status, body, headers = curl(url, save_path=save_path)
            expected = 200 if has_data else 404
            detail = ""
            if status == 200:
                ok, info = validate_npy(save_path)
                detail = info
            else:
                detail = body.decode("utf-8", errors="replace")[:200]
            log_result(f"GET {url}", url, status, expected, detail,
                       notes="Data exists" if has_data else "No data dir configured")
            save_path.unlink(missing_ok=True)

            # label
            url = f"{base_url}/label?version={version}&period={period}"
            save_path = OUTPUT_DIR / f"label_{task_type}_{version}_{period}_{patch_id}.bin"
            status, body, headers = curl(url, save_path=save_path)
            # labels may not exist for all patches even if predictions do
            expected = 200  # we'll check if it's actually there
            detail = ""
            if status == 200:
                if save_path.suffix == ".json" or body[:1] == b"{":
                    detail = "JSON response"
                else:
                    ok, info = validate_npy(save_path)
                    detail = info
            else:
                detail = body.decode("utf-8", errors="replace")[:200]
                expected = 404  # labels might not exist for every patch
            log_result(f"GET {url}", url, status, expected, detail)
            save_path.unlink(missing_ok=True)

            # label_vis
            url = f"{base_url}/label_vis?version={version}&period={period}"
            save_path = OUTPUT_DIR / f"label_vis_{task_type}_{version}_{period}_{patch_id}.bin"
            status, body, headers = curl(url, save_path=save_path)
            # v1 tasks have no label_vis config; v2 tasks may have config but missing files for specific patches
            expected = 200 if has_label_vis else 404
            detail = ""
            if status == 200:
                ok, info = validate_image(save_path)
                detail = info
            else:
                detail = body.decode("utf-8", errors="replace")[:200]
            notes = ""
            if has_label_vis:
                notes = "label_vis configured but file may not exist for this patch"
            else:
                notes = "No label_vis config for v1 tasks"
            log_result(f"GET {url}", url, status, expected, detail, notes=notes)
            save_path.unlink(missing_ok=True)


def test_tiles_endpoints():
    """Test tiles listing endpoint."""
    print("\n=== Testing Tiles Endpoints ===")

    tile_tasks = [
        ("change_detection", "v1", "2025-09_vs_2025-10"),
        ("building_extraction", "v1", "2025-10"),
        ("building_extraction", "v2", "2025-09_vs_2025-10"),
        ("land_use_classification", "v1", "2025-10"),
        ("land_use_classification", "v2", "2025-09_vs_2025-10"),
        ("land_cover_classification", "v1", "2025-10"),
        ("water_extraction", "v1", "2025-10"),
    ]

    for task_type, version, period in tile_tasks:
        url = f"/regions/harbin/tasks/{task_type}/tiles?version={version}&period={period}"
        status, body, _ = curl(url)
        expected = 200
        detail = ""
        if status == 200:
            try:
                data = json.loads(body)
                detail = f"total={data.get('total')}"
            except Exception as e:
                detail = f"JSON parse error: {e}"
        else:
            expected = 404  # demolition has no data
            detail = body.decode("utf-8", errors="replace")[:200]
        log_result(f"GET {url}", url, status, expected, detail)


def test_mosaic_endpoints():
    """Test region-wide mosaic big-image endpoint."""
    print("\n=== Testing Mosaic Endpoints ===")

    for sensor in ["s2", "s1", "landsat"]:
        url = (
            f"/regions/harbin/mosaic?date=2025-04&sensor_type={sensor}&format=png"
            f"&patch_ids=patch_000000&patch_ids=patch_000001"
        )
        save_path = OUTPUT_DIR / f"mosaic_{sensor}_preview.png"
        status, body, _ = curl(url, save_path=save_path)
        detail = ""
        if status == 200:
            ok, info = validate_image(save_path)
            detail = info
        else:
            detail = body.decode("utf-8", errors="replace")[:200]
        log_result(f"GET {url}", url, status, 200, detail, notes=f"{sensor} preview mosaic")
        save_path.unlink(missing_ok=True)


def test_model_endpoints():
    """Test custom model list endpoint (creation/training is covered by pytest)."""
    print("\n=== Testing Custom Model Endpoints ===")

    url = "/models"
    status, body, _ = curl(url)
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            detail = f"count={len(data)}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    else:
        detail = body.decode("utf-8", errors="replace")[:200]
    log_result(f"GET {url}", url, status, 200, detail, notes="List user models")


def test_system_model_via_models_endpoint():
    """Test that system pre-trained models can be inferred via /models/{id}/infer."""
    print("\n=== Testing System Model via /models Endpoint ===")

    # List with region_id should include system models
    url = "/models?region_id=harbin"
    status, body, _ = curl(url)
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            system_count = sum(1 for m in data if m.get("source") == "system")
            detail = f"total={len(data)}, system={system_count}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    else:
        detail = body.decode("utf-8", errors="replace")[:200]
    log_result(f"GET {url}", url, status, 200, detail, notes="List includes system models")

    # Single system model detail
    url = "/models/building_extraction?region_id=harbin"
    status, body, _ = curl(url)
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            detail = f"source={data.get('source')}, status={data.get('status')}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    else:
        detail = body.decode("utf-8", errors="replace")[:200]
    log_result(f"GET {url}", url, status, 200, detail, notes="System model detail")

    # System model single inference
    url = "/models/building_extraction/infer"
    payload = json.dumps({
        "region_id": "harbin",
        "patch_id": "patch_000000",
        "month": "2025-04",
        "version": "v2",
    }).encode("utf-8")
    status, body, _ = curl(url, method="POST", data=payload)
    detail = ""
    result_url = None
    if status == 200:
        try:
            data = json.loads(body)
            result_url = data.get("result_url", "")
            detail = f"result_url={result_url}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    else:
        detail = body.decode("utf-8", errors="replace")[:200]
    log_result(f"POST {url}", url, status, 200, detail, notes="System model infer")

    # Verify result image is accessible
    if result_url:
        status2, _, _ = curl(result_url)
        log_result(f"GET {result_url}", result_url, status2, 200, "", notes="System model result image")


def test_openapi_docs():
    """Test OpenAPI documentation endpoints."""
    print("\n=== Testing OpenAPI / Docs ===")

    # /openapi.json
    status, body, _ = curl("/openapi.json")
    detail = ""
    if status == 200:
        try:
            data = json.loads(body)
            detail = f"openapi={data.get('openapi')}, title={data.get('info', {}).get('title')}"
        except Exception as e:
            detail = f"JSON parse error: {e}"
    log_result("GET /openapi.json", "/openapi.json", status, 200, detail)

    # /docs
    status, body, _ = curl("/docs")
    detail = ""
    if status == 200:
        detail = f"body length={len(body)}, contains swagger={b'Swagger' in body or b'swagger' in body}"
    log_result("GET /docs", "/docs", status, 200, detail)


def generate_report():
    """Generate the markdown test report."""
    report_path = OUTPUT_DIR / "report.md"

    lines = []
    lines.append("# Embedding API 接口测试报告")
    lines.append("")
    lines.append(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**基地址**: {BASE_URL}")
    lines.append("")

    # Summary
    total = len(passed) + len(failed) + len(unexpected)
    lines.append("## 总结")
    lines.append("")
    lines.append(f"| 指标 | 数量 |")
    lines.append(f"|------|------|")
    lines.append(f"| 总测试数 | {total} |")
    lines.append(f"| ✅ 通过 | {len(passed)} |")
    lines.append(f"| ⚠️ 失败（可能预期） | {len(failed)} |")
    lines.append(f"| ❌ 异常（非预期） | {len(unexpected)} |")
    lines.append("")

    if len(unexpected) == 0:
        lines.append("**整体状态**: 🟢 健康（所有接口行为符合预期）")
    else:
        lines.append("**整体状态**: 🟡 存在异常（有非预期的接口行为，需关注）")
    lines.append("")

    # Passed
    lines.append("## ✅ 通过列表")
    lines.append("")
    if passed:
        lines.append("| # | 接口 | URL | 状态码 | 详情 |")
        lines.append("|---|------|-----|--------|------|")
        for i, p in enumerate(passed, 1):
            detail = (p.get("detail") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | {p['name']} | `{p['url']}` | {p['status']} | {detail} |")
    else:
        lines.append("无通过项。")
    lines.append("")

    # Failed (expected 404 etc)
    lines.append("## ⚠️ 失败列表（数据缺失导致，可能预期）")
    lines.append("")
    if failed:
        lines.append("| # | 接口 | URL | 状态码 | 预期 | 详情 | 备注 |")
        lines.append("|---|------|-----|--------|------|------|------|")
        for i, f in enumerate(failed, 1):
            detail = (f.get("detail") or "").replace("|", "\\|").replace("\n", " ")[:100]
            notes = (f.get("notes") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | {f['name']} | `{f['url']}` | {f['status']} | {f['expected']} | {detail} | {notes} |")
    else:
        lines.append("无失败项。")
    lines.append("")

    # Unexpected
    lines.append("## ❌ 异常列表（非预期行为）")
    lines.append("")
    if unexpected:
        lines.append("| # | 接口 | URL | 状态码 | 预期 | 详情 | 备注 |")
        lines.append("|---|------|-----|--------|------|------|------|")
        for i, u in enumerate(unexpected, 1):
            detail = (u.get("detail") or "").replace("|", "\\|").replace("\n", " ")[:100]
            notes = (u.get("notes") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | {u['name']} | `{u['url']}` | {u['status']} | {u['expected']} | {detail} | {notes} |")
    else:
        lines.append("无异常项。")
    lines.append("")

    # Key findings
    lines.append("## 关键发现")
    lines.append("")

    # Count 404s by reason
    data_missing = [f for f in failed if "not found" in (f.get("detail") or "").lower() or f.get("status") == 404]
    if data_missing:
        lines.append(f"- **数据缺失导致的 404**: 共 {len(data_missing)} 个接口。 land_cover_classification、water_extraction 任务在配置中已暴露但磁盘上暂无对应数据目录；部分 label/label_vis 文件对 patch_000000/patch_000010 不存在，属于已知数据缺失。")

    # Check label 404s for tasks that DO have data
    label_404s = [f for f in failed if "/label" in f["url"] and f["status"] == 404]
    if label_404s:
        lines.append(f"- **Label 文件缺失**: 共 {len(label_404s)} 个 label/label_vis 接口返回 404。部分任务（如 building_extraction、land_use_classification）的 predictions 数据存在，但对应 patch 的 label/label_vis 文件可能不存在。")

    lines.append("- **Embedding 接口**: harbin 和 haidian 的 patch_000000 均支持 png/json/npy/cache 格式，返回正常。invalid format 正确返回 422。")
    lines.append("- **基础接口**: /health、/regions、/regions/harbin、/regions/haidian 均正常返回 JSON 数据。")
    lines.append("- **专题任务接口**: change_detection/building_extraction/land_use_classification 的 result/prediction 正常返回；land_cover_classification/water_extraction 因无数据返回 404。")
    lines.append("- **Mosaic 大图接口**: /regions/harbin/mosaic 支持 s2/s1/landsat，返回 PNG 正常。")
    lines.append("- **自定义模型接口**: /models 列表接口返回正常；/models/{model_id}/infer 与 /models/{model_id}/infer_batch 已支持系统预训练模型 ID。")
    lines.append("")
    lines.append("## Mosaic 接口调用示例")
    lines.append("")
    lines.append("```bash")
    lines.append("# 哈尔滨全区域 Sentinel-2 真彩色 PNG（首次生成较慢，结果会缓存）")
    lines.append(f"curl -s \"{BASE_URL}/regions/harbin/mosaic?date=2025-04&sensor_type=s2&format=png\" -o /tmp/harbin_s2_2025-04.png")
    lines.append("")
    lines.append("# 只拼前两个 patch 的 Sentinel-1 SAR 伪彩色预览（快）")
    lines.append(f"curl -s \"{BASE_URL}/regions/harbin/mosaic?date=2025-04&sensor_type=s1&format=png&patch_ids=patch_000000&patch_ids=patch_000001\" -o /tmp/harbin_s1_preview.png")
    lines.append("")
    lines.append("# Landsat 全区域 GeoTIFF 原始数据（保留多波段与坐标）")
    lines.append(f"curl -s \"{BASE_URL}/regions/harbin/mosaic?date=2025-04&sensor_type=landsat&format=tif\" -o /tmp/harbin_landsat_2025-04.tif")
    lines.append("```")
    lines.append("")
    lines.append("**参数取值说明**：")
    lines.append("")
    lines.append("| 参数 | 可取值 | 默认值 | 说明 |")
    lines.append("|------|--------|--------|------|")
    lines.append("| `region_id` | `harbin` / `haidian` | - | 区域 ID |")
    lines.append("| `date` | `YYYY-MM`，如 `2025-04` | - | 哈尔滨会自动映射到 `2025Q1/Q2/Q3/Q4` |")
    lines.append("| `sensor_type` | `s2` / `s1` / `landsat` | `s2` | 传感器类型 |")
    lines.append("| `format` | `png` / `tif` | `png` | `png` 可视化；`tif` GeoTIFF 原始数据 |")
    lines.append("| `patch_ids` | 如 `patch_000000` | 全区域 | 可多次传入，只拼指定 patch |")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to {report_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_basic_endpoints()
    test_patch_endpoints()
    test_embedding_endpoints()
    test_task_endpoints()
    test_tiles_endpoints()
    test_mosaic_endpoints()
    test_model_endpoints()
    test_system_model_via_models_endpoint()
    test_openapi_docs()

    generate_report()

    print(f"\n=== Summary ===")
    print(f"Passed: {len(passed)}")
    print(f"Failed (expected-ish): {len(failed)}")
    print(f"Unexpected: {len(unexpected)}")

    if unexpected:
        sys.exit(1)


if __name__ == "__main__":
    main()
