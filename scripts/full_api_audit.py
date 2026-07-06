#!/usr/bin/env python3
"""Call every public OpenAPI endpoint and save the responses locally.

This script is intentionally black-box: it talks to a running embedding-api
service, records every response, and writes a machine-readable summary. It uses
the current OpenAPI document as the authoritative endpoint list and fails if any
route is not exercised.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from PIL import Image


BASE_URL = "http://127.0.0.1:9061"
REGION = "haidian"
PATCH_ID = "patch_000000"
PATCH_ID_2 = "patch_000001"
MONTH = "202512"
SAM3_POINT = [116.0954, 40.0628]
EXPECTED_STUBS = {
    "GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png"
}


class AuditError(RuntimeError):
    """Raised when an endpoint response does not match its expected contract."""


def _json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_response(out_dir: Path, index: int, name: str, response: requests.Response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    suffix = ".json" if "json" in content_type else ".bin"
    if "image/png" in content_type:
        suffix = ".png"
    elif "octet-stream" in content_type:
        suffix = ".npy"
    path = out_dir / f"{index:02d}_{safe_name}{suffix}"

    item: Dict[str, Any] = {
        "name": name,
        "method": response.request.method,
        "url": response.url,
        "status_code": response.status_code,
        "content_type": content_type,
        "elapsed_ms": int(response.elapsed.total_seconds() * 1000),
        "file": str(path),
    }
    if "json" in content_type:
        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text}
        _json_dump(path, data)
        if isinstance(data, dict):
            item["keys"] = sorted(data.keys())
            if "features" in data:
                item["feature_count"] = len(data.get("features") or [])
            if "results" in data:
                item["result_count"] = len(data.get("results") or [])
            for key in ("total", "success_count", "error_count", "status"):
                if key in data:
                    item[key] = data[key]
        elif isinstance(data, list):
            item["length"] = len(data)
    else:
        path.write_bytes(response.content)
        item["bytes"] = len(response.content)
        if "image/png" in content_type:
            try:
                with Image.open(path) as img:
                    item["image_width"] = img.width
                    item["image_height"] = img.height
            except Exception as exc:
                item["image_error"] = str(exc)
    return item


def _assert_json_has(item: Dict[str, Any], data: Any, keys: Iterable[str]) -> None:
    if not isinstance(data, dict):
        raise AuditError(f"{item['name']} expected JSON object")
    missing = [key for key in keys if key not in data]
    if missing:
        raise AuditError(f"{item['name']} missing keys: {missing}")


def _request(
    session: requests.Session,
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> requests.Response:
    return session.request(
        method,
        urljoin(BASE_URL, path),
        params=params,
        json=json_body,
        timeout=timeout,
    )


def _expect_2xx(name: str, response: requests.Response) -> Any:
    if not 200 <= response.status_code < 300:
        raise AuditError(f"{name} expected 2xx, got {response.status_code}: {response.text[:300]}")
    if "json" in response.headers.get("content-type", ""):
        return response.json()
    return None


def _expect_png_or_bin(name: str, response: requests.Response) -> None:
    _expect_2xx(name, response)
    if not response.content:
        raise AuditError(f"{name} returned empty binary body")
    if "image/png" in response.headers.get("content-type", ""):
        try:
            from io import BytesIO

            with Image.open(BytesIO(response.content)) as img:
                if img.width <= 0 or img.height <= 0:
                    raise AuditError(f"{name} returned invalid image size")
        except AuditError:
            raise
        except Exception as exc:
            raise AuditError(f"{name} returned unreadable PNG: {exc}") from exc


def _create_training_payload(name: str) -> Dict[str, Any]:
    features = [
        {
            "type": "Feature",
            "properties": {
                "patch_id": PATCH_ID,
                "region_id": REGION,
                "class_id": "target",
                "class_name": "目标",
                "color": "#ff0000",
                "task_type": "building_extraction",
                "month": MONTH,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [116.241, 39.886],
                    [116.247, 39.886],
                    [116.247, 39.892],
                    [116.241, 39.892],
                    [116.241, 39.886],
                ]],
            },
        },
        {
            "type": "Feature",
            "properties": {
                "patch_id": PATCH_ID,
                "region_id": REGION,
                "class_id": "background",
                "class_name": "背景",
                "color": "#808080",
                "task_type": "building_extraction",
                "month": MONTH,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [116.248, 39.892],
                    [116.253, 39.892],
                    [116.253, 39.896],
                    [116.248, 39.896],
                    [116.248, 39.892],
                ]],
            },
        },
    ]
    return {
        "name": name,
        "model_type": "classification",
        "task_type": "building_extraction",
        "region_id": REGION,
        "embedding_version": "v1",
        "epochs": 50,
        "class_ids": ["background", "target"],
        "annotations": {"type": "FeatureCollection", "features": features},
        "classes": [
            {"id": "background", "name": "背景", "color": "#808080"},
            {"id": "target", "name": "目标", "color": "#ff0000"},
        ],
    }


def main() -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(os.environ.get("API_AUDIT_ROOT", "audit_results"))
    out_dir = output_root / f"api_full_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    summary: List[Dict[str, Any]] = []
    failures: List[str] = []
    covered: List[str] = []
    index = 1

    spec = _request(session, "GET", "/openapi.json", timeout=30).json()
    openapi_ops = {
        f"{method.upper()} {path}"
        for path, methods in spec["paths"].items()
        for method in methods
    }
    _json_dump(out_dir / "openapi_paths.json", sorted(openapi_ops))

    def call(
        name: str,
        operation: str,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        expect_status: Optional[int] = None,
        required_keys: Iterable[str] = (),
        binary: bool = False,
    ) -> Any:
        nonlocal index
        response = _request(session, method, path, params=params, json_body=json_body, timeout=timeout)
        item = _save_response(out_dir, index, name, response)
        item["operation"] = operation
        summary.append(item)
        covered.append(operation)
        index += 1
        try:
            if expect_status is not None:
                if response.status_code != expect_status:
                    raise AuditError(
                        f"{name} expected {expect_status}, got {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                data = response.json() if "json" in response.headers.get("content-type", "") else None
            elif binary:
                _expect_png_or_bin(name, response)
                data = None
            else:
                data = _expect_2xx(name, response)
            if required_keys:
                _assert_json_has(item, data, required_keys)
            item["ok"] = True
            return data
        except Exception as exc:
            item["ok"] = False
            item["failure"] = str(exc)
            failures.append(str(exc))
        return None

    def assert_non_empty_versions(name: str, data: Any) -> None:
        if not isinstance(data, list):
            failures.append(f"{name} expected list response")
            return
        empty = [item.get("id") for item in data if not item.get("versions")]
        if empty:
            failures.append(f"{name} returned unavailable models with empty versions: {empty}")

    def assert_batch_success(name: str, data: Any, expected_total: int) -> None:
        if not isinstance(data, dict):
            failures.append(f"{name} expected JSON object")
            return
        if data.get("total") != expected_total:
            failures.append(f"{name} total mismatch: {data.get('total')} != {expected_total}")
        if data.get("success_count") != expected_total or data.get("error_count") != 0:
            failures.append(f"{name} did not fully succeed: {data}")
        for result in data.get("results", []):
            if result.get("status") != "success" or not result.get("result_url"):
                failures.append(f"{name} has bad result item: {result}")

    # Core region and patch APIs.
    call("health", "GET /health", "GET", "/health", required_keys=["status", "regions"])
    call("regions", "GET /regions", "GET", "/regions", required_keys=["regions"])
    call("region_detail", "GET /regions/{region_id}", "GET", f"/regions/{REGION}", required_keys=["id", "tasks"])
    call(
        "patches",
        "GET /regions/{region_id}/patches",
        "GET",
        f"/regions/{REGION}/patches",
        params={"page": 1, "page_size": 2},
        required_keys=["patches", "total"],
    )
    call(
        "patch_detail",
        "GET /regions/{region_id}/patches/{patch_id}",
        "GET",
        f"/regions/{REGION}/patches/{PATCH_ID}",
        required_keys=["patch_id", "bounds_wgs84"],
    )
    call(
        "embedding_json",
        "GET /regions/{region_id}/patches/{patch_id}/embedding",
        "GET",
        f"/regions/{REGION}/patches/{PATCH_ID}/embedding",
        params={"format": "json", "version": "v1", "month": MONTH},
        required_keys=["patch_id", "shape", "dtype"],
    )
    call(
        "mosaic_png",
        "GET /regions/{region_id}/mosaic",
        "GET",
        f"/regions/{REGION}/mosaic",
        params={"date": MONTH, "sensor_type": "s2", "format": "png", "patch_ids": PATCH_ID},
        binary=True,
        timeout=180,
    )

    # Task APIs.
    call("tasks", "GET /regions/{region_id}/tasks", "GET", f"/regions/{REGION}/tasks", required_keys=["tasks"])
    call(
        "task_summary",
        "GET /regions/{region_id}/tasks/{task_type}/summary",
        "GET",
        f"/regions/{REGION}/tasks/construction/summary",
        params={"version": "v1"},
        required_keys=["task", "version"],
    )
    call(
        "task_result_png",
        "GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result",
        "GET",
        f"/regions/{REGION}/patches/{PATCH_ID}/tasks/building_extraction/result",
        params={"format": "png", "version": "v1", "month": MONTH},
        binary=True,
    )
    call(
        "task_prediction_npy",
        "GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/prediction",
        "GET",
        f"/regions/{REGION}/patches/{PATCH_ID}/tasks/construction/prediction",
        params={"version": "v1", "period": MONTH},
        binary=True,
    )
    call(
        "task_label_json",
        "GET /regions/{region_id}/patches/{patch_id}/tasks/{task_type}/label",
        "GET",
        f"/regions/{REGION}/patches/{PATCH_ID}/tasks/construction/label",
        params={"version": "v1", "period": MONTH},
        binary=True,
    )
    tiles = call(
        "task_tiles",
        "GET /regions/{region_id}/tasks/{task_type}/tiles",
        "GET",
        f"/regions/{REGION}/tasks/building_extraction/tiles",
        params={"version": "v1", "period": MONTH},
        required_keys=["tiles", "total"],
    )
    tile_filename = (tiles or {}).get("tiles", [{}])[0].get("filename", f"{PATCH_ID}.png")
    call(
        "task_tile_by_filename",
        "GET /regions/{region_id}/tasks/{task_type}/tiles/{filename}",
        "GET",
        f"/regions/{REGION}/tasks/building_extraction/tiles/{tile_filename}",
        params={"version": "v1", "period": MONTH},
        binary=True,
    )
    call(
        "task_xyz_tile_stub",
        "GET /regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png",
        "GET",
        f"/regions/{REGION}/tasks/building_extraction/tiles/0/0/0.png",
        params={"version": "v1", "period": MONTH},
        expect_status=501,
        required_keys=["detail"],
    )

    # System model APIs and result download.
    system_models = call(
        "system_models",
        "GET /system-models",
        "GET",
        "/system-models",
        params={"region_id": REGION},
    )
    assert_non_empty_versions("system_models", system_models)
    call(
        "system_model_classes",
        "GET /system-models/{task_id}/classes",
        "GET",
        "/system-models/water_extraction/classes",
        params={"region_id": REGION, "version": "v1"},
    )
    system_infer = call(
        "system_model_infer",
        "POST /system-models/{task_id}/infer",
        "POST",
        "/system-models/water_extraction/infer",
        params={"region_id": REGION, "patch_id": PATCH_ID, "month": MONTH, "version": "v1"},
        required_keys=["result_url"],
        timeout=180,
    )
    if system_infer and system_infer.get("result_url"):
        filename = Path(system_infer["result_url"]).name
        call(
            "system_model_result_png",
            "GET /system-models/results/{filename}",
            "GET",
            f"/system-models/results/{filename}",
            binary=True,
        )

    # Model APIs: list, create, job, get, patch, infer, batch, result, delete.
    models_list = call("models_list", "GET /models", "GET", "/models", params={"region_id": REGION})
    if isinstance(models_list, list):
        empty_system = [
            item.get("id")
            for item in models_list
            if item.get("source") == "system" and not item.get("versions")
        ]
        if empty_system:
            failures.append(f"models_list returned unavailable system models: {empty_system}")
    model_data = call(
        "model_create",
        "POST /models",
        "POST",
        "/models",
        json_body=_create_training_payload(f"full_api_audit_{timestamp}"),
        required_keys=["id", "job_id", "status"],
        timeout=120,
    )
    model_id = (model_data or {}).get("id")
    job_id = (model_data or {}).get("job_id")
    if job_id:
        job_data = None
        for attempt in range(30):
            job_data = call(
                f"model_job_{attempt}",
                "GET /models/jobs/{job_id}",
                "GET",
                f"/models/jobs/{job_id}",
                required_keys=["job_id", "status", "model_id"],
            )
            if job_data and job_data.get("status") in {"completed", "failed"}:
                break
            time.sleep(1)
        if not job_data or job_data.get("status") != "completed":
            failures.append(f"model training did not complete: {job_data}")
    if model_id:
        call(
            "model_get",
            "GET /models/{model_id}",
            "GET",
            f"/models/{model_id}",
            required_keys=["id", "status"],
        )
        call(
            "model_rename",
            "PATCH /models/{model_id}",
            "PATCH",
            f"/models/{model_id}",
            json_body={"name": f"full_api_audit_renamed_{timestamp}"},
            required_keys=["status"],
        )
        custom_infer = call(
            "model_infer",
            "POST /models/{model_id}/infer",
            "POST",
            f"/models/{model_id}/infer",
            json_body={"region_id": REGION, "patch_id": PATCH_ID, "month": MONTH},
            required_keys=["result_url"],
            timeout=180,
        )
        if custom_infer and custom_infer.get("result_url"):
            filename = Path(custom_infer["result_url"]).name
            call(
                "model_result_png",
                "GET /models/results/{filename}",
                "GET",
                f"/models/results/{filename}",
                binary=True,
            )
        batch_data = call(
            "model_infer_batch",
            "POST /models/{model_id}/infer_batch",
            "POST",
            f"/models/{model_id}/infer_batch",
            json_body={
                "region_id": REGION,
                "patch_ids": [PATCH_ID, PATCH_ID_2],
                "month": MONTH,
            },
            required_keys=["total", "success_count", "error_count", "results"],
            timeout=180,
        )
        assert_batch_success("model_infer_batch", batch_data, 2)
        if isinstance(batch_data, dict):
            for idx, result in enumerate(batch_data.get("results", [])[:1]):
                if result.get("result_url"):
                    filename = Path(result["result_url"]).name
                    call(
                        f"model_batch_result_png_{idx}",
                        "GET /models/results/{filename}",
                        "GET",
                        f"/models/results/{filename}",
                        binary=True,
                    )
        call(
            "model_delete",
            "DELETE /models/{model_id}",
            "DELETE",
            f"/models/{model_id}",
            required_keys=["status"],
        )

    # Unified /models facade for system tasks.
    call(
        "models_system_get",
        "GET /models/{model_id}",
        "GET",
        "/models/water_extraction",
        params={"region_id": REGION, "version": "v1"},
        required_keys=["id", "status", "source"],
    )
    models_system_infer = call(
        "models_system_infer",
        "POST /models/{model_id}/infer",
        "POST",
        "/models/water_extraction/infer",
        json_body={"region_id": REGION, "patch_id": PATCH_ID, "month": MONTH, "version": "v1"},
        required_keys=["result_url"],
        timeout=180,
    )
    if models_system_infer and models_system_infer.get("result_url"):
        filename = Path(models_system_infer["result_url"]).name
        call(
            "models_system_result_png",
            "GET /system-models/results/{filename}",
            "GET",
            f"/system-models/results/{filename}",
            binary=True,
        )
    models_system_batch = call(
        "models_system_infer_batch",
        "POST /models/{model_id}/infer_batch",
        "POST",
        "/models/water_extraction/infer_batch",
        json_body={
            "region_id": REGION,
            "patch_ids": [PATCH_ID, PATCH_ID_2],
            "month": MONTH,
            "version": "v1",
        },
        required_keys=["total", "success_count", "error_count", "results"],
        timeout=180,
    )
    assert_batch_success("models_system_infer_batch", models_system_batch, 2)
    if isinstance(models_system_batch, dict):
        for idx, result in enumerate(models_system_batch.get("results", [])[:1]):
            if result.get("result_url"):
                filename = Path(result["result_url"]).name
                call(
                    f"models_system_batch_result_png_{idx}",
                    "GET /system-models/results/{filename}",
                    "GET",
                    f"/system-models/results/{filename}",
                    binary=True,
                )

    # SAM3 APIs.
    sam3_status_before = call(
        "sam3_status",
        "GET /regions/{region_id}/sam3/status",
        "GET",
        f"/regions/{REGION}/sam3/status",
        required_keys=["model_loaded", "cache"],
    )
    sam3_embed = call(
        "sam3_embed",
        "POST /regions/{region_id}/sam3/embed",
        "POST",
        f"/regions/{REGION}/sam3/embed",
        json_body={"patch_id": "patch_000212", "month": MONTH, "sensor_type": "s2"},
        required_keys=["embedding_id", "status", "image"],
        timeout=300,
    )
    if isinstance(sam3_embed, dict):
        image = sam3_embed.get("image") or {}
        if image.get("width") != 256 or image.get("height") != 256 or not image.get("data"):
            failures.append(f"sam3_embed returned invalid image payload: {image}")
    sam3_segment = call(
        "sam3_segment",
        "POST /regions/{region_id}/sam3/segment",
        "POST",
        f"/regions/{REGION}/sam3/segment",
        json_body={
            "date": MONTH,
            "sensor_type": "s2",
            "point_coords": [SAM3_POINT],
            "multimask_output": False,
            "include_masks": False,
        },
        required_keys=["type", "features"],
        timeout=300,
    )
    if isinstance(sam3_segment, dict):
        features = sam3_segment.get("features") or []
        if sam3_segment.get("type") != "FeatureCollection" or not features:
            failures.append(f"sam3_segment returned invalid GeoJSON: {sam3_segment}")
        for feature in features:
            geometry = feature.get("geometry") or {}
            props = feature.get("properties") or {}
            if geometry.get("type") != "Polygon":
                failures.append(f"sam3_segment feature is not Polygon: {feature}")
            for key in ("score", "bbox", "bbox_wgs84", "patch_id", "sensor_type", "date"):
                if key not in props:
                    failures.append(f"sam3_segment missing property {key}: {feature}")
    sam3_segment_masks = call(
        "sam3_segment_include_masks",
        "POST /regions/{region_id}/sam3/segment",
        "POST",
        f"/regions/{REGION}/sam3/segment",
        json_body={
            "date": MONTH,
            "sensor_type": "s2",
            "point_coords": [SAM3_POINT],
            "multimask_output": False,
            "include_masks": True,
        },
        required_keys=["type", "features", "masks"],
        timeout=300,
    )
    if isinstance(sam3_segment_masks, dict):
        masks = sam3_segment_masks.get("masks") or []
        if not masks or not masks[0].get("data"):
            failures.append(f"sam3 include_masks returned no mask data: {sam3_segment_masks}")
    sam3_status_after = call(
        "sam3_status_after",
        "GET /regions/{region_id}/sam3/status",
        "GET",
        f"/regions/{REGION}/sam3/status",
        required_keys=["model_loaded", "cache"],
    )
    if isinstance(sam3_status_after, dict):
        cache = sam3_status_after.get("cache") or {}
        if not sam3_status_after.get("model_loaded"):
            failures.append(f"sam3 status after segment says model not loaded: {sam3_status_after}")
        if "haidian_patch_000212_s2_202512" not in cache.get("entries", []):
            failures.append(f"sam3 cache missing expected entry: {cache}")

    missing = sorted(openapi_ops - set(covered))
    extra = sorted(set(covered) - openapi_ops)
    if missing:
        failures.append(f"Missing OpenAPI operations: {missing}")
    if extra:
        failures.append(f"Unknown operations in audit plan: {extra}")

    # Stronger semantic checks for aggregate responses.
    for item in summary:
        if item["operation"] in EXPECTED_STUBS:
            item["expected_stub"] = True
            item["ok"] = item["status_code"] == 501
        elif item["status_code"] >= 400:
            item.setdefault("ok", False)

    ok = not failures and all(item.get("ok") for item in summary)
    report = {
        "ok": ok,
        "base_url": BASE_URL,
        "timestamp": timestamp,
        "total_openapi_operations": len(openapi_ops),
        "covered_operations": len(set(covered)),
        "response_count": len(summary),
        "expected_stubs": sorted(EXPECTED_STUBS),
        "missing_operations": missing,
        "extra_operations": extra,
        "failures": failures,
        "responses": summary,
    }
    _json_dump(out_dir / "summary.json", report)
    _json_dump(out_dir / "summary_passed.json", [item for item in summary if item.get("ok")])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SUMMARY_DIR {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
