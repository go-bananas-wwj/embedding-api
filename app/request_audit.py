"""Request audit logging for frontend/backend integration debugging."""

import json
import logging
import os
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Iterable, List, Tuple

from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

AUDIT_LOG_PATH = Path(os.environ.get("REQUEST_AUDIT_LOG", "logs/request_audit.jsonl"))
MAX_BODY_LOG_BYTES = int(os.environ.get("REQUEST_AUDIT_MAX_BODY_BYTES", "65536"))
MAX_VALUE_CHARS = int(os.environ.get("REQUEST_AUDIT_MAX_VALUE_CHARS", "2000"))
MAX_LOG_BYTES = int(os.environ.get("REQUEST_AUDIT_ROTATE_BYTES", str(20 * 1024 * 1024)))
BACKUP_COUNT = int(os.environ.get("REQUEST_AUDIT_BACKUP_COUNT", "5"))

_SKIP_PATH_PREFIXES = (
    "/logs",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/health",
)

_SENSITIVE_KEYS = {
    "authorization",
    "x-api-key",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "secret",
}


def _audit_logger() -> logging.Logger:
    """Return a dedicated JSONL logger that does not pollute uvicorn output."""
    logger = logging.getLogger("request_audit")
    if logger.handlers:
        return logger

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        AUDIT_LOG_PATH,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _truncate_text(value: str) -> str:
    if len(value) <= MAX_VALUE_CHARS:
        return value
    return value[:MAX_VALUE_CHARS] + f"...<truncated {len(value) - MAX_VALUE_CHARS} chars>"


def _scrub(value: Any, key: str = "") -> Any:
    """Remove secrets and cap very large values before writing audit logs."""
    if key.lower() in _SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, key) for v in value[:200]]
    if isinstance(value, tuple):
        return [_scrub(v, key) for v in value[:200]]
    if isinstance(value, str):
        return _truncate_text(value)
    return value


def _multi_items(items: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, value in items:
        safe_value = _scrub(value, key)
        if key in data:
            if not isinstance(data[key], list):
                data[key] = [data[key]]
            data[key].append(safe_value)
        else:
            data[key] = safe_value
    return data


def _selected_headers(request: Request) -> Dict[str, Any]:
    """Keep integration-relevant headers while redacting credentials."""
    header_names = [
        "content-type",
        "content-length",
        "user-agent",
        "origin",
        "referer",
        "x-api-key",
        "authorization",
    ]
    return {
        name: _scrub(request.headers.get(name), name)
        for name in header_names
        if request.headers.get(name) is not None
    }


def _body_snapshot(body: bytes, content_type: str) -> Dict[str, Any]:
    if not body:
        return {"size_bytes": 0, "content": None}

    snapshot: Dict[str, Any] = {"size_bytes": len(body)}
    if len(body) > MAX_BODY_LOG_BYTES:
        snapshot["truncated"] = True
        body = body[:MAX_BODY_LOG_BYTES]
    else:
        snapshot["truncated"] = False

    if "application/json" in content_type:
        try:
            snapshot["content"] = _scrub(json.loads(body.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            snapshot["content"] = _truncate_text(body.decode("utf-8", errors="replace"))
        return snapshot

    if content_type.startswith("text/") or "application/x-www-form-urlencoded" in content_type:
        snapshot["content"] = _truncate_text(body.decode("utf-8", errors="replace"))
        return snapshot

    snapshot["content"] = f"<{content_type or 'unknown'} body omitted>"
    return snapshot


async def request_audit_middleware(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """Log request parameters and response metadata for frontend debugging."""
    if request.url.path.startswith(_SKIP_PATH_PREFIXES):
        return await call_next(request)

    start = time.perf_counter()
    # Starlette caches the body for downstream handlers. Replacing
    # ``request._receive`` here used to replay ``http.request`` forever, which
    # breaks StreamingResponse while it is waiting for ``http.disconnect``.
    body = await request.body()
    status_code = 500
    error = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        content_type = request.headers.get("content-type", "")
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": request.method,
            "path": request.url.path,
            "query": _multi_items(request.query_params.multi_items()),
            "headers": _selected_headers(request),
            "body": _body_snapshot(body, content_type),
            "client": request.client.host if request.client else None,
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
        }
        if error:
            record["error"] = error
        _audit_logger().info(json.dumps(record, ensure_ascii=False, sort_keys=True))
