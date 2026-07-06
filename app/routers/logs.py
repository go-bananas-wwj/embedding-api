"""Browser-accessible local log pages for integration debugging."""

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/logs", tags=["Logs"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
SKIP_PATH_PREFIXES = (
    "/logs",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/health",
)


def _resolve_log_path(filename: str) -> Path:
    """Resolve a log file path while keeping access inside logs/."""
    if "\x00" in filename:
        raise HTTPException(status_code=400, detail="Invalid log filename")
    target = (LOG_DIR / filename).resolve()
    try:
        target.relative_to(LOG_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid log filename")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Log file not found")
    return target


def _read_recent_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = {"parse_error": True, "raw": line.strip()}
            if str(record.get("path", "")).startswith(SKIP_PATH_PREFIXES):
                continue
            records.append(record)
            if len(records) > limit:
                records = records[-limit:]
    return records


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def logs_home() -> FileResponse:
    """Show the log landing page."""
    return FileResponse(_resolve_log_path("index.html"), media_type="text/html")


@router.get("/request-audit", include_in_schema=False)
@router.get("/request_audit", include_in_schema=False)
async def request_audit_page() -> FileResponse:
    """Show recent frontend request parameters and responses."""
    return FileResponse(_resolve_log_path("request_audit.html"), media_type="text/html")


@router.get("/request-audit-data", include_in_schema=False)
async def request_audit_data(
    limit: int = Query(200, ge=1, le=1000),
) -> JSONResponse:
    """Return recent request audit records for the browser page."""
    path = LOG_DIR / "request_audit.jsonl"
    return JSONResponse(
        {
            "records": _read_recent_jsonl(path, limit),
            "source": str(path),
            "limit": limit,
        }
    )


@router.get("/{filename:path}", include_in_schema=False)
async def get_log_file(filename: str) -> FileResponse:
    """Serve raw log files from logs/ for local debugging."""
    path = _resolve_log_path(filename)
    media_type = "text/html" if path.suffix == ".html" else "text/plain"
    return FileResponse(path, media_type=media_type)
