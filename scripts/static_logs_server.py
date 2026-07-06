#!/usr/bin/env python3
"""Static file server with clean log URLs for local API review."""

import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SKIP_PATH_PREFIXES = (
    "/logs",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/health",
)


class CleanLogRequestHandler(SimpleHTTPRequestHandler):
    """Serve /logs and /logs/request-audit without requiring .html suffixes."""

    def _send_json(self, payload) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_recent_audit_records(self, parsed) -> None:
        query = parse_qs(parsed.query)
        try:
            limit = max(1, min(int(query.get("limit", ["200"])[0]), 1000))
        except ValueError:
            limit = 200

        log_path = Path(self.directory) / "logs" / "request_audit.jsonl"
        if not log_path.exists():
            self._send_json({"records": [], "source": str(log_path), "limit": limit})
            return

        records = []
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
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
        self._send_json({"records": records, "source": str(log_path), "limit": limit})

    def send_head(self):
        parsed = urlparse(self.path)
        clean_path = parsed.path.rstrip("/")
        if clean_path == "/logs/request-audit-data":
            self._send_recent_audit_records(parsed)
            return None
        if clean_path == "/logs":
            self.path = "/logs/index.html"
        elif clean_path in {"/logs/request-audit", "/logs/request_audit"}:
            self.path = "/logs/request_audit.html"
        return super().send_head()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve project static review pages.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    os.chdir(args.root)
    handler = partial(CleanLogRequestHandler, directory=args.root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.root} at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
