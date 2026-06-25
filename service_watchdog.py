#!/usr/bin/env python3
"""Watchdog for embedding-api service.

Monitors the embedding-api service on port 9061.
If the service becomes unresponsive, gracefully restarts the process.

Usage:
    python service_watchdog.py              # Run in foreground
    nohup python service_watchdog.py &      # Run in background
    python service_watchdog.py stop         # Stop the watchdog
"""

import os
import signal
import subprocess
import sys
import time
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embedding-watchdog")

PROJECT_DIR = "/workspace/embedding-api"
PID_FILE = os.path.join(PROJECT_DIR, "service_watchdog.pid")
CHECK_INTERVAL = 30  # seconds
HEALTH_URL = "http://localhost:9061/health"
UVICORN_CMD = [
    sys.executable, "-m", "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "9061",
    "--workers", "1",
]
RESTART_BACKOFFS = [5, 10, 30, 60, 120]  # seconds


class ServiceWatchdog:
    """Manages the embedding-api process with health checks and graceful restarts."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._consecutive_failures = 0

    def _is_healthy(self) -> bool:
        """Check if embedding-api is responding on port 9061."""
        try:
            result = subprocess.run(
                ["curl", "-s", "--connect-timeout", "3", HEALTH_URL],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and '"status":"ok"' in result.stdout
        except Exception:
            return False

    def _start_service(self) -> bool:
        """Start the embedding-api process."""
        logger.info("Starting embedding-api service...")
        self._stop_service(wait=True)
        self._process = subprocess.Popen(
            UVICORN_CMD,
            cwd=PROJECT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Wait for service to become healthy
        for _ in range(20):
            time.sleep(0.5)
            if self._is_healthy():
                logger.info("Service started successfully")
                self._consecutive_failures = 0
                return True
        logger.error("Service failed to become healthy after start")
        self._consecutive_failures += 1
        return False

    def _stop_service(self, wait: bool = False) -> None:
        """Stop the service process gracefully (SIGTERM, then SIGKILL)."""
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process = None
            return
        try:
            self._process.send_signal(signal.SIGTERM)
            if wait:
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Service did not terminate gracefully, sending SIGKILL")
                    self._process.kill()
                    self._process.wait()
        except ProcessLookupError:
            pass
        self._process = None

    def run(self) -> None:
        """Main watchdog loop."""
        existing_pid = read_pid()
        if existing_pid:
            try:
                os.kill(existing_pid, 0)
                logger.error(f"Watchdog already running (PID {existing_pid})")
                sys.exit(1)
            except ProcessLookupError:
                remove_pid()

        write_pid()

        def signal_handler(signum, _frame):
            logger.info(f"Watchdog received signal {signum}, shutting down...")
            self._stop_service(wait=True)
            remove_pid()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("=" * 50)
        logger.info("Embedding API Watchdog started")
        logger.info(f"Health check URL: {HEALTH_URL}")
        logger.info(f"Check interval: {CHECK_INTERVAL}s")
        logger.info("=" * 50)

        if not self._is_healthy():
            self._start_service()

        while True:
            if not self._is_healthy():
                backoff = RESTART_BACKOFFS[
                    min(self._consecutive_failures, len(RESTART_BACKOFFS) - 1)
                ]
                logger.warning(
                    f"Service not responding! Waiting {backoff}s before restart..."
                )
                time.sleep(backoff)
                self._start_service()
                if self._is_healthy():
                    logger.info("Service restarted successfully")
                else:
                    logger.error("Service restart failed! Will retry next cycle.")
                    self._consecutive_failures += 1
            else:
                logger.debug("Service is healthy")
                self._consecutive_failures = max(0, self._consecutive_failures - 1)
            time.sleep(CHECK_INTERVAL)


def write_pid() -> None:
    """Write PID atomically using O_EXCL."""
    try:
        with open(PID_FILE, "x") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        pass


def remove_pid() -> None:
    """Remove PID file if it exists."""
    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass


def read_pid() -> Optional[int]:
    """Read PID from file."""
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def stop_watchdog() -> None:
    """Stop the running watchdog process gracefully."""
    pid = read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to watchdog PID {pid}")
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    logger.info("Watchdog stopped")
                    remove_pid()
                    return
            logger.warning("Watchdog still alive, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            logger.info("Watchdog not running")
        remove_pid()
    else:
        logger.info("No PID file found, watchdog may not be running")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_watchdog()
    else:
        watchdog = ServiceWatchdog()
        watchdog.run()
