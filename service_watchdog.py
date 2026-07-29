#!/usr/bin/env python3
"""Watchdog for embedding-api service.

Monitors the embedding-api service on port 9061.
If the service becomes unresponsive, gracefully restarts the process.

Usage:
    python service_watchdog.py              # Run in foreground
    python service_watchdog.py start        # Run persistently in background
    python service_watchdog.py status       # Show watchdog/API status
    python service_watchdog.py stop         # Stop the watchdog
"""

import logging
import os
import signal
import subprocess
import sys
import time
from typing import Optional

PROJECT_DIR = "/workspace/projects/embedding-api"
PID_FILE = os.path.join(PROJECT_DIR, "service_watchdog.pid")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
WATCHDOG_LOG_FILE = os.path.join(LOG_DIR, "watchdog.log")
WATCHDOG_CONSOLE_FILE = os.path.join(LOG_DIR, "watchdog.console.log")
SERVICE_LOG_FILE = os.path.join(LOG_DIR, "uvicorn.log")
CHECK_INTERVAL = int(os.environ.get("WATCHDOG_CHECK_INTERVAL", "15"))
HEALTH_CONNECT_TIMEOUT = int(os.environ.get("WATCHDOG_CONNECT_TIMEOUT", "5"))
HEALTH_TOTAL_TIMEOUT = int(os.environ.get("WATCHDOG_HEALTH_TIMEOUT", "20"))
FAILURE_THRESHOLD = int(os.environ.get("WATCHDOG_FAILURE_THRESHOLD", "3"))
HEALTH_URL = "http://localhost:9061/health"
UVICORN_CMD = [
    sys.executable, "-m", "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "9061",
    "--workers", "1",
]

# Enable Swagger/ReDoc in production by default.
UVICORN_ENV = {
    **os.environ,
    "DOCS_URL": "/docs",
    "REDOC_URL": "/redoc",
}
RESTART_BACKOFFS = [5, 10, 30, 60, 120]  # seconds


def configure_logging() -> logging.Logger:
    """Configure console + file logging for watchdog diagnostics."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("embedding-watchdog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(WATCHDOG_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


logger = configure_logging()


class ServiceWatchdog:
    """Manages the embedding-api process with health checks and graceful restarts."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._consecutive_failures = 0

    def _is_healthy(self) -> bool:
        """Check if embedding-api is responding on port 9061."""
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "--connect-timeout",
                    str(HEALTH_CONNECT_TIMEOUT),
                    "--max-time",
                    str(HEALTH_TOTAL_TIMEOUT),
                    HEALTH_URL,
                ],
                capture_output=True,
                text=True,
                timeout=HEALTH_TOTAL_TIMEOUT + 2,
            )
            return result.returncode == 0 and '"status":"ok"' in result.stdout
        except Exception:
            return False

    def _start_service(self) -> bool:
        """Start the embedding-api process."""
        logger.info("Starting embedding-api service...")
        self._stop_service(wait=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        service_log = open(SERVICE_LOG_FILE, "ab", buffering=0)
        self._process = subprocess.Popen(
            UVICORN_CMD,
            cwd=PROJECT_DIR,
            env=UVICORN_ENV,
            stdout=service_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        service_log.close()
        logger.info("Service log: %s", SERVICE_LOG_FILE)
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
        logger.info(f"Failure threshold: {FAILURE_THRESHOLD}")
        logger.info(f"Watchdog log: {WATCHDOG_LOG_FILE}")
        logger.info(f"Service log: {SERVICE_LOG_FILE}")
        logger.info("=" * 50)

        if not self._is_healthy():
            self._start_service()

        while True:
            process_exited = (
                self._process is not None and self._process.poll() is not None
            )
            if process_exited:
                logger.error(
                    "Service process exited with code %s; restarting immediately",
                    self._process.returncode,
                )
                self._process = None
                self._consecutive_failures = FAILURE_THRESHOLD

            if not self._is_healthy():
                self._consecutive_failures += 1
                if self._consecutive_failures >= FAILURE_THRESHOLD:
                    backoff = RESTART_BACKOFFS[
                        min(
                            self._consecutive_failures - FAILURE_THRESHOLD,
                            len(RESTART_BACKOFFS) - 1,
                        )
                    ]
                    logger.warning(
                        "Service failed %s consecutive health checks; "
                        "waiting %ss before restart",
                        self._consecutive_failures,
                        backoff,
                    )
                    time.sleep(backoff)
                    self._start_service()
                    if self._is_healthy():
                        logger.info("Service restarted successfully")
                    else:
                        logger.error("Service restart failed; will retry next cycle")
                else:
                    logger.warning(
                        "Service health check failed (%s/%s); no restart yet",
                        self._consecutive_failures,
                        FAILURE_THRESHOLD,
                    )
            else:
                self._consecutive_failures = 0
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


def process_is_alive(pid: Optional[int]) -> bool:
    """Return whether a PID exists without sending it a signal."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def start_watchdog() -> None:
    """Launch a detached watchdog that survives the invoking terminal."""
    pid = read_pid()
    if process_is_alive(pid):
        logger.info("Watchdog already running (PID %s)", pid)
        return
    remove_pid()
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(WATCHDOG_CONSOLE_FILE, "ab", buffering=0) as console:
        process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "run"],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for _ in range(50):
        time.sleep(0.1)
        pid = read_pid()
        if process_is_alive(pid):
            logger.info("Watchdog started in background (PID %s)", pid)
            return
        if process.poll() is not None:
            break
    raise RuntimeError(
        f"Watchdog failed to start; inspect {WATCHDOG_CONSOLE_FILE}"
    )


def print_status() -> None:
    """Print watchdog and API health in a human-readable form."""
    pid = read_pid()
    watchdog_status = f"running (PID {pid})" if process_is_alive(pid) else "stopped"
    api_status = "healthy" if ServiceWatchdog()._is_healthy() else "unavailable"
    print(f"watchdog: {watchdog_status}")
    print(f"api: {api_status}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "start":
        start_watchdog()
    elif command == "stop":
        stop_watchdog()
    elif command == "status":
        print_status()
    elif command == "run":
        watchdog = ServiceWatchdog()
        watchdog.run()
    else:
        raise SystemExit("Usage: service_watchdog.py [start|status|stop|run]")
