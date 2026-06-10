#!/usr/bin/env python3
"""Watchdog for embedding-api service.

Monitors the embedding-api service running on port 9061.
If the service becomes unresponsive, automatically restarts it via tmux.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embedding-watchdog")

PROJECT_DIR = "/workspace/embedding-api"
PID_FILE = os.path.join(PROJECT_DIR, "service_watchdog.pid")
CHECK_INTERVAL = 30  # seconds
HEALTH_URL = "http://localhost:9061/health"
TMUX_SESSION = "embedding-api"


def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def remove_pid():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def read_pid():
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            return int(f.read().strip())
    return None


def is_service_running():
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


def start_service():
    """Start embedding-api via tmux."""
    logger.info("Starting embedding-api service...")
    # Clean up any stale tmux session
    subprocess.run(
        ["tmux", "kill-session", "-t", TMUX_SESSION],
        capture_output=True,
    )
    time.sleep(1)
    # Create tmux session and send command
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION],
        cwd=PROJECT_DIR,
    )
    time.sleep(0.5)
    subprocess.run(
        [
            "tmux", "send-keys", "-t", TMUX_SESSION,
            "cd /workspace/embedding-api && uvicorn app.main:app --host 0.0.0.0 --port 9061 --workers 1",
            "Enter",
        ],
        cwd=PROJECT_DIR,
    )
    time.sleep(4)
    logger.info("Service start command issued")


def stop_watchdog():
    """Stop the running watchdog process."""
    pid = read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to watchdog PID {pid}")
            time.sleep(1)
            try:
                os.kill(pid, 0)
                logger.warning(f"Watchdog PID {pid} still alive, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            logger.info("Watchdog not running")
        remove_pid()
    else:
        logger.info("No PID file found, watchdog may not be running")


def run_watchdog():
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

    def signal_handler(signum, frame):
        logger.info(f"Watchdog received signal {signum}, shutting down...")
        remove_pid()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 50)
    logger.info("Embedding API Watchdog started")
    logger.info(f"Health check URL: {HEALTH_URL}")
    logger.info(f"Check interval: {CHECK_INTERVAL}s")
    logger.info("=" * 50)

    # Initial start if not running
    if not is_service_running():
        start_service()

    while True:
        if not is_service_running():
            logger.warning("Service not responding! Attempting restart...")
            start_service()
            if is_service_running():
                logger.info("Service restarted successfully")
            else:
                logger.error("Service restart failed! Will retry next cycle.")
        else:
            logger.debug("Service is healthy")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_watchdog()
    else:
        run_watchdog()
