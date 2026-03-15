"""Lightweight HTTP health/readiness server for k8s probes.

Activated by QUBX_HEALTH_PORT env var or --health-port CLI flag.
Runs in a daemon thread, no external dependencies.

Endpoints:
    GET /health → 200 if process alive (liveness probe)
    GET /ready  → 200 after on_warmup_finished; 503 during warmup (readiness probe)
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from qubx import logger


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for /health and /ready."""

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/ready":
            if self.server.ready_check():
                self._respond(200, {"status": "ready"})
            else:
                self._respond(503, {"status": "not_ready"})
        else:
            self._respond(404, {"error": "not_found"})

    def _respond(self, code: int, body: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        # Silence default stderr logging from BaseHTTPRequestHandler
        pass


class HealthServer:
    """Runs /health + /ready endpoints in a background daemon thread."""

    def __init__(self, port: int, ready_check: callable = lambda: False):
        """
        Args:
            port: TCP port to listen on
            ready_check: Callable that returns True when the strategy is ready
        """
        self._port = port
        self._ready_check = ready_check
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        self._server = HTTPServer(("0.0.0.0", self._port), _HealthHandler)
        self._server.ready_check = self._ready_check
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Health server started on port {self._port} (/health, /ready)")

    def stop(self):
        if self._server:
            self._server.shutdown()
            logger.debug("Health server stopped")
