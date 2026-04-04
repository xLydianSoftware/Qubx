"""In-process HTTP control server for bot health checks and actions.

Replaces the old HealthServer. Runs FastAPI+uvicorn in a daemon thread.
Endpoints:
    GET  /health          → liveness probe
    GET  /ready           → readiness probe
    GET  /actions         → list available actions with schemas
    POST /actions/{name}  → execute an action
"""

from __future__ import annotations

import asyncio
import os
import threading
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qubx import logger

from .executor import ActionExecutor
from .types import ActionDef

if TYPE_CHECKING:
    from qubx.core.context import StrategyContext


class ExecRequest(BaseModel):
    params: dict[str, Any] = {}


def _action_to_dict(ad: ActionDef) -> dict:
    d = {
        "name": ad.name,
        "description": ad.description,
        "category": ad.category,
        "read_only": ad.read_only,
        "dangerous": ad.dangerous,
    }
    if ad.params:
        d["params"] = [
            {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
                **({"default": p.default} if not p.required else {}),
                **({"choices": p.choices} if p.choices else {}),
                **({"items_type": p.items_type} if p.items_type else {}),
            }
            for p in ad.params
        ]
    else:
        d["params"] = []
    return d


class ControlServer:
    """Runs health + control endpoints in a background daemon thread.

    Can be started before the strategy context is ready (for liveness probes).
    The context is attached later via `attach_context()`.
    """

    def __init__(self, port: int, ready_check: Callable[[], bool] = lambda: False):
        self._port = port
        self._ready_check = ready_check
        self._ctx: StrategyContext | None = None
        self._executor: ActionExecutor | None = None
        self._command_queue: Queue = Queue()
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

        self._app = FastAPI(title="Qubx Bot Control", docs_url=None, redoc_url=None)
        self._register_routes()

    def attach_context(self, ctx: StrategyContext):
        """Attach the strategy context and wire up the command queue.

        After this call:
        - /actions and /actions/{name} endpoints become available
        - The context's data loop will drain commands from the shared queue
        """
        self._ctx = ctx
        self._executor = ActionExecutor(ctx, self._command_queue)
        ctx._command_queue = self._command_queue
        ctx._control_executor = self._executor

    def _register_routes(self):
        app = self._app

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/ready")
        def ready():
            if not self._ready_check():
                raise HTTPException(status_code=503, detail="not_ready")
            return {"status": "ready"}

        @app.get("/actions")
        def list_actions():
            if self._executor is None:
                raise HTTPException(status_code=503, detail="Context not attached yet")
            actions = self._executor.list_actions()
            return {
                "bot_id": os.environ.get("QUBX_BOT_ID", "local"),
                "actions": [_action_to_dict(a) for a in actions if not a.hidden],
            }

        @app.post("/actions/{name}")
        async def exec_action(name: str, req: ExecRequest):
            if self._executor is None:
                raise HTTPException(status_code=503, detail="Context not attached yet")
            result = await self._executor.execute(name, req.params)
            if result.status == "error":
                raise HTTPException(status_code=400, detail=result.error)
            return {
                "status": result.status,
                "data": result.data,
                "message": result.message,
            }

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="control-server")
        self._thread.start()
        logger.info(f"Control server started on port {self._port} (/health, /ready, /actions)")

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        loop.run_until_complete(self._server.serve())

    def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("Control server stopped")
