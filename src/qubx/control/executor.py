"""Thread-safe action execution dispatcher."""

from __future__ import annotations

import asyncio
import concurrent.futures
from queue import Queue
from typing import TYPE_CHECKING, Callable

from qubx import logger

from .builtin import BUILTIN_ACTIONS
from .decorator import collect_actions, execute_decorated_action
from .interfaces import IControllable
from .types import ActionDef, ActionResult

if TYPE_CHECKING:
    from qubx.core.interfaces import IStrategyContext


class CommandEvent:
    """Injected into the command queue for write actions."""

    __slots__ = ("name", "params", "future")

    def __init__(self, name: str, params: dict, future: concurrent.futures.Future[ActionResult]):
        self.name = name
        self.params = params
        self.future = future


class ActionExecutor:
    """Dispatches action calls to the correct execution context.

    Read-only actions execute directly on the caller thread.
    Write actions are enqueued on the command queue and executed
    on the strategy's processing thread.
    """

    def __init__(self, ctx: IStrategyContext, command_queue: Queue):
        self._ctx = ctx
        self._command_queue = command_queue
        self._actions: dict[str, tuple[ActionDef, Callable | None]] = {}
        self._register_builtins()
        self._register_custom()

    def _register_builtins(self):
        self._actions.update(BUILTIN_ACTIONS)

    def _register_custom(self):
        strategy = self._ctx.strategy
        # Collect @action-decorated methods
        for action_def in collect_actions(strategy):
            self._actions[action_def.name] = (action_def, None)  # handler is the decorator itself

        # Collect IControllable.get_actions()
        if isinstance(strategy, IControllable):
            for action_def in strategy.get_actions():
                self._actions[action_def.name] = (action_def, None)

    def list_actions(self) -> list[ActionDef]:
        return [ad for ad, _ in self._actions.values()]

    def resolve(self, name: str) -> tuple[ActionDef | None, Callable | None]:
        if name not in self._actions:
            return None, None
        return self._actions[name]

    async def execute(self, name: str, params: dict, timeout: float = 30.0) -> ActionResult:
        action_def, handler = self.resolve(name)
        if action_def is None:
            return ActionResult(status="error", error=f"Unknown action: {name}")

        if action_def.read_only:
            return self._execute_direct(name, params, handler)
        else:
            return await self._execute_via_queue(name, params, timeout)

    def _execute_direct(self, name: str, params: dict, handler: Callable | None) -> ActionResult:
        """Execute a read-only action directly on the current thread."""
        try:
            if handler is not None:
                result = handler(self._ctx, **params)
                if isinstance(result, ActionResult):
                    return result
                return ActionResult(status="ok", data=result)
            else:
                # Try decorated action or IControllable
                return self._execute_on_strategy(name, params)
        except Exception as e:
            return ActionResult(status="error", error=str(e))

    async def _execute_via_queue(self, name: str, params: dict, timeout: float) -> ActionResult:
        """Enqueue a write action and wait for the result."""
        future: concurrent.futures.Future[ActionResult] = concurrent.futures.Future()
        cmd = CommandEvent(name, params, future)
        self._command_queue.put(cmd)

        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
        except asyncio.TimeoutError:
            return ActionResult(status="error", error=f"Action '{name}' timed out after {timeout}s")

    def execute_command(self, cmd: CommandEvent) -> ActionResult:
        """Execute a command on the strategy thread. Called from the data loop."""
        action_def, handler = self.resolve(cmd.name)
        if action_def is None:
            return ActionResult(status="error", error=f"Unknown action: {cmd.name}")

        try:
            if handler is not None:
                result = handler(self._ctx, **cmd.params)
                if isinstance(result, ActionResult):
                    return result
                return ActionResult(status="ok", data=result)
            else:
                return self._execute_on_strategy(cmd.name, cmd.params)
        except Exception as e:
            logger.error(f"[ControlServer] :: Error executing action '{cmd.name}': {e}")
            return ActionResult(status="error", error=str(e))

    def _execute_on_strategy(self, name: str, params: dict) -> ActionResult:
        """Execute via @action decorator or IControllable."""
        strategy = self._ctx.strategy
        # Try decorated actions first
        result = execute_decorated_action(strategy, self._ctx, name, params)
        if result.status != "error" or result.error != f"Unknown action: {name}":
            return result
        # Try IControllable
        if isinstance(strategy, IControllable):
            return strategy.execute_action(self._ctx, name, params)
        return ActionResult(status="error", error=f"Unknown action: {name}")
