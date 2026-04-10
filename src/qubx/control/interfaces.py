from __future__ import annotations

from typing import TYPE_CHECKING

from .types import ActionDef, ActionResult

if TYPE_CHECKING:
    from qubx.core.interfaces import IStrategyContext


class IControllable:
    """Mixin interface for strategies that expose custom control actions.

    Strategies that implement this interface can define actions that are
    callable via the bot's control server HTTP API.
    """

    def get_actions(self) -> list[ActionDef]:
        """Return list of custom actions this strategy supports."""
        return []

    def execute_action(self, ctx: IStrategyContext, name: str, params: dict) -> ActionResult:
        """Execute a custom action by name.

        Called on the strategy's processing thread (thread-safe w.r.t. strategy state).
        """
        return ActionResult(status="error", error=f"Unknown action: {name}")
