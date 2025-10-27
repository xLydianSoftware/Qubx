"""
Event handlers for processing kernel events and updating UI.
"""

from typing import Any, Callable

from rich.markdown import Markdown
from rich.text import Text

from qubx import logger

from .widgets import OrdersTable, PositionsTable, QuotesTable, ReplOutput


class KernelEventHandler:
    """Handles kernel events and updates the UI accordingly."""

    def __init__(
        self,
        output: ReplOutput,
        positions_table: PositionsTable,
        orders_table: OrdersTable,
        quotes_table: QuotesTable,
    ):
        """
        Initialize the event handler.

        Args:
            output: REPL output widget for displaying text
            positions_table: Positions table widget for displaying positions data
            orders_table: Orders table widget for displaying orders data (optional)
            quotes_table: Quotes table widget for displaying market quotes (optional)
        """
        self.output = output
        self.positions_table = positions_table
        self.orders_table = orders_table
        self.quotes_table = quotes_table
        self._dashboard_busy = False
        self._on_dashboard_update: Callable[[], None] | None = None

    def set_dashboard_update_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when dashboard is updated."""
        self._on_dashboard_update = callback

    def handle_event(self, kind: str, payload: Any) -> None:
        """
        Handle kernel events and display them in the output.

        Args:
            kind: Event type (stream, text, markdown, error, clear, qubx_dashboard, etc.)
            payload: Event data
        """
        if kind == "qubx_dashboard":
            self._dashboard_busy = False
            if self._on_dashboard_update:
                self._on_dashboard_update()
            # Update tables with new data
            # Note: Individual widgets now handle data sanitization and errors internally
            try:
                positions = payload.get("positions", [])
                orders = payload.get("orders", [])
                quotes = payload.get("quotes", {})
                self.positions_table.update_positions(positions)
                self.orders_table.update_orders(orders)
                self.quotes_table.update_quotes(quotes)
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                self.output.write(f"[red]Dashboard update failed: {e}")
        elif kind == "stream":
            text = payload.get("text", "")
            self.output.write(Text.from_ansi(text))
        elif kind == "text":
            self.output.write(Text.from_ansi(payload))
        elif kind == "markdown":
            self.output.write(Markdown(payload))  # type: ignore
        elif kind == "error":
            tb = Text.from_ansi(payload.get("traceback", ""))
            self.output.write(f"[red]{payload.get('ename')}: {payload.get('evalue')}\n{tb}")
        elif kind == "clear":
            self.output.clear_output()
        elif kind == "debug":
            # Ignore debug messages
            pass
        else:
            # Ignore other events
            pass

    def is_dashboard_busy(self) -> bool:
        """Check if dashboard update is in progress."""
        return self._dashboard_busy

    def mark_dashboard_busy(self) -> None:
        """Mark dashboard update as in progress."""
        self._dashboard_busy = True
