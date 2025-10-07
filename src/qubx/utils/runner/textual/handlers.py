"""
Event handlers for processing kernel events and updating UI.
"""

from typing import Any, Callable

from rich.markdown import Markdown

from .widgets import PositionsTable, ReplOutput


class KernelEventHandler:
    """Handles kernel events and updates the UI accordingly."""

    def __init__(self, output: ReplOutput, positions_table: PositionsTable):
        """
        Initialize the event handler.

        Args:
            output: REPL output widget for displaying text
            positions_table: Positions table widget for displaying positions data
        """
        self.output = output
        self.positions_table = positions_table
        self._positions_busy = False
        self._on_positions_update: Callable[[], None] | None = None

    def set_positions_update_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when positions are updated."""
        self._on_positions_update = callback

    def handle_event(self, kind: str, payload: Any) -> None:
        """
        Handle kernel events and display them in the output.

        Args:
            kind: Event type (stream, text, markdown, error, clear, qubx_positions, etc.)
            payload: Event data
        """
        if kind == "qubx_positions":
            self._positions_busy = False
            if self._on_positions_update:
                self._on_positions_update()
            # Update positions table with new data
            self.positions_table.update_positions(payload)
        elif kind == "stream":
            text = payload.get("text", "")
            self.output.write(text.rstrip("\n"))
        elif kind == "text":
            self.output.write(str(payload))
        elif kind == "markdown":
            self.output.write(Markdown(payload))
        elif kind == "error":
            tb = payload.get("traceback", "")
            self.output.write(f"[red]{payload.get('ename')}: {payload.get('evalue')}\n{tb}")
        elif kind == "clear":
            self.output.clear_output()
        elif kind == "debug":
            # Ignore debug messages
            pass
        else:
            # Ignore other events
            pass

    def is_positions_busy(self) -> bool:
        """Check if positions update is in progress."""
        return self._positions_busy

    def mark_positions_busy(self) -> None:
        """Mark positions update as in progress."""
        self._positions_busy = True
