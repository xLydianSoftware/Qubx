"""Command input widget with history support and autocompletion."""

from collections import deque
from typing import TYPE_CHECKING

from textual.events import Key
from textual.widgets import Input

if TYPE_CHECKING:
    from qubx.utils.runner.textual.kernel import IPyKernel


class CommandInput(Input):
    """Input widget with command history navigation (up/down arrows) and autocomplete support."""

    def __init__(self, *args, max_history: int = 100, kernel: "IPyKernel | None" = None, **kwargs):
        """
        Initialize command input with history and autocomplete.

        Args:
            max_history: Maximum number of commands to store in history
            kernel: IPyKernel instance for code completion (optional)
            *args: Positional arguments for Input
            **kwargs: Keyword arguments for Input
        """
        super().__init__(*args, **kwargs)
        self.history: deque[str] = deque(maxlen=max_history)
        self.history_index: int = -1
        self.current_input: str = ""
        self.kernel = kernel

    def on_key(self, event: Key) -> None:
        """Handle key events for history navigation."""
        if event.key == "up":
            # Navigate to previous command in history
            if self.history and self.history_index < len(self.history) - 1:
                # Save current input if we're at the bottom of history
                if self.history_index == -1:
                    self.current_input = self.value

                self.history_index += 1
                # Access history from the end (most recent)
                self.value = self.history[-(self.history_index + 1)]
                self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()

        elif event.key == "down":
            # Navigate to next command in history
            if self.history_index > -1:
                self.history_index -= 1
                if self.history_index == -1:
                    # Restore current input
                    self.value = self.current_input
                else:
                    self.value = self.history[-(self.history_index + 1)]
                self.cursor_position = len(self.value)
            event.prevent_default()
            event.stop()

        else:
            # For any other key, reset history navigation
            if self.history_index != -1 and event.key not in ("up", "down", "shift", "ctrl", "alt"):
                self.history_index = -1
                self.current_input = ""

    def add_to_history(self, command: str) -> None:
        """
        Add a command to the history.

        Args:
            command: Command string to add to history
        """
        if command and (not self.history or command != self.history[-1]):
            # Only add non-empty and non-duplicate commands
            self.history.append(command)
        # Reset history navigation
        self.history_index = -1
        self.current_input = ""

    async def get_completions(self, code: str, cursor_pos: int) -> list[str]:
        """
        Get completions from the kernel for the current input.

        Args:
            code: The code to complete
            cursor_pos: Cursor position in the code

        Returns:
            List of completion strings
        """
        from qubx import logger

        if self.kernel is None:
            logger.warning("CommandInput: kernel is None, cannot get completions")
            return []

        try:
            logger.debug(f"CommandInput: requesting completions for '{code}' at {cursor_pos}")
            completions = await self.kernel.complete(code, cursor_pos)
            logger.debug(f"CommandInput: got {len(completions)} completions")
            return completions
        except Exception as e:
            logger.error(f"CommandInput: error getting completions: {e}", exc_info=True)
            return []
