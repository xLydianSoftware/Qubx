"""Command input widget with history support."""

from collections import deque

from textual.events import Key
from textual.widgets import Input


class CommandInput(Input):
    """Input widget with command history navigation (up/down arrows)."""

    def __init__(self, *args, max_history: int = 100, **kwargs):
        """
        Initialize command input with history.

        Args:
            max_history: Maximum number of commands to store in history
            *args: Positional arguments for Input
            **kwargs: Keyword arguments for Input
        """
        super().__init__(*args, **kwargs)
        self.history: deque[str] = deque(maxlen=max_history)
        self.history_index: int = -1
        self.current_input: str = ""

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
