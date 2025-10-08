"""REPL output widget for displaying kernel output."""

from textual.widgets import RichLog


class ReplOutput(RichLog):
    """REPL output widget with clear functionality and line limit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clear_output(self):
        """Clear all output from the REPL."""
        self.clear()
