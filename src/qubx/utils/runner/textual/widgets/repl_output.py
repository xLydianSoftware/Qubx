"""REPL output widget for displaying kernel output."""

from rich.text import Text
from textual.app import ScreenStackError
from textual.binding import Binding
from textual.widgets import TextArea


class ReplOutput(TextArea):
    """REPL output widget with text selection support, line numbers, and smart auto-scroll."""

    BINDINGS = [
        Binding("ctrl+e", "scroll_to_end", "Scroll to End", show=True),
    ]

    DEFAULT_CSS = """
    ReplOutput {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self, max_lines: int = 10000, *args, **kwargs):
        """
        Initialize the REPL output widget.

        Args:
            max_lines: Maximum number of lines to keep in the output (default: 10,000)
            *args: Additional positional arguments for TextArea
            **kwargs: Additional keyword arguments for TextArea
        """
        # Remove unsupported parameters for TextArea
        kwargs.pop("markup", None)
        kwargs.pop("wrap", None)
        kwargs.pop("max_lines", None)

        super().__init__(
            read_only=True,
            show_line_numbers=True,
            # language="python",
            *args,
            **kwargs,
        )
        self._max_lines = max_lines
        self._counter = 0

    def write(self, content: str | Text, *args, **kwargs):
        """
        Write content to the output with smart auto-scroll.

        Args:
            content: Text content to write (string or Rich Text object)
            *args: Additional positional arguments (ignored for compatibility)
            **kwargs: Additional keyword arguments (ignored for compatibility)
        """
        # Check if user is at the last line BEFORE appending
        should_autoscroll = self.cursor_at_last_line

        # Convert Rich Text to plain string
        if isinstance(content, Text):
            # Try to preserve ANSI color codes if available
            try:
                # Get ANSI representation if possible
                text_str = str(content)
            except Exception:
                # Fallback to plain text
                text_str = content.plain
        else:
            text_str = str(content)

        # Append text at document end using insert()
        self.insert(text_str + "\n", location=self.document.end)
        self._counter += 1

        if self._counter % 1000 == 0:
            # Apply line limit if needed
            self._apply_line_limit()

        # If user was at last line, follow the new content
        if should_autoscroll:
            self.cursor_location = self.document.end
            # Only scroll if widget is mounted in an app
            if self.is_attached:
                self.scroll_cursor_visible(animate=False)
        # Otherwise, leave cursor/scroll position unchanged

    def _apply_line_limit(self):
        """Trim lines from the beginning if the output exceeds max_lines."""
        lines = self.text.split("\n")
        if len(lines) > self._max_lines:
            # Keep only the last max_lines
            trimmed_text = "\n".join(lines[-self._max_lines :])
            self.text = trimmed_text

    def clear_output(self):
        """Clear all output from the REPL."""
        self.clear()

    def action_scroll_to_end(self) -> None:
        """Scroll to the end of the output (triggered by Ctrl+E)."""
        self.cursor_location = self.document.end
        if self.is_attached:
            self.scroll_cursor_visible(animate=False)

    def _watch_selection(self, old: "TextArea.Selection", new: "TextArea.Selection") -> None:
        """Override to handle ScreenStackError during app shutdown."""
        try:
            super()._watch_selection(old, new)
        except ScreenStackError:
            # Suppress error when app is shutting down and screen is gone
            pass
