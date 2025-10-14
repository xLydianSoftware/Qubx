"""Debug log widget for displaying debug messages in the TUI."""

import logging
from datetime import datetime

from rich.text import Text
from textual.widgets import RichLog


class DebugLog(RichLog):
    """Debug log widget for displaying formatted debug messages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_lines = 1000

    def write_debug(self, level: str, message: str):
        """
        Write a debug message with timestamp and level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Color based on level
        if level == "DEBUG":
            color = "dim white"
        elif level == "INFO":
            color = "cyan"
        elif level == "WARNING":
            color = "yellow"
        elif level == "ERROR":
            color = "red"
        else:
            color = "white"

        # Format message
        text = Text()
        text.append(f"[{timestamp}] ", style="dim")
        text.append(level.ljust(7), style=f"bold {color}")
        text.append(f" {message}")

        # Write to log
        self.write(text)

        # Trim old lines if needed
        if len(self._lines) > self._max_lines:
            # Clear and keep last max_lines
            lines = list(self._lines)[-self._max_lines:]
            self.clear()
            for line in lines:
                self.write(line)

    def clear_debug(self):
        """Clear all debug messages."""
        self.clear()


class TextualLogHandler(logging.Handler):
    """Custom logging handler that writes to a DebugLog widget."""

    def __init__(self, debug_log: DebugLog):
        """
        Initialize the handler.

        Args:
            debug_log: DebugLog widget to write to
        """
        super().__init__()
        self.debug_log = debug_log
        self.setLevel(logging.DEBUG)

    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to the debug widget.

        Args:
            record: Log record to emit
        """
        try:
            # Format the message
            msg = self.format(record)

            # Post to Textual's thread-safe message queue
            # Use call_from_thread to safely update from any thread
            if hasattr(self.debug_log, "app") and self.debug_log.app:
                self.debug_log.app.call_from_thread(
                    self.debug_log.write_debug,
                    record.levelname,
                    msg
                )
        except Exception:
            # Silently ignore errors to prevent infinite loops
            pass
