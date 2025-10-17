"""REPL output widget for displaying kernel output."""

import subprocess

from rich.text import Text
from textual.widgets import RichLog


class ReplOutput(RichLog):
    """REPL output widget with clear functionality, line limit, and copy support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lines_buffer: list[str] = []

    def write(self, content: str | Text, *args, **kwargs):
        """Write content to the log and store in buffer for copying."""
        # Convert to plain text for buffer storage
        if isinstance(content, Text):
            text_line = content.plain
        else:
            text_line = str(content)

        self._lines_buffer.append(text_line)

        # Call parent's write method (which is actually 'write' from RichLog)
        super().write(content, *args, **kwargs)

    def clear_output(self):
        """Clear all output from the REPL."""
        self.clear()
        self._lines_buffer.clear()

    def copy_last_lines(self, n: int = 50) -> bool:
        """
        Copy the last N lines to the clipboard.

        Args:
            n: Number of lines to copy (default: 50)

        Returns:
            True if successful, False otherwise
        """
        if not self._lines_buffer:
            return False

        # Get last n lines
        lines_to_copy = self._lines_buffer[-n:]
        text_to_copy = "\n".join(lines_to_copy)

        # Try to copy to clipboard using platform-specific commands
        try:
            # Try xclip (Linux)
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text_to_copy.encode(),
                check=True,
                capture_output=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        try:
            # Try xsel (Linux alternative)
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text_to_copy.encode(),
                check=True,
                capture_output=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        try:
            # Try pbcopy (macOS)
            subprocess.run(
                ["pbcopy"],
                input=text_to_copy.encode(),
                check=True,
                capture_output=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        return False

    def copy_all_lines(self) -> bool:
        """
        Copy all lines to the clipboard.

        Returns:
            True if successful, False otherwise
        """
        return self.copy_last_lines(len(self._lines_buffer))
