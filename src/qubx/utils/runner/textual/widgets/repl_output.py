"""REPL output widget for displaying kernel output.

RichLog (not TextArea): renders Rich/ANSI color for the kernel log stream — a TextArea is a
plain-text editor and drops the color from ``Text.from_ansi(...)``. Output-only, append-only;
RichLog handles max-line trimming; the tail-following below is ours — RichLog's own auto_scroll
jumps to the end unconditionally.
"""

import re
from collections import deque

from rich.text import Text
from textual.binding import Binding
from textual.widgets import RichLog

from qubx.utils.runner.textual.widgets.log_filter import LEVELS

# - loguru writes "{ts} [ LEVEL ] module | ..."; TEXT-level lines carry no prefix at all.
#   Bounded so a bracketed word inside a message body cannot be mistaken for the level.
_LEVEL_RE = re.compile(r"^.{0,40}?\[\s*([A-Z]+)\s*\]")
_FILTERABLE = {name for name, _ in LEVELS}


class ReplOutput(RichLog):
    """Colored, append-only REPL/log output with smart auto-scroll."""

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
        # - every line is kept so a level can be switched back on without losing history
        self._buffer: deque[tuple[str | None, str | Text]] = deque(maxlen=max_lines)
        self._enabled: set[str] | None = None  # None = show everything
        # drop TextArea-era kwargs that RichLog doesn't accept / we set explicitly
        kwargs.pop("markup", None)
        kwargs.pop("wrap", None)
        kwargs.pop("max_lines", None)
        super().__init__(
            *args,
            max_lines=max_lines,
            wrap=True,
            markup=False,  # content arrives as Rich Text / plain str, not markup strings
            highlight=False,
            auto_scroll=False,  # - per-write, in `write`, so scrolling up is not undone
            **kwargs,
        )

    def write(self, content: str | Text, *args, **kwargs):
        """
        Append a line, rendering Rich/ANSI color. Extra args ignored for call-site compatibility.

        RichLog's own `auto_scroll` jumps to the tail on every write, which drags the view back
        down while you are reading further up. Follow the tail only while already at it.
        """
        level = self._level_of(content)
        self._buffer.append((level, content))
        if not self._shown(level):
            return self
        return super().write(content, scroll_end=self.is_vertical_scroll_end)

    @staticmethod
    def _level_of(content: str | Text) -> str | None:
        """
        The level a line reports, or None for output that carries no log prefix.
        """
        m = _LEVEL_RE.match(content.plain if isinstance(content, Text) else str(content))
        return m.group(1) if m else None

    def _shown(self, level: str | None) -> bool:
        """
        Only the four toggleable levels can be hidden.

        Unprefixed output (TEXT, tracebacks, kernel stdout) and levels with no toggle of their
        own (CRITICAL, SUCCESS) always show — a filter must never swallow a critical error.
        """
        if self._enabled is None or level is None or level not in _FILTERABLE:
            return True
        return level in self._enabled

    def apply_levels(self, enabled: set[str]) -> None:
        """
        Re-render the buffer with only `enabled` levels visible.
        """
        self._enabled = set(enabled)
        self.clear()
        for level, content in self._buffer:
            if self._shown(level):
                super().write(content, scroll_end=False)
        self.scroll_end(animate=False)

    def clear_output(self):
        """Clear all output."""
        self._buffer.clear()
        self.clear()

    def action_scroll_to_end(self) -> None:
        """Jump to the tail (Ctrl+E) — useful after scrolling up."""
        self.scroll_end(animate=False)
