"""REPL output widget for displaying kernel output.

RichLog (not TextArea): renders Rich/ANSI color for the kernel log stream — a TextArea is a
plain-text editor and drops the color from ``Text.from_ansi(...)``. Output-only, append-only;
RichLog handles max-line trimming and smart auto-scroll (follows the tail only when scrolled to it).
"""

from rich.text import Text
from textual.binding import Binding
from textual.widgets import RichLog


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
            auto_scroll=True,  # follow the tail when the user is at the bottom
            **kwargs,
        )

    def write(self, content: str | Text, *args, **kwargs):
        """Append a line, rendering Rich/ANSI color. Extra args ignored for call-site compatibility."""
        return super().write(content)

    def clear_output(self):
        """Clear all output."""
        self.clear()

    def action_scroll_to_end(self) -> None:
        """Jump to the tail (Ctrl+E) — useful after scrolling up."""
        self.scroll_end(animate=False)
