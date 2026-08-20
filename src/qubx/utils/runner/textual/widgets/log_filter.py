"""
Clickable level toggles for the log pane: [E][W][I][D].

Loguru writes the level into the line itself (``{ts} [ ERROR ] module | ...``), so filtering
is done on the rendered text rather than on log records — the pane receives the kernel's
stdout, not a logging stream.
"""

from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static

# - buckets in the order they are shown; the letter is what the toggle displays
LEVELS: tuple[tuple[str, str], ...] = (
    ("ERROR", "E"),
    ("WARNING", "W"),
    ("INFO", "I"),
    ("DEBUG", "D"),
)


class LogLevelFilter(Horizontal):
    """
    A row of level toggles. Emits `Changed` with the levels that remain enabled.
    """

    DEFAULT_CSS = """
    LogLevelFilter {
        height: 1;
        dock: top;
        background: $panel;
    }
    LogLevelFilter > Static {
        width: 4;
        height: 1;
        content-align: center middle;
    }
    LogLevelFilter > Static.on { color: $success; text-style: bold; }
    LogLevelFilter > Static.off { color: $text-disabled; }
    LogLevelFilter > #log-filter-label { width: auto; color: $text-muted; }
    
    """

    class Changed(Message):
        """Posted when a level is toggled."""

        def __init__(self, enabled: set[str]) -> None:
            super().__init__()
            self.enabled = enabled

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enabled: set[str] = {name for name, _ in LEVELS}

    def compose(self):
        yield Static("Filter: ", id="log-filter-label")
        for name, letter in LEVELS:
            # - markup=False: "[E]" is a valid Textual markup tag and renders as nothing
            yield Static(f"[{letter}]", id=f"log-filter-{name.lower()}", classes="on", markup=False)

    @property
    def enabled(self) -> set[str]:
        return set(self._enabled)

    def on_click(self, event) -> None:
        widget = event.widget
        if widget is None or not (wid := widget.id) or not wid.startswith("log-filter-"):
            return
        name = wid.removeprefix("log-filter-").upper()
        if name not in {n for n, _ in LEVELS}:
            return
        if name in self._enabled:
            self._enabled.discard(name)
            widget.set_classes("off")
        else:
            self._enabled.add(name)
            widget.set_classes("on")
        self.post_message(self.Changed(self.enabled))
