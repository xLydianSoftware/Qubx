"""
Shared Qubx dark theme for all Textual TUI applications.

Import and register in any App subclass::

    from qubx.cli.theme import QUBX_DARK

    class MyApp(App):
        DARK = True

        def on_mount(self) -> None:
            self.register_theme(QUBX_DARK)
            self.theme = QUBX_DARK.name

CSS variables provided (use as ``$variable-name`` in TCSS)::

    Core palette
    ─────────────────────────────────────────────────────────
    $background       #000000   pure black screen background
    $surface          #0a0a0a   widget / even-row background
    $panel            #0d0d0d   header / odd-row background
    $boost            #0d1f0d   cursor / focused element fill
    $primary          #00cc66   main green (labels, headings)
    $secondary        #00aa44   secondary green (buttons, tree)
    $accent           #00ff88   bright green (selected, cursor text)
    $success          #00ff88   success / profit indicator
    $warning          #ffcc00   warning / neutral indicator
    $error            #cc3333   error / loss indicator
    $foreground       #cccccc   default text

    Custom extras (via Theme.variables)
    ─────────────────────────────────────────────────────────
    $border-green     #1e3a1e   subtle dark-green border / separator
"""

from textual.theme import Theme


QUBX_DARK = Theme(
    name="qubx-dark",
    dark=True,
    # - palette
    primary="#00cc66",
    secondary="#00aa44",
    accent="#00ff88",
    # - backgrounds
    background="#000000",
    surface="#0a0a0a",
    panel="#0d0d0d",
    boost="#0d1f0d",
    # - status colors
    success="#00ff88",
    warning="#ffcc00",
    error="#cc3333",
    # - text
    foreground="#cccccc",
    # - custom variables accessible as $border-green in TCSS
    variables={
        "border-green": "#1e3a1e",
    },
)
