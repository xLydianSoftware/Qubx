"""
Main Textual application for running strategies with Jupyter kernel integration.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input

from qubx import logger

from .handlers import KernelEventHandler
from .init_code import generate_init_code
from .kernel import IPyKernel
from .widgets import PositionsTable, ReplOutput


class TextualStrategyApp(App[None]):
    """Main Textual application for running strategies with kernel interaction."""

    CSS_PATH = Path(__file__).parent / "styles.tcss"

    BINDINGS = [
        Binding("ctrl+l", "clear_repl", "Clear REPL", show=True),
        Binding("ctrl+c", "interrupt", "Interrupt", show=True),
        Binding("p", "toggle_positions", "Positions", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, config_file: Path, account_file: Path | None, paper: bool, restore: bool) -> None:
        """
        Initialize the Textual strategy app.

        Args:
            config_file: Path to the strategy configuration file
            account_file: Optional path to the account configuration file
            paper: Whether to run in paper trading mode
            restore: Whether to restore the strategy state
        """
        super().__init__()
        self.config_file = config_file
        self.account_file = account_file
        self.paper = paper
        self.restore = restore
        self.kernel = IPyKernel()
        self.output: ReplOutput
        self.input: Input
        self.positions_panel: Vertical
        self.positions_table: PositionsTable
        self.positions_visible = False
        self.event_handler: KernelEventHandler

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Setup event handler
        self.event_handler = KernelEventHandler(self.output, self.positions_table)

        # Add welcome message
        self.output.write("[bold cyan]Qubx Strategy Runner[/bold cyan]")
        self.output.write("[dim]Type Python commands below and press Enter to execute[/dim]")
        self.output.write("")

        # Start the kernel
        await self.kernel.start()
        self.kernel.register(self.event_handler.handle_event)

        # Initialize the strategy context within the kernel
        self.output.write("[yellow]Initializing strategy context...")
        try:
            # Pre-load the context and helpers into the kernel
            init_code = generate_init_code(
                self.config_file,
                self.account_file,
                self.paper,
                self.restore,
            )
            self.kernel.execute(init_code, silent=False)

            self.output.write("[green]✓ Strategy context initialized and ready!")
            self.output.write("[cyan]Available objects: ctx, S (strategy), portfolio(), orders(), trade(), exit()")
        except Exception as e:
            self.output.write(f"[red]Failed to initialize strategy: {e}")
            logger.exception("Strategy initialization failed")

        # Start interval timer for positions updates (1 second)
        self.set_interval(1.0, self._request_positions)

    async def on_unmount(self) -> None:
        """Clean up when app is closing."""
        # The kernel will handle stopping the context via the exit() function
        await self.kernel.stop()

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="output-container"):
                self.output = ReplOutput(id="output", wrap=True, markup=True, max_lines=10000)
                yield self.output
            self.positions_panel = Vertical(id="positions-panel")
            with self.positions_panel:
                self.positions_table = PositionsTable(id="positions-table")
                self.positions_table.setup_columns()
                yield self.positions_table
        with Vertical(id="input-container"):
            self.input = Input(placeholder=">>> Type Python code here and press Enter", id="input")
            yield self.input
        yield Footer()

    # ------------------- Event handlers ---------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        code = event.value.strip()
        if not code:
            return
        self.input.value = ""
        # Echo input
        self.output.write(f"[bold cyan]>>> {code}")
        self.kernel.execute(code)

    # ------------------- Actions ---------------------

    def action_clear_repl(self) -> None:
        """Clear the output."""
        self.output.clear_output()

    async def action_interrupt(self) -> None:
        """Send interrupt signal to the kernel."""
        await self.kernel.interrupt()
        self.output.write("[orange1]⚠ KeyboardInterrupt sent to kernel")

    def action_toggle_positions(self) -> None:
        """Toggle the positions panel visibility."""
        self.positions_visible = not self.positions_visible
        if self.positions_visible:
            self.positions_panel.add_class("visible")
        else:
            self.positions_panel.remove_class("visible")

    # ------------------- Helper methods ----------------

    def _request_positions(self) -> None:
        """Request positions update from the kernel (called by interval timer)."""
        if not self.positions_visible or self.event_handler.is_positions_busy():
            return
        self.event_handler.mark_positions_busy()
        self.kernel.execute("emit_positions()", silent=True)
