"""
Main Textual application for running strategies with Jupyter kernel integration.
"""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from qubx import logger

from .handlers import KernelEventHandler
from .init_code import generate_init_code, generate_mock_init_code
from .kernel import IPyKernel
from .widgets import CommandInput, OrdersTable, PositionsTable, QuotesTable, ReplOutput


class TextualStrategyApp(App[None]):
    """Main Textual application for running strategies with kernel interaction."""

    CSS_PATH = Path(__file__).parent / "styles.tcss"

    BINDINGS = [
        Binding("ctrl+l", "clear_repl", "Clear REPL", show=True),
        Binding("ctrl+c", "interrupt", "Interrupt", show=True),
        Binding("ctrl+y", "copy_output", "Copy Output", show=True),
        Binding("p", "toggle_positions", "Positions", show=True),
        Binding("o", "toggle_orders", "Orders", show=True),
        Binding("m", "toggle_market", "Market", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        config_file: Path,
        account_file: Path | None,
        paper: bool,
        restore: bool,
        test_mode: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the Textual strategy app.

        Args:
            config_file: Path to the strategy configuration file
            account_file: Optional path to the account configuration file
            paper: Whether to run in paper trading mode
            restore: Whether to restore the strategy state
            test_mode: Whether to run in test mode (skips strategy initialization)
        """
        super().__init__(*args, **kwargs)
        self.config_file = config_file
        self.account_file = account_file
        self.paper = paper
        self.restore = restore
        self.test_mode = test_mode
        self.kernel = IPyKernel()
        self.output: ReplOutput
        self.input: CommandInput
        self.positions_panel: Vertical
        self.positions_table: PositionsTable
        self.orders_panel: Vertical
        self.orders_table: OrdersTable
        self.market_panel: Vertical
        self.quotes_table: QuotesTable
        self.positions_visible = False
        self.orders_visible = False
        self.market_visible = False
        self.event_handler: KernelEventHandler

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Setup event handler
        self.event_handler = KernelEventHandler(self.output, self.positions_table, self.orders_table, self.quotes_table)

        # Add welcome message
        self.output.write("[bold cyan]Qubx Strategy Runner[/bold cyan]")
        self.output.write("[dim]Type Python commands below and press Enter to execute[/dim]")
        self.output.write("")

        # Start the kernel
        await self.kernel.start()
        self.kernel.register(self.event_handler.handle_event)

        # Initialize the strategy context within the kernel
        if self.test_mode:
            self.output.write("[yellow]Initializing test mode...")
            try:
                init_code = generate_mock_init_code()
                self.kernel.execute(init_code, silent=False)
                self.output.write("[green]✓ Test mode initialized!")
            except Exception as e:
                self.output.write(f"[red]Failed to initialize test mode: {e}")
                logger.exception("Test mode initialization failed")
        else:
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
            except Exception as e:
                self.output.write(f"[red]Failed to initialize strategy: {e}")
                logger.exception("Strategy initialization failed")

        # Start interval timer for dashboard updates (1 second)
        self.set_interval(1.0, self._request_dashboard)

    async def on_unmount(self) -> None:
        """Clean up when app is closing."""
        # The kernel will handle stopping the context via the exit() function
        await self.kernel.stop()

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=True)
        with Vertical(id="content-wrapper"):
            with Horizontal(id="main-container"):
                # Output on the left
                with Vertical(id="output-container"):
                    self.output = ReplOutput(id="output", wrap=True, markup=True, max_lines=10000)
                    yield self.output
                # Vertical layout for positions/orders stacked up/down on the right
                with Vertical(id="tables-container", classes="tables-column"):
                    # Positions panel
                    self.positions_panel = Vertical(id="positions-panel", classes="side-panel")
                    with self.positions_panel:
                        self.positions_table = PositionsTable(id="positions-table")
                        self.positions_table.setup_columns()
                        yield self.positions_table
                    # Orders panel
                    self.orders_panel = Vertical(id="orders-panel", classes="side-panel")
                    with self.orders_panel:
                        self.orders_table = OrdersTable(id="orders-table")
                        self.orders_table.setup_columns()
                        yield self.orders_table
                    # Market data panel
                    self.market_panel = Vertical(id="market-panel", classes="side-panel")
                    with self.market_panel:
                        self.quotes_table = QuotesTable(id="quotes-table")
                        self.quotes_table.setup_columns()
                        yield self.quotes_table
            with Vertical(id="input-container"):
                self.input = CommandInput(placeholder=">>> Type Python code here and press Enter", id="input")
                yield self.input
        yield Footer()

    # ------------------- Event handlers ---------------
    def on_input_submitted(self, event: CommandInput.Submitted) -> None:
        """Handle Enter key in the input field."""
        code = event.value.strip()
        if not code:
            return
        # Add to command history
        self.input.add_to_history(code)
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

    def action_copy_output(self) -> None:
        """Copy the last 50 lines of output to clipboard."""
        success = self.output.copy_last_lines(50)
        if success:
            self.output.write("[green]✓ Last 50 lines copied to clipboard")
        else:
            self.output.write("[red]✗ Failed to copy (xclip/xsel/pbcopy not found)")

    def action_toggle_positions(self) -> None:
        """Toggle the positions panel visibility."""
        self.positions_visible = not self.positions_visible
        if self.positions_visible:
            self.positions_panel.add_class("visible")
        else:
            self.positions_panel.remove_class("visible")
        self._update_tables_container_visibility()

    def action_toggle_orders(self) -> None:
        """Toggle the orders panel visibility."""
        self.orders_visible = not self.orders_visible
        if self.orders_visible:
            self.orders_panel.add_class("visible")
        else:
            self.orders_panel.remove_class("visible")
        self._update_tables_container_visibility()

    def _update_tables_container_visibility(self) -> None:
        """Show tables container if either positions or orders is visible."""
        tables_container = self.query_one("#tables-container")
        if self.positions_visible or self.orders_visible:
            tables_container.add_class("visible")
        else:
            tables_container.remove_class("visible")

    def action_toggle_market(self) -> None:
        """Toggle the market data panel visibility."""
        self.market_visible = not self.market_visible
        if self.market_visible:
            self.market_panel.add_class("visible")
        else:
            self.market_panel.remove_class("visible")

    # ------------------- Helper methods ----------------

    def _request_dashboard(self) -> None:
        """Request dashboard update from the kernel (called by interval timer)."""
        if self.event_handler.is_dashboard_busy():
            return
        self.event_handler.mark_dashboard_busy()
        self.kernel.execute("emit_dashboard()", silent=True)
