"""
Main Textual application for running strategies with Jupyter kernel integration.
"""

import asyncio
import concurrent.futures
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header
from textual_autocomplete import DropdownItem

from qubx import logger

from .handlers import KernelEventHandler
from .init_code import generate_init_code, generate_mock_init_code
from .kernel import IPyKernel
from .widgets import CommandInput, DebugLog, OrdersTable, PositionsTable, QuotesTable, ReplOutput


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
        Binding("d", "toggle_debug", "Debug", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        config_file: Path,
        account_file: Path | None,
        paper: bool,
        restore: bool,
        connection_file: Path | None = None,
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
            connection_file: Optional path to existing kernel connection file
            test_mode: Whether to run in test mode (skips strategy initialization)
        """
        super().__init__(*args, **kwargs)
        self.config_file = config_file
        self.account_file = account_file
        self.paper = paper
        self.restore = restore
        self.connection_file = connection_file
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
        self.debug_panel: Vertical
        self.debug_log: DebugLog
        self.positions_visible = False
        self.orders_visible = False
        self.market_visible = False
        self.debug_visible = False
        self.event_handler: KernelEventHandler
        self._completion_cache: dict[str, list[str]] = {}
        self._completion_task: asyncio.Task | None = None
        self._log_handler_id: int | None = None
        self._original_stdout = None
        self._original_stderr = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Setup event handler
        self.event_handler = KernelEventHandler(self.output, self.positions_table, self.orders_table, self.quotes_table)

        # Add welcome message
        self.output.write("[bold cyan]Qubx Strategy Runner[/bold cyan]")
        self.output.write("[dim]Type Python commands below and press Enter to execute[/dim]")
        self.output.write("")

        # Start or connect to kernel
        if self.connection_file:
            # Connect to existing kernel (already initialized)
            self.output.write(f"[yellow]Connecting to existing kernel: {self.connection_file}")
            logger.debug("About to connect to existing kernel")
            try:
                await self.kernel.connect_to_existing(str(self.connection_file))
                logger.debug("Kernel connection established")
                self.kernel.register(self.event_handler.handle_event)
                logger.debug("Event handler registered")
                self.output.write("[green]✓ Connected to existing kernel!")
                logger.info("Connected to existing kernel: {}", self.connection_file)
            except Exception as e:
                self.output.write(f"[red]Failed to connect to kernel: {e}")
                logger.exception("Kernel connection failed")
                raise

            # Retrieve and replay output history
            self.output.write("[dim]Retrieving session history...")
            logger.debug("About to get output history")
            history = await self.kernel.get_output_history()
            logger.debug(f"Got history: {len(history) if history else 0} entries")
            if history:
                self.output.write(f"[dim]Replaying {len(history)} history entries...")
                logger.debug("Replaying history entries")
                for entry in history:
                    msg_type = entry.get("type")
                    content = entry.get("content")
                    if msg_type == "text":
                        self.output.write(str(content))
                    elif msg_type == "dashboard":
                        # Dashboard data will be picked up by event handler
                        pass
                logger.debug("History replay complete")
                self.output.write("[green]✓ History restored!")
            else:
                self.output.write("[dim]No history found")
                logger.debug("No history found")
        else:
            # Start new kernel and initialize it
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

        # Setup debug log handler
        logger.debug("Setting up debug log handler")

        # Save current handlers (including file debug log)
        # We only want to remove console handlers, not file handlers
        # Unfortunately logger.remove() removes ALL, so we skip it
        # The TUI sink will be added and will receive logs alongside file handler

        # Create a sink function that writes to the debug log
        def debug_sink(message):
            """Sink function for loguru to write to debug log."""
            record = message.record
            level = record["level"].name
            msg = record["message"]
            self.debug_log.write_debug(level, msg)

        # Add TUI debug panel sink (logs go to both file and TUI)
        self._log_handler_id = logger.add(debug_sink, level="DEBUG")
        logger.debug("Debug log handler setup complete")

        # Start interval timer for dashboard updates (1 second)
        logger.debug("Starting dashboard update timer")
        self.set_interval(1.0, self._request_dashboard)
        logger.debug("on_mount() complete")

    async def on_unmount(self) -> None:
        """Clean up when app is closing."""
        logger.debug("on_unmount() called")

        # Shutdown thread pool executor
        self._executor.shutdown(wait=False)
        logger.debug("Thread pool executor shutdown")

        # Remove TUI debug log handler (but keep file handler)
        if self._log_handler_id is not None:
            logger.remove(self._log_handler_id)
            logger.debug("TUI log handler removed")

        # Only stop the kernel if we created it (not if we just connected to it)
        # When connection_file is set, the kernel is managed externally
        if not self.connection_file:
            logger.debug("Stopping kernel we created")
            # The kernel will handle stopping the context via the exit() function
            await self.kernel.stop()
            logger.debug("Kernel stopped")
        else:
            logger.debug("Stopping client channels (external kernel)")
            # Just stop the client channels, don't shutdown the kernel
            if self.kernel.kc:
                self.kernel.kc.stop_channels()
            logger.debug("Client channels stopped")

        logger.debug("on_unmount() complete")

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
                    # Debug log panel
                    self.debug_panel = Vertical(id="debug-panel", classes="side-panel")
                    with self.debug_panel:
                        self.debug_log = DebugLog(id="debug-log", wrap=True, markup=True, max_lines=1000)
                        yield self.debug_log
            with Vertical(id="input-container"):
                self.input = CommandInput(
                    placeholder=">>> Type Python code here and press Enter", id="input", kernel=self.kernel
                )
                yield self.input
                # AutoComplete widget with kernel-powered completions
                # yield AutoComplete(self.input, candidates=self._get_completions_callback)
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
        """Show tables container if any panel is visible."""
        tables_container = self.query_one("#tables-container")
        if self.positions_visible or self.orders_visible or self.market_visible or self.debug_visible:
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
        self._update_tables_container_visibility()

    def action_toggle_debug(self) -> None:
        """Toggle the debug log panel visibility."""
        self.debug_visible = not self.debug_visible
        if self.debug_visible:
            self.debug_panel.add_class("visible")
        else:
            self.debug_panel.remove_class("visible")
        self._update_tables_container_visibility()

    # ------------------- Helper methods ----------------

    def _get_completions_callback(self, target_state) -> list[DropdownItem]:
        """
        Callback for AutoComplete to get completion candidates from the kernel.

        Args:
            target_state: TargetState object from textual-autocomplete with input text

        Returns:
            List of DropdownItem objects for the autocomplete dropdown
        """
        value = target_state.text
        cursor_pos = target_state.cursor_position

        logger.debug(f"Completion callback: text='{value}', cursor_pos={cursor_pos}")

        if not value:
            logger.debug("Empty value, returning no completions")
            return []

        # Check cache first - return cached results immediately
        if value in self._completion_cache:
            cached = self._completion_cache[value]
            logger.debug(f"Using cached completions: {len(cached)} items")
            return [DropdownItem(main=str(c)) for c in cached]

        # Function to run async completion in a separate thread with its own event loop
        def run_completion_sync():
            """Run the async completion in a new event loop."""

            async def get_completions():
                try:
                    logger.debug(f"Requesting completions from kernel for '{value}' at pos {cursor_pos}")
                    completions = await self.input.get_completions(value, cursor_pos)
                    logger.debug(f"Kernel returned {len(completions)} completions")
                    return completions
                except Exception as e:
                    logger.error(f"Error getting completions: {e}", exc_info=True)
                    return []

            # Create a new event loop for this thread and run the coroutine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(get_completions())
                return result
            finally:
                loop.close()

        # Submit to thread pool and wait with timeout
        try:
            logger.debug("Submitting completion request to thread pool")
            future = self._executor.submit(run_completion_sync)
            completions = future.result(timeout=0.5)

            logger.debug(f"Got {len(completions)} completions from thread pool")
            # Cache the results
            self._completion_cache[value] = completions
            return [DropdownItem(main=str(c)) for c in completions]
        except concurrent.futures.TimeoutError:
            logger.warning("Completion request timed out after 0.5s")
            return []
        except Exception as e:
            logger.error(f"Error in completion callback: {e}", exc_info=True)
            return []

    def _request_dashboard(self) -> None:
        """Request dashboard update from the kernel (called by interval timer)."""
        if self.event_handler.is_dashboard_busy():
            return
        self.event_handler.mark_dashboard_busy()
        self.kernel.execute("emit_dashboard()", silent=True)
