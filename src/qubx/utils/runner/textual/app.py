"""
Main Textual application for running strategies with Jupyter kernel integration.
"""

import asyncio
import concurrent.futures
import sys
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult, ScreenStackError
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
        Binding("p", "toggle_positions", "Positions", show=True),
        Binding("o", "toggle_orders", "Orders", show=True),
        Binding("m", "toggle_market", "Market", show=True),
        # Binding("d", "toggle_debug", "Debug", show=True),
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
        kernel: IPyKernel | None = None,
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
            kernel: Optional pre-connected kernel instance
        """
        super().__init__(*args, **kwargs)
        self.config_file = config_file
        self.account_file = account_file
        self.paper = paper
        self.restore = restore
        self.connection_file = connection_file
        self.test_mode = test_mode
        self.kernel = kernel if kernel is not None else IPyKernel()
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
        self.output.write(Text("Qubx Strategy Runner", style="bold cyan"))
        self.output.write(Text("Type Python commands below and press Enter to execute", style="dim"))
        self.output.write("")

        # Check if kernel is already connected (passed from outside)
        if self.kernel.is_connected():
            self.output.write(Text("✓ Using pre-connected kernel!", style="green"))
            # Start iopub listener on this event loop (it wasn't started during pre-connection)
            self.kernel.start_iopub_listener()
            self.kernel.register(self.event_handler.handle_event)

            # Retrieve and display output history
            self.output.write(Text("Retrieving output history...", style="yellow"))
            # history = await self.kernel.get_output_history()
            # if history:
            #     self.output.write(f"[green]✓ Retrieved {len(history)} history entries")
            #     # Display recent history (last 50 entries)
            #     for entry in history[-50:]:
            #         entry_type = entry.get("type", "text")
            #         content = entry.get("content", "")
            #         if entry_type == "stream" and isinstance(content, dict):
            #             text = content.get("text", "")
            #             self.output.write(text)
            #         elif entry_type == "text":
            #             self.output.write(str(content))
            #         elif entry_type == "html":
            #             self.output.write(str(content))  # Display HTML as text for now
            #         elif entry_type == "error" and isinstance(content, dict):
            #             self.output.write(f"[red]{content.get('ename', 'Error')}: {content.get('evalue', '')}")
            # else:
            #     self.output.write("[dim]No previous history found")
        else:
            # Start a new kernel
            self.output.write(Text("Starting new kernel...", style="yellow"))
            await self.kernel.start()
            self.kernel.register(self.event_handler.handle_event)
            init_code = generate_init_code(self.config_file, self.account_file, self.paper, self.restore)
            self.kernel.execute(init_code, silent=False)

        # Setup debug log handler
        # Remove all default handlers (console output)
        logger.remove()

        # Create a sink function that writes to the debug log
        def debug_sink(message):
            """Sink function for loguru to write to debug log."""
            record = message.record
            level = record["level"].name
            msg = record["message"]
            self.debug_log.write_debug(level, msg)

        # Add ONLY the debug panel sink - all logs go to TUI only
        self._log_handler_id = logger.add(debug_sink, level="DEBUG")

        # Initialize the strategy context within the kernel (only if NOT connecting to existing or pre-connected)
        if not self.connection_file and not self.kernel.is_connected():
            if self.test_mode:
                self.output.write(Text("Initializing test mode...", style="yellow"))
                try:
                    init_code = generate_mock_init_code()
                    self.kernel.execute(init_code, silent=False)
                    self.output.write(Text("✓ Test mode initialized!", style="green"))
                except Exception as e:
                    self.output.write(Text(f"Failed to initialize test mode: {e}", style="red"))
                    logger.exception("Test mode initialization failed")
            else:
                self.output.write(Text("Initializing strategy context...", style="yellow"))
                try:
                    # Pre-load the context and helpers into the kernel
                    init_code = generate_init_code(
                        self.config_file,
                        self.account_file,
                        self.paper,
                        self.restore,
                    )
                    self.kernel.execute(init_code, silent=False)

                    self.output.write(Text("✓ Strategy context initialized and ready!", style="green"))
                except Exception as e:
                    self.output.write(Text(f"Failed to initialize strategy: {e}", style="red"))
                    logger.exception("Strategy initialization failed")

        # Start interval timer for dashboard updates (1 second)
        self.set_interval(1.0, self._request_dashboard)

    async def on_unmount(self) -> None:
        """Clean up when app is closing."""
        # Shutdown thread pool executor
        self._executor.shutdown(wait=False)

        # Remove custom log handler
        if self._log_handler_id is not None:
            logger.remove(self._log_handler_id)

        # Restore default logger configuration (remove all and re-add default)
        logger.remove()
        logger.add(sys.stderr, level="INFO")  # Restore normal INFO level logging

        # Stop the strategy context BEFORE shutting down the kernel
        # This ensures on_stop() and notifiers are called
        logger.info("[TextualApp] :: Stopping strategy context...")
        try:
            # Execute ctx.stop() in the kernel to trigger proper shutdown
            self.kernel.execute("if 'ctx' in globals() and ctx is not None: ctx.stop()", silent=True)
            # Give it a moment to process
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"[TextualApp] :: Failed to stop context: {e}")

        # Now stop the kernel
        await self.kernel.stop()

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=True)
        with Vertical(id="content-wrapper"):
            with Horizontal(id="main-container"):
                # Output on the left
                with Vertical(id="output-container"):
                    self.output = ReplOutput(id="output", max_lines=10000)
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
                    # self.debug_panel = Vertical(id="debug-panel", classes="side-panel")
                    # with self.debug_panel:
                    #     self.debug_log = DebugLog(id="debug-log", wrap=True, markup=True, max_lines=1000)
                    #     yield self.debug_log
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
        self.output.write(Text(f">>> {code}", style="bold cyan"))
        self.kernel.execute(code)

    # ------------------- Actions ---------------------

    def action_clear_repl(self) -> None:
        """Clear the output."""
        self.output.clear_output()

    async def action_interrupt(self) -> None:
        """Send interrupt signal to the kernel."""
        await self.kernel.interrupt()
        self.output.write(Text("⚠ KeyboardInterrupt sent to kernel", style="orange1"))

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

    def action_quit(self) -> None:
        """Quit the application cleanly."""
        self.exit()

    def _handle_exception(self, error: Exception) -> None:
        """Handle exceptions, suppressing ScreenStackError during shutdown."""
        if isinstance(error, ScreenStackError):
            # Suppress this error during shutdown - it's expected when widgets
            # try to access screen after it's been removed
            return
        super()._handle_exception(error)

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
        try:
            if self.event_handler.is_dashboard_busy():
                return
            self.event_handler.mark_dashboard_busy()
            self.kernel.execute("emit_dashboard()", silent=True)
        except Exception as e:
            logger.error(f"Error requesting dashboard update: {e}", exc_info=True)
        finally:
            self.event_handler.mark_dashboard_ready()
