"""
Textual-based TUI for running qubx strategies with Jupyter kernel integration.

Features:
- Live REPL output panel for kernel interaction
- Application logs panel for qubx logger output
- Input bar for executing Python code in the kernel
- Real-time strategy monitoring
"""

import asyncio
import traceback
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input
from textual.widgets import RichLog as TuiLog

from qubx import logger


# --------------------------- Kernel wrapper ---------------------------------
class IPyKernel:
    """Wrapper around AsyncKernelManager for managing Jupyter kernel lifecycle."""

    def __init__(self) -> None:
        self.km = None  # AsyncKernelManager
        self.kc = None  # AsyncKernelClient
        self.iopub_task: asyncio.Task | None = None
        self.callbacks: list = []  # callables receiving (kind, payload)

    async def start(self) -> None:
        """Start the Jupyter kernel and its channels."""
        from jupyter_client import AsyncKernelManager

        self.km = AsyncKernelManager(kernel_name="python3")
        await self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        # Ensure kernel is ready
        await self.kc.wait_for_ready()
        self.iopub_task = asyncio.create_task(self._drain_iopub())

    async def stop(self) -> None:
        """Stop the kernel and its channels."""
        if self.kc:
            self.kc.stop_channels()
        if self.km:
            await self.km.shutdown_kernel(now=False)
        if self.iopub_task:
            self.iopub_task.cancel()

    def register(self, cb):
        """Register a callback to receive kernel events."""
        self.callbacks.append(cb)

    def execute(self, code: str, *, silent: bool = False) -> str:
        """Execute code in the kernel and return the message ID."""
        if not self.kc:
            raise RuntimeError("Kernel client not ready")
        msg_id = self.kc.execute(code, allow_stdin=False, silent=silent)
        return msg_id

    async def interrupt(self) -> None:
        """Send interrupt signal to the kernel."""
        if self.km:
            await self.km.interrupt_kernel()

    async def _drain_iopub(self):
        """Continuously drain iopub messages from the kernel."""
        try:
            while True:
                msg = await self.kc.get_iopub_msg()
                msg_type = msg["header"]["msg_type"]
                content = msg["content"]

                # Skip certain message types
                if msg_type in ("status", "comm_open", "comm_msg", "comm_close", "execute_input"):
                    continue

                # Handle different message types
                if msg_type == "stream":
                    self._emit("stream", {"name": content.get("name"), "text": content.get("text", "")})
                elif msg_type in ("display_data", "execute_result"):
                    data = content.get("data", {})
                    # Check for custom MIME type for positions
                    if "application/x-qubx-positions+json" in data:
                        self._emit("qubx_positions", data["application/x-qubx-positions+json"])
                    elif "text/markdown" in data:
                        self._emit("markdown", data["text/markdown"])
                    elif "text/plain" in data:
                        self._emit("text", data["text/plain"])
                    else:
                        self._emit("text", str(data))
                elif msg_type == "error":
                    self._emit(
                        "error",
                        {
                            "ename": content.get("ename"),
                            "evalue": content.get("evalue"),
                            "traceback": "\n".join(content.get("traceback", [])),
                        },
                    )
                elif msg_type == "clear_output":
                    self._emit("clear", {})
                else:
                    self._emit("debug", {"msg_type": msg_type, "content": content})
        except asyncio.CancelledError:
            return
        except Exception:
            self._emit("fatal", traceback.format_exc())

    def _emit(self, kind: str, payload: Any) -> None:
        """Emit an event to all registered callbacks."""
        for cb in list(self.callbacks):
            try:
                cb(kind, payload)
            except Exception:
                logger.exception("Callback failed")


# --------------------------- UI Widgets -------------------------------------
class ReplOutput(TuiLog):
    """REPL output widget with clear functionality and line limit."""

    def __init__(self, *args, **kwargs):
        # Set max_lines to limit history
        kwargs['max_lines'] = 10000
        super().__init__(*args, **kwargs)

    def clear_output(self):
        """Clear all output from the REPL."""
        self.clear()


# --------------------------- Main Textual App -------------------------------
class TextualStrategyApp(App[None]):
    """Main Textual application for running strategies with kernel interaction."""

    CSS = """
    * {
        transition: none;
        scrollbar-gutter: stable;
    }

    Screen {
        layout: vertical;
    }

    #main-container {
        height: 1fr;
        layout: horizontal;
    }

    #output-container {
        width: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #output-container:focus-within {
        border: solid $primary;
    }

    #positions-panel {
        width: 50%;
        border: solid $success;
        padding: 1;
        display: none;
    }

    #positions-panel.visible {
        display: block;
    }

    #positions-table {
        height: 1fr;
    }

    DataTable {
        scrollbar-gutter: stable;
        border: none;
    }

    DataTable:focus {
        border: none;
    }

    DataTable:hover {
        border: none;
    }

    #input-container {
        height: auto;
        padding: 1 2;
        background: $background;
    }

    Input {
        height: 3;
        border: solid $accent;
        background: $surface;
        color: $text;
    }

    Input:focus {
        border: solid $accent;
        background: $surface;
        color: $text;
    }

    Input.-invalid {
        border: solid $accent;
    }

    Input.-valid {
        border: solid $accent;
    }

    Input > .input--placeholder {
        color: $text-muted;
    }

    Input > .input--cursor {
        background: $accent;
        color: $text;
        text-style: none;
    }

    Footer {
        background: $panel;
    }

    RichLog {
        height: 1fr;
        background: $surface;
        scrollbar-gutter: stable;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_repl", "Clear REPL", show=True),
        Binding("ctrl+c", "interrupt", "Interrupt", show=True),
        Binding("p", "toggle_positions", "Positions", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, config_file: Path, account_file: Path | None, paper: bool, restore: bool) -> None:
        super().__init__()
        self.config_file = config_file
        self.account_file = account_file
        self.paper = paper
        self.restore = restore
        self.kernel = IPyKernel()
        self.ctx = None  # Strategy context
        self.output: ReplOutput
        self.input: Input
        self.positions_panel: Vertical
        self.positions_table: DataTable
        self.positions_visible = False
        self._positions_busy = False

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Add welcome message
        self.output.write("[bold cyan]Qubx Strategy Runner[/bold cyan]")
        self.output.write("[dim]Type Python commands below and press Enter to execute[/dim]")
        self.output.write("")

        # Start the kernel
        await self.kernel.start()
        self.kernel.register(self.on_kernel_event)

        # Initialize the strategy context within the kernel
        self.output.write("[yellow]Initializing strategy context...")
        try:
            # Pre-load the context and helpers into the kernel
            init_code = self._generate_init_code()
            self.kernel.execute(init_code, silent=False)

            self.output.write("[green]✓ Strategy context initialized and ready!")
            self.output.write(
                "[cyan]Available objects: ctx, S (strategy), portfolio(), orders(), trade(), exit()"
            )
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
                self.output = ReplOutput(id="output", wrap=True, markup=True)
                yield self.output
            self.positions_panel = Vertical(id="positions-panel")
            with self.positions_panel:
                self.positions_table = DataTable(id="positions-table", cursor_type="row")
                self.positions_table.add_columns(
                    "Symbol", "Side", "Qty", "Avg Px", "Last Px", "PnL", "Mkt Value"
                )
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

    # ------------------- Kernel events ----------------
    def on_kernel_event(self, kind: str, payload: Any) -> None:
        """Handle kernel events and display them in the output."""
        if kind == "qubx_positions":
            self._positions_busy = False
            # Schedule the update on the Textual thread
            self.call_later(self._update_positions_table, payload)
        elif kind == "stream":
            text = payload.get("text", "")
            self.output.write(text.rstrip("\n"))
        elif kind == "text":
            self.output.write(str(payload))
        elif kind == "markdown":
            self.output.write(Markdown(payload))
        elif kind == "error":
            tb = payload.get("traceback", "")
            self.output.write(f"[red]{payload.get('ename')}: {payload.get('evalue')}\n{tb}")
        elif kind == "clear":
            self.output.clear_output()
        elif kind == "debug":
            # Ignore debug messages
            pass
        else:
            # Ignore other events
            pass

    # ------------------- Helper methods ----------------

    def _request_positions(self) -> None:
        """Request positions update from the kernel (called by interval timer)."""
        if not self.positions_visible or self._positions_busy:
            return
        self._positions_busy = True
        self.kernel.execute("emit_positions()", silent=True)

    def _update_positions_table(self, rows: list[dict]) -> None:
        """Update the positions table with new data from the kernel."""
        # Only update if there are actual changes
        if not rows:
            return

        # Clear rows only (keep column headers)
        self.positions_table.clear(columns=False)

        # Sort by market value (descending)
        sorted_rows = sorted(rows, key=lambda r: abs(r.get("mkt_value", 0)), reverse=True)

        # Add rows
        for r in sorted_rows:
            self.positions_table.add_row(
                r["symbol"],
                r["side"],
                str(r["qty"]),
                str(r["avg_px"]),
                str(r["last_px"]),
                f"{r['pnl']:.2f}",
                f"{r['mkt_value']:.3f}",
            )

    def _generate_init_code(self) -> str:
        """Generate initialization code to inject context into the kernel."""
        # Convert paths to strings for the code generation
        config_path_str = str(self.config_file.absolute())
        account_path_str = str(self.account_file.absolute()) if self.account_file else "None"

        return f"""
import pandas as pd
import sys
from pathlib import Path
from qubx import logger
from qubx.core.basics import Instrument, Position
from qubx.core.context import StrategyContext
from qubx.core.interfaces import IStrategyContext
from qubx.utils.misc import dequotify, add_project_to_system_path
from qubx.utils.runner.runner import run_strategy_yaml
import nest_asyncio

# Apply nest_asyncio for nested event loops
nest_asyncio.apply()

pd.set_option('display.max_colwidth', None, 'display.max_columns', None, 'display.width', 1000)

# Initialize the strategy context
config_file = Path('{config_path_str}')
account_file = Path('{account_path_str}') if '{account_path_str}' != 'None' else None

# Add project to system path
add_project_to_system_path()
add_project_to_system_path(str(config_file.parent.parent))
add_project_to_system_path(str(config_file.parent))

# Run the strategy
ctx = run_strategy_yaml(config_file, account_file, paper={self.paper}, restore={self.restore}, blocking=False)
S = ctx.strategy

def _pos_to_dict(p: Position):
    mv = round(p.notional_value, 3)
    return dict(
        MktValue=mv,
        Position=round(p.quantity, p.instrument.size_precision),
        PnL=p.total_pnl(),
        AvgPrice=round(p.position_avg_price_funds, p.instrument.price_precision),
        LastPrice=round(p.last_update_price, p.instrument.price_precision),
    )

def portfolio(all=True):
    from tabulate import tabulate
    d = dict()
    for s, p in ctx.get_positions().items():
        if p.quantity != 0.0 or all:
            d[dequotify(s.symbol, s.quote)] = _pos_to_dict(p)
    d = pd.DataFrame.from_dict(d, orient='index')
    if d.empty:
        print('-(no open positions yet)-')
        return
    d = d.sort_values('MktValue', ascending=False)
    print(tabulate(d, ['MktValue', 'Position', 'PnL', 'AvgPrice', 'LastPrice'], tablefmt='rounded_grid'))

def orders():
    if (_orders := ctx.get_orders()):
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        for k, (i, o) in enumerate(_orders.items()):
            print(f" [{{k}}] {{i}} {{o.status}} {{o.side}} {{o.instrument.symbol}} {{o.quantity}} @ {{o.price}} - {{o.time}}")
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

def trade(instrument, qty: float, price=None, tif='gtc'):
    return ctx.trade(instrument if isinstance(instrument, Instrument) else instrument._instrument, qty, price, tif)

__exit = exit
def exit():
    ctx.stop()
    __exit()

# Positions emit helper for Textual UI
from IPython.display import display

_CUSTOM_MIME_POS = "application/x-qubx-positions+json"

def _positions_as_records(all=True):
    rows = []
    try:
        for s, p in ctx.get_positions().items():
            if p.quantity != 0.0 or all:
                # Handle NaN values by converting to None or 0
                pnl = p.total_pnl()
                if not isinstance(pnl, (int, float)) or (isinstance(pnl, float) and (pnl != pnl or pnl == float('inf') or pnl == float('-inf'))):
                    pnl = 0.0

                mkt_value = p.notional_value
                if not isinstance(mkt_value, (int, float)) or (isinstance(mkt_value, float) and (mkt_value != mkt_value or mkt_value == float('inf') or mkt_value == float('-inf'))):
                    mkt_value = 0.0

                rows.append({{
                    "symbol": s.symbol,
                    "side": "LONG" if p.quantity > 0 else ("SHORT" if p.quantity < 0 else "FLAT"),
                    "qty": round(p.quantity, s.size_precision),
                    "avg_px": round(p.position_avg_price_funds, s.price_precision),
                    "last_px": round(p.last_update_price, s.price_precision),
                    "pnl": round(pnl, 2),
                    "mkt_value": round(mkt_value, 3),
                }})
    except Exception:
        pass  # Context not ready yet
    return rows

def emit_positions(all=True):
    \"\"\"Publish current positions via custom MIME for Textual to capture.\"\"\"
    try:
        data = {{ _CUSTOM_MIME_POS: _positions_as_records(all=all) }}
        display(data, raw=True)
    except Exception:
        pass  # Silently fail if context is not ready

print(f"Strategy initialized: {{ctx.strategy.__class__.__name__}}")
print(f"Instruments: {{[i.symbol for i in ctx.instruments]}}")
"""


# --------------------------- Entry point ------------------------------------
def run_strategy_yaml_in_textual(
    config_file: Path, account_file: Path | None = None, paper: bool = False, restore: bool = False
) -> None:
    """
    Run a strategy in a Textual TUI with Jupyter kernel integration.

    Args:
        config_file: Path to the strategy configuration file
        account_file: Path to the account configuration file
        paper: Whether to run in paper trading mode
        restore: Whether to restore the strategy state
    """
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        logger.error("Can't find <r>nest_asyncio</r> module - try to install it first")
        return

    try:
        from jupyter_client import AsyncKernelManager  # noqa: F401
    except ImportError:
        logger.error("Can't find <r>jupyter_client</r> module - try to install it first")
        return

    logger.info("Running strategy in Textual TUI mode")

    # Create and run the app
    app = TextualStrategyApp(config_file, account_file, paper, restore)

    try:
        app.run()
    except Exception as e:
        logger.error(f"Textual app failed: {e}")
        raise
