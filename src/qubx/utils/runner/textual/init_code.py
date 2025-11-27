"""
Strategy initialization code generation for Jupyter kernel.

Generates Python code to initialize strategy context and helper functions
that will be executed in the kernel.
"""

from pathlib import Path


def generate_init_code(
    config_file: Path,
    account_file: Path | None,
    paper: bool,
    restore: bool,
) -> str:
    """
    Generate initialization code to inject strategy context into the kernel.

    Args:
        config_file: Path to the strategy configuration file
        account_file: Optional path to the account configuration file
        paper: Whether to run in paper trading mode
        restore: Whether to restore the strategy state

    Returns:
        Python code string to be executed in the kernel
    """
    # Convert paths to strings for the code generation
    config_path_str = str(config_file.absolute())
    account_path_str = str(account_file.absolute()) if account_file else "None"

    return f"""
import pandas as pd
import sys
import time
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

# ===== Output History Tracking =====
# Store REPL output history for new connections
_qubx_output_history = []
_qubx_output_max = 1000  # Maximum history entries to keep

def _qubx_store_output(msg_type: str, content):
    \"\"\"Store an output entry in history.\"\"\"
    global _qubx_output_history
    _qubx_output_history.append({{
        'timestamp': time.time(),
        'type': msg_type,
        'content': content
    }})
    # Keep history bounded
    if len(_qubx_output_history) > _qubx_output_max:
        _qubx_output_history.pop(0)

# Hook into IPython output system
from IPython import get_ipython
_ipython = get_ipython()
if _ipython:
    def _qubx_post_execute_hook():
        \"\"\"Capture output after execution.\"\"\"
        # This hook is called after each cell execution
        # History is automatically tracked by IPython, we just store metadata
        pass

    _ipython.events.register('post_execute', _qubx_post_execute_hook)

# Initialize the strategy context
config_file = Path('{config_path_str}')
account_file = Path('{account_path_str}') if '{account_path_str}' != 'None' else None

# Add project to system path
add_project_to_system_path()
add_project_to_system_path(str(config_file.parent.parent))
add_project_to_system_path(str(config_file.parent))

# Run the strategy
ctx = run_strategy_yaml(config_file, account_file, paper={paper}, restore={restore}, blocking=False)
S = ctx.strategy

def _sanitize_number(value):
    \"\"\"Convert NaN or infinity to 0.0 for safe JSON serialization.\"\"\"
    if not isinstance(value, (int, float)):
        return 0.0
    if value != value or value == float('inf') or value == float('-inf'):
        return 0.0
    return value

def _pos_to_dict(p: Position):
    mv = round(p.notional_value, 3)
    return dict(
        MktValue=mv,
        Position=round(p.quantity, p.instrument.size_precision),
        PnL=p.pnl,
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

# Unified dashboard emit helper for Textual UI
from IPython.display import display

_CUSTOM_MIME_DASHBOARD = "application/x-qubx-dashboard+json"

def _positions_as_records(all=True):
    \"\"\"Convert positions to records for dashboard.\"\"\"
    rows = []
    try:
        for s, p in ctx.get_positions().items():
            if p.quantity != 0.0 or all:
                rows.append({{
                    "exchange": s.exchange,
                    "symbol": s.symbol,
                    "side": "LONG" if p.quantity > 0 else ("SHORT" if p.quantity < 0 else "FLAT"),
                    "leverage": _sanitize_number(ctx.get_leverage(s)),
                    "qty": _sanitize_number(round(p.quantity, s.size_precision)),
                    "avg_px": _sanitize_number(round(p.position_avg_price_funds, s.price_precision)),
                    "last_px": _sanitize_number(round(p.last_update_price, s.price_precision)),
                    "pnl": _sanitize_number(round(p.pnl, 2)),
                    "mkt_value": _sanitize_number(round(p.notional_value, 3)),
                }})
    except Exception:
        pass  # Context not ready yet
    return rows

def _orders_as_records():
    \"\"\"Convert orders to records for dashboard.\"\"\"
    rows = []
    try:
        orders = ctx.get_orders()
        if orders:
            for order_id, order in orders.items():
                rows.append({{
                    "exchange": order.instrument.exchange,
                    "symbol": order.instrument.symbol,
                    "side": order.side,
                    "type": order.type,
                    "qty": _sanitize_number(order.quantity),
                    "price": _sanitize_number(order.price) if order.price else None,
                    "status": order.status,
                    "time": str(order.time),
                    "id": order_id,
                }})
    except Exception:
        pass  # Context not ready yet
    return rows

def _quotes_as_records():
    \"\"\"Convert quotes to records for dashboard.\"\"\"
    quotes = {{}}
    try:
        for instrument in ctx.get_instruments():
            quote = ctx.quote(instrument)
            if quote:
                key = f"{{instrument.exchange}}:{{instrument.symbol}}"
                spread = quote.ask - quote.bid if quote.bid and quote.ask else 0.0
                spread_pct = (spread / quote.bid * 100) if quote.bid and quote.bid > 0 else 0.0
                quotes[key] = {{
                    "exchange": instrument.exchange,
                    "symbol": instrument.symbol,
                    "bid": _sanitize_number(round(quote.bid, instrument.price_precision)) if quote.bid else None,
                    "ask": _sanitize_number(round(quote.ask, instrument.price_precision)) if quote.ask else None,
                    "spread": _sanitize_number(round(spread, instrument.price_precision)),
                    "spread_pct": _sanitize_number(round(spread_pct, 4)),
                }}
    except Exception:
        pass  # Context not ready yet
    return quotes

def emit_dashboard(all=True, debug=False):
    \"\"\"Publish unified dashboard data via custom MIME for Textual to capture.\"\"\"
    try:
        data = {{
            "positions": _positions_as_records(all=all),
            "orders": _orders_as_records(),
            "quotes": _quotes_as_records(),
        }}

        # Let strategy inject custom data
        if hasattr(S, 'get_dashboard_data'):
            custom = S.get_dashboard_data(ctx)
            if custom:
                data["custom"] = custom

        display({{ _CUSTOM_MIME_DASHBOARD: data }}, raw=True)
    except Exception as e:
        if debug:
            import traceback
            print(f"emit_dashboard error: {{e}}")
            print(traceback.format_exc())
        # Silently fail if context is not ready

print(f"Strategy initialized: {{ctx.strategy.__class__.__name__}}")
print(f"Instruments: {{[i.symbol for i in ctx.instruments]}}")
print(f"Available: ctx, S (strategy), portfolio(), orders(), trade(), emit_dashboard(), exit()")
"""


def generate_mock_init_code() -> str:
    """
    Generate minimal initialization code for testing without a real strategy.

    This creates an empty kernel environment with mock dashboard functions
    but skips strategy loading and initialization.

    Returns:
        Python code string to be executed in the kernel for testing
    """
    return """
import pandas as pd
import time
from IPython.display import display

pd.set_option('display.max_colwidth', None, 'display.max_columns', None, 'display.width', 1000)

# ===== Output History Tracking =====
_qubx_output_history = []
_qubx_output_max = 1000

def _qubx_store_output(msg_type: str, content):
    global _qubx_output_history
    _qubx_output_history.append({
        'timestamp': time.time(),
        'type': msg_type,
        'content': content
    })
    if len(_qubx_output_history) > _qubx_output_max:
        _qubx_output_history.pop(0)

# Mock context objects
ctx = None
S = None

# Unified dashboard emit helper for Textual UI (mock version)
_CUSTOM_MIME_DASHBOARD = "application/x-qubx-dashboard+json"

def emit_dashboard(all=True, debug=False):
    \"\"\"Mock dashboard emitter that returns empty data.\"\"\"
    data = {
        "positions": [],
        "orders": [],
        "quotes": {},
        "custom": {}
    }
    display({ _CUSTOM_MIME_DASHBOARD: data }, raw=True)

def portfolio(all=True):
    print('-(no open positions - test mode)-')

def orders():
    print('-(no orders - test mode)-')

def trade(instrument, qty: float, price=None, tif='gtc'):
    print(f'-(test mode: would trade {qty} of {instrument})-')
    return None

def exit():
    pass

print("Test mode: Minimal kernel environment initialized")
print("Available: emit_dashboard(), portfolio(), orders(), trade(), exit()")
"""
