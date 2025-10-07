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
ctx = run_strategy_yaml(config_file, account_file, paper={paper}, restore={restore}, blocking=False)
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
