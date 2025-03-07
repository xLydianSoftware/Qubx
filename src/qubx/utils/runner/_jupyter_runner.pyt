import qubx
%qubxd

%load_ext autoreload
%autoreload 2

import pandas as pd
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from qubx.utils.misc import add_project_to_system_path, red, green, yellow, blue, magenta, cyan
from qubx.core.basics import Instrument, Position
from qubx.core.context import StrategyContext
from qubx.core.interfaces import IPositionGathering, IPositionSizer, PositionsTracker
from qubx.utils.misc import dequotify, quotify
from qubx.utils.runner.runner import run_strategy_yaml
from qubx.pandaz.utils import *
import qubx.pandaz.ta as pta
import qubx.ta.indicators as ta

pd.set_option('display.max_colwidth', None, 'display.max_columns', None, 'display.width', 1000) # type: ignore

config_file = Path('{config_file}')
add_project_to_system_path()
add_project_to_system_path(str(config_file.parent.parent))
add_project_to_system_path(str(config_file.parent))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
account_file = Path('{account_file}') if '{account_file}' != 'None' else None
ctx: StrategyContext = run_strategy_yaml(Path('{config_file}'), account_file, {paper}, {restore}) # type: ignore
assert ctx is not None, 'Strategy context is not created'

S = ctx.strategy

def _pos_to_dict(p: Position):
    mv = round(p.market_value_funds, 3)
    return dict(
            Position=round(p.quantity, p.instrument.size_precision),  
            PnL=p.total_pnl(), 
            AvgPrice=round(p.position_avg_price_funds, p.instrument.price_precision), 
            LastPrice=round(p.last_update_price, p.instrument.price_precision),
            MktValue=mv)


class ActiveInstrument:
    def __init__(self, ctx: StrategyContext, instrument: Instrument):
        self._instrument = instrument
        # - bad hack - need to provide getters for that
        self._tracker: PositionsTracker = ctx._processing_manager._position_tracker # type: ignore
        self._sizer: IPositionSizer = self._tracker.get_position_sizer()
        self._gathering: IPositionGathering = ctx._processing_manager._position_gathering # type: ignore
    
    def exchange(self):
        return self._instrument.exchange

    def trade(self, qty: float, price=None, tif='gtc'):
        return ctx.trade(self._instrument, qty, price, tif)

    def signal(self, s: float, price: float | None = None, 
               stop: float | None = None,
               take: float | None = None,
               comment: str = ''):
        _targets = self._tracker.process_signals(ctx, [self._instrument.signal(s, price, stop, take, comment=comment)])
        self._gathering.alter_positions(ctx, _targets)

    def __le__(self, other: float):
        self.signal(other)
        return self

    def close(self):
        if (p:=ctx.get_position(self._instrument)).quantity != 0:
            ctx.trade(self._instrument, -p.quantity)

    def cancel(self):
        for o in ctx.get_orders(self._instrument).values():
            ctx.cancel_order(o.id)

    def orders(self):
        for i, o in ctx.get_orders(self._instrument).items():
            print("\t" + green(i) + " " + o.status + " " + o.side + " " + green(o.instrument.symbol) + " " +  str(o.quantity) + " @ " + str(o.price) + " - " +  blue(str(o.time)))

    def ohlc(self, timeframe: str, length: int = 50):
        return ctx.ohlc(self._instrument, timeframe, length)

    def __repr__(self):
        p = ctx.get_position(self._instrument)
        q = ctx.quote(self._instrument)
        _present = dict()
        if p:
            _present = _pos_to_dict(p)

        if q:
            _present['bid'] = str(q.bid) # type: ignore
            _present['ask'] = str(q.ask) # type: ignore

        return pd.DataFrame(_present, index=[self._instrument.symbol]).to_string()


def _pollute_caller_globals(ctx: StrategyContext):
    import sys
    d = sys._getframe(1).f_globals

    for i in ctx.instruments:
        d[i.symbol] = ActiveInstrument(ctx, i)


_pollute_caller_globals(ctx)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def orders(instrument: Instrument | ActiveInstrument | None=None):
    if (_orders:=ctx.get_orders()):
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        for i, o in _orders.items():
            print("\t" + green(i) + " " + blue(o.status) + " " + green(o.side) + " " + red(o.instrument.symbol) + " " +  str(o.quantity) + " @ " + str(o.price) + " - " +  blue(str(o.time)))
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    # return ctx.get_orders(instrument if isinstance(instrument, Instrument) or instrument is None else instrument._instrument)

def trade(instrument: Instrument | ActiveInstrument, qty: float, price=None, tif='gtc'):
    return ctx.trade(instrument if isinstance(instrument, Instrument) else instrument._instrument, qty, price, tif)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def portfolio(all=True):
    from tabulate import tabulate

    d = dict()
    for s, p in ctx.get_positions().items():
        mv = round(p.market_value_funds, 3)
        if mv != 0.0 or all:
            d[dequotify(s.symbol)] = _pos_to_dict(p)

    d = pd.DataFrame.from_dict(d, orient='index')
    # d = d[d['PnL'] != 0.0]
    if d.empty:
        print('-(no open positions yet)-')
        return

    d = d.sort_values('PnL' ,ascending=False)
    # d = pd.concat((d, pd.Series(dict(TOTAL=d['PnL'].sum()), name='PnL'))).fillna('')
    d = pd.concat((d, scols(pd.Series(dict(TOTAL=d['PnL'].sum()), name='PnL'), pd.Series(dict(TOTAL=d['MktValue'].sum()), name='MktValue')))).fillna('')
    print(tabulate(d, ['Position', 'PnL', 'AvgPrice', 'LastPrice', 'MktValue'], tablefmt='rounded_grid'))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
__exit = exit
def exit():
    ctx.stop()
    __exit()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

portfolio()
orders()
