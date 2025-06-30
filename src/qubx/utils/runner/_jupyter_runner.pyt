import qubx
%qubxd

# %load_ext autoreload
# %autoreload 2

import pandas as pd
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from qubx import logger
from qubx.utils.misc import add_project_to_system_path, red, green, yellow, blue, magenta, cyan
from qubx.core.basics import Instrument, Position
from qubx.core.context import StrategyContext
from qubx.core.interfaces import IPositionGathering, IPositionSizer, IStrategyContext, PositionsTracker
from qubx.utils.misc import dequotify, quotify
from qubx.utils.runner.runner import run_strategy_yaml
from qubx.pandaz.utils import *
import qubx.pandaz.ta as pta
import qubx.ta.indicators as ta
from qubx.core.lookups import lookup
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


pd.set_option('display.max_colwidth', None, 'display.max_columns', None, 'display.width', 1000) # type: ignore

# - remove projects folder from path (poisoned by qubxd)
import sys
for s in list(sys.path):
    if './projects' in s:
        sys.path.remove(s)

config_file = Path('{config_file}')
# add_project_to_system_path()
# add_project_to_system_path(str(config_file.parent.parent))
add_project_to_system_path(str(config_file.parent))
# add_project_to_system_path('.')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
account_file = Path('{account_file}') if '{account_file}' != 'None' else None
ctx: StrategyContext = run_strategy_yaml(Path('{config_file}'), account_file, {paper}, {restore}) # type: ignore
assert ctx is not None, 'Strategy context is not created'

def _pollute_caller_globals(ctx: StrategyContext, n_level=1):
    for i in ctx.instruments:
        globals()[i.symbol] = ActiveInstrument(ctx, i)

S = ctx.strategy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - another dirty hack - need to intercept call to original on_universe_change and then call original
import inspect
def __interceptor_on_universe_change(func):

    def _intercepted(ctx: IStrategyContext, added: list[Instrument], removed: list[Instrument]):
        # logger.info(f">>> intercepted " + func.__name__ +  " added " + str(added) + " removed " + str(removed))
        result = func(ctx, added, removed)
        _globs = globals()
        for i in added:
            _globs[i.symbol] = ActiveInstrument(ctx, i)

        for i in removed:
            _globs.pop(i.symbol)

        # print new portfolio
        print(" - New Universe - ")
        portfolio(True)

        return result
    return _intercepted

for x in inspect.getmembers(S, (inspect.ismethod)):
    if x[0] == 'on_universe_change':
        setattr(S, x[0], __interceptor_on_universe_change(getattr(S, x[0])))
        break
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def _pos_to_dict(p: Position):
    mv = round(p.market_value_funds, 3)
    return dict(
            Position=round(p.quantity, p.instrument.size_precision),  
            PnL=p.total_pnl(), 
            AvgPrice=round(p.position_avg_price_funds, p.instrument.price_precision), 
            LastPrice=round(p.last_update_price, p.instrument.price_precision),
            MktValue=mv)


class ActiveInstrument:
    _tracker: PositionsTracker | None = None
    _gathering: IPositionGathering | None = None
    _sizer: IPositionSizer | None = None

    def __init__(self, ctx: StrategyContext, instrument: Instrument):
        self._instrument = instrument
        # - hack - need to provide getters for that
        self._tracker = ctx._processing_manager._position_tracker # type: ignore
        try:
            self._gathering = ctx._processing_manager._position_gathering # type: ignore
        except Exception as e:
            pass
        try:
            self._sizer = self._tracker.get_position_sizer()
        except Exception as e:
            pass
    
    def exchange(self):
        return self._instrument.exchange

    def trade(self, qty: float, price=None, tif='gtc', **options):
        return ctx.trade(self._instrument, qty, price, tif, **options)

    def trade_a(self, qty: float, price=None, tif='gtc', **options):
        return ctx.trade_async(self._instrument, qty, price, tif, **options)

    def signal(self, s: float, price: float | None = None, 
               stop: float | None = None,
               take: float | None = None,
               comment: str = ''):
        _targets = self._tracker.process_signals(ctx, [self._instrument.signal(ctx, s, price, stop, take, comment=comment)])
        if self._gathering:
            self._gathering.alter_positions(ctx, _targets)
        else:
            logger.error("No configured position gathering found - cannot alter positions !")

    def __le__(self, other: float):
        self.signal(other)
        return self

    def quote(self):
        return ctx.quote(self._instrument)

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


_pollute_caller_globals(ctx)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def orders(instrument: Instrument | ActiveInstrument | None=None):
    if (_orders:=ctx.get_orders()):
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        for k, (i, o) in enumerate(_orders.items()):
            print(" [" + str(k) + "] " + "\t" + green(i) + " " + blue(o.status) + " " + green(o.side) + " " + red(o.instrument.symbol) + " " +  str(o.quantity) + " @ " + str(o.price) + " - " +  blue(str(o.time)))
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    # return ctx.get_orders(instrument if isinstance(instrument, Instrument) or instrument is None else instrument._instrument)


def trade(instrument: Instrument | ActiveInstrument, qty: float, price=None, tif='gtc'):
    return ctx.trade(instrument if isinstance(instrument, Instrument) else instrument._instrument, qty, price, tif)


def trade_a(instrument: Instrument | ActiveInstrument, qty: float, price=None, tif='gtc'):
    return ctx.trade_async(instrument if isinstance(instrument, Instrument) else instrument._instrument, qty, price, tif)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def portfolio(all=True):
    from tabulate import tabulate

    d = dict()
    for s, p in ctx.get_positions().items():
        mv = round(p.market_value_funds, 3)
        if p.quantity != 0.0 or all:
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

# - let's add some magic -
def __plus(ctx, s: str) -> IStrategyContext:
    i = lookup.find_symbol(ctx.exchanges[0], quotify(s))
    if i is not None:
        ctx.set_universe(ctx.instruments + [i])
    return ctx

def __minus(ctx, s: str) -> IStrategyContext:
    i = lookup.find_symbol(ctx.exchanges[0], quotify(s))
    if i is not None and i in ctx.instruments:
        _n_list = ctx.instruments
        _n_list.remove(i)
        ctx.set_universe(_n_list)
    return ctx

IStrategyContext.__add__ = __plus
IStrategyContext.__sub__ = __minus


from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.core.getipython import get_ipython
@magics_class
class IntMagics(Magics):

    @line_magic
    def lo(self, line: str):
        orders()

    @line_cell_magic
    def lp(self, line: str):
        portfolio(any(x in line.lower() for x in ['true', 'all']))

    @line_cell_magic
    def add(self, line: str):
        ctx + line.strip()

    @line_cell_magic
    def remove(self, line: str):
        ctx - line.strip()

    @line_cell_magic
    def cncl(self, line: str):
        order_n = int(line.strip())

        if (_orders:=ctx.get_orders()):
            for k, (i, o) in enumerate(_orders.items()):
                if order_n == k or order_n == o.id:
                    ctx.cancel_order(o.id)
                    break

get_ipython().register_magics(IntMagics)


portfolio()
orders()
