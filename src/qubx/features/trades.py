from collections import defaultdict

import pandas as pd

from qubx.core.basics import Instrument, Trade
from qubx.core.interfaces import IMarketManager
from qubx.core.series import RollingSum

from .core import FeatureProvider
from .utils import check_interval


class TradePrice(FeatureProvider):
    name: str = "TradePrice"

    _trades: dict[Instrument, list[Trade]]

    def __init__(self, timeframe: str = "1s", **kwargs):
        super().__init__(timeframe=timeframe, **kwargs)

    def inputs(self) -> list[str]:
        return ["trade"]

    def outputs(self) -> list[str]:
        return [self.get_output_name(tf=self.timeframe)]

    def on_start(self, ctx: IMarketManager) -> None:
        self._trades = defaultdict(list)

    def calculate(self, instrument: Instrument, trade: Trade) -> float | None:
        trade_buffer = self._trades[instrument]
        if not trade_buffer:
            trade_buffer.append(trade)
            return None

        if check_interval(trade_buffer[0].time, trade.time, self.timeframe):
            _avg_price = sum([trade.price * trade.size for trade in trade_buffer]) / sum(
                [trade.size for trade in trade_buffer]
            )
            trade_buffer.clear()
            trade_buffer.append(trade)
            return _avg_price

        trade_buffer.append(trade)
        return None


class TradeVolumeImbalance(FeatureProvider):
    name: str = "TVI"
    timeframe: str = "1s"
    trade_period: str = "1Min"

    # rolling sum of buy and sell quantities
    _buy_sum: dict[Instrument, RollingSum]
    _sell_sum: dict[Instrument, RollingSum]

    # trade buffer for the last second
    _trades: dict[Instrument, list[Trade]]

    # def warmup(self) -> pd.Timedelta:
    #     return pd.Timedelta("30Min")

    def inputs(self) -> list[str]:
        return ["trade"]

    def outputs(self) -> list[str]:
        return [self.get_output_name(self.trade_period, tf=self.timeframe)]

    def on_start(self, ctx: IMarketManager) -> None:
        _trade_delta = pd.Timedelta(self.trade_period)
        _chunk_period = pd.Timedelta(self.timeframe)
        _chunks = int(_trade_delta / _chunk_period)
        self._buys = defaultdict(lambda: RollingSum(_chunks))
        self._sells = defaultdict(lambda: RollingSum(_chunks))
        self._trades = defaultdict(list)

    def calculate(self, ctx: IMarketManager, instrument: Instrument, trade: Trade) -> float | None:
        trade_buffer = self._trades[instrument]
        if trade_buffer:
            _earliest_trade = trade_buffer[0]
            _earliest_time = pd.Timestamp(_earliest_trade.time).floor(self.timeframe)
            _last_time = pd.Timestamp(trade.time).floor(self.timeframe)
            if _earliest_time != _last_time:
                val = self._calculate(instrument, trade_buffer)
                trade_buffer.clear()
                trade_buffer.append(trade)
                return val

        trade_buffer.append(trade)
        return None

    def _calculate(self, instrument: Instrument, trades: list[Trade]) -> float:
        buy_sum = self._buys[instrument]
        sell_sum = self._sells[instrument]
        buy_volume = buy_sum.update(
            sum([trade.size for trade in trades if trade.side == 1]),
            new_item_started=True,
        )
        sell_volume = sell_sum.update(
            sum([trade.size for trade in trades if trade.side == -1]),
            new_item_started=True,
        )
        if buy_sum.is_init_stage or sell_sum.is_init_stage:
            return 0.0
        return (buy_volume - sell_volume) / (buy_volume + sell_volume)
