from collections import defaultdict

from qubx.core.basics import Instrument
from qubx.core.interfaces import IMarketManager
from qubx.core.series import OrderBook

from .core import FeatureProvider
from .utils import check_interval


class OrderbookMidPrice(FeatureProvider):
    name: str = "OrderbookMidPrice"

    def __init__(self, timeframe: str = "1s", **kwargs):
        super().__init__(timeframe=timeframe, **kwargs)

    def inputs(self) -> list[str]:
        return ["orderbook"]

    def outputs(self) -> list[str]:
        return [self.name]

    def calculate(self, instrument: Instrument, orderbook: OrderBook) -> float:
        return orderbook.mid_price()


class OrderbookImbalance(FeatureProvider):
    name: str = "OBI"
    depths: list[int] = [1, 5, 10, 20]

    _orderbooks: dict[Instrument, list[OrderBook]]

    def __init__(self, timeframe: str = "1s", **kwargs):
        super().__init__(timeframe=timeframe, **kwargs)

    def inputs(self) -> list[str]:
        return ["orderbook"]

    def outputs(self) -> list[str]:
        return [self.get_output_name(depth) for depth in self.depths]

    def on_start(self, ctx: IMarketManager) -> None:
        self._orderbooks = defaultdict(list)

    def calculate(self, instrument: Instrument, orderbook: OrderBook) -> dict[str, float] | None:
        buffer = self._orderbooks[instrument]
        if not buffer:
            buffer.append(orderbook)
            return None

        obis = None
        if check_interval(buffer[0].time, orderbook.time, self.timeframe):
            obis = {self.get_output_name(depth): self._calculate(depth, buffer) for depth in self.depths}
            buffer.clear()

        buffer.append(orderbook)
        return obis

    def _calculate(self, depth: int, orderbooks: list[OrderBook]) -> float:
        bid_sizes = [ob.bids[:depth].sum() for ob in orderbooks]
        ask_sizes = [ob.asks[:depth].sum() for ob in orderbooks]
        bid_size, ask_size = (
            sum(bid_sizes) / len(bid_sizes),
            sum(ask_sizes) / len(ask_sizes),
        )
        return (bid_size - ask_size) / (bid_size + ask_size)
