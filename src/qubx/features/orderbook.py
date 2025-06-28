from qubx.core.basics import Instrument
from qubx.core.interfaces import IMarketManager
from qubx.core.series import OrderBook

from .core import FeatureProvider


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
    timeframe: str = "1s"
    depths: list[int] = [1, 5, 10, 20]

    def inputs(self) -> list[str]:
        return ["orderbook"]

    def outputs(self) -> list[str]:
        return [self.get_output_name(depth) for depth in self.depths]

    def calculate(self, ctx: IMarketManager, instrument: Instrument, orderbook: OrderBook) -> dict[str, float] | None:
        obis = {self.get_output_name(depth): self._calculate(depth, orderbook) for depth in self.depths}
        return obis

    def _calculate(self, depth: int, orderbook: OrderBook) -> float:
        bid_sizes = orderbook.bids[:depth].sum()
        ask_sizes = orderbook.asks[:depth].sum()
        return bid_sizes - ask_sizes
