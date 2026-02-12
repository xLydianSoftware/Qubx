from collections import defaultdict

import numpy as np
import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType, Instrument, MarketEvent, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer
from qubx.data.registry import StorageRegistry


class DataSubscriptionStrategy(IStrategy):
    base: str = "ohlc(1h)"
    additional: list[str] = []
    schedule: str = "1h"

    _n_features: int
    _n_events: int
    _data_hits: dict[str, int]

    def on_init(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription(self.base)

        # - subscribe on some arbitrary type
        for sb in self.additional:
            initializer.subscribe(sb)

        if self.schedule:
            initializer.set_event_schedule(self.schedule)

        self._data_hits = defaultdict(lambda: 0)

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> list[Signal] | Signal | None:
        self._data_hits[data.type] += 1

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
        self._data_hits["event"] += 1


class TestSimulator:
    def test_external_subscription(self):
        stor = StorageRegistry.get("csv::tests/data/storages/multi/")

        rr = stor.get_reader("BINANCE.UM", "SWAP")
        assert "features" in rr.get_data_types("BTCUSDT")

        simulate(
            (s := DataSubscriptionStrategy(base="ohlc(1h)", additional=["features"], schedule="1h")),
            data=stor,
            capital=1000,
            start="2026-01-01 00:00",
            stop="2026-01-01 05:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="INFO",
            silent=True,
        )

        assert s._data_hits["event"] == 5
        assert s._data_hits["features"] == 5
