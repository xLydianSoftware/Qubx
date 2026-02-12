import numpy as np
import pandas as pd

from qubx import logger
from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType, Instrument, MarketEvent, Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer
from qubx.data.registry import StorageRegistry


class ExternalFeaturesSubscription(IStrategy):
    n_features: int
    n_events: int

    def on_init(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription("ohlc(1h)")
        initializer.set_event_schedule("1h")

        # - subscribe on some arbitrary type
        initializer.subscribe("features")

        self.n_features = 0
        self.n_events = 0

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> list[Signal] | Signal | None:
        if data.type == "features":
            logger.info(data)
            self.n_features += 1

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
        self.n_events += 1


class TestSimulator:
    def test_external_subscription(self):
        stor = StorageRegistry.get("csv::tests/data/storages/multi/")

        rr = stor.get_reader("BINANCE.UM", "SWAP")
        assert "features" in rr.get_data_types("BTCUSDT")

        simulate(
            (s := ExternalFeaturesSubscription()),
            data=stor,
            capital=1000,
            start="2026-01-01 00:00",
            stop="2026-01-01 05:00",
            instruments=["BINANCE.UM:SWAP:BTCUSDT"],
            debug="INFO",
            silent=True,
        )

        assert s.n_events == 5
        assert s.n_features == 5
