import asyncio
from pprint import pprint
from threading import Thread
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from ccxt.pro import Exchange

from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.core.basics import CtrlChannel, DataType, ITimeProvider, dt_64
from qubx.core.exceptions import QueueTimeout
from qubx.core.lookups import lookup
from qubx.core.mixins.subscription import SubscriptionManager
from tests.qubx.core.utils_test import DummyTimeProvider

OHLCV_RESPONSE = {
    "ETH/USDT:USDT": {"5m": [[1731239700000, 3222.69, 3227.58, 3218.18, 3220.01, 2866.3094, 10000.0, 5000.0]]}
}


async def async_sleep(*args, seconds: int = 1, **kwargs):
    await asyncio.sleep(seconds)


async def return_ohlcv(*args, **kwargs):
    await asyncio.sleep(0.1)
    return OHLCV_RESPONSE


class MockExchange(Exchange):
    def __init__(self):
        self.name = "mock_exchange"
        self.asyncio_loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.asyncio_loop.run_forever, daemon=True)
        self.thread.start()
        self.watch_ohlcv_for_symbols = AsyncMock()
        self.watch_ohlcv_for_symbols.side_effect = return_ohlcv
        self.watch_trades_for_symbols = AsyncMock()
        self.watch_order_book_for_symbols = AsyncMock()
        self.find_timeframe = MagicMock(return_value="1m")
        self.fetch_ohlcv = AsyncMock(return_value=[])


class TestCcxtExchangeConnector:
    connector: CcxtDataProvider

    @pytest.fixture(autouse=True)
    def setup(self):
        # Create event loop
        self.mock_exchange = MockExchange()
        self.fixed_time = np.datetime64("2023-01-01T00:00:00.000000000")

        self.connector = CcxtDataProvider(
            exchange=self.mock_exchange, time_provider=DummyTimeProvider(), channel=CtrlChannel("test")
        )
        self.sub_manager = SubscriptionManager(self.connector)

        # return from setup
        yield

        # teardown
        self.mock_exchange.asyncio_loop.stop()

    def test_subscribe(self):
        # Create test instrument
        i1, i2 = lookup.find_symbol("BINANCE.UM", "BTCUSDT"), lookup.find_symbol("BINANCE.UM", "ETHUSDT")
        assert i1 is not None and i2 is not None

        # Subscribe to different data types
        # self.connector.subscribe([i1, i2], "trade", warmup_period="1m")
        # self.connector.subscribe([i1], "orderbook", warmup_period="1m")
        # self.connector.subscribe([i2], "orderbook", warmup_period="1m")
        self.sub_manager.subscribe(DataType.OHLC["15Min"], [i2])
        self.sub_manager.subscribe(DataType.OHLC["15Min"], [i1])

        # Commit subscriptions
        self.sub_manager.commit()

        # Verify subscriptions were added
        # assert i1 in self.connector._subscriptions["trade"]
        # assert i1 in self.connector._subscriptions["orderbook"]
        assert i1 in self.connector._subscriptions[DataType.OHLC["15Min"]]

        channel = self.connector.channel
        events = []
        max_count = 10
        count = 0
        while True:
            try:
                events.append(channel.receive(2))
                count += 1
            except QueueTimeout:
                break
            if count > max_count:
                break

        assert len(events) > 0
        pprint(events)

        # Verify exchange methods were called
        # self.mock_exchange.watch_ohlcv_for_symbols.assert_awaited()
