from unittest.mock import Mock, call

import pytest

from qubx.core.basics import DataType, Instrument
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.mixins.subscription import SubscriptionManager
from qubx.health.dummy import DummyHealthMonitor


class TestSubscriptionStuff:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.exchange = "BINANCE.UM"
        self.mock_broker = Mock()
        self.mock_broker.is_simulated_trading = False
        self.mock_broker.get_subscribed_instruments.return_value = set()
        self.mock_broker.exchange.return_value = self.exchange
        self.mock_time_provider = Mock()
        self.mock_time_provider.time.return_value = 0.0
        self.manager = SubscriptionManager(
            self.mock_time_provider, [self.mock_broker], DummyHealthMonitor(), StrategyState()
        )

    def _get_instrument(self, symbol: str) -> Instrument:
        instr = lookup.find_symbol(self.exchange, symbol)
        assert instr is not None
        return instr

    def test_sub_types(self):
        trade, _ = DataType.from_str("trade")
        assert trade == DataType.TRADE

        ohlc, params = DataType.from_str("ohlc(1Min)")
        assert ohlc == DataType.OHLC
        assert params == {"timeframe": "1Min"}

        ob, params = DataType.from_str("orderbook(0.01, 100)")
        assert ob == DataType.ORDERBOOK
        assert params == {"tick_size_pct": 0.01, "depth": 100}

        assert DataType.from_str("quote") == (DataType.QUOTE, {})
        assert DataType.from_str("liquidation") == (DataType.LIQUIDATION, {})
        assert DataType.from_str("orderbook") == (DataType.ORDERBOOK, {})
        assert DataType.from_str(DataType.TRADE) == (DataType.TRADE, {})

    def test_basic_subscription(self):
        instrument = self._get_instrument("BTCUSDT")
        self.manager.subscribe(DataType.ORDERBOOK, instrument)
        self.manager.commit()

        self.mock_broker.subscribe.assert_called_once_with(DataType.ORDERBOOK, {instrument}, reset=True)

    def test_warmup_subscription(self):
        instrument = self._get_instrument("BTCUSDT")
        warmup_config = {DataType.ORDERBOOK: "1d"}
        self.manager.set_warmup(warmup_config)
        self.manager.subscribe(DataType.ORDERBOOK, instrument)
        self.manager.commit()

        expected_warmup = {(DataType.ORDERBOOK, instrument): "1d"}
        self.mock_broker.warmup.assert_called_once_with(expected_warmup)

    def test_multiple_subscriptions(self):
        instruments = [self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")]
        self.manager.subscribe(DataType.ORDERBOOK, instruments)
        self.manager.subscribe(DataType.TRADE, instruments[0])
        self.manager.commit()

        expected_calls = [
            call(DataType.ORDERBOOK, set(instruments), reset=True),
            call(DataType.TRADE, set([instruments[0]]), reset=True),
        ]
        self.mock_broker.subscribe.assert_has_calls(expected_calls, any_order=True)

    def test_unsubscribe(self):
        instrument = self._get_instrument("BTCUSDT")
        self.mock_broker.get_subscribed_instruments.return_value = {instrument}

        self.manager.unsubscribe(DataType.ORDERBOOK, instrument)
        self.manager.commit()

        self.mock_broker.subscribe.assert_called_once_with(DataType.ORDERBOOK, set(), reset=True)

    def test_global_subscription(self):
        instruments = {self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")}
        self.mock_broker.get_subscribed_instruments.side_effect = lambda x=None: (instruments if x is None else set())

        self.manager.set_warmup({DataType.TRADE: "1d"})
        self.manager.subscribe(DataType.TRADE)
        self.manager.subscribe(DataType.ORDERBOOK)
        self.manager.commit()

        self.mock_broker.warmup.assert_called_once_with({(DataType.TRADE, i): "1d" for i in instruments})

        expected_calls = [
            call(DataType.TRADE, instruments, reset=True),
            call(DataType.ORDERBOOK, instruments, reset=True),
        ]
        self.mock_broker.subscribe.assert_has_calls(expected_calls, any_order=True)

    def test_subscribe_all(self):
        instruments = {self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")}
        self.mock_broker.get_subscribed_instruments.return_value = set()
        self.mock_broker.get_subscriptions.return_value = [DataType.TRADE, DataType.ORDERBOOK]

        self.manager.subscribe(DataType.ALL, list(instruments))
        self.manager.commit()

        expected_calls = [
            call(DataType.TRADE, instruments, reset=True),
            call(DataType.ORDERBOOK, instruments, reset=True),
        ]
        self.mock_broker.subscribe.assert_has_calls(expected_calls, any_order=True)

    def test_ohlc_warmup(self):
        instruments = {self._get_instrument("BTCUSDT"), self._get_instrument("ETHUSDT")}
        self.mock_broker.get_subscribed_instruments.return_value = set()
        self.mock_broker.get_subscriptions.return_value = [DataType.OHLC]

        # make sure that ohlc warmups are called even if base subscription is not ohlc
        self.manager.set_base_subscription(DataType.TRADE)
        self.manager.set_warmup({DataType.OHLC["1h"]: "30d", DataType.OHLC["1m"]: "1d", DataType.TRADE: "10m"})
        self.manager.subscribe(self.manager.get_base_subscription(), list(instruments))
        self.manager.commit()

        assert self.manager.get_base_subscription() == DataType.TRADE

        expected_warmup = (
            {(DataType.OHLC["1h"], i): "30d" for i in instruments}
            | {(DataType.OHLC["1m"], i): "1d" for i in instruments}
            | {(DataType.TRADE, i): "10m" for i in instruments}
        )
        self.mock_broker.warmup.assert_called_once_with(expected_warmup)
