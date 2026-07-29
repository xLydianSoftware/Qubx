from collections import Counter
from unittest.mock import Mock, call, patch

import pytest

from qubx.core.basics import DataType, Instrument
from qubx.core.interfaces import StrategyState, StreamHealth
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


class TestSubscriptionStaleness:
    """A.4: the staleness monitor prefers producer-side StreamHealth ages when the ledger has
    entries for an exchange's market-data stream (fixing the blocked-consumer false positive —
    processing.py:892's on_data_arrival call is consumer-side and can't tell "loop is busy" apart
    from "stream is dead"), falls back to today's consumer-side is_stale() when the ledger has no
    entries (third-party data providers, e.g. xdata, record nothing — CRITICAL back-compat), and
    skips the whole [1/4]..[4/4] remediation cycle when the shared channel is backlogged.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.exchange = "BINANCE.UM"
        self.instrument = lookup.find_symbol(self.exchange, "BTCUSDT")
        assert self.instrument is not None

        self.data_provider = Mock()
        self.data_provider.is_simulation = False
        self.data_provider.is_connected.return_value = True
        self.data_provider.exchange.return_value = self.exchange
        # Only "quote" carries a subscribed instrument; "orderbook"/"trade" are skipped
        # (not subscribed) so each test only has one stream in play.
        self.data_provider.get_subscribed_instruments.side_effect = (
            lambda data_type: [self.instrument] if data_type == "quote" else []
        )

        self.hm = Mock()
        self.hm.get_stream_health.return_value = StreamHealth(ages={}, violations=Counter())
        self.hm.get_queue_size.return_value = 0
        self.hm.is_stale.return_value = False

        self.manager = SubscriptionManager(Mock(), [self.data_provider], self.hm, StrategyState())

    def test_producer_side_fresh_skips_consumer_check_and_resub(self):
        # The ledger has a fresh event for "quote" -> trusted directly; the per-instrument
        # consumer-side check (which a blocked ProcessorThread would fool) is never consulted.
        self.hm.get_stream_health.return_value = StreamHealth(ages={"quote": (5.0, 5.0)}, violations=Counter())

        self.manager._monitor_subscription_status()

        self.hm.is_stale.assert_not_called()
        self.data_provider.unsubscribe.assert_not_called()
        self.data_provider.subscribe.assert_not_called()

    def test_producer_side_stale_flags_without_consumer_check(self):
        # The ledger itself shows the stream is genuinely stale (event_age past threshold) ->
        # flagged directly, still without ever consulting the consumer-side timestamp.
        self.hm.get_stream_health.return_value = StreamHealth(ages={"quote": (700.0, 700.0)}, violations=Counter())

        with patch("qubx.core.mixins.subscription.time.sleep"):
            self.manager._monitor_subscription_status()

        self.hm.is_stale.assert_not_called()
        self.data_provider.unsubscribe.assert_called_once_with("quote", {self.instrument})
        self.data_provider.subscribe.assert_called_once_with("quote", {self.instrument})

    def test_falls_back_to_consumer_side_when_ledger_has_no_entries(self):
        # No producer-side entries for this exchange/stream (e.g. a third-party data provider
        # like xdata that never calls record_stream_event) -> today's exact behavior, unchanged.
        self.hm.get_stream_health.return_value = StreamHealth(ages={}, violations=Counter())
        self.hm.is_stale.return_value = True

        with patch("qubx.core.mixins.subscription.time.sleep"):
            self.manager._monitor_subscription_status()

        self.hm.is_stale.assert_called_once_with(self.instrument, "quote")
        self.data_provider.unsubscribe.assert_called_once_with("quote", {self.instrument})
        self.data_provider.subscribe.assert_called_once_with("quote", {self.instrument})

    def test_falls_back_and_stays_healthy_when_consumer_side_not_stale(self):
        self.hm.get_stream_health.return_value = StreamHealth(ages={}, violations=Counter())
        self.hm.is_stale.return_value = False

        self.manager._monitor_subscription_status()

        self.data_provider.unsubscribe.assert_not_called()

    def test_backlog_guard_skips_resub_cycle(self):
        # Even a genuinely stale stream must not trigger resub churn while the shared channel is
        # backlogged — that's extra loop load exactly when the loop is already overloaded.
        self.hm.is_stale.return_value = True
        self.hm.get_queue_size.return_value = 5_000

        self.manager._monitor_subscription_status()

        self.data_provider.unsubscribe.assert_not_called()
        self.data_provider.subscribe.assert_not_called()

    def test_backlog_at_threshold_does_not_skip(self):
        # Strictly greater-than: a backlog exactly at the configured threshold still runs.
        self.hm.is_stale.return_value = True
        self.hm.get_queue_size.return_value = 1000  # default threshold

        with patch("qubx.core.mixins.subscription.time.sleep"):
            self.manager._monitor_subscription_status()

        self.data_provider.unsubscribe.assert_called_once_with("quote", {self.instrument})

    def test_backlog_guard_is_configurable(self):
        manager = SubscriptionManager(
            Mock(), [self.data_provider], self.hm, StrategyState(), resub_backlog_threshold=10
        )
        self.hm.is_stale.return_value = True
        self.hm.get_queue_size.return_value = 20

        manager._monitor_subscription_status()

        self.data_provider.unsubscribe.assert_not_called()
