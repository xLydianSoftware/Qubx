from collections import defaultdict
from unittest.mock import Mock, call

import pytest

from qubx.connectors.ccxt.subscription_manager import SubscriptionManager as CcxtSubscriptionManager
from qubx.core.basics import DataType, Instrument
from qubx.core.interfaces import StrategyState
from qubx.core.lookups import lookup
from qubx.core.mixins.subscription import SubscriptionManager, _CommitPlan
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

    def _snapshot_plan(self) -> _CommitPlan:
        """Snapshot the pending state into a plan exactly the way commit() does."""
        m = self.manager
        return _CommitPlan(
            stream_subscriptions={s: set(i) for s, i in m._pending_stream_subscriptions.items()},
            stream_unsubscriptions={s: set(i) for s, i in m._pending_stream_unsubscriptions.items()},
            global_subscriptions=set(m._pending_global_subscriptions),
            global_unsubscriptions=set(m._pending_global_unsubscriptions),
        )

    def test_get_updated_subs_returns_canonically_sorted_order(self):
        """Regression: commit() iterates _get_updated_subs() and, in the backtester,
        whichever subscription type is processed LAST decides whether the instrument's
        OME quote gets re-primed or wiped (BasicSimulatedExchange.on_subscribe evicts the
        primed quote on every call; only OHLC-like data re-primes it, funding_payment can't).
        Raw set()-union order is PYTHONHASHSEED-dependent, so the list must be sorted().
        """
        instrument = self._get_instrument("BTCUSDT")

        # - order A: as frab would build it up (global ohlc sub, then global funding sub, ...)
        self.manager._pending_global_subscriptions.add("ohlc(1h)")
        self.manager._pending_global_subscriptions.add("funding_payment")
        self.manager._pending_stream_subscriptions["orderbook"].add(instrument)
        self.manager._pending_stream_unsubscriptions["trade"].add(instrument)
        result_a = self.manager._get_updated_subs(self._snapshot_plan())

        # - order B: same four type strings, rebuilt from scratch in reverse insertion order
        self.manager._pending_global_subscriptions = set()
        self.manager._pending_global_unsubscriptions = set()
        self.manager._pending_stream_subscriptions = defaultdict(set)
        self.manager._pending_stream_unsubscriptions = defaultdict(set)

        self.manager._pending_stream_unsubscriptions["trade"].add(instrument)
        self.manager._pending_stream_subscriptions["orderbook"].add(instrument)
        self.manager._pending_global_subscriptions.add("funding_payment")
        self.manager._pending_global_subscriptions.add("ohlc(1h)")
        result_b = self.manager._get_updated_subs(self._snapshot_plan())

        expected = ["funding_payment", "ohlc(1h)", "orderbook", "trade"]
        assert result_a == expected
        assert result_b == expected

    def test_new_instrument_does_not_rewarm_whole_universe(self):
        """Regression for #371: a warmup-only sub (has a warmup spec but is never itself
        a live subscription -- e.g. ohlc(1h) riding a live trade base sub) must warm only
        the newly-added instrument when one more instrument joins an already-subscribed
        base sub, not the whole existing universe."""
        btc = self._get_instrument("BTCUSDT")
        eth = self._get_instrument("ETHUSDT")
        sol = self._get_instrument("SOLUSDT")

        self.manager.set_base_subscription(DataType.TRADE)
        self.manager.set_warmup({DataType.OHLC["1h"]: "3d"})
        self.mock_broker.get_subscribed_instruments.side_effect = (
            lambda sub=None: {btc, eth} if sub is None or sub == DataType.TRADE else set()
        )

        # - btc/eth already subscribed to the live base sub from an earlier commit
        self.manager.subscribe(DataType.TRADE, [btc, eth])
        self.manager.commit()
        self.mock_broker.warmup.reset_mock()

        # - one new instrument joins
        self.manager.subscribe(DataType.TRADE, sol)
        self.manager.commit()

        expected_warmup = {(DataType.OHLC["1h"], sol): "3d"}
        self.mock_broker.warmup.assert_called_once_with(expected_warmup)

    def test_new_global_subscription_warms_full_universe(self):
        """A genuinely NEW global subscription (armed via plan.global_subscriptions) must
        still warm the whole current universe -- the #371 fix only narrows the scope for
        subs armed solely because _new_instruments is non-empty."""
        btc = self._get_instrument("BTCUSDT")
        eth = self._get_instrument("ETHUSDT")
        sol = self._get_instrument("SOLUSDT")
        instruments = {btc, eth, sol}
        self.mock_broker.get_subscribed_instruments.side_effect = lambda sub=None: instruments if sub is None else set()

        self.manager.set_warmup({DataType.FUNDING_PAYMENT: "1d"})
        self.manager.subscribe(DataType.FUNDING_PAYMENT)
        self.manager.commit()

        expected_warmup = {(DataType.FUNDING_PAYMENT, i): "1d" for i in instruments}
        self.mock_broker.warmup.assert_called_once_with(expected_warmup)

    def test_initial_boot_warms_all_instruments(self):
        """First-ever commit: every instrument is 'new' (all land in
        plan.stream_subscriptions), so the #371 fix's narrow _new_instruments-only scope
        still warms everyone -- there is no already-subscribed instrument to exclude."""
        btc = self._get_instrument("BTCUSDT")
        eth = self._get_instrument("ETHUSDT")
        sol = self._get_instrument("SOLUSDT")
        instruments = {btc, eth, sol}

        self.manager.set_base_subscription(DataType.TRADE)
        self.manager.set_warmup({DataType.OHLC["1h"]: "3d"})
        self.manager.subscribe(DataType.TRADE, list(instruments))
        self.manager.commit()

        expected_warmup = {(DataType.OHLC["1h"], i): "3d" for i in instruments}
        self.mock_broker.warmup.assert_called_once_with(expected_warmup)

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


class TestSubscriptionWatchdog:
    """The stale-instrument watchdog is the last independent recovery path.

    Two ways it disarmed itself:
      * it skipped any provider reporting not-connected, and a wedged CCXT provider reports exactly
        that (stop_stream clears the stream-enabled flag before it blocks) - the 2026-07-28 freeze;
      * it looked instruments up by the bare base type ("orderbook"), while the fleet subscribes as
        "orderbook(0, 1)" and every provider lookup is an exact-key match - so it collected zero
        instruments and never even consulted is_stale.

    These tests drive a REAL ccxt SubscriptionManager keyed with the parameterised type. A mocked
    provider keyed on "orderbook" passes under both the broken and the fixed lookup, which is how
    the second bug shipped. (That state class is plain Python - importing it pulls in no ccxt.)
    """

    BASE_SUB = DataType.ORDERBOOK[0, 1]  # "orderbook(0, 1)" - what the platform actually subscribes

    def _manager(self, data_provider, health_monitor):
        time_provider = Mock()
        time_provider.time.return_value = 0.0
        state = StrategyState()
        state.is_on_warmup_finished_called = True
        return SubscriptionManager(time_provider, [data_provider], health_monitor, state, monitor_interval_seconds=1e6)

    def _provider(self, instrument, connected: bool, sub_type: str | None = None):
        """A provider whose subscription queries go through the real CcxtSubscriptionManager."""
        state = CcxtSubscriptionManager()
        state.add_subscription(sub_type or self.BASE_SUB, [instrument])
        state.mark_subscription_active(sub_type or self.BASE_SUB)

        provider = Mock()
        provider.is_simulation = False
        provider.is_connected.return_value = connected
        provider.exchange.return_value = "BINANCE.UM"
        provider.get_subscriptions.side_effect = state.get_subscriptions
        provider.get_subscribed_instruments.side_effect = state.get_subscribed_instruments
        return provider

    def _stale_health(self):
        health = Mock()
        health.is_stale.side_effect = lambda instr, dt: dt == "orderbook"
        return health

    def test_a_disconnected_provider_is_still_checked_and_recovered(self):
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        provider = self._provider(instrument, connected=False)
        manager = self._manager(provider, self._stale_health())

        manager._monitor_subscription_status()

        provider.unsubscribe.assert_called_once_with(self.BASE_SUB, {instrument})
        provider.subscribe.assert_called_once_with(self.BASE_SUB, {instrument})

    def test_staleness_is_checked_against_the_base_type(self):
        """Health thresholds and last-event times are keyed by base type, the provider by full key."""
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        provider = self._provider(instrument, connected=True)
        health = self._stale_health()
        manager = self._manager(provider, health)

        manager._monitor_subscription_status()

        health.is_stale.assert_called_once_with(instrument, "orderbook")

    def test_fresh_instruments_are_left_alone(self):
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        provider = self._provider(instrument, connected=True)
        health = Mock()
        health.is_stale.return_value = False
        manager = self._manager(provider, health)

        manager._monitor_subscription_status()

        provider.unsubscribe.assert_not_called()
        provider.subscribe.assert_not_called()

    def test_unwatched_data_types_are_ignored(self):
        """Only quote/orderbook/trade have staleness thresholds; ohlc must not be touched."""
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        provider = self._provider(instrument, connected=True, sub_type=DataType.OHLC["1m"])
        health = Mock()
        health.is_stale.return_value = True
        manager = self._manager(provider, health)

        manager._monitor_subscription_status()

        health.is_stale.assert_not_called()
        provider.unsubscribe.assert_not_called()

    def test_simulation_providers_are_still_skipped(self):
        instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instrument is not None
        provider = self._provider(instrument, connected=True)
        provider.is_simulation = True
        manager = self._manager(provider, self._stale_health())

        manager._monitor_subscription_status()

        provider.unsubscribe.assert_not_called()
        provider.subscribe.assert_not_called()
