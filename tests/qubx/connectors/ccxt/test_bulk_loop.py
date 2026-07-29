"""Bulk REST loop routing (quantkit#106).

Verifies the three seams that move bulk REST off the realtime websocket loop:

- the exchange-manager cache keys on the loop, so one venue holds a realtime AND a
  bulk instance (same instance on repeated calls, distinct across loops);
- CcxtDataProvider wires warmup + get_ohlc history to the bulk instance/loop while
  subscriptions keep the realtime one;
- CcxtConnector runs account snapshots and hist-deals recovery on the bulk loop while
  order paths stay on the realtime loop.

Loop identity is asserted with real BackgroundEventLoops and fake exchanges that
record ``asyncio.get_running_loop()`` inside their fetch coroutines.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import clear_exchange_manager_cache, get_ccxt_exchange_manager
from qubx.connectors.ccxt.handlers.ohlc import OhlcDataHandler
from qubx.connectors.plugin import BuildContext
from qubx.core.basics import CtrlChannel, Instrument, MarketType
from qubx.health.dummy import DummyHealthMonitor
from qubx.utils.misc import BackgroundEventLoop, get_bulk_rest_loop
from tests.qubx.core.utils_test import DummyTimeProvider


def _instrument() -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTC/USDT:USDT",
        tick_size=0.1,
        lot_size=0.001,
        min_size=0.001,
    )


# --------------------------------------------------------------------------- #
# Factory cache keys on the loop
# --------------------------------------------------------------------------- #
class TestExchangeManagerCachePerLoop:
    @pytest.fixture(autouse=True)
    def _clean_cache(self):
        clear_exchange_manager_cache()
        yield
        clear_exchange_manager_cache()

    @pytest.fixture
    def two_loops(self):
        a, b = asyncio.new_event_loop(), asyncio.new_event_loop()
        yield a, b
        a.close()
        b.close()

    def _manager(self, loop):
        return get_ccxt_exchange_manager(
            "okx",
            health_monitor=DummyHealthMonitor(),
            time_provider=DummyTimeProvider(),
            loop=loop,
        )

    def test_same_loop_returns_cached_instance(self, two_loops):
        loop, _ = two_loops
        assert self._manager(loop) is self._manager(loop)

    def test_different_loop_returns_distinct_instance(self, two_loops):
        loop_a, loop_b = two_loops
        em_a, em_b = self._manager(loop_a), self._manager(loop_b)
        assert em_a is not em_b
        # Distinct ccxt exchange objects, each bound to its own loop (aiohttp sessions
        # are loop-affine — instances must never be shared across loops).
        assert em_a.exchange is not em_b.exchange
        assert em_a.exchange.asyncio_loop is loop_a
        assert em_b.exchange.asyncio_loop is loop_b

    def test_recreation_params_keep_the_instance_loop(self, two_loops):
        """force_recreation must recreate the exchange on the SAME loop the manager was
        created for (the bulk manager never triggers this automatically, but the
        factory params must stay coherent for both)."""
        loop_a, loop_b = two_loops
        em_a, em_b = self._manager(loop_a), self._manager(loop_b)
        assert em_a._factory_params["loop"] is loop_a
        assert em_b._factory_params["loop"] is loop_b


# --------------------------------------------------------------------------- #
# Data provider: warmup + history on the bulk instance/loop
# --------------------------------------------------------------------------- #
def _build_data_provider(monkeypatch=None):
    """CcxtDataProvider with the factory patched to hand out one mock manager per loop.

    Returns (data_provider, managers_by_loop, calls) where calls collects the factory
    call kwargs.
    """
    managers: dict = {}
    calls: list[dict] = []

    def fake_get_manager(**kwargs):
        calls.append(kwargs)
        loop = kwargs["loop"]
        key = id(loop)
        if key not in managers:
            em = Mock()
            em.exchange = Mock()
            em.exchange.name = "TEST"
            em.exchange.apiKey = None
            em.exchange.asyncio_loop = loop
            managers[key] = em
        return managers[key]

    credentials = Mock()
    settings = Mock()
    settings.testnet = False
    credentials.get_exchange_settings = Mock(return_value=settings)

    channel = CtrlChannel("test")
    channel.control.set()

    ctx = BuildContext(
        exchange_name="TEST",
        time_provider=DummyTimeProvider(),
        channel=channel,
        credentials=credentials,
        health_monitor=DummyHealthMonitor(),
        loop=Mock(name="realtime-loop"),
    )
    with patch("qubx.connectors.ccxt.factory.get_ccxt_exchange_manager", side_effect=fake_get_manager):
        dp = CcxtDataProvider(ctx, max_ws_retries=3, warmup_timeout=10)
    return dp, managers, calls


class TestDataProviderBulkWiring:
    def test_bulk_manager_is_distinct_and_on_bulk_loop(self):
        dp, _, calls = _build_data_provider()
        assert dp._bulk_exchange_manager is not dp._exchange_manager
        # Second factory call requested the process-wide BulkRestLoop
        bulk_calls = [c for c in calls if c["loop"] is get_bulk_rest_loop().loop]
        assert len(bulk_calls) == 1
        assert dp._bulk_exchange_manager.exchange.asyncio_loop is get_bulk_rest_loop().loop

    def test_warmup_service_bound_to_bulk_manager(self):
        """Warmup coroutines submit to the warmup service's manager loop — binding the
        bulk manager (and a handler factory holding the same manager) IS the reroute."""
        dp, _, _ = _build_data_provider()
        assert dp._warmup_service._exchange_manager is dp._bulk_exchange_manager
        assert dp._warmup_service._handler_factory is dp._bulk_handler_factory
        assert dp._bulk_handler_factory._exchange_manager is dp._bulk_exchange_manager

    def test_subscription_handlers_stay_on_realtime_manager(self):
        dp, _, _ = _build_data_provider()
        assert dp._data_type_handler_factory._exchange_manager is dp._exchange_manager

    def test_get_ohlc_history_runs_on_bulk_loop(self):
        """ctx.ohlc's REST history request executes on the bulk loop, not the realtime one."""
        dp, _, _ = _build_data_provider()

        seen_loops: list[asyncio.AbstractEventLoop] = []

        class LoopRecordingHandler(OhlcDataHandler):
            def __init__(self):  # bypass BaseDataTypeHandler init — only the coroutine matters
                pass

            async def get_historical_ohlc(self, instrument, timeframe, nbarsback):
                seen_loops.append(asyncio.get_running_loop())
                return []

        with patch.object(dp._bulk_handler_factory, "get_handler", return_value=LoopRecordingHandler()):
            bars = dp.get_ohlc(_instrument(), "1m", 10)

        assert bars == []
        assert seen_loops == [get_bulk_rest_loop().loop]

    def test_warmup_executes_on_bulk_loop(self):
        """End-to-end: execute_warmup drives the handler coroutine on the bulk loop."""
        dp, _, _ = _build_data_provider()

        seen_loops: list[asyncio.AbstractEventLoop] = []

        async def tracking_warmup(**kwargs):
            seen_loops.append(asyncio.get_running_loop())

        handler = Mock()
        handler.warmup = tracking_warmup
        with patch.object(dp._bulk_handler_factory, "get_handler", return_value=handler):
            dp.warmup({("ohlc", _instrument()): "1h"})

        assert seen_loops == [get_bulk_rest_loop().loop]


# --------------------------------------------------------------------------- #
# Connector: snapshot + hist-deals on the bulk loop, orders on the realtime loop
# --------------------------------------------------------------------------- #
class _FakeExchange:
    """Minimal ccxt stand-in whose fetch coroutines record their running loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.name = "binance"
        self.has: dict = {}
        self.markets: dict = {}
        self.apiKey = "k"
        self.asyncio_loop = loop
        self.seen_loops: dict[str, asyncio.AbstractEventLoop] = {}

    async def fetch_open_orders(self, params=None):
        self.seen_loops["fetch_open_orders" + ("_trigger" if params else "")] = asyncio.get_running_loop()
        return []

    async def fetch_positions(self):
        self.seen_loops["fetch_positions"] = asyncio.get_running_loop()
        return []

    async def fetch_balance(self):
        self.seen_loops["fetch_balance"] = asyncio.get_running_loop()
        return {"total": {}, "used": {}}

    async def fetch_my_trades(self, symbol, since=None):
        self.seen_loops["fetch_my_trades"] = asyncio.get_running_loop()
        return []


@pytest.fixture
def rt_and_bulk_loops():
    rt = BackgroundEventLoop(name="test-realtime")
    bulk = BackgroundEventLoop(name="test-bulk")
    yield rt, bulk
    rt.stop()
    bulk.stop()


def _make_live_connector(rt: BackgroundEventLoop, bulk: BackgroundEventLoop):
    rt_exchange = _FakeExchange(rt.loop)
    bulk_exchange = _FakeExchange(bulk.loop)

    em, bulk_em = Mock(), Mock()
    em.exchange = rt_exchange
    bulk_em.exchange = bulk_exchange

    sent: list = []
    channel = Mock()
    channel.send = Mock(side_effect=lambda e: sent.append(e))

    conn = CcxtConnector(
        exchange_name="BINANCE.UM",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        bulk_exchange_manager=bulk_em,
        data_provider=Mock(),
    )
    return conn, sent, rt_exchange, bulk_exchange


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


class TestConnectorBulkRouting:
    def test_loop_properties_route_to_distinct_loops(self, rt_and_bulk_loops):
        rt, bulk = rt_and_bulk_loops
        conn, _, _, _ = _make_live_connector(rt, bulk)
        assert conn._loop.loop is rt.loop
        assert conn._bulk_loop.loop is bulk.loop

    def test_snapshot_fetches_run_on_bulk_loop(self, rt_and_bulk_loops):
        rt, bulk = rt_and_bulk_loops
        conn, sent, rt_exchange, bulk_exchange = _make_live_connector(rt, bulk)

        conn.request_snapshot()
        _wait_until(lambda: len(sent) == 1)

        assert rt_exchange.seen_loops == {}  # realtime instance untouched
        assert set(bulk_exchange.seen_loops) == {
            "fetch_open_orders",
            "fetch_open_orders_trigger",
            "fetch_positions",
            "fetch_balance",
        }
        assert all(loop is bulk.loop for loop in bulk_exchange.seen_loops.values())

    def test_hist_deals_fetch_runs_on_bulk_loop(self, rt_and_bulk_loops):
        import numpy as np

        rt, bulk = rt_and_bulk_loops
        conn, _, rt_exchange, bulk_exchange = _make_live_connector(rt, bulk)

        conn.request_hist_deals(_instrument(), np.datetime64("2026-07-01T00:00:00"))
        _wait_until(lambda: "fetch_my_trades" in bulk_exchange.seen_loops)

        assert bulk_exchange.seen_loops["fetch_my_trades"] is bulk.loop
        assert rt_exchange.seen_loops == {}

    def test_order_status_probe_stays_on_realtime_loop(self, rt_and_bulk_loops):
        """Latency-critical order paths keep the realtime loop — only bulk-class REST moved."""
        rt, bulk = rt_and_bulk_loops
        conn, _, _, _ = _make_live_connector(rt, bulk)

        seen: list[asyncio.AbstractEventLoop] = []

        async def probe():
            seen.append(asyncio.get_running_loop())

        conn._spawn(probe())
        _wait_until(lambda: len(seen) == 1)
        assert seen[0] is rt.loop

    def test_without_bulk_manager_falls_back_to_single_instance(self, rt_and_bulk_loops):
        """No bulk manager (tests / bespoke constructions) -> pre-bulk-loop behavior:
        everything on the realtime instance."""
        rt, _ = rt_and_bulk_loops
        rt_exchange = _FakeExchange(rt.loop)
        em = Mock()
        em.exchange = rt_exchange
        conn = CcxtConnector(
            exchange_name="BINANCE.UM",
            channel=Mock(send=Mock()),
            time_provider=DummyTimeProvider(),
            exchange_manager=em,
            data_provider=Mock(),
        )
        assert conn._bulk_em is conn._em
        assert conn._bulk_loop.loop is rt.loop


# --------------------------------------------------------------------------- #
# create_ccxt_connector passes a bulk manager on the BulkRestLoop
# --------------------------------------------------------------------------- #
class TestCreateConnectorBulkManager:
    def test_factory_builds_bulk_manager_on_bulk_loop(self):
        from qubx.connectors.ccxt.factory import create_ccxt_connector
        from qubx.connectors.plugin import ConnectorBuildContext

        calls: list[dict] = []

        def fake_get_manager(**kwargs):
            calls.append(kwargs)
            em = Mock()
            em.exchange = Mock()
            em.exchange.asyncio_loop = kwargs["loop"]
            return em

        creds_provider = Mock()
        creds = Mock()
        creds.testnet = False
        creds.api_key, creds.secret = "k", "s"
        creds.model_extra = None
        creds_provider.get_exchange_credentials = Mock(return_value=creds)

        rt_loop = Mock(name="realtime-loop")
        ctx = ConnectorBuildContext(
            exchange_name="BINANCE.UM",
            time_provider=DummyTimeProvider(),
            channel=Mock(),
            credentials=creds_provider,
            health_monitor=DummyHealthMonitor(),
            loop=rt_loop,
            rate_limiter=None,
            data_provider=Mock(),
        )
        with patch("qubx.connectors.ccxt.factory.get_ccxt_exchange_manager", side_effect=fake_get_manager):
            conn = create_ccxt_connector(ctx)

        assert [c["loop"] for c in calls] == [rt_loop, get_bulk_rest_loop().loop]
        assert conn._em is not conn._bulk_em
        assert conn._bulk_em.exchange.asyncio_loop is get_bulk_rest_loop().loop

    def test_shared_rate_limiter_attached_to_both_instances(self):
        from qubx.connectors.ccxt.factory import create_ccxt_connector
        from qubx.connectors.plugin import ConnectorBuildContext

        managers: list = []

        def fake_get_manager(**kwargs):
            em = Mock()
            em.exchange = Mock()
            em.exchange.asyncio_loop = kwargs["loop"]
            managers.append(em)
            return em

        creds_provider = Mock()
        creds = Mock()
        creds.testnet = False
        creds.api_key, creds.secret = "k", "s"
        creds.model_extra = None
        creds_provider.get_exchange_credentials = Mock(return_value=creds)

        rate_limiter = Mock()
        ctx = ConnectorBuildContext(
            exchange_name="BINANCE.UM",
            time_provider=DummyTimeProvider(),
            channel=Mock(),
            credentials=creds_provider,
            health_monitor=DummyHealthMonitor(),
            loop=Mock(name="realtime-loop"),
            rate_limiter=rate_limiter,
            data_provider=Mock(),
        )
        with patch("qubx.connectors.ccxt.factory.get_ccxt_exchange_manager", side_effect=fake_get_manager):
            create_ccxt_connector(ctx)

        assert len(managers) == 2
        for em in managers:
            em.attach_rate_limiter.assert_called_once_with(rate_limiter)


def test_fake_exchange_is_awaitable_sanity():
    """Guard: the _FakeExchange coroutines behave like ccxt's (awaitable, loop-aware)."""
    loop = asyncio.new_event_loop()
    try:
        ex = _FakeExchange(loop)
        result = loop.run_until_complete(ex.fetch_balance())
        assert result == {"total": {}, "used": {}}
        assert ex.seen_loops["fetch_balance"] is loop
    finally:
        loop.close()
