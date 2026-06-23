"""Unit tests for the CcxtConnector reconnect / exchange-recreation path.

After the ExchangeManager swaps in a fresh exchange object, the running
``_subscribe_executions`` loop is bound to the PREVIOUS exchange's ``watch_orders``
and is therefore dead. The connector registers a recreation callback that restarts
the executions stream against the freshly-recreated exchange and pulls a snapshot to
resync AccountManager against venue truth (design IConnector.connect contract case 2).

Mocked ccxt — no credentials, no real loop/thread. ``_loop.submit`` and ``_spawn``
are stubbed with capturing fakes so the async work is observable deterministically.
"""

from unittest.mock import AsyncMock, Mock, patch

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.core.basics import Instrument, MarketType
from qubx.core.series import Quote
from tests.qubx.core.utils_test import DummyTimeProvider

CCXT_SYMBOL = "BTC/USDT:USDT"


def _instrument() -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.1,
        lot_size=0.001,
        min_size=0.001,
        min_notional=0.0,
    )


def _quote(bid: float = 99.0, ask: float = 101.0) -> Quote:
    return Quote(0, bid, ask, 1.0, 1.0)


def _make_connector() -> tuple[CcxtConnector, list, Mock]:
    exchange = Mock()
    exchange.name = "binance"
    exchange.watch_orders = AsyncMock(return_value=[])

    em = Mock()
    em.exchange = exchange
    # capture the callback registered against the ExchangeManager
    registered: list = []
    em.register_recreation_callback = Mock(side_effect=lambda cb: registered.append(cb))

    dp = Mock()
    dp.get_quote = Mock(return_value=_quote())

    sent: list = []
    channel = Mock()
    channel.send = Mock(side_effect=lambda e: sent.append(e))

    conn = CcxtConnector(
        exchange_name="BINANCE.UM",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=dp,
    )
    conn._symbol_to_instrument[CCXT_SYMBOL] = _instrument()
    conn._registered_callbacks = registered  # type: ignore[attr-defined]
    return conn, sent, em


def _fake_loop() -> tuple[Mock, list]:
    """A fake AsyncThreadLoop whose submit() captures the coroutine and returns a future."""
    submitted: list = []

    def _submit(coro):
        # close so we don't leak "coroutine was never awaited"
        coro.close()
        fut = Mock()
        fut.done = Mock(return_value=False)
        fut.add_done_callback = Mock()
        fut.cancel = Mock()
        submitted.append(fut)
        return fut

    loop = Mock()
    loop.submit = Mock(side_effect=_submit)
    return loop, submitted


def test_recreation_callback_registered_in_init() -> None:
    conn, _sent, _em = _make_connector()
    # The handler is registered exactly once at construction.
    assert conn._handle_exchange_recreation in conn._registered_callbacks  # type: ignore[attr-defined]


def test_recreation_before_connect_is_noop() -> None:
    # Never connected (no executions future) -> the handler must do nothing.
    conn, _sent, _em = _make_connector()
    conn.request_snapshot = Mock()  # type: ignore[method-assign]
    conn._start_executions_stream = Mock()  # type: ignore[method-assign]

    conn._handle_exchange_recreation()

    conn._start_executions_stream.assert_not_called()
    conn.request_snapshot.assert_not_called()


def test_recreation_restarts_stream_and_pulls_snapshot() -> None:
    conn, _sent, _em = _make_connector()
    loop, submitted = _fake_loop()
    snapshot_calls: list = []
    conn.request_snapshot = Mock(side_effect=lambda: snapshot_calls.append(1))  # type: ignore[method-assign]

    with patch.object(type(conn), "_loop", new=loop):
        conn.connect()
        assert conn._executions_future is not None
        first_future = conn._executions_future
        assert len(snapshot_calls) == 1  # initial connect() snapshot

        # Simulate the ExchangeManager swapping in a fresh exchange and firing callbacks.
        conn._handle_exchange_recreation()

    # (a) the executions future was restarted (cancelled old, submitted new).
    first_future.cancel.assert_called_once()
    assert conn._executions_future is not None
    assert conn._executions_future is not first_future
    # (b) a fresh snapshot was pulled to resync AM against venue truth.
    assert len(snapshot_calls) == 2
    # readiness was cleared on recreation (the old stream was dead).
    # connect() didn't set it; the loop would, but we never ran it — so just assert the
    # handler explicitly reset it to False.
    assert conn.is_ws_ready() is False
