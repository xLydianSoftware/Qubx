"""Unit tests for the CcxtConnector read side (commit 7.2).

Mocked ccxt — no credentials or network. Async work is driven deterministically:
``_spawn`` is replaced with a capturing stub and the captured coroutine is awaited
directly (mirroring the write-side harness), so no real thread/loop boundary is
crossed. WS updates are exercised by feeding raw ccxt order dicts straight into the
synchronous ``_handle_ws_order`` handler.
"""

from unittest.mock import AsyncMock, Mock, patch

import ccxt
import pytest

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Balance, Instrument, MarketType, OrderRequest, OrderStatus, Position
from qubx.core.events import (
    AccountSnapshotEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
)
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


def _make_connector(*, exchange: Mock | None = None) -> tuple[CcxtConnector, list, Mock]:
    if exchange is None:
        exchange = Mock()
        exchange.create_order = AsyncMock(return_value={})
        exchange.has = {"editOrder": True}
    exchange.name = "binance"

    em = Mock()
    em.exchange = exchange

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
    # Pre-seed the symbol->instrument cache so the WS/snapshot converters don't need a
    # full ccxt market dict to resolve the instrument.
    conn._symbol_to_instrument[CCXT_SYMBOL] = _instrument()

    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))
    conn._captured = captured  # type: ignore[attr-defined]
    return conn, sent, exchange


async def _drive(conn: CcxtConnector) -> None:
    for coro in conn._captured:  # type: ignore[attr-defined]
        await coro
    conn._captured.clear()  # type: ignore[attr-defined]


def _ws_order(
    *,
    status: str,
    cid: str = "qubx_BTCUSDT_1",
    venue_id: str = "VENUE123",
    side: str = "buy",
    amount: float = 1.0,
    price: float = 100.0,
    trades: list | None = None,
) -> dict:
    return {
        "info": {},
        "id": venue_id,
        "clientOrderId": cid,
        "symbol": CCXT_SYMBOL,
        "timestamp": 1700000000000,
        "type": "limit",
        "timeInForce": "GTC",
        "side": side,
        "price": price,
        "amount": amount,
        "cost": 0.0,
        "status": status,
        "trades": trades or [],
    }


def _ws_trade(trade_id: str = "T1", amount: float = 0.5, price: float = 100.0) -> dict:
    return {
        "id": trade_id,
        "order": "VENUE123",
        "timestamp": 1700000000000,
        "side": "buy",
        "amount": amount,
        "price": price,
        "takerOrMaker": "taker",
        "fee": {"cost": 0.01, "currency": "USDT"},
    }


def _order_request(**overrides) -> OrderRequest:
    kw = dict(
        instrument=_instrument(),
        quantity=1.0,
        price=100.0,
        order_type="LIMIT",
        side="BUY",
        time_in_force="gtc",
        client_id="qubx_BTCUSDT_1",
        options={},
    )
    kw.update(overrides)
    return OrderRequest(**kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# (a) WS order updates -> typed lifecycle events
# --------------------------------------------------------------------------- #
def test_ws_open_emits_accepted() -> None:
    conn, sent, _ = _make_connector()
    conn._handle_ws_order(_ws_order(status="open"))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderAcceptedEvent)
    assert ev.client_order_id == "qubx_BTCUSDT_1"
    assert ev.venue_order_id == "VENUE123"


def test_ws_open_twice_emits_accepted_only_once() -> None:
    conn, sent, _ = _make_connector()
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_order(_ws_order(status="open"))
    assert sum(isinstance(e, OrderAcceptedEvent) for e in sent) == 1


def test_ws_partial_fill_first_emits_accepted_then_partially_filled() -> None:
    conn, sent, _ = _make_connector()
    raw = _ws_order(status="open", trades=[_ws_trade("T1", amount=0.5)])
    # Outer status "open" but inner partial: ccxt reports partially_filled in the
    # canonical status field for a partial.
    raw["status"] = "open"
    raw["info"] = {"status": "PARTIALLY_FILLED"}
    conn._handle_ws_order(raw)
    # A fill seen with no prior venue ack synthesizes ACCEPTED first, so the strategy's
    # on_order_update sees ACCEPTED before the fill and the order lifecycle stays ordered.
    assert len(sent) == 2
    assert isinstance(sent[0], OrderAcceptedEvent)
    ev = sent[1]
    assert isinstance(ev, OrderPartiallyFilledEvent)
    assert ev.fill.trade_id == "T1"
    assert ev.fill.amount == 0.5
    assert ev.client_order_id == "qubx_BTCUSDT_1"


def test_ws_filled_first_emits_accepted_then_filled() -> None:
    conn, sent, _ = _make_connector()
    raw = _ws_order(status="closed", trades=[_ws_trade("T9", amount=1.0)])
    conn._handle_ws_order(raw)
    # Fill-first → ACCEPTED synthesized ahead of the terminal fill.
    assert len(sent) == 2
    assert isinstance(sent[0], OrderAcceptedEvent)
    ev = sent[1]
    assert isinstance(ev, OrderFilledEvent)
    assert ev.fill.trade_id == "T9"
    assert ev.fill.amount == 1.0


def test_ws_filled_multiple_trades_partials_then_filled() -> None:
    conn, sent, _ = _make_connector()
    raw = _ws_order(
        status="closed",
        trades=[_ws_trade("T1", amount=0.4), _ws_trade("T2", amount=0.6)],
    )
    conn._handle_ws_order(raw)
    # Fill-first: ACCEPTED, then one PartiallyFilled per non-final trade, then Filled.
    assert len(sent) == 3
    assert isinstance(sent[0], OrderAcceptedEvent)
    assert isinstance(sent[1], OrderPartiallyFilledEvent)
    assert isinstance(sent[2], OrderFilledEvent)
    assert sent[1].fill.trade_id == "T1"
    assert sent[2].fill.trade_id == "T2"


def test_ws_canceled_emits_canceled() -> None:
    conn, sent, _ = _make_connector()
    conn._handle_ws_order(_ws_order(status="canceled"))
    assert isinstance(sent[0], OrderCanceledEvent)
    assert sent[0].venue_order_id == "VENUE123"


def test_ws_expired_emits_expired() -> None:
    conn, sent, _ = _make_connector()
    conn._handle_ws_order(_ws_order(status="expired"))
    assert isinstance(sent[0], OrderExpiredEvent)


def test_ws_rejected_emits_rejected() -> None:
    conn, sent, _ = _make_connector()
    conn._handle_ws_order(_ws_order(status="rejected"))
    assert isinstance(sent[0], OrderRejectedEvent)
    assert sent[0].client_order_id == "qubx_BTCUSDT_1"


# --------------------------------------------------------------------------- #
# (b) request_snapshot -> one AccountSnapshotEvent
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_request_snapshot_emits_account_snapshot() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_open_orders = AsyncMock(return_value=[_ws_order(status="open", cid="qubx_x", venue_id="V1")])
    exchange.fetch_positions = AsyncMock(return_value=[{"raw": "pos"}])
    exchange.fetch_balance = AsyncMock(
        return_value={"total": {"USDT": 1000.0}, "used": {"USDT": 100.0}}
    )
    exchange.markets = {}
    conn, sent, _ = _make_connector(exchange=exchange)

    pos = Position(instrument=_instrument())
    with patch(
        "qubx.connectors.ccxt.connector.ccxt_convert_positions", return_value=[pos]
    ) as conv_pos:
        conn.request_snapshot()
        await _drive(conn)
        conv_pos.assert_called_once()

    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, AccountSnapshotEvent)
    snap = ev.snapshot
    assert snap.exchange == "BINANCE.UM"
    assert snap.as_of is not None
    assert len(snap.open_orders) == 1
    assert snap.open_orders[0].venue_order_id == "V1"
    assert snap.positions == [pos]
    assert len(snap.balances) == 1
    bal = snap.balances[0]
    assert isinstance(bal, Balance)
    assert bal.currency == "USDT"
    assert bal.total == 1000.0
    assert bal.locked == 100.0


@pytest.mark.asyncio
async def test_request_snapshot_failed_leg_left_none() -> None:
    # A failing fetch leg must not sink the others: positions left None (not wiped).
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.fetch_positions = AsyncMock(side_effect=ccxt.NetworkError("boom"))
    exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 5.0}, "used": {"USDT": 0.0}})
    exchange.markets = {}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_snapshot()
    await _drive(conn)

    snap = sent[0].snapshot
    assert snap.open_orders == []
    assert snap.positions is None  # failed leg omitted, not wiped
    assert len(snap.balances) == 1


# --------------------------------------------------------------------------- #
# (c) request_order_status -> lifecycle event / not-found reject
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_request_order_status_emits_event() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(return_value=_ws_order(status="closed", trades=[_ws_trade("TX", amount=1.0)]))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(client_order_id="qubx_BTCUSDT_1", venue_order_id="VENUE123")
    await _drive(conn)

    exchange.fetch_order.assert_awaited_once_with("VENUE123", None)
    # Reconcile fetch of a filled order with no prior ack: ACCEPTED then the terminal fill.
    assert isinstance(sent[0], OrderAcceptedEvent)
    assert isinstance(sent[1], OrderFilledEvent)
    assert sent[1].fill.trade_id == "TX"


@pytest.mark.asyncio
async def test_request_order_status_not_found_emits_reject_with_both_ids() -> None:
    # A reconcile fetch that comes back OrderNotFound emits OrderRejectedEvent carrying both
    # ids, so AM can resolve the order by client_order_id or venue_order_id and terminate it.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(side_effect=ccxt.OrderNotFound("unknown"))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(client_order_id="qubx_BTCUSDT_1", venue_order_id="VENUE123")
    await _drive(conn)

    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderRejectedEvent)
    assert ev.reason == "reconcile: order not present at venue"
    assert ev.client_order_id == "qubx_BTCUSDT_1"
    assert ev.venue_order_id == "VENUE123"


@pytest.mark.asyncio
async def test_request_order_status_by_venue_id_only_emits_event() -> None:
    # Reconcile a venue-id-only order (no client_order_id): fetch by venue id and surface it.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(return_value=_ws_order(status="closed", trades=[_ws_trade("TX", amount=1.0)]))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(venue_order_id="VENUE123")
    await _drive(conn)

    exchange.fetch_order.assert_awaited_once_with("VENUE123", None)
    assert isinstance(sent[0], OrderAcceptedEvent)
    assert isinstance(sent[1], OrderFilledEvent)


@pytest.mark.asyncio
async def test_request_order_status_network_error_leaves_inflight() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(side_effect=ccxt.NetworkError("timeout"))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(client_order_id="qubx_BTCUSDT_1", venue_order_id="VENUE123")
    await _drive(conn)

    assert sent == []


def test_request_order_status_raises_when_no_id_given() -> None:
    conn, _, _ = _make_connector()
    with pytest.raises(Exception):
        conn.request_order_status()
    with pytest.raises(Exception):
        conn.request_order_status(client_order_id="")


# --------------------------------------------------------------------------- #
# (d) order cache: submit populates it; cancel/update resolve the cached symbol;
#     terminal WS update evicts it
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_submit_populates_cache() -> None:
    conn, _sent, _ = _make_connector()
    conn.submit_order(_order_request())
    await _drive(conn)
    assert "qubx_BTCUSDT_1" in conn._orders
    cached = conn._orders["qubx_BTCUSDT_1"]
    assert cached.ccxt_symbol == CCXT_SYMBOL
    assert cached.side == "buy"
    assert cached.type == "limit"


@pytest.mark.asyncio
async def test_cancel_resolves_cached_symbol() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.create_order = AsyncMock(return_value={})
    exchange.cancel_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, _sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)
    conn.cancel_order(client_order_id="qubx_BTCUSDT_1", venue_order_id="VENUE123")
    await _drive(conn)

    # Closes the commit-3 gap: cancel_order now awaited WITH the cached ccxt symbol.
    exchange.cancel_order.assert_awaited_once_with("VENUE123", CCXT_SYMBOL)


@pytest.mark.asyncio
async def test_update_edit_resolves_cached_symbol_and_side() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.create_order = AsyncMock(return_value={})
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, _sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)
    conn.update_order(client_order_id="qubx_BTCUSDT_1", venue_order_id="VENUE123", price=101.0, quantity=2.0)
    await _drive(conn)

    kwargs = exchange.edit_order.await_args.kwargs
    assert kwargs["symbol"] == CCXT_SYMBOL
    assert kwargs["side"] == "buy"
    assert kwargs["type"] == "limit"


def test_terminal_ws_update_evicts_cache() -> None:
    conn, _sent, _ = _make_connector()
    # First an open update populates the cache + venue index.
    conn._handle_ws_order(_ws_order(status="open"))
    assert "qubx_BTCUSDT_1" in conn._orders
    assert conn._venue_to_cid.get("VENUE123") == "qubx_BTCUSDT_1"
    # A terminal update evicts both.
    conn._handle_ws_order(_ws_order(status="canceled"))
    assert "qubx_BTCUSDT_1" not in conn._orders
    assert "VENUE123" not in conn._venue_to_cid


def test_ws_external_order_materialized_in_cache() -> None:
    conn, _sent, _ = _make_connector()
    conn._handle_ws_order(_ws_order(status="open", cid="manual-123", venue_id="V_EXT"))
    assert "manual-123" in conn._orders
    assert conn._orders["manual-123"].ccxt_symbol == CCXT_SYMBOL


# --------------------------------------------------------------------------- #
# (e) connect() triggers an initial snapshot + starts the WS subscription
# --------------------------------------------------------------------------- #
def test_connect_triggers_initial_snapshot_and_subscription() -> None:
    conn, _sent, _ = _make_connector()
    submitted: list = []
    fut = Mock()
    fut.done = Mock(return_value=False)
    fut.add_done_callback = Mock()
    loop = Mock()
    loop.submit = Mock(side_effect=lambda coro: submitted.append(coro) or fut)

    # _loop is a property (resolves the exchange loop); patch it on the class.
    with patch.object(type(conn), "_loop", new=loop):
        conn.connect()

    # WS execution subscription started on the exchange loop.
    assert len(submitted) == 1
    # Initial snapshot requested (fire-and-forget via the _spawn capture).
    assert len(conn._captured) == 1  # type: ignore[attr-defined]
    # Close the coroutines we never awaited (avoid "coroutine was never awaited").
    submitted[0].close()
    conn._captured[0].close()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_is_ws_ready_reflects_stream_state() -> None:
    conn, _sent, exchange = _make_connector()
    assert conn.is_ws_ready() is False  # not connected yet

    # One successful watch_orders round-trip flips readiness; then channel closes.
    calls = {"n": 0}

    async def _watch():
        calls["n"] += 1
        if calls["n"] == 1:
            return [_ws_order(status="open")]
        conn.channel.control.is_set = Mock(return_value=False)
        return []

    exchange.watch_orders = AsyncMock(side_effect=_watch)
    conn.channel.control = Mock()
    conn.channel.control.is_set = Mock(return_value=True)

    await conn._subscribe_executions()
    # Loop exited cleanly; readiness was True during the run and reset on exit.
    assert conn.is_ws_ready() is False
    assert calls["n"] >= 1


# --------------------------------------------------------------------------- #
# (f) connector -> AccountManager seam: a snapshot the connector emits must
#     apply cleanly to a REAL AccountManager (the order's status must be an
#     OrderStatus enum, or AccountState.add_order -> order.status.is_terminal
#     raises AttributeError).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_snapshot_applies_to_real_account_manager() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_open_orders = AsyncMock(
        return_value=[_ws_order(status="open", cid="qubx_BTCUSDT_1", venue_id="V1")]
    )
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 1000.0}, "used": {"USDT": 100.0}})
    exchange.markets = {}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_snapshot()
    await _drive(conn)

    assert len(sent) == 1
    event = sent[0]
    assert isinstance(event, AccountSnapshotEvent)
    # Sanity: the converted open order carries an OrderStatus enum, not a raw string.
    assert event.snapshot.open_orders[0].status is OrderStatus.ACCEPTED

    # Apply the connector-emitted snapshot to a REAL AccountManager. This is the seam
    # the unit tests skipped: AccountState.add_order calls order.status.is_terminal,
    # which only exists on OrderStatus (a raw string would raise AttributeError here).
    strategy = Mock()
    am = SimulatedAccountManager(
        connectors={"BINANCE.UM": object()},
        strategy=strategy,
        time=DummyTimeProvider(),
    )
    am.apply(event)  # must NOT raise

    # The open order was registered against the real account state (materialized
    # EXTERNAL since its cid does not match the qubx- recovered prefix).
    registered = am.find_order_by_id("V1")
    assert registered is not None
    assert registered.status is OrderStatus.ACCEPTED
    assert am.get_balance("USDT", exchange="BINANCE.UM").total == 1000.0
