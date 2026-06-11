"""Unit tests for the CcxtConnector read side (commit 7.2).

Mocked ccxt — no credentials or network. Async work is driven deterministically:
``_spawn`` is replaced with a capturing stub and the captured coroutine is awaited
directly (mirroring the write-side harness), so no real thread/loop boundary is
crossed. WS updates are exercised by feeding raw ccxt order dicts straight into the
synchronous ``_handle_ws_order`` handler.
"""

import asyncio
import contextlib
import math
from unittest.mock import AsyncMock, Mock, patch

import ccxt
import ccxt.pro
import numpy as np
import pytest

from qubx import logger
from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.connectors.ccxt.exchanges._two_stream import _TwoStreamCcxtConnector
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Balance, Instrument, MarketType, OrderOrigin, OrderRequest, OrderStatus, Position
from qubx.core.events import (
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    PositionUpdateEvent,
)
from qubx.core.series import Quote
from tests.qubx.connectors.ccxt.data.ccxt_responses import BINANCE_MARKETS
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


def _make_connector(
    *, exchange: Mock | None = None, cls: type[CcxtConnector] = CcxtConnector
) -> tuple[CcxtConnector, list, Mock]:
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

    conn = cls(
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
    # on_order sees ACCEPTED before the fill and the order lifecycle stays ordered.
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
    exchange.fetch_balance = AsyncMock(return_value={"total": {"USDT": 1000.0}, "used": {"USDT": 100.0}})
    exchange.markets = {}
    conn, sent, _ = _make_connector(exchange=exchange)

    pos = Position(instrument=_instrument())
    with patch("qubx.connectors.ccxt.connector.ccxt_convert_positions", return_value=[pos]) as conv_pos:
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
@pytest.mark.parametrize(
    "ids",
    [
        {"client_order_id": "qubx_BTCUSDT_1", "venue_order_id": "VENUE123"},
        {"venue_order_id": "VENUE123"},
    ],
    ids=["both_ids", "venue_id_only"],
)
@pytest.mark.asyncio
async def test_request_order_status_emits_event(ids: dict) -> None:
    # Reconcile by both ids or by venue id alone: fetch by venue id and surface the order.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(return_value=_ws_order(status="closed", trades=[_ws_trade("TX", amount=1.0)]))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(**ids)
    await _drive(conn)

    exchange.fetch_order.assert_awaited_once_with("VENUE123", None)
    # Reconcile fetch of a filled order with no prior ack: ACCEPTED then the terminal fill.
    assert isinstance(sent[0], OrderAcceptedEvent)
    assert isinstance(sent[1], OrderFilledEvent)
    assert sent[1].fill.trade_id == "TX"


@pytest.mark.asyncio
async def test_request_order_status_filled_without_trades_emits_status_only_fill() -> None:
    # F7 rescue: Binance fetch_order payloads typically carry no embedded trades — the
    # FILLED status must still be rescued via a fill=None terminal event (nothing is
    # booked; the next snapshot's position reconcile corrects the ledger).
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(return_value=_ws_order(status="closed"))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(client_order_id="qubx_BTCUSDT_1", venue_order_id="VENUE123")
    await _drive(conn)

    assert isinstance(sent[0], OrderAcceptedEvent)
    assert isinstance(sent[1], OrderFilledEvent)
    assert sent[1].fill is None


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


@pytest.mark.asyncio
async def test_request_order_status_cid_only_uncached_uses_cloid_variant_with_instrument() -> None:
    # R7 lost-ack case: only the cloid is known and the connector never cached the order
    # (snapshot-materialized RECOVERED order). The fetch must go through ccxt's
    # client-order-id variant (Binance rejects a cloid passed as orderId with -1102) with
    # the symbol resolved from the caller-provided instrument.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order_with_client_order_id = AsyncMock(return_value=_ws_order(status="open"))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(client_order_id="qubx_BTCUSDT_1", instrument=_instrument())
    await _drive(conn)

    exchange.fetch_order_with_client_order_id.assert_awaited_once_with("qubx_BTCUSDT_1", CCXT_SYMBOL)
    exchange.fetch_order.assert_not_called()
    assert isinstance(sent[0], OrderAcceptedEvent)
    assert sent[0].venue_order_id == "VENUE123"  # venue id recovered from the cloid fetch


@pytest.mark.asyncio
async def test_request_order_status_cid_only_upgrades_to_cached_venue_id() -> None:
    # When the cache already learned the venue id (WS ack seen), a cid-only request is
    # upgraded to the plain venue-id fetch — the more reliable path across venues.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(return_value=_ws_order(status="open"))
    conn, sent, _ = _make_connector(exchange=exchange)
    conn._handle_ws_order(_ws_order(status="open"))  # cache learns VENUE123 + symbol
    sent.clear()

    conn.request_order_status(client_order_id="qubx_BTCUSDT_1")
    await _drive(conn)

    exchange.fetch_order.assert_awaited_once_with("VENUE123", CCXT_SYMBOL)


@pytest.mark.asyncio
async def test_request_order_status_cid_only_not_found_emits_reject() -> None:
    # Venue says the cloid is unknown -> the existing OrderNotFound reject synthesis path.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order_with_client_order_id = AsyncMock(side_effect=ccxt.OrderNotFound("unknown"))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.request_order_status(client_order_id="qubx_BTCUSDT_1", instrument=_instrument())
    await _drive(conn)

    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderRejectedEvent)
    assert ev.code == "OrderNotFound"
    assert ev.client_order_id == "qubx_BTCUSDT_1"


@pytest.mark.asyncio
async def test_request_order_status_venue_refusal_logs_warning_no_event() -> None:
    # A BadRequest-family refusal means the fetch itself was unanswerable — state UNKNOWN,
    # so nothing may be emitted, but it must be operator-visible (WARNING, not silent).
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_order = AsyncMock(side_effect=ccxt.BadRequest("binance -1102 bad orderId"))
    conn, sent, _ = _make_connector(exchange=exchange)

    records: list = []
    sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
    try:
        conn.request_order_status(client_order_id="qubx_BTCUSDT_1", venue_order_id="VENUE123")
        await _drive(conn)
    finally:
        logger.remove(sink_id)

    assert sent == []
    assert any("refused by venue" in r["message"] for r in records)


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


@pytest.mark.asyncio
async def test_update_cid_only_uses_cloid_edit_variant() -> None:
    # R7 mirror on the write side: an update before the venue ack (no venue id anywhere)
    # must go through ccxt's client-order-id edit variant with the cached
    # symbol/type/side — a cloid passed to edit_order as orderId is Binance -1102.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.create_order = AsyncMock(return_value={})
    exchange.edit_order_with_client_order_id = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)
    sent.clear()
    conn.update_order(client_order_id="qubx_BTCUSDT_1", price=101.0, quantity=2.0)
    await _drive(conn)

    exchange.edit_order_with_client_order_id.assert_awaited_once_with(
        "qubx_BTCUSDT_1", CCXT_SYMBOL, "limit", "buy", 2.0, 101.0
    )
    exchange.edit_order.assert_not_called()
    assert isinstance(sent[0], OrderUpdatedEvent)
    assert sent[0].venue_order_id == "VENUE123"  # venue id recovered from the edit response


@pytest.mark.asyncio
async def test_update_cid_only_upgrades_to_cached_venue_id() -> None:
    # When the cache already knows the venue id, a cid-only update edits by venue id.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, _sent, _ = _make_connector(exchange=exchange)
    conn._handle_ws_order(_ws_order(status="open"))  # cache learns VENUE123 + symbol

    conn.update_order(client_order_id="qubx_BTCUSDT_1", price=101.0, quantity=2.0)
    await _drive(conn)

    assert exchange.edit_order.await_args.kwargs["id"] == "VENUE123"
    exchange.edit_order_with_client_order_id.assert_not_called()


@pytest.mark.asyncio
async def test_snapshot_seeds_order_cache() -> None:
    # R7 third leg: a snapshot can be the only place the connector sees a RECOVERED/
    # EXTERNAL order — it must seed the venue-call cache so later cancel/update/status
    # calls resolve the symbol and venue id instead of dying on ArgumentsRequired.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.fetch_open_orders = AsyncMock(return_value=[_ws_order(status="open", cid="recovered-1", venue_id="V9")])
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_balance = AsyncMock(return_value={"total": {}, "used": {}})
    exchange.markets = {}
    exchange.cancel_order = AsyncMock(return_value={"id": "V9"})
    exchange.cancel_order_with_client_order_id = AsyncMock(return_value={"id": "V9"})
    conn, _sent, _ = _make_connector(exchange=exchange)

    conn.request_snapshot()
    await _drive(conn)

    assert "recovered-1" in conn._orders
    assert conn._orders["recovered-1"].ccxt_symbol == CCXT_SYMBOL
    assert conn._venue_to_cid.get("V9") == "recovered-1"

    # A cid-only cancel of the snapshot-seen order now resolves venue id + symbol.
    conn.cancel_order(client_order_id="recovered-1")
    await _drive(conn)
    exchange.cancel_order.assert_not_called()  # cid-only cancel stays on the cloid path
    exchange.cancel_order_with_client_order_id.assert_awaited_once_with("recovered-1", CCXT_SYMBOL)


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


def _arm_control(conn) -> None:
    conn.channel.control = Mock()
    conn.channel.control.is_set = Mock(return_value=True)


@pytest.mark.asyncio
async def test_is_ws_ready_true_on_quiet_stream() -> None:
    # R4: ccxt's watch_orders future resolves only on actual order traffic — a quiet
    # account never gets a message. Readiness must be optimistic (set once the loop
    # drives the watch), or AM liveness reconnects every threshold window forever.
    conn, _sent, exchange = _make_connector()
    assert conn.is_ws_ready() is False  # not connected yet

    block = asyncio.Event()
    exchange.watch_orders = AsyncMock(side_effect=block.wait)  # never yields a message
    _arm_control(conn)

    task = asyncio.ensure_future(conn._subscribe_executions())
    await asyncio.sleep(0.01)
    assert conn.is_ws_ready() is True  # ready with zero messages delivered

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_ws_ready_clears_on_persistent_network_errors() -> None:
    # R6: a stream that keeps erroring must stop reporting ready (so AM liveness can
    # reconnect) and must give up after max retries instead of spinning forever.
    conn, _sent, exchange = _make_connector()
    exchange.watch_orders = AsyncMock(side_effect=ccxt.NetworkError("connection reset"))
    _arm_control(conn)

    with patch("asyncio.sleep", new=AsyncMock()):
        await conn._subscribe_executions()

    assert conn.is_ws_ready() is False
    assert exchange.watch_orders.await_count == conn.max_ws_retries


@pytest.mark.asyncio
async def test_ws_ready_clears_and_logs_loud_on_auth_error() -> None:
    # A revoked key is unrecoverable by retrying — readiness must clear AND the
    # operator must see an explicit ERROR pointing at the keys.
    conn, _sent, exchange = _make_connector()
    exchange.watch_orders = AsyncMock(side_effect=ccxt.AuthenticationError("key revoked"))
    _arm_control(conn)

    records: list = []
    sink_id = logger.add(lambda m: records.append(m.record), level="ERROR")
    try:
        with patch("asyncio.sleep", new=AsyncMock()):
            await conn._subscribe_executions()
    finally:
        logger.remove(sink_id)

    assert conn.is_ws_ready() is False
    assert any("authentication failed" in r["message"] for r in records)


@pytest.mark.asyncio
async def test_ws_ready_recovers_after_transient_error() -> None:
    # A transient error clears readiness only for the backoff window: the loop re-arms
    # it when it re-drives the watch, before any message arrives. Clean exit resets it.
    conn, _sent, exchange = _make_connector()
    observed: list[tuple[str, bool]] = []

    async def _watch():
        if exchange.watch_orders.await_count == 1:
            raise ccxt.NetworkError("blip")
        observed.append(("rearmed", conn.is_ws_ready()))
        conn.channel.control.is_set = Mock(return_value=False)
        return []

    async def _backoff(_delay):
        observed.append(("backoff", conn.is_ws_ready()))

    exchange.watch_orders = AsyncMock(side_effect=_watch)
    _arm_control(conn)

    with patch("asyncio.sleep", new=AsyncMock(side_effect=_backoff)):
        await conn._subscribe_executions()

    assert observed == [("backoff", False), ("rearmed", True)]
    assert conn.is_ws_ready() is False  # reset on clean exit


@pytest.mark.asyncio
async def test_ws_ready_true_while_quiet_after_transient_error() -> None:
    # The R4/R6 composition pin: ccxt rejects pending watch futures with NetworkError on
    # every routine WS connection close (Binance drops user-data streams ~daily). On a
    # quiet account the silent re-subscription delivers no message, so readiness must
    # re-arm on the re-drive itself — or AM liveness recreates the exchange per drop.
    conn, _sent, exchange = _make_connector()
    block = asyncio.Event()

    async def _watch():
        if exchange.watch_orders.await_count == 1:
            raise ccxt.NetworkError("connection closed by remote server")
        await block.wait()  # re-watch parks forever: no order traffic

    exchange.watch_orders = AsyncMock(side_effect=_watch)
    _arm_control(conn)

    real_sleep = asyncio.sleep
    try:
        with patch("asyncio.sleep", new=AsyncMock()):
            task = asyncio.ensure_future(conn._subscribe_executions())
            await real_sleep(0.01)
            assert conn.is_ws_ready() is True  # re-armed while parked, no message needed
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


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
    exchange.fetch_open_orders = AsyncMock(return_value=[_ws_order(status="open", cid="qubx_BTCUSDT_1", venue_id="V1")])
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
    am = SimulatedAccountManager(
        connectors={"BINANCE.UM": object()},
        base_currencies={"BINANCE.UM": "USDT"},
        time=DummyTimeProvider(),
    )
    am.apply(event)  # must NOT raise

    # The open order was registered against the real account state (materialized
    # RECOVERED since its cid carries the qubx_ framework prefix).
    registered = am.find_order_by_id("V1")
    assert registered is not None
    assert registered.status is OrderStatus.ACCEPTED
    assert registered.origin is OrderOrigin.RECOVERED
    assert registered.client_order_id == "qubx_BTCUSDT_1"
    assert am.get_balance("USDT", exchange="BINANCE.UM").total == 1000.0


# --------------------------------------------------------------------------- #
# (g) F26 — WS position/balance pushes + account-stream composition
# --------------------------------------------------------------------------- #
def _ws_position(*, contracts: float = 0.03, side: str = "long", ps: str = "BOTH", ts: int = 1700000000000) -> dict:
    # ccxt unified position as pro/binance.parse_ws_position emits it for an
    # ACCOUNT_UPDATE P entry: contracts is abs, side carries the sign, markPrice /
    # maintenanceMargin are None, timestamp is the event time E (stamped in
    # handle_positions), and the raw P entry rides in info (incl. hedge-mode ps).
    return {
        "info": {"s": "ETHUSDT", "pa": "0.03", "ep": "3383.73", "ps": ps},
        "symbol": "ETH/USDT:USDT",
        "contracts": contracts,
        "side": side,
        "entryPrice": 3383.73,
        "unrealizedPnl": 1.5,
        "markPrice": None,
        "maintenanceMargin": None,
        "timestamp": ts,
    }


def _ws_balance(*, reason: str = "ORDER", event_time: int = 1700000000123) -> dict:
    # ccxt unified watch_balance result: a currency cache plus the raw last venue
    # message in info — for futures, the ACCOUNT_UPDATE with the changed assets in a.B.
    return {
        "info": {
            "e": "ACCOUNT_UPDATE",
            "E": event_time,
            "T": event_time - 3,
            "a": {
                "m": reason,
                "B": [
                    {"a": "USDT", "wb": "122624.125", "cw": "100.1"},
                    {"a": "BNB", "wb": "10.5", "cw": "10.5"},
                ],
                "P": [],
            },
        },
        "timestamp": event_time,
        "USDT": {"free": None, "used": None, "total": 122624.125},
    }


def test_ws_position_push_emits_position_update_event() -> None:
    conn, sent, exchange = _make_connector()
    # Production contract: the factory stamps the framework exchange name onto the ccxt
    # exchange ({"name": exchange}), so converted instruments route to the right state.
    exchange.name = "BINANCE.UM"
    exchange.markets = BINANCE_MARKETS

    conn._handle_ws_position(_ws_position(contracts=0.03, side="short"))

    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, PositionUpdateEvent)
    assert ev.position.quantity == -0.03  # signed from contracts/side
    assert ev.position.instrument.symbol == "ETHUSDT"
    assert ev.position.instrument.exchange == "BINANCE.UM"
    assert ev.instrument == ev.position.instrument
    assert ev.as_of == np.datetime64(1700000000000, "ms")  # venue event time E


def test_ws_position_push_hedge_mode_skipped() -> None:
    conn, sent, exchange = _make_connector()
    exchange.markets = BINANCE_MARKETS
    conn._handle_ws_position(_ws_position(ps="LONG"))
    assert sent == []


def test_ws_balance_push_emits_per_asset_events() -> None:
    conn, sent, _ = _make_connector()

    conn._handle_ws_balances(_ws_balance(reason="FUNDING_FEE"))

    assert len(sent) == 2
    by_ccy = {e.balance.currency: e for e in sent}
    ev = by_ccy["USDT"]
    assert isinstance(ev, BalanceUpdateEvent)
    assert ev.balance.exchange == "BINANCE.UM"
    assert ev.balance.total == 122624.125
    # Futures pushes carry no free/locked split: NaN tells the reducer total-only.
    assert math.isnan(ev.balance.free) and math.isnan(ev.balance.locked)
    assert ev.reason == "FUNDING_FEE"
    assert ev.as_of == np.datetime64(1700000000123, "ms")  # venue event time E
    assert by_ccy["BNB"].balance.total == 10.5


def test_ws_balance_push_without_account_update_payload_skipped() -> None:
    # A non-futures payload (spot balanceUpdate: info.a is the asset string) must not emit.
    conn, sent, _ = _make_connector()
    conn._handle_ws_balances({"info": {"e": "balanceUpdate", "E": 1, "a": "IOTX", "d": "0.4"}, "timestamp": 1})
    assert sent == []


def _record_streams(conn: CcxtConnector) -> list[dict]:
    """Replace _run_ws_loop with a recorder so _subscribe_executions composition can be
    asserted without driving real watch loops."""
    recorded: list[dict] = []

    async def _fake_loop(**kwargs) -> None:
        recorded.append(kwargs)

    conn._run_ws_loop = _fake_loop  # type: ignore[method-assign]
    return recorded


def _binance_exchange(*, has: dict, options: dict) -> Mock:
    """Exchange mock that passes the isinstance(ex, ccxt.pro.binance) D4 gate."""
    exchange = Mock(spec=ccxt.pro.binance)
    exchange.has = has
    exchange.options = options
    return exchange


@pytest.mark.asyncio
async def test_account_streams_derivatives_venue_adds_position_and_balance_loops() -> None:
    exchange = _binance_exchange(
        has={"watchPositions": True, "watchBalance": True},
        options={"defaultSubType": "linear"},  # binanceusdm: defaultType stays 'spot'
    )
    conn, _, _ = _make_connector(exchange=exchange)
    recorded = _record_streams(conn)

    await conn._subscribe_executions()

    assert [k["stream"] for k in recorded] == ["executions", "positions", "balance"]
    # Only the orders loop owns liveness; watch_balance resolves a single dict.
    assert [k["mark_ready"] for k in recorded] == [True, False, False]
    assert [k.get("iterate", True) for k in recorded] == [True, True, False]
    assert recorded[1]["handle"] == conn._handle_ws_position
    assert recorded[2]["handle"] == conn._handle_ws_balances
    # ccxt's own positions snapshot fetch is disabled — AM owns the snapshot fetch.
    assert exchange.options["watchPositions"] == {"fetchPositionsSnapshot": False, "awaitPositionsSnapshot": False}


@pytest.mark.asyncio
async def test_account_streams_spot_venue_no_push_loops() -> None:
    exchange = _binance_exchange(
        has={"watchPositions": True, "watchBalance": True},
        options={"defaultType": "spot"},
    )
    conn, _, _ = _make_connector(exchange=exchange)
    recorded = _record_streams(conn)

    await conn._subscribe_executions()

    assert [k["stream"] for k in recorded] == ["executions"]


@pytest.mark.asyncio
async def test_account_streams_derivatives_without_watch_support_no_push_loops() -> None:
    exchange = _binance_exchange(has={"editOrder": True}, options={"defaultSubType": "linear"})
    conn, _, _ = _make_connector(exchange=exchange)
    recorded = _record_streams(conn)

    await conn._subscribe_executions()

    assert [k["stream"] for k in recorded] == ["executions"]


@pytest.mark.asyncio
async def test_account_streams_non_binance_derivatives_venue_no_push_loops() -> None:
    # D4 scope pin: a non-Binance derivatives venue (hyperliquid-like) advertises the
    # same has[] capability flags, but the push handlers only parse Binance
    # ACCOUNT_UPDATE shapes — only the executions loop may compose.
    exchange = Mock()
    exchange.has = {"watchPositions": True, "watchBalance": True}
    exchange.options = {"defaultType": "swap"}
    conn, _, _ = _make_connector(exchange=exchange)
    recorded = _record_streams(conn)

    await conn._subscribe_executions()

    assert [k["stream"] for k in recorded] == ["executions"]


@pytest.mark.asyncio
async def test_two_stream_subscribes_orders_and_trades_only() -> None:
    # Even on a derivatives venue advertising the push streams, the split-venue
    # connector keeps exactly its two loops (F26 push streams are Binance-only — D4).
    exchange = Mock()
    exchange.has = {"watchPositions": True, "watchBalance": True}
    exchange.options = {"defaultType": "swap"}
    conn, _, _ = _make_connector(exchange=exchange, cls=_TwoStreamCcxtConnector)
    recorded = _record_streams(conn)

    await conn._subscribe_executions()

    assert [k["stream"] for k in recorded] == ["orders", "my_trades"]
    assert [k["mark_ready"] for k in recorded] == [True, False]
    assert recorded[1]["handle"] == conn._handle_ws_trade
