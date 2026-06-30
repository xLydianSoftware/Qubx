"""Unit tests for the CcxtConnector write side (commit 7.1).

Mocked ccxt — no credentials or network. The connector fires venue calls on an
asyncio loop via ``_spawn``; tests replace ``_spawn`` with a capturing stub and
await the captured coroutine directly, so the async work is driven
deterministically without crossing a real thread/loop boundary.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import ccxt
import pytest

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    MarketType,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from qubx.core.connector import IConnector
from qubx.core.events import (
    OrderAcceptedEvent,
    OrderCanceledEvent,
    OrderCancelRejectedEvent,
    OrderRejectedEvent,
    OrderUpdatedEvent,
    OrderUpdateRejectedEvent,
)
from qubx.core.exceptions import BadRequest, InvalidOrderParameters
from qubx.core.series import Quote
from tests.qubx.core.utils_test import DummyTimeProvider


def _instrument(min_notional: float = 0.0) -> Instrument:
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
        min_notional=min_notional,
    )


def _quote(bid: float = 99.0, ask: float = 101.0) -> Quote:
    return Quote(0, bid, ask, 1.0, 1.0)


def _make_connector(
    *,
    exchange: Mock | None = None,
    data_provider: Mock | None = None,
) -> tuple[CcxtConnector, list, Mock]:
    """Build a connector with a capturing channel and a mocked exchange.

    Returns (connector, sent_events, exchange). ``_spawn`` is replaced with a
    capture so tests can await the coroutine themselves.
    """
    if exchange is None:
        exchange = Mock()
        exchange.create_order = AsyncMock(return_value={})
        exchange.cancel_order = AsyncMock(return_value={})
        exchange.cancel_order_with_client_order_id = AsyncMock(return_value={})
        exchange.edit_order = AsyncMock(return_value={})
        exchange.has = {"editOrder": True}

    em = Mock()
    em.exchange = exchange

    if data_provider is None:
        data_provider = Mock()
        data_provider.get_quote = Mock(return_value=_quote())

    sent: list = []
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock(side_effect=lambda e: sent.append(e))

    conn = CcxtConnector(
        exchange_name="BINANCE.UM",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=data_provider,
    )

    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))
    conn._captured = captured  # type: ignore[attr-defined]

    # _run_sync (leverage / margin) is synchronous from the caller's POV: drive the
    # coroutine to completion on a throwaway loop so tests need no real loop thread.
    def _run_sync(coro, timeout=None):
        return asyncio.new_event_loop().run_until_complete(coro)

    conn._run_sync = Mock(side_effect=_run_sync)
    return conn, sent, exchange


async def _drive(conn: CcxtConnector) -> None:
    """Await all coroutines captured by the stubbed _spawn."""
    for coro in conn._captured:  # type: ignore[attr-defined]
        await coro
    conn._captured.clear()  # type: ignore[attr-defined]


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


def _order(
    *,
    client_order_id: str = "qubx_BTCUSDT_1",
    venue_order_id: str | None = None,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    quantity: float = 1.0,
    price: float | None = 100.0,
) -> Order:
    """Build an ACCEPTED order to hand to the connector's cancel/update/status calls.

    The connector reads symbol/side/type/ids straight off it (it keeps no order cache).
    """
    return Order(
        client_order_id=client_order_id,
        type=order_type,
        instrument=_instrument(),
        quantity=quantity,
        side=side,
        time_in_force="gtc",
        status=OrderStatus.ACCEPTED,
        venue_order_id=venue_order_id,
        price=price,
    )


# --------------------------------------------------------------------------- #
# (10) protocol conformance
# --------------------------------------------------------------------------- #
def test_isinstance_iconnector() -> None:
    conn, _, _ = _make_connector()
    assert isinstance(conn, IConnector)


# --------------------------------------------------------------------------- #
# (1) submit_order builds the correct ccxt payload
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_submit_builds_payload_limit_with_client_id() -> None:
    conn, _sent, exchange = _make_connector()
    conn.submit_order(_order_request())
    await _drive(conn)

    exchange.create_order.assert_awaited_once()
    payload = exchange.create_order.await_args.kwargs
    assert payload["symbol"] == "BTC/USDT:USDT"
    assert payload["type"] == "limit"
    assert payload["side"] == "buy"
    assert payload["amount"] == 1.0
    assert payload["price"] == 100.0
    assert payload["params"]["clientOrderId"] == "qubx_BTCUSDT_1"
    assert payload["params"]["timeInForce"] == "GTC"
    assert payload["params"]["type"] == "swap"  # futures
    assert "reduceOnly" not in payload["params"]


@pytest.mark.asyncio
async def test_submit_payload_reduce_only_and_trigger() -> None:
    # order_type arrives UPPERCASE from the trading manager (OrderType StrEnum) — the trigger
    # detection must be case-insensitive (a lowercase-only startswith dropped triggerPrice live).
    conn, _sent, exchange = _make_connector()
    conn.submit_order(_order_request(order_type="STOP_LIMIT", price=120.0, options={"reduceOnly": True}))
    await _drive(conn)

    payload = exchange.create_order.await_args.kwargs
    assert payload["params"]["reduceOnly"] is True
    assert payload["params"]["triggerPrice"] == 120.0
    assert payload["type"] == "limit"  # stop_ prefix stripped


@pytest.mark.asyncio
async def test_submit_stop_market_sets_trigger_price() -> None:
    # The live AtrRiskTracker case: a STOP_MARKET (uppercase) must carry triggerPrice or Binance
    # rejects "requires a triggerPrice extra param for a stop_market order".
    conn, _sent, exchange = _make_connector()
    conn.submit_order(_order_request(order_type="STOP_MARKET", price=58331.5))
    await _drive(conn)

    payload = exchange.create_order.await_args.kwargs
    assert payload["params"]["triggerPrice"] == 58331.5
    assert payload["type"] == "market"  # stop_ prefix stripped


@pytest.mark.asyncio
async def test_submit_gtx_buy_price_adjustment() -> None:
    conn, _sent, exchange = _make_connector()
    # GTX BUY priced >= ask (101) must be nudged 1 tick below ask -> 101 - 0.1
    conn.submit_order(_order_request(price=105.0, time_in_force="gtx"))
    await _drive(conn)

    payload = exchange.create_order.await_args.kwargs
    assert payload["params"]["timeInForce"] == "GTX"
    assert payload["price"] == pytest.approx(101.0 - 0.1)


# --------------------------------------------------------------------------- #
# (2) framework-side validation RAISES synchronously
# --------------------------------------------------------------------------- #
def test_submit_raises_on_zero_quantity() -> None:
    conn, _, exchange = _make_connector()
    with pytest.raises(InvalidOrderParameters):
        conn.submit_order(_order_request(quantity=0.0))
    exchange.create_order.assert_not_awaited()


def test_submit_raises_when_quote_unavailable() -> None:
    dp = Mock()
    dp.get_quote = Mock(return_value=None)
    conn, _, exchange = _make_connector(data_provider=dp)
    with pytest.raises(BadRequest):
        conn.submit_order(_order_request())
    exchange.create_order.assert_not_awaited()


def test_submit_raises_below_min_notional() -> None:
    dp = Mock()
    dp.get_quote = Mock(return_value=_quote())  # mid ~100
    conn, _, exchange = _make_connector(data_provider=dp)
    req = _order_request(instrument=_instrument(min_notional=10_000.0), quantity=1.0)
    with pytest.raises(InvalidOrderParameters):
        conn.submit_order(req)
    exchange.create_order.assert_not_awaited()


def test_submit_raises_missing_price_for_limit() -> None:
    conn, _, exchange = _make_connector()
    with pytest.raises(InvalidOrderParameters):
        conn.submit_order(_order_request(price=None))
    exchange.create_order.assert_not_awaited()


# --------------------------------------------------------------------------- #
# (3) venue verdict from create_order -> OrderRejectedEvent (NOT raised)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_submit_venue_verdict_emits_rejected_not_raised() -> None:
    exchange = Mock()
    exchange.create_order = AsyncMock(side_effect=ccxt.InsufficientFunds("not enough margin"))
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())  # must not raise
    await _drive(conn)

    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderRejectedEvent)
    assert ev.client_order_id == "qubx_BTCUSDT_1"
    assert "not enough margin" in ev.reason
    assert ev.code == "InsufficientFunds"


@pytest.mark.asyncio
async def test_submit_generic_exchange_error_emits_rejected() -> None:
    exchange = Mock()
    exchange.create_order = AsyncMock(side_effect=ccxt.ExchangeError("boom"))
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)

    assert isinstance(sent[0], OrderRejectedEvent)


@pytest.mark.parametrize(
    "error",
    [
        ccxt.RateLimitExceeded("slow down"),
        ccxt.ExchangeNotAvailable("down"),
        ccxt.OnMaintenance("maintenance window"),
    ],
)
@pytest.mark.asyncio
async def test_submit_networkerror_subclass_verdicts_emit_rejected(error) -> None:
    # RateLimitExceeded / ExchangeNotAvailable / OnMaintenance are ccxt NetworkError
    # *subclasses* but they are venue verdicts (the venue actively refused), so they
    # must emit OrderRejectedEvent — NOT be swallowed as transient and left inflight.
    exchange = Mock()
    exchange.create_order = AsyncMock(side_effect=error)
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)

    assert len(sent) == 1
    assert isinstance(sent[0], OrderRejectedEvent)
    assert sent[0].client_order_id == "qubx_BTCUSDT_1"


# --------------------------------------------------------------------------- #
# (4) successful create with venue id -> OrderAcceptedEvent
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_submit_success_emits_accepted() -> None:
    exchange = Mock()
    exchange.create_order = AsyncMock(
        return_value={
            "id": "VENUE123",
            "clientOrderId": "qubx_BTCUSDT_1",
            "status": "NEW",
            "side": "buy",
            "type": "limit",
            "amount": 1.0,
            "price": 100.0,
            "timestamp": 1700000000000,
            "cost": 0.0,
            "timeInForce": "GTC",
            "info": {},
        }
    )
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)

    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderAcceptedEvent)
    assert ev.venue_order_id == "VENUE123"
    assert ev.client_order_id == "qubx_BTCUSDT_1"


@pytest.mark.asyncio
async def test_submit_no_id_emits_nothing() -> None:
    exchange = Mock()
    exchange.create_order = AsyncMock(return_value={"id": None})
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)
    assert sent == []  # WS read side will surface the ack later


# --------------------------------------------------------------------------- #
# (5) cancel_order
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_cancel_by_venue_id_emits_canceled() -> None:
    # An order carrying a venue id cancels through the venue-id endpoint, with the ccxt
    # symbol the connector reads straight off the order's instrument (no cache).
    exchange = Mock()
    exchange.cancel_order = AsyncMock(return_value={"id": "VENUE123", "clientOrderId": "qubx_BTCUSDT_1"})
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.cancel_order(_order(venue_order_id="VENUE123"))
    await _drive(conn)

    exchange.cancel_order.assert_awaited_once_with("VENUE123", "BTC/USDT:USDT")
    assert isinstance(sent[0], OrderCanceledEvent)
    assert sent[0].venue_order_id == "VENUE123"


@pytest.mark.asyncio
async def test_cancel_stop_order_uses_trigger_surface() -> None:
    # A STOP order lives on the venue's conditional/algo surface; the cancel must pass
    # params={'trigger': True} (driven by order.type), else Binance answers -2011 and the
    # live stop can't be cancelled (e.g. on a flatten signal).
    exchange = Mock()
    exchange.cancel_order = AsyncMock(return_value={"id": "VENUE123", "clientOrderId": "qubx_BTCUSDT_1"})
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.cancel_order(_order(venue_order_id="VENUE123", order_type=OrderType.STOP_MARKET))
    await _drive(conn)

    exchange.cancel_order.assert_awaited_once_with("VENUE123", "BTC/USDT:USDT", params={"trigger": True})
    assert isinstance(sent[0], OrderCanceledEvent)


@pytest.mark.asyncio
async def test_cancel_stop_order_by_cloid_uses_trigger_surface() -> None:
    # Same for the cloid path (venue id not seen yet).
    exchange = Mock()
    exchange.cancel_order_with_client_order_id = AsyncMock(return_value={"id": "VENUE123"})
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.cancel_order(_order(venue_order_id=None, order_type=OrderType.STOP_MARKET))
    await _drive(conn)

    exchange.cancel_order_with_client_order_id.assert_awaited_once_with(
        "qubx_BTCUSDT_1", "BTC/USDT:USDT", params={"trigger": True}
    )
    assert isinstance(sent[0], OrderCanceledEvent)


@pytest.mark.asyncio
async def test_cancel_by_cloid_uses_cloid_endpoint() -> None:
    # No venue id yet (ack not seen) -> cloid endpoint, symbol from the order's instrument.
    exchange = Mock()
    exchange.cancel_order_with_client_order_id = AsyncMock(return_value={"id": "VENUE123"})
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.cancel_order(_order(venue_order_id=None))
    await _drive(conn)

    exchange.cancel_order_with_client_order_id.assert_awaited_once_with("qubx_BTCUSDT_1", "BTC/USDT:USDT")
    assert isinstance(sent[0], OrderCanceledEvent)


@pytest.mark.asyncio
async def test_cancel_venue_reject_emits_cancel_rejected() -> None:
    exchange = Mock()
    # OperationRejected on an already-filled order -> definitive failure (no retry)
    exchange.cancel_order = AsyncMock(side_effect=ccxt.OperationRejected("Order already filled"))
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    # The reject carries both ids (the order always has its cid) so AM can revert the order
    # from PENDING_CANCEL by either.
    conn.cancel_order(_order(venue_order_id="VENUE123"))
    await _drive(conn)

    assert len(sent) == 1
    assert isinstance(sent[0], OrderCancelRejectedEvent)
    assert sent[0].client_order_id == "qubx_BTCUSDT_1"
    assert sent[0].venue_order_id == "VENUE123"  # both ids carried so AM routes by either


@pytest.mark.asyncio
async def test_cancel_cloid_network_error_leaves_inflight_no_reject() -> None:
    # A transient network error on a cloid cancel is an UNKNOWN outcome (the cancel may
    # still have landed): leave the order inflight, do NOT emit a terminal cancel-reject
    # (which would wrongly revert PENDING_CANCEL -> ACCEPTED).
    exchange = Mock()
    exchange.cancel_order_with_client_order_id = AsyncMock(side_effect=ccxt.NetworkError("timeout"))
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.cancel_order(_order(venue_order_id=None))
    await _drive(conn)

    assert sent == []


@pytest.mark.asyncio
async def test_submit_network_error_leaves_inflight_no_reject() -> None:
    # Transient network error is an UNKNOWN outcome, not a venue verdict: the order
    # must be left inflight (no terminal OrderRejectedEvent) for AM to reconcile.
    exchange = Mock()
    exchange.create_order = AsyncMock(side_effect=ccxt.NetworkError("connection reset"))
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.submit_order(_order_request())
    await _drive(conn)

    assert sent == []


@pytest.mark.asyncio
async def test_update_network_error_leaves_inflight_no_reject() -> None:
    exchange = Mock()
    exchange.edit_order = AsyncMock(side_effect=ccxt.NetworkError("timeout"))
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.update_order(_order(venue_order_id="VENUE123"), price=123.0)
    await _drive(conn)

    assert sent == []


# --------------------------------------------------------------------------- #
# (6) update_order
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_update_direct_edit_emits_updated() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.update_order(_order(venue_order_id="VENUE123"), price=102.0, quantity=2.0)
    await _drive(conn)

    # symbol/side/type come straight off the order — the venue-id edit endpoint gets them all.
    exchange.edit_order.assert_awaited_once_with(
        id="VENUE123", symbol="BTC/USDT:USDT", type="limit", side="buy", amount=2.0, price=102.0, params={}
    )
    ev = sent[0]
    assert isinstance(ev, OrderUpdatedEvent)
    assert ev.client_order_id == "qubx_BTCUSDT_1"
    assert ev.venue_order_id == "VENUE123"
    assert ev.new_price == 102.0
    assert ev.new_quantity == 2.0


@pytest.mark.asyncio
async def test_update_by_cloid_uses_cloid_edit_endpoint() -> None:
    # No venue id yet -> ccxt's client-order-id edit variant, with symbol/side/type off the order.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.edit_order_with_client_order_id = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.update_order(_order(venue_order_id=None), price=102.0, quantity=2.0)
    await _drive(conn)

    exchange.edit_order_with_client_order_id.assert_awaited_once_with(
        "qubx_BTCUSDT_1", "BTC/USDT:USDT", "limit", "buy", 2.0, 102.0
    )
    assert isinstance(sent[0], OrderUpdatedEvent)


@pytest.mark.asyncio
async def test_update_edit_venue_reject_emits_update_rejected() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.edit_order = AsyncMock(side_effect=ccxt.InvalidOrder("cannot edit"))
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.update_order(_order(venue_order_id="VENUE123"), price=102.0)
    await _drive(conn)

    assert isinstance(sent[0], OrderUpdateRejectedEvent)
    assert sent[0].venue_order_id == "VENUE123"  # both ids carried so AM routes by either


@pytest.mark.asyncio
async def test_update_cancel_recreate_path_rejects_without_touching_live_order() -> None:
    # Exchange without editOrder support -> cancel+recreate path, not yet wired, so it must
    # reject WITHOUT cancelling: cancelling first would leave the order dead at the venue
    # while telling the strategy "still alive".
    exchange = Mock()
    exchange.has = {"editOrder": False}
    exchange.cancel_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.update_order(_order(venue_order_id="VENUE123"), price=102.0, quantity=2.0)
    await _drive(conn)

    exchange.cancel_order.assert_not_awaited()  # live order left untouched
    assert isinstance(sent[0], OrderUpdateRejectedEvent)


# --------------------------------------------------------------------------- #
# (7) make_client_id prefix
# --------------------------------------------------------------------------- #
def test_make_client_id_adds_prefix() -> None:
    conn, _, _ = _make_connector()
    assert conn.make_client_id("abc123") == "qubx_abc123"
    assert conn.make_client_id("qubx_BTCUSDT_1") == "qubx_BTCUSDT_1"


# --------------------------------------------------------------------------- #
# (8) set_instrument_leverage / set_margin_mode call ccxt + return bool
# --------------------------------------------------------------------------- #
def test_set_leverage_calls_ccxt_returns_true() -> None:
    exchange = Mock()
    exchange.set_leverage = AsyncMock(return_value={})
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    ok = conn.set_instrument_leverage(_instrument(), 5.0)
    assert ok is True
    exchange.set_leverage.assert_awaited_once_with(5.0, "BTC/USDT:USDT")


def test_set_leverage_returns_false_on_error() -> None:
    exchange = Mock()
    exchange.set_leverage = AsyncMock(side_effect=ccxt.ExchangeError("nope"))
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    assert conn.set_instrument_leverage(_instrument(), 5.0) is False


def test_set_margin_mode_calls_ccxt_returns_true() -> None:
    exchange = Mock()
    exchange.set_margin_mode = AsyncMock(return_value={})
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    ok = conn.set_margin_mode(_instrument(), "isolated")
    assert ok is True
    exchange.set_margin_mode.assert_awaited_once_with("isolated", "BTC/USDT:USDT")
