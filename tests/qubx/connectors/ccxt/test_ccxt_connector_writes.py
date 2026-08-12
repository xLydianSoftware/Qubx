"""Unit tests for the CcxtConnector write side (commit 7.1).

Mocked ccxt — no credentials or network. The connector fires venue calls on an
asyncio loop via ``_spawn``; tests replace ``_spawn`` with a capturing stub and
await the captured coroutine directly, so the async work is driven
deterministically without crossing a real thread/loop boundary.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import ccxt
import pytest

from qubx.connectors.ccxt.connector import CcxtConnector, _LeverageInfo
from qubx.connectors.ccxt.rate_limits import _default_endpoint_costs
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
from qubx.core.errors import VenueOperationError
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
from qubx.rate_limiting import EndpointCosts, ExchangeRateLimitConfig, ExchangeRateLimiter, PoolConfig
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
    rate_limiter: ExchangeRateLimiter | None = None,
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
    em.rate_limiter = rate_limiter

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
async def test_cancel_acked_order_gone_at_venue_emits_nothing_no_retry() -> None:
    # An ACKED order (has venue id) that the venue already removed (filled/expired/canceled)
    # before our cancel lands answers -2011 "Unknown order sent". Because it was acked, that is
    # DEFINITIVE ("gone"), not the submit/cancel race — so the connector must NOT retry and must
    # emit nothing (the WS terminal + snapshot reconciler resolve it), avoiding the 32s retry storm.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.cancel_order = AsyncMock(
        side_effect=ccxt.ExchangeError('binanceusdm {"code":-2011,"msg":"Unknown order sent."}')
    )
    conn, sent, _ = _make_connector(exchange=exchange)

    with patch("qubx.connectors.ccxt.connector.asyncio.sleep", AsyncMock()):
        conn.cancel_order(_order(venue_order_id="VENUE123"))
        await _drive(conn)

    exchange.cancel_order.assert_awaited_once()  # - definitive 'gone' -> no retry storm
    assert sent == []  # - neither canceled nor cancel-rejected; WS/snapshot resolve the terminal


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
    # negative control for the gate-timeout cancel: only a rate-limited reject is coded
    assert sent[0].code is None


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


@pytest.mark.asyncio
async def test_update_replacement_dialect_sends_total_minus_filled() -> None:
    # Hyperliquid-style modify is a venue-side cancel+replace: the amend amount is the
    # REPLACEMENT order's size (remaining), while the framework speaks totals. The
    # connector must translate on the wire and still echo the requested TOTAL on the ack.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.AMEND_QUANTITY_DIALECT = "replacement"
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    order.filled_quantity = 0.4
    conn.update_order(order, price=102.0, quantity=2.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(1.6)  # total - filled on the wire
    ev = sent[0]
    assert isinstance(ev, OrderUpdatedEvent)
    assert ev.new_quantity == 2.0  # requested TOTAL on the event


@pytest.mark.asyncio
async def test_update_total_dialect_passes_total_verbatim() -> None:
    # Binance/Gate amend quantity IS the new total (executedQty preserved): passthrough.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT  # plain Mock auto-creates attrs; ensure absent
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    order.filled_quantity = 0.4
    conn.update_order(order, price=102.0, quantity=2.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(2.0)
    assert sent[0].new_quantity == 2.0


@pytest.mark.asyncio
async def test_update_price_only_resolves_quantity_from_order_and_echoes_none() -> None:
    # Binance requires both quantity and price on modify — a price-only update sends the
    # order's current total on the wire but the ack event says "quantity unchanged".
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    conn.update_order(order, price=102.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(2.0)  # resolved from the order
    assert kwargs["price"] == 102.0
    ev = sent[0]
    assert isinstance(ev, OrderUpdatedEvent)
    assert ev.new_price == 102.0
    assert ev.new_quantity is None  # unchanged — reducer keeps its total


@pytest.mark.asyncio
async def test_update_quantity_only_resolves_price_from_order() -> None:
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    conn.update_order(order, quantity=3.0)
    await _drive(conn)

    _, kwargs = exchange.edit_order.await_args
    assert kwargs["amount"] == pytest.approx(3.0)
    assert kwargs["price"] == order.price  # resolved from the order
    ev = sent[0]
    assert ev.new_price is None
    assert ev.new_quantity == 3.0


@pytest.mark.asyncio
async def test_update_silent_cancel_response_emits_canceled_not_updated() -> None:
    # Binance/Gate documented behavior: an amend whose total lands at/below executedQty
    # CANCELS the order (no reject). Racing fills can trigger this despite the mixin
    # pre-check — the connector must surface the truth as a cancel, not an update-ack.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123", "status": "canceled"})
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.update_order(_order(venue_order_id="VENUE123"), price=102.0, quantity=2.0)
    await _drive(conn)

    assert len(sent) == 1
    assert isinstance(sent[0], OrderCanceledEvent)
    assert sent[0].venue_order_id == "VENUE123"


def test_update_replacement_dialect_zero_remaining_raises_synchronously() -> None:
    # A fully-filled order under the replacement dialect translates to a <= 0 wire
    # amount, which is meaningless to send — raise synchronously (no venue call spawned)
    # rather than let the venue reject it asynchronously.
    exchange = Mock()
    exchange.has = {"editOrder": True}
    exchange.AMEND_QUANTITY_DIALECT = "replacement"
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    conn, sent, _ = _make_connector(exchange=exchange)

    order = _order(venue_order_id="VENUE123")
    order.quantity = 2.0
    order.filled_quantity = 2.0
    with pytest.raises(ValueError):
        conn.update_order(order, quantity=2.0)

    exchange.edit_order.assert_not_awaited()
    assert sent == []


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
@pytest.mark.asyncio
async def test_set_leverage_sends_the_venue_call_off_thread() -> None:
    """The caller is the ProcessorThread, so the request is spawned, not awaited: the
    method returns before the venue has seen anything."""
    exchange = Mock()
    exchange.set_leverage = AsyncMock(return_value={})
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    assert conn.set_instrument_leverage(_instrument(), 5.0) is None
    exchange.set_leverage.assert_not_awaited()

    await _drive(conn)
    exchange.set_leverage.assert_awaited_once_with(5, "BTC/USDT:USDT")


@pytest.mark.asyncio
async def test_set_leverage_reports_a_refusal_on_the_channel() -> None:
    """Nothing is waiting on a return value, so a venue refusal would be invisible
    without an event."""
    exchange = Mock()
    exchange.set_leverage = AsyncMock(side_effect=ccxt.ExchangeError("nope"))
    exchange.has = {"editOrder": True}
    conn, sent, _ = _make_connector(exchange=exchange)

    conn.set_instrument_leverage(_instrument(), 5.0)
    await _drive(conn)

    errors = [e for _, dtype, e, _ in sent if dtype == "error"]
    assert len(errors) == 1
    assert isinstance(errors[0], VenueOperationError)
    assert errors[0].operation == "set_instrument_leverage"
    assert "nope" in str(errors[0].error)


@pytest.mark.asyncio
async def test_set_leverage_is_skipped_when_the_venue_already_has_it() -> None:
    """Most of a universe is already at the wanted leverage on any tick after the first;
    sending those anyway was the bulk of the ~1.2s-per-instrument cost."""
    exchange = Mock()
    exchange.set_leverage = AsyncMock(return_value={})
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)
    conn._leverage_cache["BTC/USDT:USDT"] = _LeverageInfo(configured=5, maximum=20)

    conn.set_instrument_leverage(_instrument(), 5.0)

    await _drive(conn)
    exchange.set_leverage.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_leverage_is_clamped_to_the_cached_venue_maximum() -> None:
    exchange = Mock()
    exchange.set_leverage = AsyncMock(return_value={})
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)
    conn._leverage_cache["BTC/USDT:USDT"] = _LeverageInfo(configured=2, maximum=10)

    conn.set_instrument_leverage(_instrument(), 50.0)
    await _drive(conn)

    # - int on the wire: Binance rejects a float with -1102 (measured live 2026-08-07,
    #   every clamped request failed because the clamp produced a float)
    exchange.set_leverage.assert_awaited_once_with(10, "BTC/USDT:USDT")


@pytest.mark.asyncio
async def test_a_successful_set_updates_the_cache() -> None:
    """Without this the same value is re-sent every tick until the hourly refresh."""
    exchange = Mock()
    exchange.set_leverage = AsyncMock(return_value={})
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)
    conn._leverage_cache["BTC/USDT:USDT"] = _LeverageInfo(configured=2, maximum=10)

    conn.set_instrument_leverage(_instrument(), 5.0)
    await _drive(conn)

    assert conn._leverage_cache["BTC/USDT:USDT"] == _LeverageInfo(configured=5, maximum=10)

    conn.set_instrument_leverage(_instrument(), 5.0)
    await _drive(conn)
    assert exchange.set_leverage.await_count == 1


def test_set_margin_mode_calls_ccxt_returns_true() -> None:
    exchange = Mock()
    exchange.set_margin_mode = AsyncMock(return_value={})
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    ok = conn.set_margin_mode(_instrument(), "isolated")
    assert ok is True
    exchange.set_margin_mode.assert_awaited_once_with("isolated", "BTC/USDT:USDT")


# --------------------------------------------------------------------------- #
# (9) per-instrument venue-setting reads (off the venue position row)
# --------------------------------------------------------------------------- #
def _position_row(**over) -> dict:
    row = {
        "symbol": "BTC/USDT:USDT",
        "leverage": 10.0,
        "marginMode": "isolated",
        "info": {"adlQuantile": "2", "maxNotionalValue": "1000000"},
    }
    row.update(over)
    return row


def test_get_instrument_leverage_reads_position_row() -> None:
    exchange = Mock()
    exchange.fetch_positions = AsyncMock(return_value=[_position_row()])
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    assert conn.get_instrument_leverage(_instrument()) == 10.0
    exchange.fetch_positions.assert_awaited_once_with(["BTC/USDT:USDT"])


def test_get_max_instrument_notional_reads_position_row() -> None:
    exchange = Mock()
    exchange.fetch_positions = AsyncMock(return_value=[_position_row()])
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    assert conn.get_max_instrument_notional(_instrument()) == 1_000_000.0


def test_get_margin_mode_reads_position_row() -> None:
    exchange = Mock()
    exchange.fetch_positions = AsyncMock(return_value=[_position_row(marginMode="cross")])
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    assert conn.get_margin_mode(_instrument()) == "cross"


def test_get_adl_level_reads_position_info() -> None:
    exchange = Mock()
    exchange.fetch_positions = AsyncMock(return_value=[_position_row()])
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    assert conn.get_adl_level(_instrument()) == 2


def test_reads_none_inf_when_no_position() -> None:
    exchange = Mock()
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.has = {"editOrder": True}
    conn, _, _ = _make_connector(exchange=exchange)

    assert conn.get_instrument_leverage(_instrument()) is None
    assert conn.get_max_instrument_notional(_instrument()) == float("inf")
    assert conn.get_margin_mode(_instrument()) is None
    assert conn.get_adl_level(_instrument()) is None


# (11) order-count budget — every write charges the account-scoped `orders` pool
_ORDERS_CAPACITY = 5


@pytest.fixture
def order_limiter():
    """Real limiter (in-memory backend) with a small account-scoped ``orders`` pool.

    ``cooldown`` sits far above ``gate_max_wait`` so a gate closed by a 429 is still closed
    when the acquire gives up — otherwise the timeout path would be unobservable.
    """
    config = ExchangeRateLimitConfig(
        pools={
            "ccxt_rest": PoolConfig("ccxt_rest", "ip", 100, 100.0, cooldown=5.0),
            "orders": PoolConfig("orders", "account", _ORDERS_CAPACITY, 1.0, cooldown=5.0),
        },
        endpoint_map=_default_endpoint_costs(),
        default_costs=EndpointCosts([("ccxt_rest", 1)]),
        gate_max_wait=0.05,
    )
    limiter = ExchangeRateLimiter("BINANCE.UM", config)
    yield limiter
    limiter.reset_gates()


def _record_acquires(limiter: ExchangeRateLimiter, monkeypatch) -> list[str]:
    """Record acquired endpoints while still draining the real pools."""
    seen: list[str] = []
    real_acquire = limiter.acquire

    async def _recording(endpoint: str, **kw) -> None:
        seen.append(endpoint)
        await real_acquire(endpoint, **kw)

    monkeypatch.setattr(limiter, "acquire", _recording)
    return seen


def _write_exchange() -> Mock:
    """Exchange whose write calls all answer with a venue ack."""
    exchange = Mock()
    exchange.has = {"editOrder": True}
    del exchange.AMEND_QUANTITY_DIALECT
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
    exchange.cancel_order = AsyncMock(return_value={"id": "VENUE123", "clientOrderId": "qubx_BTCUSDT_1"})
    exchange.edit_order = AsyncMock(return_value={"id": "VENUE123"})
    return exchange


class TestOrderRateLimiting:
    """Order endpoints debit the account-scoped ``orders`` pool — nothing did before.

    The IP weight of the same request is charged separately by the throttle override inside
    ccxt's ``fetch2``, which is why these endpoints cost ``[("orders", 1)]`` and nothing else.
    """

    @pytest.mark.asyncio
    async def test_submit_acquires_one_order_slot(self, order_limiter, monkeypatch) -> None:
        acquires = _record_acquires(order_limiter, monkeypatch)
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)

        conn.submit_order(_order_request())
        await _drive(conn)

        assert acquires == ["create_order"]
        exchange.create_order.assert_awaited_once()
        assert isinstance(sent[0], OrderAcceptedEvent)

    @pytest.mark.asyncio
    async def test_cancel_acquires_one_order_slot(self, order_limiter, monkeypatch) -> None:
        acquires = _record_acquires(order_limiter, monkeypatch)
        retrying = _write_exchange()
        # one transient failure: an acquire inside _cancel_with_retry's loop would show up twice
        retrying.cancel_order = AsyncMock(
            side_effect=[
                ccxt.NetworkError("connection reset"),
                {"id": "VENUE123", "clientOrderId": "qubx_BTCUSDT_1"},
            ]
        )
        conn, sent, exchange = _make_connector(exchange=retrying, rate_limiter=order_limiter)
        conn.cancel_retry_interval = 0

        conn.cancel_order(_order(venue_order_id="VENUE123"))
        await _drive(conn)

        assert exchange.cancel_order.await_count == 2
        # one slot per cancel *operation*, not per retry inside _cancel_with_retry
        assert acquires == ["cancel_order"]
        assert isinstance(sent[0], OrderCanceledEvent)

    @pytest.mark.asyncio
    async def test_update_acquires_one_order_slot(self, order_limiter, monkeypatch) -> None:
        acquires = _record_acquires(order_limiter, monkeypatch)
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)

        conn.update_order(_order(venue_order_id="VENUE123"), price=102.0)
        await _drive(conn)

        assert acquires == ["edit_order"]
        exchange.edit_order.assert_awaited_once()
        assert isinstance(sent[0], OrderUpdatedEvent)

    @pytest.mark.asyncio
    async def test_submit_gate_timeout_rejects_without_calling_the_venue(self, order_limiter) -> None:
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        order_limiter.report_limit_hit(pool_name="orders", reason="venue 429")
        assert order_limiter.is_gate_closed("orders")

        conn.submit_order(_order_request())
        await _drive(conn)

        exchange.create_order.assert_not_awaited()  # rejected before reaching the venue
        assert len(sent) == 1
        assert isinstance(sent[0], OrderRejectedEvent)
        assert sent[0].code == "RateLimitGateTimeout"
        assert "orders" in sent[0].reason

    @pytest.mark.asyncio
    async def test_cancel_gate_timeout_rejects_without_calling_the_venue(self, order_limiter) -> None:
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        order_limiter.report_limit_hit(pool_name="orders", reason="venue 429")

        conn.cancel_order(_order(venue_order_id="VENUE123"))
        await _drive(conn)

        exchange.cancel_order.assert_not_awaited()
        assert len(sent) == 1
        assert isinstance(sent[0], OrderCancelRejectedEvent)
        assert sent[0].reason.startswith("rate limited: ")
        assert "orders" in sent[0].reason
        # coded like submit/update, so a reject filter keyed on code catches rate-limited cancels
        assert sent[0].code == "RateLimitGateTimeout"

    @pytest.mark.asyncio
    async def test_update_gate_timeout_rejects_without_calling_the_venue(self, order_limiter) -> None:
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        order_limiter.report_limit_hit(pool_name="orders", reason="venue 429")

        conn.update_order(_order(venue_order_id="VENUE123"), price=102.0)
        await _drive(conn)

        exchange.edit_order.assert_not_awaited()
        assert len(sent) == 1
        assert isinstance(sent[0], OrderUpdateRejectedEvent)
        assert sent[0].code == "RateLimitGateTimeout"

    @pytest.mark.asyncio
    async def test_closed_rest_gate_does_not_block_an_order(self, order_limiter, monkeypatch) -> None:
        # A read-path 429 closes the IP gate; orders cost ``[("orders", 1)]`` only, so placement
        # stays open. Pin it — widening that cost list must be a deliberate decision.
        acquires = _record_acquires(order_limiter, monkeypatch)
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        order_limiter.report_limit_hit(pool_name="ccxt_rest", reason="read-path 429")
        assert order_limiter.is_gate_closed("ccxt_rest")

        conn.submit_order(_order_request())
        await _drive(conn)

        assert acquires == ["create_order"]
        exchange.create_order.assert_awaited_once()
        assert isinstance(sent[0], OrderAcceptedEvent)

    @pytest.mark.asyncio
    async def test_no_rate_limiter_still_submits(self) -> None:
        """Negative control: without a limiter the write path is untouched."""
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=None)

        conn.submit_order(_order_request())
        await _drive(conn)

        exchange.create_order.assert_awaited_once()
        assert isinstance(sent[0], OrderAcceptedEvent)

    @pytest.mark.asyncio
    async def test_order_pool_is_actually_drained(self, order_limiter) -> None:
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)

        for _ in range(3):
            conn.submit_order(_order_request())
        await _drive(conn)

        assert exchange.create_order.await_count == 3
        assert len(sent) == 3
        state = await order_limiter.get_pool_state("orders")
        assert state is not None
        assert state["consumed"] == 3.0
        assert state["remaining"] < _ORDERS_CAPACITY  # tokens really left the bucket

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            ccxt.RateLimitExceeded("binance -1015 too many new orders"),
            ccxt.DDoSProtection("EOrder:Rate limit exceeded"),
        ],
        ids=["binance_-1015", "kraken_EOrder"],
    )
    async def test_venue_order_limit_closes_only_the_orders_gate(self, order_limiter, error) -> None:
        """The pool is a proactive model: only the venue's own verdict reveals budget another actor
        on the account spent. ccxt maps that verdict to two sibling types, so both need wiring.
        """
        conn, _sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        exchange.create_order = AsyncMock(side_effect=error)

        conn.submit_order(_order_request())
        await _drive(conn)

        assert order_limiter.is_gate_closed("orders")
        assert not order_limiter.is_gate_closed("ccxt_rest"), "an order-budget breach says nothing about the IP pool"

    @pytest.mark.asyncio
    async def test_venue_order_limit_on_update_closes_the_orders_gate(self, order_limiter) -> None:
        conn, _sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        exchange.edit_order = AsyncMock(side_effect=ccxt.RateLimitExceeded("binance -1015"))

        conn.update_order(_order(venue_order_id="VENUE123"), price=102.0)
        await _drive(conn)

        assert order_limiter.is_gate_closed("orders")
        assert not order_limiter.is_gate_closed("ccxt_rest")

    @pytest.mark.asyncio
    async def test_venue_order_limit_on_cancel_closes_the_orders_gate(self, order_limiter) -> None:
        conn, _sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        exchange.cancel_order = AsyncMock(side_effect=ccxt.DDoSProtection("EOrder:Rate limit exceeded"))
        conn.cancel_timeout = 0  # one attempt, no retry backoff

        conn.cancel_order(_order(venue_order_id="VENUE123"))
        await _drive(conn)

        assert order_limiter.is_gate_closed("orders")
        assert not order_limiter.is_gate_closed("ccxt_rest")

    @pytest.mark.asyncio
    async def test_reports_the_orders_pool_by_name_not_by_endpoint(self, order_limiter, monkeypatch) -> None:
        """``endpoint=`` would close every pool in that endpoint's cost list, not just ``orders``."""
        hits: list[dict] = []
        monkeypatch.setattr(order_limiter, "report_limit_hit", lambda **kw: hits.append(kw))
        conn, _sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        exchange.create_order = AsyncMock(side_effect=ccxt.RateLimitExceeded("binance -1015"))

        conn.submit_order(_order_request())
        await _drive(conn)

        assert len(hits) == 1
        assert hits[0]["pool_name"] == "orders"
        assert "endpoint" not in hits[0]
        assert "-1015" in hits[0]["reason"]

    @pytest.mark.asyncio
    async def test_non_rate_limit_venue_error_leaves_the_gate_open(self, order_limiter) -> None:
        """Negative control: a rejection is not a budget breach."""
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=order_limiter)
        exchange.create_order = AsyncMock(side_effect=ccxt.InsufficientFunds("balance too low"))

        conn.submit_order(_order_request())
        await _drive(conn)

        assert isinstance(sent[0], OrderRejectedEvent)
        assert not order_limiter.is_gate_closed("orders")

    @pytest.mark.asyncio
    async def test_venue_order_limit_without_a_limiter_still_rejects(self) -> None:
        """Negative control: reporting is skipped, the reject still goes out."""
        conn, sent, exchange = _make_connector(exchange=_write_exchange(), rate_limiter=None)
        exchange.create_order = AsyncMock(side_effect=ccxt.RateLimitExceeded("binance -1015"))

        conn.submit_order(_order_request())
        await _drive(conn)

        assert isinstance(sent[0], OrderRejectedEvent)


