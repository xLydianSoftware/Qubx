"""Unit tests for ``CcxtConnector.convert_currency`` — the venue cash operation.

No credentials, no network: the exchange is a real ccxt instance with one hand-written
spot market injected (so ``market()`` / ``amount_to_precision`` / ``price_to_precision``
are the real implementations) and its venue calls replaced by mocks. The connector fires
the conversion on the exchange loop via ``_spawn``; tests capture that coroutine and await
it themselves, so the async work runs without a real thread/loop boundary.
"""

import io
from unittest.mock import AsyncMock, Mock

import ccxt
import pytest

from qubx import logger
from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.core.basics import CtrlChannel
from qubx.core.events import AccountMessage, CurrencyConversionEvent
from tests.qubx.core.utils_test import DummyTimeProvider

USDC_USDT_MARKET = {
    "id": "USDCUSDT",
    "symbol": "USDC/USDT",
    "base": "USDC",
    "quote": "USDT",
    "baseId": "USDC",
    "quoteId": "USDT",
    "type": "spot",
    "spot": True,
    "margin": True,
    "swap": False,
    "future": False,
    "option": False,
    "contract": False,
    "active": True,
    "precision": {"amount": 1.0, "price": 1e-05},
    "limits": {
        "amount": {"min": 1.0, "max": 1e7},
        "cost": {"min": 5.0, "max": 9e6},
        "price": {"min": 0.8, "max": 1.2},
    },
}

FILLED_SELL = {
    "id": "28457",
    "status": "closed",
    "symbol": "USDC/USDT",
    "side": "sell",
    "amount": 6844.0,
    "filled": 6844.0,
    "remaining": 0.0,
    "average": 1.0002,
    "cost": 6845.37,
}


def _exchange(
    *,
    bid: float = 1.0002,
    ask: float = 1.00021,
    order: dict | None = None,
    order_error: Exception | None = None,
) -> ccxt.binance:
    """A real ccxt binance with injected markets; only the venue calls are mocked."""
    ex = ccxt.binance()
    ex.set_markets([USDC_USDT_MARKET])
    ex.fetch_bids_asks = AsyncMock(return_value={"USDC/USDT": {"symbol": "USDC/USDT", "bid": bid, "ask": ask}})
    if order_error is not None:
        ex.create_order = AsyncMock(side_effect=order_error)
    else:
        ex.create_order = AsyncMock(return_value=order if order is not None else FILLED_SELL)
    return ex


def _make_connector(exchange: ccxt.binance | None = None) -> tuple[CcxtConnector, list, ccxt.binance]:
    exchange = exchange if exchange is not None else _exchange()

    em = Mock()
    em.exchange = exchange
    em.rate_limiter = None

    sent: list = []
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock(side_effect=lambda e: sent.append(e))

    conn = CcxtConnector(
        exchange_name="BINANCE.PM",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=Mock(),
    )

    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))
    conn._captured = captured  # type: ignore[attr-defined]
    # the post-fill balance refresh is asserted on directly; stubbing it keeps _drive to the
    # conversion's own coroutine
    conn.request_snapshot = Mock()
    return conn, sent, exchange


async def _drive(conn: CcxtConnector) -> None:
    for coro in conn._captured:  # type: ignore[attr-defined]
        await coro
    conn._captured.clear()  # type: ignore[attr-defined]


def _record(sent: list):
    [event] = [e for e in sent if isinstance(e, CurrencyConversionEvent)]
    return event.conversion


# --------------------------------------------------------------------------- #
# the caller is never blocked
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_call_returns_an_id_without_touching_the_venue() -> None:
    """The ProcessorThread must keep draining its queue: the venue round trip happens on the
    exchange loop and the outcome comes back as an event."""
    conn, sent, exchange = _make_connector()

    conversion_id = conn.convert_currency("USDC", "USDT", 6844.38)

    assert conversion_id
    exchange.create_order.assert_not_awaited()
    exchange.fetch_bids_asks.assert_not_awaited()
    assert sent == []

    await _drive(conn)

    assert _record(sent).conversion_id == conversion_id


@pytest.mark.asyncio
async def test_conversion_event_is_not_an_account_message() -> None:
    """AccountManager.apply() is typed to accept only AccountMessage — staying off that marker
    is what keeps a conversion from becoming an order and a position."""
    conn, sent, _ = _make_connector()

    conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    [event] = sent
    assert isinstance(event, CurrencyConversionEvent)
    assert not isinstance(event, AccountMessage)
    assert event.instrument is None


# --------------------------------------------------------------------------- #
# order construction
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_sell_conversion_submits_ioc_limit_priced_off_the_bid() -> None:
    conn, sent, exchange = _make_connector()

    conversion_id = conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    payload = exchange.create_order.await_args.kwargs
    assert payload["symbol"] == "USDC/USDT"
    assert payload["type"] == "limit"
    assert payload["side"] == "sell"
    assert payload["amount"] == 6844.0  # rounded down to the market's 1-unit step
    assert payload["price"] == pytest.approx(0.9992)  # bid 1.0002 less the 10bps default
    assert payload["params"]["timeInForce"] == "IOC"
    assert payload["params"]["clientOrderId"] == conversion_id  # traceable in venue history

    record = _record(sent)
    assert record.status == "FILLED"
    assert record.filled_from == pytest.approx(6844.0)
    assert record.filled_to == pytest.approx(6845.37)
    assert record.avg_price == pytest.approx(1.0002)
    assert record.venue_order_id == "28457"


@pytest.mark.asyncio
async def test_buy_conversion_uses_the_inverse_market_and_sizes_in_base() -> None:
    """Only USDC/USDT is listed, so converting USDT->USDC must BUY that market and size the
    order in USDC — the amount asked for is USDT to spend."""
    exchange = _exchange(
        order={
            "id": "77",
            "status": "closed",
            "side": "buy",
            "amount": 4993.0,
            "filled": 4993.0,
            "remaining": 0.0,
            "average": 1.00031,
            "cost": 4994.55,
        }
    )
    conn, sent, _ = _make_connector(exchange)

    conn.convert_currency("USDT", "USDC", 5000.0)
    await _drive(conn)

    payload = exchange.create_order.await_args.kwargs
    assert payload["symbol"] == "USDC/USDT"
    assert payload["side"] == "buy"
    assert payload["price"] == pytest.approx(1.00121)  # ask 1.00021 plus the 10bps default
    # spend at most 5000 USDT at the limit price -> 4993 USDC (rounded down to the step)
    assert payload["amount"] == 4993.0

    record = _record(sent)
    assert record.from_currency == "USDT"
    assert record.to_currency == "USDC"
    assert record.filled_from == pytest.approx(4994.55)  # USDT spent
    assert record.filled_to == pytest.approx(4993.0)  # USDC received


@pytest.mark.asyncio
async def test_explicit_limit_price_skips_the_book_read() -> None:
    """An absolute bound the caller already knows (0.995 for a stable pair) needs no quote —
    and must be used verbatim rather than blended with the book."""
    conn, _sent, exchange = _make_connector()

    conn.convert_currency("USDC", "USDT", 6844.38, limit_price=0.995)
    await _drive(conn)

    exchange.fetch_bids_asks.assert_not_awaited()
    assert exchange.create_order.await_args.kwargs["price"] == pytest.approx(0.995)


@pytest.mark.asyncio
async def test_markets_are_loaded_before_the_pair_is_resolved() -> None:
    """A trading-only connector (market data via xdata) may never have loaded markets: the
    conversion must load them rather than report the pair as unlisted."""
    exchange = _exchange()
    exchange.markets = None
    exchange.markets_by_id = None

    async def _load_markets(reload=False, params={}):
        exchange.set_markets([USDC_USDT_MARKET])
        return exchange.markets

    exchange.load_markets = AsyncMock(side_effect=_load_markets)
    conn, sent, _ = _make_connector(exchange)

    conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    exchange.load_markets.assert_awaited()
    assert _record(sent).status == "FILLED"


# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_partial_fill_is_reported_not_retried() -> None:
    """IOC leaves nothing resting, so a short fill is a result the caller acts on — the
    connector never retries it."""
    exchange = _exchange(
        order={
            "id": "91",
            "status": "canceled",  # IOC: the unfilled remainder is killed
            "amount": 6844.0,
            "filled": 1200.0,
            "remaining": 5644.0,
            "average": 1.0002,
            "cost": 1200.24,
        }
    )
    conn, sent, _ = _make_connector(exchange)

    conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    record = _record(sent)
    assert record.status == "PARTIAL"
    assert record.filled_from == pytest.approx(1200.0)
    assert record.filled_to == pytest.approx(1200.24)
    assert exchange.create_order.await_count == 1


@pytest.mark.asyncio
async def test_zero_fill_is_reported_as_unfilled() -> None:
    exchange = _exchange(
        order={"id": "92", "status": "canceled", "amount": 6844.0, "filled": 0.0, "remaining": 6844.0, "cost": 0.0}
    )
    conn, sent, _ = _make_connector(exchange)

    conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    record = _record(sent)
    assert record.status == "UNFILLED"
    assert record.filled_from == 0.0
    assert record.filled_to == 0.0
    assert record.avg_price is None


@pytest.mark.asyncio
async def test_venue_refusal_arrives_as_a_failed_record() -> None:
    """Nothing waits on a return value any more, so a refusal must reach the caller as the
    same event a fill would — never as a dead-lettered exception."""
    conn, sent, _ = _make_connector(_exchange(order_error=ccxt.InsufficientFunds("balance too low")))

    conversion_id = conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    record = _record(sent)
    assert record.conversion_id == conversion_id
    assert record.status == "FAILED"
    assert record.filled_from == 0.0
    assert "balance too low" in (record.failure_reason or "")


@pytest.mark.asyncio
async def test_unknown_pair_arrives_as_a_failed_record() -> None:
    conn, sent, exchange = _make_connector()

    conn.convert_currency("BTC", "USDT", 1.0)
    await _drive(conn)

    record = _record(sent)
    assert record.status == "FAILED"
    assert "BTC" in (record.failure_reason or "")
    exchange.create_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_amount_below_the_market_floor_arrives_as_a_failed_record() -> None:
    conn, sent, exchange = _make_connector()

    conn.convert_currency("USDC", "USDT", 3.0)  # the market's notional floor is 5.0
    await _drive(conn)

    record = _record(sent)
    assert record.status == "FAILED"
    assert "notional" in (record.failure_reason or "")
    exchange.create_order.assert_not_awaited()


# --------------------------------------------------------------------------- #
# argument rejections stay synchronous (the framework's rejection boundary)
# --------------------------------------------------------------------------- #
def test_non_positive_amount_raises_without_spawning() -> None:
    conn, sent, _ = _make_connector()

    with pytest.raises(ValueError, match="positive"):
        conn.convert_currency("USDC", "USDT", 0.0)

    assert conn._captured == []  # type: ignore[attr-defined]
    assert sent == []


def test_converting_a_currency_into_itself_raises() -> None:
    conn, _sent, _ = _make_connector()

    with pytest.raises(ValueError, match="USDT"):
        conn.convert_currency("USDT", "USDT", 100.0)

    assert conn._captured == []  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# side effects
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_filled_conversion_requests_a_fresh_balance_snapshot() -> None:
    """The caller's next tick must not read the pre-conversion balances — otherwise a top-up
    loop converts twice before the poller catches up."""
    conn, _sent, _ = _make_connector()

    conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    conn.request_snapshot.assert_called_once_with(include_orders=False)


@pytest.mark.asyncio
async def test_unfilled_conversion_requests_no_snapshot() -> None:
    exchange = _exchange(
        order={"id": "92", "status": "canceled", "amount": 6844.0, "filled": 0.0, "remaining": 6844.0, "cost": 0.0}
    )
    conn, _sent, _ = _make_connector(exchange)

    conn.convert_currency("USDC", "USDT", 6844.38)
    await _drive(conn)

    conn.request_snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_conversion_is_logged_with_its_outcome() -> None:
    """Real money moved with no order record behind it: the bot log is the only local trace
    if the strategy ignores the callback."""
    conn, _sent, _ = _make_connector()

    sink = io.StringIO()
    sink_id = logger.add(sink, level="INFO")
    try:
        conn.convert_currency("USDC", "USDT", 6844.38)
        await _drive(conn)
    finally:
        logger.remove(sink_id)

    logged = sink.getvalue()
    assert "6844" in logged and "USDC" in logged and "USDT" in logged and "FILLED" in logged
