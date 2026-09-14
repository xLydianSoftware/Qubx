"""Unit tests for ``CcxtConnector.convert_currency`` — the venue cash operation.

No credentials, no network: the exchange is a real ccxt instance with one hand-written
spot market injected (so ``market()`` / ``amount_to_precision`` / ``price_to_precision``
are the real implementations) and its two async calls replaced by mocks.
"""

import asyncio
import io
from unittest.mock import AsyncMock, Mock

import ccxt
import pytest

from qubx import logger
from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.core.basics import CtrlChannel
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


def _exchange(
    *,
    bid: float = 1.0002,
    ask: float = 1.00021,
    order: dict | None = None,
    markets: list[dict] | None = None,
) -> ccxt.binance:
    """A real ccxt binance with injected markets; only the two venue calls are mocked."""
    ex = ccxt.binance()
    ex.set_markets(markets if markets is not None else [USDC_USDT_MARKET])
    ex.fetch_bids_asks = AsyncMock(return_value={"USDC/USDT": {"symbol": "USDC/USDT", "bid": bid, "ask": ask}})
    ex.create_order = AsyncMock(
        return_value=order
        if order is not None
        else {
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
    )
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

    # _run_sync is synchronous from the caller's POV: drive the coroutine on a throwaway
    # loop so the test needs no real loop thread (same shim the writes tests use).
    def _run_sync(coro, timeout=None):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    conn._run_sync = Mock(side_effect=_run_sync)

    # the post-fill balance refresh goes off-thread via _spawn; capture the coroutine (and
    # close it) instead of scheduling it on a loop this test has no thread for.
    def _spawn(coro):
        coro.close()

    conn._spawn = Mock(side_effect=_spawn)
    return conn, sent, exchange


def test_sell_conversion_submits_ioc_limit_priced_off_the_bid() -> None:
    conn, _sent, exchange = _make_connector()

    result = conn.convert_currency("USDC", "USDT", 6844.38)

    payload = exchange.create_order.await_args.kwargs
    assert payload["symbol"] == "USDC/USDT"
    assert payload["type"] == "limit"
    assert payload["side"] == "sell"
    assert payload["amount"] == 6844.0  # rounded down to the market's 1-unit step
    assert payload["price"] == pytest.approx(0.9992)  # bid 1.0002 less the 10bps default
    assert payload["params"]["timeInForce"] == "IOC"

    assert result.status == "FILLED"
    assert result.filled_from == pytest.approx(6844.0)
    assert result.filled_to == pytest.approx(6845.37)
    assert result.avg_price == pytest.approx(1.0002)
    assert result.venue_order_id == "28457"


def test_buy_conversion_uses_the_inverse_market_and_sizes_in_base() -> None:
    """Only USDC/USDT is listed, so converting USDT->USDC must BUY that market and size
    the order in USDC — the amount asked for is USDT to spend."""
    exchange = _exchange(
        order={
            "id": "77",
            "status": "closed",
            "symbol": "USDC/USDT",
            "side": "buy",
            "amount": 4995.0,
            "filled": 4995.0,
            "remaining": 0.0,
            "average": 1.00031,
            "cost": 4996.55,
        }
    )
    conn, _sent, _ = _make_connector(exchange)

    result = conn.convert_currency("USDT", "USDC", 5000.0)

    payload = exchange.create_order.await_args.kwargs
    assert payload["symbol"] == "USDC/USDT"
    assert payload["side"] == "buy"
    assert payload["price"] == pytest.approx(1.00121)  # ask 1.00021 plus the 10bps default
    # spend at most 5000 USDT at the limit price -> 4993 USDC (rounded down to the step)
    assert payload["amount"] == 4993.0

    assert result.from_currency == "USDT"
    assert result.to_currency == "USDC"
    assert result.filled_from == pytest.approx(4996.55)  # USDT spent
    assert result.filled_to == pytest.approx(4995.0)  # USDC received


def test_partial_fill_is_reported_not_raised() -> None:
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
    conn, _sent, _ = _make_connector(exchange)

    result = conn.convert_currency("USDC", "USDT", 6844.38)

    assert result.status == "PARTIAL"
    assert result.filled_from == pytest.approx(1200.0)
    assert result.filled_to == pytest.approx(1200.24)
    assert exchange.create_order.await_count == 1


def test_zero_fill_is_reported_as_unfilled() -> None:
    exchange = _exchange(
        order={"id": "92", "status": "canceled", "amount": 6844.0, "filled": 0.0, "remaining": 6844.0, "cost": 0.0}
    )
    conn, _sent, _ = _make_connector(exchange)

    result = conn.convert_currency("USDC", "USDT", 6844.38)

    assert result.status == "UNFILLED"
    assert result.filled_from == 0.0
    assert result.filled_to == 0.0
    assert result.avg_price is None


def test_explicit_limit_price_skips_the_book_read() -> None:
    """An absolute bound the caller already knows (0.995 for a stable pair) needs no quote —
    and must be used verbatim rather than blended with the book."""
    conn, _sent, exchange = _make_connector()

    conn.convert_currency("USDC", "USDT", 6844.38, limit_price=0.995)

    exchange.fetch_bids_asks.assert_not_awaited()
    assert exchange.create_order.await_args.kwargs["price"] == pytest.approx(0.995)


def test_unknown_pair_raises_before_submitting() -> None:
    conn, _sent, exchange = _make_connector()

    with pytest.raises(ValueError, match="BTC"):
        conn.convert_currency("BTC", "USDT", 1.0)

    exchange.create_order.assert_not_awaited()
    exchange.fetch_bids_asks.assert_not_awaited()


def test_amount_below_market_min_notional_raises_before_submitting() -> None:
    """The venue would reject it anyway; refusing here keeps the caller's mistake local."""
    conn, _sent, exchange = _make_connector()

    with pytest.raises(ValueError, match="notional"):
        conn.convert_currency("USDC", "USDT", 3.0)  # market floor is 5.0

    exchange.create_order.assert_not_awaited()


def test_non_positive_amount_raises() -> None:
    conn, _sent, exchange = _make_connector()

    with pytest.raises(ValueError, match="positive"):
        conn.convert_currency("USDC", "USDT", 0.0)

    exchange.create_order.assert_not_awaited()


def test_conversion_emits_no_events_and_registers_no_order() -> None:
    """The isolation guarantee: a conversion moves cash, so nothing reaches the channel the
    AccountManager drains — no order, no deal, no position."""
    conn, sent, _exchange = _make_connector()

    conn.convert_currency("USDC", "USDT", 6844.38)

    assert sent == []


def test_filled_conversion_requests_a_fresh_balance_snapshot() -> None:
    """The caller (and its next scheduled tick) must not read the pre-conversion balances —
    otherwise a top-up loop converts twice before the poller catches up."""
    conn, _sent, _exchange = _make_connector()
    conn.request_snapshot = Mock()

    conn.convert_currency("USDC", "USDT", 6844.38)

    conn.request_snapshot.assert_called_once_with(include_orders=False)


def test_unfilled_conversion_requests_no_snapshot() -> None:
    exchange = _exchange(
        order={"id": "92", "status": "canceled", "amount": 6844.0, "filled": 0.0, "remaining": 6844.0, "cost": 0.0}
    )
    conn, _sent, _ = _make_connector(exchange)
    conn.request_snapshot = Mock()

    conn.convert_currency("USDC", "USDT", 6844.38)

    conn.request_snapshot.assert_not_called()


def test_markets_are_loaded_before_the_pair_is_resolved() -> None:
    """A trading-only connector (market data via xdata) may never have loaded markets: the
    conversion must load them rather than report the pair as unlisted."""
    exchange = _exchange()
    exchange.markets = None
    exchange.markets_by_id = None

    async def _load_markets(reload=False, params={}):
        exchange.set_markets([USDC_USDT_MARKET])
        return exchange.markets

    exchange.load_markets = AsyncMock(side_effect=_load_markets)
    conn, _sent, _ = _make_connector(exchange)

    result = conn.convert_currency("USDC", "USDT", 6844.38)

    exchange.load_markets.assert_awaited()
    assert result.status == "FILLED"


def test_conversion_is_logged_with_its_outcome() -> None:
    """Real money moved with no order record behind it: the bot log is the only local trace
    if the caller dies before it reports."""
    conn, _sent, _exchange = _make_connector()

    sink = io.StringIO()
    sink_id = logger.add(sink, level="INFO")
    try:
        conn.convert_currency("USDC", "USDT", 6844.38)
    finally:
        logger.remove(sink_id)

    logged = sink.getvalue()
    assert "6844" in logged and "USDC" in logged and "USDT" in logged and "FILLED" in logged
