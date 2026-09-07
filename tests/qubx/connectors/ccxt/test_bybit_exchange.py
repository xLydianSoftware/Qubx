"""Bybit wire-dialect tests: the real ccxt request builders driven against an injected market.

The ``upstream_*`` tests are canaries — they fail once ccxt fixes a defect an override works around.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, Mock, patch

import ccxt
import ccxt.pro as cxp
import numpy as np
import pytest
from ccxt.base.errors import ArgumentsRequired, BadRequest, NotSupported, OrderNotFound

from qubx.connectors.ccxt.exchanges import CUSTOM_CONNECTORS, EXCHANGE_ALIASES, BybitF
from qubx.connectors.ccxt.exchanges.bybit.connector import BybitCcxtConnector
from qubx.connectors.ccxt.utils import ccxt_convert_funding_rate, prepare_ccxt_order_payload
from qubx.core.basics import Quote
from qubx.core.lookups import lookup


def _swap_market() -> dict:
    return {
        "id": "BTCUSDT",
        "symbol": "BTC/USDT:USDT",
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "baseId": "BTC",
        "quoteId": "USDT",
        "settleId": "USDT",
        "type": "swap",
        "spot": False,
        "margin": False,
        "swap": True,
        "future": False,
        "option": False,
        "active": True,
        "contract": True,
        "linear": True,
        "inverse": False,
        "contractSize": 1.0,
        "precision": {"amount": 0.001, "price": 0.1},
        "limits": {"amount": {"min": 0.001}, "price": {}, "cost": {}},
        "info": {},
    }


@pytest.fixture
def bybit() -> BybitF:
    ex = BybitF()
    ex.set_markets([_swap_market()])
    return ex


def test_registration():
    assert EXCHANGE_ALIASES["bybit.f"] == "bybit_f"
    assert cxp.bybit_f is BybitF
    assert "bybit_f" in cxp.exchanges
    assert CUSTOM_CONNECTORS["bybit.f"] is BybitCcxtConnector
    assert CUSTOM_CONNECTORS["bybit"] is BybitCcxtConnector


def test_id_is_unchanged_so_rate_limits_and_sandbox_still_resolve(bybit):
    assert bybit.id == "bybit"


def test_uta_fetch_order_is_acknowledged(bybit):
    assert bybit.describe()["options"]["fetchOrder"]["acknowledged"] is True


def test_cancel_by_client_id_sends_order_link_id(bybit):
    request = bybit.cancel_order_request("", "BTC/USDT:USDT", {"clientOrderId": "qubx_x_1"})
    assert request["orderLinkId"] == "qubx_x_1"
    assert "orderId" not in request
    assert "clientOrderId" not in request
    assert request["category"] == "linear"


def test_cancel_by_native_order_link_id_spelling(bybit):
    request = bybit.cancel_order_request("", "BTC/USDT:USDT", {"orderLinkId": "qubx_x_1"})
    assert request["orderLinkId"] == "qubx_x_1"
    assert "orderId" not in request


def test_cancel_by_venue_id_is_untouched(bybit):
    request = bybit.cancel_order_request("V1", "BTC/USDT:USDT", {})
    assert request["orderId"] == "V1"
    assert "orderLinkId" not in request


def test_upstream_still_has_the_defect_the_override_fixes():
    """If this fails, ccxt fixed it upstream and the cancel override can be dropped."""
    ex = cxp.bybit()
    ex.set_markets([_swap_market()])
    request = ex.cancel_order_request("", "BTC/USDT:USDT", {"clientOrderId": "c"})
    assert request["orderId"] == ""
    assert "orderLinkId" not in request


def test_post_only_refusal_is_reported_as_rejected(bybit):
    """Bybit accepts a crossing post-only order then cancels it; normalised to rejected."""
    raw = {
        "orderId": "X1",
        "orderLinkId": "q1",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderStatus": "Cancelled",
        "rejectReason": "EC_PostOnlyWillTakeLiquidity",
        "qty": "0.01",
        "cumExecQty": "0",
        "price": "100",
        "createdTime": "1",
        "updatedTime": "2",
    }
    market = bybit.market("BTC/USDT:USDT")
    assert bybit.parse_order(dict(raw), market)["status"] == "rejected"
    # upstream canary: if this ever reports "rejected" too, the override can be dropped
    upstream = cxp.bybit()
    upstream.set_markets([_swap_market()])
    assert upstream.parse_order(dict(raw), market)["status"] == "canceled"


def test_an_ordinary_cancel_is_still_a_cancel(bybit):
    raw = {
        "orderId": "X2",
        "orderLinkId": "q2",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderStatus": "Cancelled",
        "rejectReason": "EC_NoError",
        "qty": "0.01",
        "cumExecQty": "0",
        "price": "100",
        "createdTime": "1",
        "updatedTime": "2",
    }
    assert bybit.parse_order(raw, bybit.market("BTC/USDT:USDT"))["status"] == "canceled"


def test_order_with_no_reject_reason_is_unaffected(bybit):
    raw = {
        "orderId": "X3",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderStatus": "Cancelled",
        "qty": "0.01",
        "cumExecQty": "0",
        "price": "100",
        "createdTime": "1",
        "updatedTime": "2",
    }
    assert bybit.parse_order(raw, bybit.market("BTC/USDT:USDT"))["status"] == "canceled"


@pytest.mark.asyncio
async def test_fetch_order_by_cid_returns_only_a_matching_row(bybit):
    """A row that does not carry the requested cid must not be applied to it."""
    calls = []

    async def _open(symbol=None, since=None, limit=None, params={}):
        calls.append(params.get("orderLinkId"))
        return [{"clientOrderId": "someone-else", "id": "Z9"}]

    async def _closed(symbol=None, since=None, limit=None, params={}):
        return [{"clientOrderId": "wanted", "id": "OK"}]

    bybit.fetch_open_orders = _open
    bybit.fetch_canceled_and_closed_orders = _closed

    row = await bybit.fetch_order("", "BTC/USDT:USDT", {"clientOrderId": "wanted"})
    assert row["id"] == "OK"
    assert calls == ["wanted"]


def test_fetch_margin_mode_is_the_read_surface(bybit):
    """The connector's get_margin_mode override depends on it."""
    assert bybit.has["fetchMarginMode"] is True


def test_parse_position_still_nulls_margin_mode(bybit):
    """Bybit deprecated per-position tradeMode on a UTA."""
    row = bybit.parse_position(
        {"symbol": "BTCUSDT", "side": "Buy", "size": "0", "positionIdx": 0},
        bybit.market("BTC/USDT:USDT"),
    )
    assert row["marginMode"] is None


@pytest.mark.parametrize(
    "venue_value,expected",
    [("isolated", "isolated"), ("cross", "cross"), ("portfolio", None), (None, None)],
)
def test_get_margin_mode_reads_account_info(venue_value, expected):
    conn = object.__new__(BybitCcxtConnector)
    conn.exchange_name = "BYBIT.F"
    conn._em = Mock()
    conn._em.exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": venue_value})
    conn._run_sync = lambda coro, timeout=None: asyncio.new_event_loop().run_until_complete(coro)

    instrument = lookup.find_symbol("BYBIT.F", "ETHUSDT")
    assert conn.get_margin_mode(instrument) == expected
    conn._em.exchange.fetch_margin_mode.assert_awaited_once_with("ETH/USDT:USDT")


def test_get_margin_mode_survives_a_venue_error():
    conn = object.__new__(BybitCcxtConnector)
    conn.exchange_name = "BYBIT.F"
    conn._em = Mock()
    conn._em.exchange.fetch_margin_mode = AsyncMock(side_effect=RuntimeError("boom"))
    conn._run_sync = lambda coro, timeout=None: asyncio.new_event_loop().run_until_complete(coro)

    assert conn.get_margin_mode(lookup.find_symbol("BYBIT.F", "ETHUSDT")) is None


class _StubWsClient:
    """The slice of ccxt's ws Client that handle_order_book touches."""

    def __init__(self, url: str = "wss://stream.bybit.com/v5/public/linear"):
        self.url = url
        self.resolved: list[tuple] = []

    def resolve(self, message, message_hash):
        self.resolved.append((message, message_hash))


def _book_message(topic: str, levels: int) -> dict:
    return {
        "topic": topic,
        "type": "snapshot",
        "ts": 1673272861686,
        "data": {
            "s": "BTCUSDT",
            "b": [[str(100 - i), "1"] for i in range(levels)],
            "a": [[str(101 + i), "1"] for i in range(levels)],
        },
    }


def _upstream() -> cxp.bybit:
    ex = cxp.bybit()
    ex.set_markets([_swap_market()])
    return ex


@pytest.mark.parametrize("side,direction", [("buy", 1), ("sell", 2)])
def test_stop_market_carries_the_trigger_direction(bybit, side, direction):
    """1 = fire on a rise, 2 = on a fall. A BUY stop rests above the market, a SELL below."""
    request = bybit.create_order_request("BTC/USDT:USDT", "market", side, 0.01, 100.0, {"triggerPrice": 105.0})
    assert request["triggerDirection"] == direction
    assert request["triggerPrice"] == "105"
    assert request["orderType"] == "Market"
    assert "timeInForce" not in request


@pytest.mark.parametrize("side,direction", [("buy", 1), ("sell", 2)])
def test_stop_limit_carries_the_trigger_direction(bybit, side, direction):
    request = bybit.create_order_request(
        "BTC/USDT:USDT", "limit", side, 0.01, 100.0, {"triggerPrice": 105.0, "timeInForce": "GTC"}
    )
    assert request["triggerDirection"] == direction
    assert request["price"] == "100"


def test_the_stop_price_spelling_is_recognised_too(bybit):
    request = bybit.create_order_request("BTC/USDT:USDT", "market", "sell", 0.01, 100.0, {"stopPrice": 95.0})
    assert request["triggerDirection"] == 2


def test_an_explicit_trigger_direction_is_left_alone(bybit):
    """A strategy can override the derivation through OrderRequest.options."""
    request = bybit.create_order_request(
        "BTC/USDT:USDT", "market", "buy", 0.01, 100.0, {"triggerPrice": 95.0, "triggerDirection": "descending"}
    )
    assert request["triggerDirection"] == 2


def test_the_callers_params_are_not_mutated(bybit):
    """The connector reuses payload["params"]; injecting in place would leak across retries."""
    params = {"triggerPrice": 105.0}
    bybit.create_order_request("BTC/USDT:USDT", "market", "buy", 0.01, 100.0, params)
    assert params == {"triggerPrice": 105.0}


@pytest.mark.parametrize(
    "order_type,params",
    [("limit", {"timeInForce": "GTC"}), ("market", {}), ("limit", {"postOnly": True})],
)
def test_a_non_trigger_request_is_byte_identical_to_upstream(bybit, order_type, params):
    ours = bybit.create_order_request("BTC/USDT:USDT", order_type, "buy", 0.01, 100.0, dict(params))
    theirs = _upstream().create_order_request("BTC/USDT:USDT", order_type, "buy", 0.01, 100.0, dict(params))
    assert ours == theirs


def test_create_order_request_still_takes_seven_positionals(bybit):
    """ccxt's create_orders and pro's create_order_ws both pass isUTA positionally."""
    request = bybit.create_order_request("BTC/USDT:USDT", "market", "buy", 0.01, 100.0, {"triggerPrice": 105.0}, True)
    assert request["triggerDirection"] == 1


def test_upstream_refuses_every_contract_trigger_order():
    """Upstream requires triggerDirection and never derives it."""
    with pytest.raises(ArgumentsRequired, match="triggerDirection"):
        _upstream().create_order_request("BTC/USDT:USDT", "market", "buy", 0.01, 100.0, {"triggerPrice": 105.0})


@pytest.mark.parametrize("side,direction", [("BUY", 1), ("SELL", 2)])
def test_a_riskctrl_shaped_stop_reaches_the_wire(bybit, side, direction):
    """The real payload builder's output, straight into the real request builder."""
    instrument = lookup.find_symbol("BYBIT.F", "BTCUSDT")
    assert instrument is not None
    payload = prepare_ccxt_order_payload(
        instrument=instrument,
        order_side=side,
        order_type="STOP_MARKET",
        amount=0.01,
        price=105000.0,
        client_id="qubx_stop_1",
        time_in_force="gtc",
        quote=Quote(0, 100000.0, 100001.0, 1.0, 1.0),
        reduce_only=False,
    )
    request = bybit.create_order_request(**payload)
    assert request["triggerDirection"] == direction
    assert request["triggerPrice"] == "105000"
    assert request["orderType"] == "Market"
    assert request["orderLinkId"] == "qubx_stop_1"
    # a market-typed order carries neither, and Bybit rejects both on one
    assert "timeInForce" not in request
    assert "postOnly" not in request


@pytest.mark.parametrize("method", ["fetchOpenOrders", "fetchPositions"])
def test_the_snapshot_fetches_have_an_explicit_pagination_cap(bybit, method):
    """Read through ccxt's own lookup, not the options dict, so a rename upstream fails here."""
    assert bybit.handle_option_and_params({}, method, "paginationCalls", 10) == [20, {}]


@pytest.mark.parametrize("method", ["fetchOpenOrders", "fetchPositions"])
def test_paginate_itself_is_never_declared_in_options(bybit, method):
    """paginate must ride params: handle_option_and_params drops the key only when it finds it
    there, so an options-driven paginate makes fetch_paginated_call_cursor recurse forever."""
    assert bybit.handle_option_and_params({}, method, "paginate") == [None, {}]


@pytest.mark.asyncio
async def test_cursor_pagination_stops_at_the_declared_cap():
    """A venue cursor that never ends must not page forever. Real ccxt, stubbed transport."""
    exchange = BybitF({"enableRateLimit": False})
    exchange.options["defaultType"] = "swap"
    calls: list[dict] = []

    async def _load_markets(reload=False, params={}):
        exchange.markets, exchange.markets_by_id = {}, {}
        return {}

    async def _endpoint(params={}):
        calls.append(dict(params))
        page = len(calls)
        return {
            "retCode": 0,
            "result": {
                "category": "linear",
                "nextPageCursor": f"cur{page}",
                "list": [
                    {
                        "orderId": f"V{page}_{i}",
                        "orderLinkId": f"c{page}_{i}",
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "orderType": "Limit",
                        "price": "100",
                        "qty": "1",
                        "cumExecQty": "0",
                        "orderStatus": "New",
                        "createdTime": "1700000000000",
                        "updatedTime": "1700000000000",
                        "nextPageCursor": f"cur{page}",
                    }
                    for i in range(50)
                ],
            },
        }

    exchange.load_markets = _load_markets
    exchange.privateGetV5OrderRealtime = _endpoint
    try:
        orders = await exchange.fetch_open_orders(params={"paginate": True})
    finally:
        await exchange.close()

    assert len(calls) == 20
    assert len(orders) == 1000
    # the seam's key is consumed by the first frame and never reaches the venue
    assert all("paginate" not in p for p in calls)
    assert all("paginationCalls" not in p for p in calls)


@pytest.mark.parametrize(
    "position_side,order_side",
    # Bybit's S is the POSITION side; Binance reports the liquidating ORDER side, so it is inverted
    [("Buy", "sell"), ("Sell", "buy")],
)
def test_all_liquidation_side_is_inverted_to_the_order_side(bybit, position_side, order_side):
    row = {"T": 1739502302929, "s": "BTCUSDT", "S": position_side, "v": "20000", "p": "0.04499"}
    parsed = bybit.parse_ws_liquidation(dict(row))
    assert parsed["side"] == order_side
    assert parsed["symbol"] == "BTC/USDT:USDT"
    assert parsed["contracts"] == 20000.0
    assert parsed["price"] == 0.04499
    assert parsed["timestamp"] == 1739502302929


def test_upstream_reads_the_all_liquidation_side_as_a_literal_s():
    """ccxt passes 'S' as safe_string_lower's DEFAULT argument, not as a second key."""
    row = {"T": 1739502302929, "s": "BTCUSDT", "S": "Sell", "v": "20000", "p": "0.04499"}
    assert _upstream().parse_ws_liquidation(row)["side"] == "s"


def test_the_single_symbol_liquidation_row_is_also_inverted(bybit):
    """The inversion is about Bybit's field meaning, so it applies to both topics."""
    row = {"price": "0.03803", "side": "Buy", "size": "1637", "symbol": "BTCUSDT", "updatedTime": 1673251091822}
    assert _upstream().parse_ws_liquidation(dict(row))["side"] == "buy"
    assert bybit.parse_ws_liquidation(dict(row))["side"] == "sell"


@pytest.mark.asyncio
async def test_quotes_come_off_the_tickers_topic(bybit):
    seen: dict = {}

    async def _watch_tickers(symbols=None, params={}):
        seen["symbols"] = symbols
        return {
            "BTC/USDT:USDT": {"symbol": "BTC/USDT:USDT", "bid": 100.0, "ask": 101.0},
            "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "bid": None, "ask": 3.0},
        }

    bybit.watch_tickers = _watch_tickers
    quotes = await bybit.watch_bids_asks(["BTC/USDT:USDT", "ETH/USDT:USDT"])

    assert seen["symbols"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    # keyed by symbol, as QuoteDataHandler iterates it; a half-populated ticker is dropped
    assert list(quotes) == ["BTC/USDT:USDT"]
    # and nothing was written into the book cache the L2 stream owns
    assert bybit.orderbooks == {}


@pytest.mark.asyncio
async def test_un_watch_bids_asks_releases_the_tickers_topic(bybit):
    seen: dict = {}

    async def _un_watch_tickers(symbols=None, params={}):
        seen["symbols"] = symbols
        return "released"

    bybit.un_watch_tickers = _un_watch_tickers
    assert await bybit.un_watch_bids_asks(["BTC/USDT:USDT"]) == "released"
    assert seen["symbols"] == ["BTC/USDT:USDT"]
    assert bybit.has["unWatchBidsAsks"] is True


@pytest.mark.asyncio
async def test_upstream_has_no_unwatch_for_bids_asks():
    """QuoteDataHandler branches on hasattr, and the base class stub raises."""
    upstream = _upstream()
    assert upstream.has.get("unWatchBidsAsks") is not True
    with pytest.raises(NotSupported):
        await upstream.un_watch_bids_asks(["BTC/USDT:USDT"])


def test_upstreams_bidask_stream_collapses_the_l2_book():
    """orderbook.1 writes into the same cache the depth-50 subscription fills, resolving the
    L2 waiter from the collapsed book."""
    upstream = _upstream()
    client = _StubWsClient()
    upstream.handle_order_book(client, _book_message("orderbook.50.BTCUSDT", 50))
    assert len(upstream.orderbooks["BTC/USDT:USDT"]["bids"]) == 50

    upstream.handle_order_book(client, _book_message("orderbook.1.BTCUSDT", 1))
    assert len(upstream.orderbooks["BTC/USDT:USDT"]["bids"]) == 1
    # the L2 waiter is resolved with the collapsed book before the limit=='1' branch runs
    assert client.resolved[-2][1] == "orderbook:BTC/USDT:USDT"
    assert len(client.resolved[-2][0]["bids"]) == 1


@pytest.mark.parametrize(
    "requested,depth", [(None, 50), (1, 1), (25, 50), (50, 50), (100, 200), (200, 200), (500, 1000), (2000, 1000)]
)
def test_ws_depth_rounds_up_to_a_depth_bybit_serves(bybit, requested, depth):
    assert bybit._ws_depth(requested) == depth


@pytest.mark.asyncio
async def test_the_book_subscription_uses_a_legal_depth(bybit):
    seen: dict = {}

    async def _watch_topics(url, hashes, topics, params):
        seen["topics"] = topics
        return bybit.order_book({})

    bybit.watch_topics = _watch_topics
    await bybit.watch_order_book_for_symbols(["BTC/USDT:USDT"], 100)
    assert seen["topics"] == ["orderbook.200.BTCUSDT"]


@pytest.mark.asyncio
async def test_upstream_rejects_a_binance_shaped_depth():
    """orderbook_limit: 100 is a hard BadRequest at subscribe, then retried forever."""
    upstream = _upstream()
    with pytest.raises(BadRequest):
        await upstream.watch_order_book_for_symbols(["BTC/USDT:USDT"], 100)
    await upstream.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,topic", [(None, "orderbook.50.BTCUSDT"), (100, "orderbook.200.BTCUSDT")])
async def test_the_unsubscribe_names_the_topic_that_was_subscribed(bybit, requested, topic):
    seen: dict = {}

    async def _un_watch_topics(url, topic_name, symbols, hashes, sub_hashes, topics, params):
        seen["topics"] = topics
        seen["params"] = params
        return "released"

    bybit.un_watch_topics = _un_watch_topics
    await bybit.un_watch_order_book_for_symbols(["BTC/USDT:USDT"], limit=requested)
    assert seen["topics"] == [topic]
    assert "limit" not in seen["params"]


@pytest.mark.asyncio
async def test_the_single_symbol_unsubscribe_forwards_the_depth(bybit):
    """Upstream's un_watch_order_book takes params as its second positional, which here is limit."""
    seen: dict = {}

    async def _un_watch_topics(url, topic_name, symbols, hashes, sub_hashes, topics, params):
        seen["topics"] = topics
        return "released"

    bybit.un_watch_topics = _un_watch_topics
    await bybit.un_watch_order_book("BTC/USDT:USDT", 100)
    assert seen["topics"] == ["orderbook.200.BTCUSDT"]


@pytest.mark.asyncio
async def test_upstream_unsubscribes_a_depth_it_never_subscribed():
    seen: dict = {}
    upstream = _upstream()

    async def _un_watch_topics(url, topic_name, symbols, hashes, sub_hashes, topics, params):
        seen["topics"] = topics

    upstream.un_watch_topics = _un_watch_topics
    await upstream.un_watch_order_book_for_symbols(["BTC/USDT:USDT"])
    # 500 is not even a depth Bybit lets you subscribe to
    assert seen["topics"] == ["orderbook.500.BTCUSDT"]
    await upstream.close()


@pytest.mark.parametrize("method", ["un_watch_order_book_for_symbols", "un_watch_order_book"])
def test_the_unsubscribe_exposes_limit_to_the_handler(bybit, method):
    """OrderBookDataHandler probes inspect.signature for `limit` before passing a depth."""
    assert "limit" in inspect.signature(getattr(bybit, method)).parameters


def _eth_market() -> dict:
    return {**_swap_market(), "id": "ETHUSDT", "symbol": "ETH/USDT:USDT", "base": "ETH", "baseId": "ETH"}


def _funding_exchange(interval_minutes: int = 480) -> BybitF:
    """A BybitF whose markets carry the fundingInterval ccxt reads the interval off."""
    ex = BybitF({"enableRateLimit": False})
    ex.set_markets([{**m, "info": {"fundingInterval": interval_minutes}} for m in (_swap_market(), _eth_market())])
    return ex


def _tickers_response(next_funding_time: str = "1700000000000") -> dict:
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "linear",
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lastPrice": "60000",
                    "markPrice": "60002",
                    "indexPrice": "60001",
                    "fundingRate": "0.0001",
                    "nextFundingTime": next_funding_time,
                }
            ],
        },
        "time": 1699999999000,
    }


def test_funding_rates_are_declared_as_watchable(bybit):
    """FundingRateDataHandler awaits watch_funding_rates unconditionally."""
    assert bybit.has["watchFundingRates"] is True


@pytest.mark.asyncio
async def test_upstream_has_no_funding_stream_at_all():
    """NotSupported is an ExchangeError, so the connection manager re-awaits it at 1 Hz forever."""
    upstream = _upstream()
    assert upstream.has.get("watchFundingRates") is not True
    with pytest.raises(NotSupported):
        await upstream.watch_funding_rates(["BTC/USDT:USDT"])
    await upstream.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interval_minutes,interval", [(480, "8h"), (240, "4h"), (60, "1h")])
async def test_watch_funding_rates_emits_a_convertible_rate(interval_minutes, interval):
    """The framework converter reads nextFundingTime; ccxt's bybit parser writes fundingTimestamp."""
    ex = _funding_exchange(interval_minutes)
    ex.publicGetV5MarketTickers = AsyncMock(return_value=_tickers_response())
    try:
        rates = await ex.watch_funding_rates(["BTC/USDT:USDT"])
    finally:
        await ex.close()

    info = rates["BTC/USDT:USDT"]
    assert info["nextFundingTime"] == 1700000000000
    rate = ccxt_convert_funding_rate(info)
    assert rate.rate == 0.0001
    assert rate.interval == interval
    assert rate.next_funding_time == np.datetime64("2023-11-14T22:13:20", "ns")
    assert rate.time == np.datetime64("2023-11-14T22:13:19", "ns")
    assert rate.mark_price == 60002.0


@pytest.mark.asyncio
async def test_the_upstream_rate_is_unusable_without_the_mapping():
    """ccxt's own row has no nextFundingTime key."""
    ex = _funding_exchange()
    ex.publicGetV5MarketTickers = AsyncMock(return_value=_tickers_response())
    try:
        raw = await ex.fetch_funding_rates(["BTC/USDT:USDT"])
    finally:
        await ex.close()
    assert "nextFundingTime" not in raw["BTC/USDT:USDT"]
    with pytest.raises(KeyError):
        ccxt_convert_funding_rate(raw["BTC/USDT:USDT"])


@pytest.mark.asyncio
async def test_the_all_symbols_subscription_polls_the_whole_universe():
    """The poller polls __all__ with []; bybit's fetch does market(symbols[0]) on that, so an
    empty list must map back to None."""
    ex = _funding_exchange()
    seen: list = []

    async def _fetch(symbols=None, params={}):
        seen.append(symbols)
        return {}

    ex.fetch_funding_rates = _fetch
    try:
        assert await ex.watch_funding_rates(None) == {}
    finally:
        await ex.close()
    assert seen == [None]


@pytest.mark.asyncio
async def test_un_watch_funding_rates_takes_no_arguments():
    """The handler's cleanup calls it bare — a required symbols argument would raise there."""
    ex = _funding_exchange()
    ex.publicGetV5MarketTickers = AsyncMock(return_value=_tickers_response())
    try:
        await ex.watch_funding_rates(["BTC/USDT:USDT"])
        assert ex._funding_rate_adapter is not None
        await ex.un_watch_funding_rates()
    finally:
        await ex.close()
    assert ex._funding_rate_adapter is None


@pytest.mark.asyncio
async def test_un_watch_funding_rates_keeps_the_poller_for_the_remaining_symbols():
    ex = _funding_exchange()
    ex.publicGetV5MarketTickers = AsyncMock(return_value=_tickers_response())
    try:
        await ex.watch_funding_rates(["BTC/USDT:USDT", "ETH/USDT:USDT"])
        await ex.un_watch_funding_rates(["ETH/USDT:USDT"])
        assert ex._funding_rate_adapter is not None
        await ex.un_watch_funding_rates(["BTC/USDT:USDT"])
    finally:
        await ex.close()
    assert ex._funding_rate_adapter is None


def test_bulk_liquidations_are_declared_as_watchable(bybit):
    """LiquidationDataHandler awaits watch_liquidations_for_symbols unconditionally."""
    assert bybit.has["watchLiquidationsForSymbols"] is True


@pytest.mark.asyncio
async def test_upstream_has_no_bulk_liquidation_stream():
    """The same 1 Hz NotSupported retry wedge as funding."""
    upstream = _upstream()
    assert upstream.has["watchLiquidationsForSymbols"] is False
    with pytest.raises(NotSupported):
        await upstream.watch_liquidations_for_symbols(["BTC/USDT:USDT"])
    await upstream.close()


@pytest.mark.asyncio
async def test_the_bulk_liquidation_subscription_names_the_all_liquidation_topics(bybit):
    seen: dict = {}
    rows = [{"symbol": "BTC/USDT:USDT", "side": "sell"}]

    async def _watch_topics(url, hashes, topics, params):
        seen.update(url=url, hashes=hashes, topics=topics)
        return rows

    bybit.watch_topics = _watch_topics
    assert await bybit.watch_liquidations_for_symbols(["BTC/USDT:USDT"]) is rows
    # the hash is the one ccxt's own handle_liquidation resolves
    assert seen["hashes"] == ["liquidations::BTC/USDT:USDT"]
    assert seen["topics"] == ["allLiquidation.BTCUSDT"]
    assert seen["url"].endswith("/v5/public/linear")


@pytest.mark.asyncio
async def test_the_bulk_liquidation_subscription_batches_every_symbol(bybit):
    """One topic per symbol on one connection — the handler hands us the whole batch."""
    seen: dict = {}
    bybit.set_markets([_swap_market(), _eth_market()])

    async def _watch_topics(url, hashes, topics, params):
        seen["topics"] = topics
        return []

    bybit.watch_topics = _watch_topics
    await bybit.watch_liquidations_for_symbols(["BTC/USDT:USDT", "ETH/USDT:USDT"])
    assert seen["topics"] == ["allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT"]


@pytest.mark.asyncio
async def test_the_bulk_liquidation_subscription_refuses_an_empty_batch(bybit):
    with pytest.raises(ArgumentsRequired):
        await bybit.watch_liquidations_for_symbols([])


@pytest.mark.asyncio
async def test_the_liquidation_topic_agrees_with_the_single_symbol_watch(bybit):
    """A mixed subscription must not end up on two different topics for one symbol."""
    seen: list = []

    async def _watch_topics(url, hashes, topics, params):
        seen.append(topics)
        return []

    bybit.watch_topics = _watch_topics
    await bybit.watch_liquidations_for_symbols(["BTC/USDT:USDT"])
    await bybit.watch_liquidations("BTC/USDT:USDT")
    assert seen[0] == seen[1]


def _uta(ex: BybitF) -> BybitF:
    ex.is_unified_enabled = AsyncMock(return_value=[False, True])
    return ex


def _order_row(order_id: str = "V1") -> dict:
    return {
        "orderId": order_id,
        "orderLinkId": "qubx_stop_1",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Market",
        "stopOrderType": "Stop",
        "triggerPrice": "105000",
        "price": "0",
        "qty": "0.01",
        "cumExecQty": "0.01",
        "orderStatus": "Filled",
        "createdTime": "1700000000000",
        "updatedTime": "1700000000001",
    }


@pytest.mark.asyncio
async def test_a_triggered_stop_is_found_after_the_stop_order_filter_misses():
    """A fired stop leaves the conditional book but still reads back STOP_MARKET, so the
    trigger=True lookup answers nothing."""
    ex = _uta(_funding_exchange())
    calls: list[dict] = []

    async def _endpoint(params={}):
        calls.append(dict(params))
        if params.get("orderFilter") == "StopOrder":
            return {"retCode": 0, "result": {"list": []}}
        return {"retCode": 0, "result": {"list": [_order_row()]}}

    ex.privateGetV5OrderRealtime = _endpoint
    try:
        order = await ex.fetch_order("V1", "BTC/USDT:USDT", {"trigger": True})
    finally:
        await ex.close()

    assert order["id"] == "V1"
    assert order["status"] == "closed"
    assert [c.get("orderFilter") for c in calls] == ["StopOrder", None]


@pytest.mark.asyncio
async def test_upstream_reports_a_triggered_stop_as_not_found():
    """The defect: one StopOrder-filtered call, and OrderNotFound on a live order."""
    ex = _uta(_upstream())
    ex.options["fetchOrder"] = {"acknowledged": True}
    ex.privateGetV5OrderRealtime = AsyncMock(return_value={"retCode": 0, "result": {"list": []}})
    try:
        with pytest.raises(OrderNotFound):
            await ex.fetch_order("V1", "BTC/USDT:USDT", {"trigger": True})
    finally:
        await ex.close()
    assert ex.privateGetV5OrderRealtime.await_count == 1


@pytest.mark.asyncio
async def test_the_stop_alias_falls_through_too():
    ex = _uta(_funding_exchange())
    calls: list[dict] = []

    async def _endpoint(params={}):
        calls.append(dict(params))
        return {"retCode": 0, "result": {"list": [] if params.get("orderFilter") else [_order_row()]}}

    ex.privateGetV5OrderRealtime = _endpoint
    try:
        assert (await ex.fetch_order("V1", "BTC/USDT:USDT", {"stop": True}))["id"] == "V1"
    finally:
        await ex.close()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_plain_lookup_is_never_retried():
    """A genuinely missing order must stay one call and one OrderNotFound."""
    ex = _uta(_funding_exchange())
    ex.privateGetV5OrderRealtime = AsyncMock(return_value={"retCode": 0, "result": {"list": []}})
    try:
        with pytest.raises(OrderNotFound):
            await ex.fetch_order("V1", "BTC/USDT:USDT")
    finally:
        await ex.close()
    assert ex.privateGetV5OrderRealtime.await_count == 1


@pytest.mark.asyncio
async def test_a_missing_trigger_order_still_raises_after_both_lookups():
    ex = _uta(_funding_exchange())
    ex.privateGetV5OrderRealtime = AsyncMock(return_value={"retCode": 0, "result": {"list": []}})
    try:
        with pytest.raises(OrderNotFound):
            await ex.fetch_order("V1", "BTC/USDT:USDT", {"trigger": True})
    finally:
        await ex.close()
    assert ex.privateGetV5OrderRealtime.await_count == 2


@pytest.mark.asyncio
async def test_a_triggered_stop_is_found_by_client_id(bybit):
    """The cid path carries the filter into the order-list endpoints too."""
    calls: list[dict] = []

    async def _open(symbol=None, since=None, limit=None, params={}):
        calls.append(dict(params))
        return [] if params.get("trigger") else [{"clientOrderId": "qubx_stop_1", "id": "V1"}]

    async def _closed(symbol=None, since=None, limit=None, params={}):
        calls.append(dict(params))
        return []

    bybit.fetch_open_orders = _open
    bybit.fetch_canceled_and_closed_orders = _closed

    row = await bybit.fetch_order("", "BTC/USDT:USDT", {"clientOrderId": "qubx_stop_1", "trigger": True})
    assert row["id"] == "V1"
    # both filtered endpoints first, then the unfiltered retry
    assert [c.get("trigger") for c in calls] == [True, True, None]
    assert all(c["orderLinkId"] == "qubx_stop_1" for c in calls)


@pytest.mark.asyncio
async def test_the_caller_params_are_not_mutated_by_the_fall_through(bybit):
    async def _rows(symbol=None, since=None, limit=None, params={}):
        return []

    bybit.fetch_open_orders = _rows
    bybit.fetch_canceled_and_closed_orders = _rows
    params = {"clientOrderId": "c1", "trigger": True}
    with pytest.raises(OrderNotFound):
        await bybit.fetch_order("", "BTC/USDT:USDT", params)
    assert params == {"clientOrderId": "c1", "trigger": True}


def _leverage_error(code: str) -> BadRequest:
    return BadRequest(f'bybit {{"retCode":{code},"retMsg":"leverage not modified"}}')


@pytest.mark.asyncio
async def test_a_no_op_set_leverage_is_treated_as_success(bybit):
    """Bybit answers a redundant set with 110043; every other venue no-ops silently."""
    with patch.object(cxp.bybit, "set_leverage", AsyncMock(side_effect=_leverage_error("110043"))):
        assert await bybit.set_leverage(3, "BTC/USDT:USDT") == {}


@pytest.mark.asyncio
async def test_a_genuine_leverage_refusal_still_raises(bybit):
    """110013 is the venue actually refusing; only 110043 is a confirmation."""
    with patch.object(cxp.bybit, "set_leverage", AsyncMock(side_effect=_leverage_error("110013"))):
        with pytest.raises(BadRequest):
            await bybit.set_leverage(3, "BTC/USDT:USDT")


@pytest.mark.asyncio
async def test_a_non_bad_request_failure_still_raises(bybit):
    """The code must be matched on a BadRequest, not found anywhere in any error string."""
    with patch.object(
        cxp.bybit, "set_leverage", AsyncMock(side_effect=ccxt.ExchangeError("110043 in a network error"))
    ):
        with pytest.raises(ccxt.ExchangeError):
            await bybit.set_leverage(3, "BTC/USDT:USDT")
