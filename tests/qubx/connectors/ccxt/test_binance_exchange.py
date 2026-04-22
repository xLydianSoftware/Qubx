"""Unit tests for Qubx's Binance exchange overrides.

Pins two related fixes:
1. Binance USDⓈ-M WebSocket URL split (/public, /market, /private) — rolled
   out by Binance 2026-04-23. ccxt 4.5.44+ rewrites URLs at subscribe time.
2. BinanceQV.un_watch_bids_asks delegation — upstream removed get_message_hash
   in 4.5.20, which broke the previous override; it now delegates to
   watch_multi_ticker_helper(isUnsubscribe=True).

Everything here runs fully offline. `load_markets()` is bypassed because
GitHub-hosted CI runners are blocked by Binance (HTTP 451 from
fapi.binance.com/fapi/v1/exchangeInfo); instead we preseed two minimal
USDⓈ-M market entries via `set_markets()` — the only thing the watch_*
paths need from market data is `lowercaseId`/`symbol`/contract flags.
"""

import asyncio

import pytest

from qubx.connectors.ccxt.exchanges.binance.exchange import BinanceQVUSDM


def run(coro):
    return asyncio.run(coro)


def _usdm_market(base: str, amount_precision: float, price_precision: float) -> dict:
    """Minimal USDⓈ-M swap market dict that satisfies ccxt's downstream lookups."""
    return {
        "id": f"{base}USDT",
        "lowercaseId": f"{base.lower()}usdt",
        "symbol": f"{base}/USDT:USDT",
        "base": base,
        "quote": "USDT",
        "settle": "USDT",
        "baseId": base,
        "quoteId": "USDT",
        "settleId": "USDT",
        "type": "swap",
        "spot": False,
        "margin": False,
        "swap": True,
        "future": False,
        "option": False,
        "contract": True,
        "linear": True,
        "inverse": False,
        "subType": "linear",
        "active": True,
        "taker": 0.0004,
        "maker": 0.0002,
        "contractSize": 1.0,
        "expiry": None,
        "expiryDatetime": None,
        "strike": None,
        "optionType": None,
        "precision": {"amount": amount_precision, "price": price_precision},
        "limits": {
            "amount": {"min": amount_precision, "max": None},
            "price": {"min": None, "max": None},
            "cost": {"min": None, "max": None},
            "leverage": {"min": 1, "max": 125},
        },
        "info": {},
        "created": None,
        "marginModes": {"cross": True, "isolated": True},
    }


@pytest.fixture
def offline_binance_usdm():
    """BinanceQVUSDM with preseeded markets — no network calls."""
    ex = BinanceQVUSDM({"options": {"defaultType": "future"}})
    ex.set_markets([
        _usdm_market("BTC", 0.001, 0.1),
        _usdm_market("ETH", 0.001, 0.01),
    ])
    yield ex
    run(ex.close())


class TestBinanceUsdmWsUrlSplit:
    """Verify every Qubx-used futures subscription routes through the category-split URLs."""

    @staticmethod
    def _capture_ws_url(exchange, subscribe_coro_factory):
        captured = []

        def capture(url):
            captured.append(url)
            raise RuntimeError("STOP_BEFORE_CONNECT")

        exchange.client = capture

        async def go():
            try:
                await subscribe_coro_factory()
            except RuntimeError as e:
                if str(e) != "STOP_BEFORE_CONNECT":
                    raise

        run(go())
        assert captured, "exchange.client(url) was never called"
        return captured[-1]

    def test_book_ticker_routes_to_public_ws(self, offline_binance_usdm):
        ex = offline_binance_usdm
        url = self._capture_ws_url(ex, lambda: ex.watch_bids_asks(["BTC/USDT:USDT"]))
        assert url.startswith("wss://fstream.binance.com/public/ws/"), url

    def test_depth_routes_to_public_ws(self, offline_binance_usdm):
        ex = offline_binance_usdm
        url = self._capture_ws_url(ex, lambda: ex.watch_order_book("BTC/USDT:USDT"))
        assert url.startswith("wss://fstream.binance.com/public/ws/"), url

    def test_agg_trade_routes_to_market_ws(self, offline_binance_usdm):
        ex = offline_binance_usdm
        url = self._capture_ws_url(ex, lambda: ex.watch_trades("BTC/USDT:USDT"))
        assert url.startswith("wss://fstream.binance.com/market/ws/"), url

    def test_kline_routes_to_market_ws(self, offline_binance_usdm):
        ex = offline_binance_usdm
        url = self._capture_ws_url(ex, lambda: ex.watch_ohlcv("BTC/USDT:USDT", "1m"))
        assert url.startswith("wss://fstream.binance.com/market/ws/"), url

    def test_ticker_routes_to_market_ws(self, offline_binance_usdm):
        ex = offline_binance_usdm
        url = self._capture_ws_url(ex, lambda: ex.watch_ticker("BTC/USDT:USDT"))
        assert url.startswith("wss://fstream.binance.com/market/ws/"), url

    def test_mark_price_routes_to_market_ws(self, offline_binance_usdm):
        ex = offline_binance_usdm
        url = self._capture_ws_url(ex, lambda: ex.watch_mark_prices(["BTC/USDT:USDT"]))
        assert url.startswith("wss://fstream.binance.com/market/ws/"), url

    def test_private_ws_url_uses_split_path(self, offline_binance_usdm):
        url = offline_binance_usdm.get_private_ws_url("future", "DUMMY_LISTEN_KEY")
        assert url == "wss://fstream.binance.com/private/ws?listenKey=DUMMY_LISTEN_KEY"

    def test_spot_and_delivery_hosts_are_unchanged(self, offline_binance_usdm):
        # Binance's migration only affects USDⓈ-M. Spot and Coin-M stay legacy.
        ex = offline_binance_usdm
        assert ex.get_ws_url("spot", "public") == "wss://stream.binance.com:9443/ws"
        assert ex.get_ws_url("delivery", "public") == "wss://dstream.binance.com/ws"


class TestBinanceUnWatchBidsAsks:
    """Verify BinanceQV.un_watch_bids_asks delegates correctly to upstream's helper."""

    def test_un_watch_bids_asks_produces_correct_unsubscribe_request(self, offline_binance_usdm):
        exchange = offline_binance_usdm
        captured = {}

        async def fake_watch_multiple(url, messageHashes, request, subscribeHashes, subscription):
            captured.update(
                url=url,
                messageHashes=messageHashes,
                request=request,
                subscribeHashes=subscribeHashes,
                subscription=subscription,
            )
            return "ACK"

        exchange.watch_multiple = fake_watch_multiple

        result = run(exchange.un_watch_bids_asks(["BTC/USDT:USDT", "ETH/USDT:USDT"]))

        assert result == "ACK"

        # URL must use the new /public/ws split path (post-2026-04-23 migration).
        assert captured["url"].startswith("wss://fstream.binance.com/public/ws/")

        # Request body is a Binance UNSUBSCRIBE with lowercase symbol@channel params.
        assert captured["request"]["method"] == "UNSUBSCRIBE"
        assert set(captured["request"]["params"]) == {"btcusdt@bookTicker", "ethusdt@bookTicker"}
        assert isinstance(captured["request"]["id"], int)

        # subscription.subMessageHashes must mirror what watch_bids_asks subscribe
        # stored, so ccxt's client correctly evicts the subscription on ACK.
        sub = captured["subscription"]
        assert sub["unsubscribe"] is True
        assert sub["subMessageHashes"] == [
            "bidask:bookTicker@BTC/USDT:USDT",
            "bidask:bookTicker@ETH/USDT:USDT",
        ]

        # messageHashes (what the returned future resolves on) use the unsubscribe:: prefix.
        assert sub["messageHashes"] == [
            "unsubscribe::bidask:bookTicker@BTC/USDT:USDT",
            "unsubscribe::bidask:bookTicker@ETH/USDT:USDT",
        ]

    def test_un_watch_bids_asks_does_not_call_removed_get_message_hash(self, offline_binance_usdm):
        # Regression guard: get_message_hash was removed in ccxt 4.5.20 and the
        # previous override called it, raising AttributeError. Delegation path
        # must not rely on it.
        exchange = offline_binance_usdm
        assert not hasattr(exchange, "get_message_hash")

        async def fake_watch_multiple(*args, **kwargs):
            return "ACK"

        exchange.watch_multiple = fake_watch_multiple
        run(exchange.un_watch_bids_asks(["BTC/USDT:USDT"]))
