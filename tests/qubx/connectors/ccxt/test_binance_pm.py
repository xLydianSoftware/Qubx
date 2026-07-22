"""Unit tests for Binance portfolio-margin (BINANCE.PM) support.

Pins the PM routing fix and the PM-specific connector subclass:

1. ``BinancePortfolioMargin`` options must resolve every param-less unified call onto
   the linear/papi path. ccxt resolves the market type from ``defaultType`` BEFORE
   consulting ``portfolioMargin``, and its spot/margin branches short-circuit — the
   previous ``defaultType: "margin"`` sent the user-data stream to the spot ws-api
   (zero UM events) and venue-wide open orders to the papi MARGIN surface.
2. ``parse_balance_custom`` pins realized-only PM balance semantics (totalWalletBalance)
   even though the routed type is linear.
3. ``parse_position_risk`` defaults marginMode to cross (papi payloads carry neither
   marginType nor isolatedMargin; PM is always cross).
4. ``BinancePmCcxtConnector._extract_venue_figures`` reads the papiGetAccount dict the
   exchange class grafts into fetch_balance's ``info`` (uniMMR sentinel mapped to None).
5. The factory resolves ``CUSTOM_CONNECTORS`` by venue name first, canonical second.
6. ``ccxt_extract_leverage_settings`` accepts papi's ``maxNotional`` spelling.

Everything here runs fully offline.
"""

import asyncio
import io
from unittest.mock import MagicMock

from qubx import logger
from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.connectors.ccxt.exchanges import CUSTOM_CONNECTORS
from qubx.connectors.ccxt.exchanges.binance.connector import BinancePmCcxtConnector
from qubx.connectors.ccxt.exchanges.binance.exchange import BinancePortfolioMargin
from qubx.connectors.ccxt.factory import get_ccxt_connector
from qubx.connectors.ccxt.utils import ccxt_extract_leverage_settings


def run(coro):
    # NOT asyncio.run: that clears the thread's current event loop on exit, breaking
    # later tests in the same worker that rely on asyncio.get_event_loop()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _pm_exchange() -> BinancePortfolioMargin:
    return BinancePortfolioMargin({"apiKey": "k", "secret": "s"})


class TestBinancePmRouting:
    def test_options_route_linear(self):
        ex = _pm_exchange()
        try:
            assert ex.options["defaultType"] == "swap"
            assert ex.options["defaultSubType"] == "linear"
            assert ex.options["portfolioMargin"] is True
            # papi ws base must exist for the PM user-data stream
            assert ex.urls["api"]["ws"]["papi"].startswith("wss://")
        finally:
            run(ex.close())

    def test_watch_and_fetch_paths_resolve_linear(self):
        """The exact decisions watch_orders / fetch_open_orders make param-less: any
        non-linear outcome regresses to the spot ws-api / papi margin surfaces."""
        ex = _pm_exchange()
        try:
            for method in ("watchOrders", "watchBalance", "authenticate", "fetchOpenOrders"):
                t, params = ex.handle_market_type_and_params(method, None, {})
                st, _ = ex.handle_sub_type_and_params(method, None, params)
                assert ex.is_linear(t, st), f"{method} resolved type={t} subType={st} — not linear"
                assert t not in ("spot", "margin")
        finally:
            run(ex.close())


class TestBinancePmBalanceParse:
    PM_BALANCE_ROW = {
        "asset": "USDT",
        "totalWalletBalance": "386.01",
        "crossMarginFree": "386.01",
        "crossMarginLocked": "1.5",
        "crossMarginBorrowed": "0",
        "crossMarginInterest": "0",
        "umWalletBalance": "300.0",
        "umUnrealizedPNL": "5.0",
        "cmWalletBalance": "0",
        "cmUnrealizedPNL": "0",
    }

    def test_pm_totals_are_realized_only(self):
        ex = _pm_exchange()
        try:
            # type "linear" is what the swap defaultType routes — the PM pin must still
            # produce whole-account realized-only totals, not umWallet + unrealized PnL
            parsed = ex.parse_balance_custom([self.PM_BALANCE_ROW], "linear", None, True)
            assert parsed["USDT"]["total"] == 386.01
            assert parsed["USDT"]["used"] == 1.5
        finally:
            run(ex.close())

    def test_non_pm_passthrough(self):
        ex = _pm_exchange()
        try:
            futures_account = {
                "assets": [{"asset": "USDT", "availableBalance": "10", "initialMargin": "2", "walletBalance": "12"}]
            }
            parsed = ex.parse_balance_custom(futures_account, None, None, False)
            assert parsed["USDT"]["total"] == 12.0
        finally:
            run(ex.close())


class TestBinancePmPositionRisk:
    PAPI_ROW = {
        "symbol": "BTCUSDT",
        "positionAmt": "0.01",
        "entryPrice": "44525.0",
        "markPrice": "45464.17",
        "unRealizedProfit": "9.39",
        "liquidationPrice": "38007.16",
        "leverage": "100",
        "positionSide": "BOTH",
        "updateTime": 1707371879042,
        "maxNotionalValue": "500000.0",
        "notional": "454.64",
    }

    MARKET = {
        "id": "BTCUSDT",
        "symbol": "BTC/USDT:USDT",
        "contract": True,
        "linear": True,
        "contractSize": 1.0,
        "precision": {"amount": 0.001, "price": 0.1},
    }

    def test_margin_mode_defaults_to_cross(self):
        ex = _pm_exchange()
        try:
            parsed = ex.parse_position_risk(self.PAPI_ROW, self.MARKET)
            assert parsed["marginMode"] == "cross"
        finally:
            run(ex.close())

    def test_explicit_margin_mode_is_kept(self):
        ex = _pm_exchange()
        try:
            row = dict(self.PAPI_ROW, marginType="isolated", isolatedMargin="1.0")
            parsed = ex.parse_position_risk(row, self.MARKET)
            assert parsed["marginMode"] == "isolated"
        finally:
            run(ex.close())


class TestBinancePmVenueFigures:
    @staticmethod
    def _connector() -> BinancePmCcxtConnector:
        return BinancePmCcxtConnector(
            exchange_name="BINANCE.UM",
            channel=MagicMock(),
            time_provider=MagicMock(),
            exchange_manager=MagicMock(),
            data_provider=MagicMock(),
        )

    def test_reads_grafted_papi_account(self):
        figures = self._connector()._extract_venue_figures(
            {
                "info": {
                    "balance": [{"asset": "USDT"}],
                    "account": {
                        "accountEquity": "404.51",
                        "totalAvailableBalance": "359.52",
                        "uniMMR": "731.2",
                        "accountMaintMargin": "0.55",
                        "virtualMaxWithdrawAmount": "358.0",
                    },
                }
            }
        )
        assert figures == (404.51, 359.52, 731.2, 358.0)

    def test_unimmr_sentinel_maps_to_none(self):
        # no positions: accountMaintMargin 0 and uniMMR is a 99999999 sentinel
        figures = self._connector()._extract_venue_figures(
            {
                "info": {
                    "balance": [],
                    "account": {
                        "accountEquity": "404.51",
                        "totalAvailableBalance": "404.51",
                        "uniMMR": "99999999",
                        "accountMaintMargin": "0.0",
                        "virtualMaxWithdrawAmount": "404.51",
                    },
                }
            }
        )
        assert figures == (404.51, 404.51, None, 404.51)

    def test_missing_graft_degrades_to_none(self):
        # raw papiGetBalance list info (graft failed) must not sink the snapshot
        assert self._connector()._extract_venue_figures({"info": [{"asset": "USDT"}]}) == (None, None, None, None)


DOGE_MARKET = {
    "id": "DOGEUSDT",
    "lowercaseId": "dogeusdt",
    "symbol": "DOGE/USDT:USDT",
    "base": "DOGE",
    "quote": "USDT",
    "settle": "USDT",
    "baseId": "DOGE",
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
    "precision": {"amount": 1.0, "price": 0.00001},
    "limits": {
        "amount": {"min": 1.0, "max": None},
        "price": {"min": None, "max": None},
        "cost": {"min": None, "max": None},
        "leverage": {"min": 1, "max": 75},
    },
    "info": {},
    "created": None,
    "marginModes": {"cross": True, "isolated": False},
}

ALGO_RESPONSE = {
    "algoId": 4000001785373649,
    "algoType": "CONDITIONAL",
    "clientAlgoId": "qubx_DOGEUSDT_1",
    "orderType": "STOP_MARKET",
    "symbol": "DOGEUSDT",
    "side": "SELL",
    "positionSide": "BOTH",
    "timeInForce": "GTC",
    "quantity": "300",
    "algoStatus": "NEW",
    "actualOrderId": "",
    "actualPrice": "0",
    "actualQty": "0",
    "triggerPrice": "0.050640",
    "price": "0.000000",
    "reduceOnly": False,
    "createTime": 1784722758752,
    "updateTime": 1784722758752,
}


def _pm_exchange_with_markets():
    """PM exchange with preseeded DOGE market and a captured raw-request transport —
    algo calls (place/cancel/query) never have implicit ccxt methods, so everything
    goes through ``request``; capture it to assert paths and payloads offline."""
    ex = _pm_exchange()
    ex.set_markets([DOGE_MARKET])
    calls = []

    async def fake_request(path, api="public", method="GET", params={}, *args, **kwargs):
        calls.append((path, api, method, dict(params)))
        return dict(ALGO_RESPONSE)

    ex.request = fake_request
    return ex, calls


class TestPmAlgoOrders:
    """PM conditional orders migrated to the papi Algo Service on 2026-04-28 (old
    um/conditional/* endpoints 404). Live-verified 2026-07-22: trigger param is
    ``triggerPrice``, client id is ``clientAlgoId``, cancel/query by ``algoId``."""

    def test_stop_market_routes_to_algo_endpoint(self):
        # the exact shape prepare_ccxt_order_payload emits for STOP_MARKET
        ex, calls = _pm_exchange_with_markets()
        try:
            order = run(
                ex.create_order(
                    "DOGE/USDT:USDT",
                    "market",
                    "sell",
                    300,
                    0.05064,
                    {"triggerPrice": 0.05064, "clientOrderId": "qubx_DOGEUSDT_1", "timeInForce": "GTC", "type": "swap"},
                )
            )
            path, api, method, req = calls[0]
            assert (path, api, method) == ("um/algo/order", "papi", "POST")
            assert req["type"] == "STOP_MARKET"
            assert req["algoType"] == "CONDITIONAL"
            assert req["triggerPrice"] == "0.05064"
            assert req["clientAlgoId"] == "qubx_DOGEUSDT_1"
            assert "stopPrice" not in req and "price" not in req
            assert order["id"] == "4000001785373649"
            assert order["status"] == "open"
            assert order["triggerPrice"] == 0.05064
        finally:
            run(ex.close())

    def test_stop_limit_carries_price_and_tif(self):
        ex, calls = _pm_exchange_with_markets()
        try:
            run(ex.create_order("DOGE/USDT:USDT", "limit", "sell", 300, 0.0501, {"triggerPrice": 0.05064}))
            _, _, _, req = calls[0]
            assert req["type"] == "STOP"
            assert req["price"] == "0.0501"
            assert req["timeInForce"] == "GTC"
        finally:
            run(ex.close())

    def test_non_trigger_orders_unaffected(self):
        ex, calls = _pm_exchange_with_markets()
        try:
            try:
                run(ex.create_order("DOGE/USDT:USDT", "limit", "buy", 300, 0.05, {}))
            except Exception:
                pass  # goes to the real papi order path (network) — only assert no algo call
            assert calls == []
        finally:
            run(ex.close())

    def test_cancel_by_algo_id_and_by_client_id(self):
        ex, calls = _pm_exchange_with_markets()
        try:
            run(ex.cancel_order("4000001785373649", "DOGE/USDT:USDT", {"trigger": True}))
            # the cid path used by cancel_order_with_client_order_id (id='')
            run(ex.cancel_order("", "DOGE/USDT:USDT", {"trigger": True, "clientOrderId": "qubx_DOGEUSDT_1"}))
            (p1, _, m1, r1), (p2, _, m2, r2) = calls
            assert (p1, m1, r1.get("algoId")) == ("um/algo/order", "DELETE", "4000001785373649")
            assert (p2, m2, r2.get("clientAlgoId")) == ("um/algo/order", "DELETE", "qubx_DOGEUSDT_1")
            assert "algoId" not in r2
        finally:
            run(ex.close())

    def test_fetch_order_and_open_orders_route_to_algo_queries(self):
        ex, calls = _pm_exchange_with_markets()
        try:
            run(ex.fetch_order("4000001785373649", "DOGE/USDT:USDT", {"trigger": True}))
            assert calls[-1][:3] == ("um/algo/algoOrder", "papi", "GET")

            async def fake_list(path, api="public", method="GET", params={}, *args, **kwargs):
                calls.append((path, api, method, dict(params)))
                return [dict(ALGO_RESPONSE)]

            ex.request = fake_list
            orders = run(ex.fetch_open_orders("DOGE/USDT:USDT", params={"trigger": True}))
            assert calls[-1][:3] == ("um/algo/openAlgoOrders", "papi", "GET")
            assert len(orders) == 1 and orders[0]["clientOrderId"] == "qubx_DOGEUSDT_1"
        finally:
            run(ex.close())

    def test_algo_status_mapping(self):
        ex, _ = _pm_exchange_with_markets()
        try:
            for raw_status, unified in [("NEW", "open"), ("CANCELED", "canceled"), ("TRIGGERED", "closed")]:
                parsed = ex.parse_algo_order(dict(ALGO_RESPONSE, algoStatus=raw_status))
                assert parsed["status"] == unified, raw_status
        finally:
            run(ex.close())


class TestVenueErrorLogging:
    def test_merge_open_orders_survives_html_error_text(self):
        """Venue errors can carry raw HTML (Binance 404 pages). With a colorized sink,
        f-stringing that into loguru's format string raises ValueError inside the
        snapshot task — the positional-args form must survive it."""
        connector = TestBinancePmVenueFigures._connector()
        html_error = Exception("404 Not Found <!DOCTYPE html>\n<html>\n<head></head></html>")
        sink_id = logger.add(io.StringIO(), colorize=True, level="WARNING")
        try:
            assert connector._merge_open_orders(html_error, []) is None
            assert connector._merge_open_orders([], html_error) is None
        finally:
            logger.remove(sink_id)


class TestPmAdlLevels:
    """PM has no adl in positionRisk — one bulk adlQuantile call per snapshot stamps
    Position.adl_level and refreshes a cache; get_adl_level is a dict read (never a
    blocking venue call — safe to iterate from the strategy thread)."""

    ADL_ROWS = [
        {"symbol": "DOGEUSDT", "adlQuantile": {"LONG": 2, "SHORT": 1}},
        {"symbol": "BTCUSDT", "adlQuantile": {"BOTH": 3}},
    ]

    @staticmethod
    def _connector_with_adl(rows):
        connector = TestBinancePmVenueFigures._connector()

        async def fake_adl(*args, **kwargs):
            return rows

        connector._em.exchange.papiGetUmAdlQuantile = fake_adl
        connector._em.exchange.market = MagicMock(side_effect=Exception("markets not loaded"))
        return connector

    @staticmethod
    def _position(symbol: str) -> MagicMock:
        pos = MagicMock()
        pos.instrument.symbol = symbol
        pos.adl_level = None
        pos.leverage = 1.0
        pos.max_notional = 1.0  # short-circuits the base leverage fill (no fetch_leverages)
        return pos

    def test_fill_stamps_positions_and_cache(self):
        connector = self._connector_with_adl(self.ADL_ROWS)
        doge, btc = self._position("DOGEUSDT"), self._position("BTCUSDT")
        run(connector._fill_adl_levels([doge, btc]))
        assert doge.adl_level == 2  # max(LONG, SHORT)
        assert btc.adl_level == 3
        assert connector.get_adl_level(doge.instrument) == 2
        assert connector.get_adl_level(btc.instrument) == 3

    def test_get_adl_level_never_calls_venue(self):
        connector = self._connector_with_adl(self.ADL_ROWS)
        instrument = MagicMock()
        instrument.symbol = "DOGEUSDT"
        assert connector.get_adl_level(instrument) is None  # empty cache, no network
        connector._run_sync = MagicMock(side_effect=AssertionError("must not block on venue"))
        assert connector.get_adl_level(instrument) is None

    def test_fetch_failure_keeps_previous_cache(self):
        connector = self._connector_with_adl(self.ADL_ROWS)
        doge = self._position("DOGEUSDT")
        run(connector._fill_adl_levels([doge]))

        async def boom(*args, **kwargs):
            raise RuntimeError("venue down")

        connector._em.exchange.papiGetUmAdlQuantile = boom
        fresh = self._position("DOGEUSDT")
        run(connector._fill_adl_levels([fresh]))
        assert connector.get_adl_level(doge.instrument) == 2  # cache preserved
        assert fresh.adl_level is None  # nothing stamped on failure


class TestPmConnectorResolution:
    def test_registered_for_venue_name(self):
        assert CUSTOM_CONNECTORS["binance.pm"] is BinancePmCcxtConnector

    def test_factory_resolves_venue_before_canonical(self):
        kwargs = dict(
            channel=MagicMock(),
            time_provider=MagicMock(),
            exchange_manager=MagicMock(),
            data_provider=MagicMock(),
        )
        pm = get_ccxt_connector("BINANCE.UM", venue_name="BINANCE.PM", **kwargs)
        assert type(pm) is BinancePmCcxtConnector
        assert pm.exchange_name == "BINANCE.UM"  # account events stay keyed canonical

        um = get_ccxt_connector("BINANCE.UM", venue_name="BINANCE.UM", **kwargs)
        assert type(um) is CcxtConnector

        um_no_venue = get_ccxt_connector("BINANCE.UM", **kwargs)
        assert type(um_no_venue) is CcxtConnector


class TestLeverageSettingsSpelling:
    def test_papi_max_notional_spelling(self):
        rows = [
            {"symbol": "BTC/USDT:USDT", "longLeverage": 5, "info": {"maxNotional": "480000000"}},
            {"symbol": "ETH/USDT:USDT", "longLeverage": 10, "info": {"maxNotionalValue": "10000000"}},
        ]
        settings = ccxt_extract_leverage_settings(rows)
        assert settings["BTC/USDT:USDT"] == (5.0, 480000000.0)
        assert settings["ETH/USDT:USDT"] == (10.0, 10000000.0)
