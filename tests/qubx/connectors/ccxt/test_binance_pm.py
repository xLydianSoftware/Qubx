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
    return asyncio.run(coro)


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


class TestPmTriggerOrders:
    def test_conditional_create_order_fails_fast(self):
        """The papi conditional API is suspended (404 live-verified 2026-07-22) — a stop
        order must raise NotSupported synchronously, before any network call."""
        import ccxt

        ex = _pm_exchange()
        try:
            for kwargs in (
                {"params": {"triggerPrice": 0.05}},
                {"params": {"stopLossPrice": 0.05}},
                {"type": "stop_market", "params": {}},
            ):
                order_type = kwargs.get("type", "market")
                try:
                    run(ex.create_order("DOGE/USDT:USDT", order_type, "sell", 150, None, kwargs["params"]))
                    raise AssertionError(f"expected NotSupported for {kwargs}")
                except ccxt.NotSupported:
                    pass
        finally:
            run(ex.close())

    def test_trigger_open_orders_short_circuit(self):
        # papi /um/conditional/openOrders 404s (live-verified 2026-07-22) — the PM
        # override must not touch the exchange at all
        connector = TestBinancePmVenueFigures._connector()
        assert run(connector._fetch_trigger_open_orders()) == []
        connector._em.exchange.fetch_open_orders.assert_not_called()


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
