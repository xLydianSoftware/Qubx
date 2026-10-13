"""BINANCE spot traded from a BINANCE.PM account (``markets: [BINANCE.UM, BINANCE]``).

Pins the ccxt routing the spot leg relies on (spot market + portfolioMargin -> papi
cross-margin endpoints, margin execution reports parse as the spot symbol) and the PM
connector's spot behaviour: reduceOnly stripped, holdings from cross-margin balances,
margin open orders in the snapshot, spot skipped for leverage/ADL/amend. Offline.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock

import ccxt
import pytest

from qubx.connectors.ccxt.exchanges.binance.connector import BinancePmCcxtConnector
from qubx.connectors.ccxt.exchanges.binance.exchange import BinancePortfolioMargin
from qubx.core.basics import CtrlChannel, Order, OrderRequest, OrderSide, OrderStatus, OrderType
from qubx.core.lookups import lookup
from qubx.core.series import Quote
from tests.qubx.core.utils_test import DummyTimeProvider


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _market(symbol: str, market_id: str, base: str, kind: str) -> dict:
    spot = kind == "spot"
    return {
        "id": market_id,
        "symbol": symbol,
        "base": base,
        "quote": "USDT",
        "settle": None if spot else "USDT",
        "baseId": base,
        "quoteId": "USDT",
        "settleId": None if spot else "USDT",
        "type": kind,
        "spot": spot,
        "margin": spot,
        "swap": not spot,
        "future": False,
        "option": False,
        "contract": not spot,
        "linear": None if spot else True,
        "inverse": None if spot else False,
        "subType": None if spot else "linear",
        "active": True,
        "contractSize": None if spot else 1.0,
        "expiry": None,
        "expiryDatetime": None,
        "strike": None,
        "optionType": None,
        "precision": {"amount": 0.1 if spot else 1.0, "price": 0.0001},
        "limits": {
            "amount": {"min": 0.1 if spot else 1.0, "max": None},
            "price": {"min": None, "max": None},
            "cost": {"min": 5.0, "max": None},
            "leverage": {"min": None, "max": None},
        },
        "info": {"orderTypes": ["LIMIT", "LIMIT_MAKER", "MARKET", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]}
        if spot
        else {},
        "created": None,
    }


TRX_SPOT = _market("TRX/USDT", "TRXUSDT", "TRX", "spot")
TRX_SWAP = _market("TRX/USDT:USDT", "TRXUSDT", "TRX", "swap")
USDC_SPOT = _market("USDC/USDT", "USDCUSDT", "USDC", "spot")


def _pm_exchange() -> BinancePortfolioMargin:
    ex = BinancePortfolioMargin({"name": "BINANCE.PM", "apiKey": "k", "secret": "s"})
    ex.set_markets([TRX_SPOT, TRX_SWAP, USDC_SPOT])
    return ex


MARGIN_ORDER = {
    "symbol": "TRXUSDT",
    "orderId": 101,
    "clientOrderId": "qubx_TRXUSDT_1",
    "price": "0.3",
    "origQty": "60",
    "executedQty": "0",
    "cummulativeQuoteQty": "0",
    "status": "NEW",
    "timeInForce": "GTC",
    "type": "LIMIT",
    "side": "BUY",
    "time": 1784722758752,
    "updateTime": 1784722758752,
}


class TestCcxtSpotRouting:
    """What ccxt 4.5.50 does with a spot market on the PM class — verified in its source."""

    def test_spot_create_order_goes_to_papi_margin(self):
        ex = _pm_exchange()
        try:
            ex.papiPostMarginOrder = AsyncMock(return_value=MARGIN_ORDER)
            ex.papiPostUmOrder = AsyncMock(side_effect=AssertionError("UM endpoint for a spot order"))
            order = run(ex.create_order("TRX/USDT", "limit", "buy", 60, 0.3, {"clientOrderId": "qubx_TRXUSDT_1"}))
            request = ex.papiPostMarginOrder.await_args.args[0]
            assert request["symbol"] == "TRXUSDT"
            assert request["newClientOrderId"] == "qubx_TRXUSDT_1"
            assert order["symbol"] == "TRX/USDT"
        finally:
            run(ex.close())

    def test_ccxt_forwards_reduce_only_on_spot(self):
        # why the connector strips it: ccxt only drops reduceOnly for marginMode/margin type
        ex = _pm_exchange()
        try:
            request = ex.create_order_request("TRX/USDT", "market", "sell", 60, None, {"reduceOnly": True})
            assert request.get("reduceOnly") is True
        finally:
            run(ex.close())

    def test_spot_cancel_query_trades_and_open_orders_go_to_papi_margin(self):
        ex = _pm_exchange()
        try:
            ex.papiDeleteMarginOrder = AsyncMock(return_value=MARGIN_ORDER | {"status": "CANCELED"})
            ex.papiGetMarginOrder = AsyncMock(return_value=MARGIN_ORDER)
            ex.papiGetMarginMyTrades = AsyncMock(return_value=[])
            ex.papiGetMarginOpenOrders = AsyncMock(return_value=[MARGIN_ORDER])
            ex.papiGetUmOpenOrders = AsyncMock(side_effect=AssertionError("UM book queried for a spot symbol"))
            run(ex.cancel_order("101", "TRX/USDT"))
            run(ex.fetch_order("101", "TRX/USDT"))
            run(ex.fetch_my_trades("TRX/USDT", since=1784722758000))
            (row,) = run(ex.fetch_open_orders("TRX/USDT"))
            assert row["symbol"] == "TRX/USDT"
            for call in (ex.papiDeleteMarginOrder, ex.papiGetMarginOrder, ex.papiGetMarginMyTrades):
                call.assert_awaited_once()
        finally:
            run(ex.close())

    def test_margin_rows_parse_as_spot_and_um_rows_as_swap(self):
        ex = _pm_exchange()
        try:
            assert ex.parse_order(MARGIN_ORDER)["symbol"] == "TRX/USDT"
            um_row = MARGIN_ORDER | {"positionSide": "BOTH", "cumQuote": "0"}
            assert ex.parse_order(um_row)["symbol"] == "TRX/USDT:USDT"
        finally:
            run(ex.close())

    def test_pm_stream_execution_report_parses_as_spot(self):
        ex = _pm_exchange()
        try:
            report = {
                "e": "executionReport",
                "s": "TRXUSDT",
                "c": "qubx_TRXUSDT_1",
                "S": "BUY",
                "o": "MARKET",
                "q": "60",
                "p": "0",
                "X": "FILLED",
                "x": "TRADE",
                "i": 101,
                "l": "60",
                "z": "60",
                "L": "0.3",
                "n": "0.06",
                "N": "TRX",
                "T": 1784722758752,
                "t": 555,
                "m": False,
                "O": 1784722758700,
            }
            assert ex.parse_ws_order(report)["symbol"] == "TRX/USDT"
            futures_report = {"s": "TRXUSDT", "ps": "BOTH", "X": "NEW", "x": "NEW", "i": 1, "T": 1}
            assert ex.parse_ws_order(futures_report)["symbol"] == "TRX/USDT:USDT"
        finally:
            run(ex.close())


def _instr(exchange: str):
    instrument = lookup.find_symbol(exchange, "TRXUSDT")
    assert instrument is not None
    return instrument


def _connector(*, spot: bool = True, exchange=None, quote: Quote | None = None):
    em = Mock()
    em.exchange = exchange if exchange is not None else Mock()
    em.rate_limiter = None
    data_provider = Mock()
    data_provider.get_quote = Mock(return_value=quote)
    sent: list = []
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock(side_effect=lambda e: sent.append(e))
    conn = BinancePmCcxtConnector(
        exchange_name="BINANCE.UM",
        markets=("BINANCE.UM", "BINANCE") if spot else (),
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=data_provider,
    )
    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))
    conn._captured = captured  # type: ignore[attr-defined]
    return conn, sent


def _request(instrument, *, side: str = "SELL", options: dict | None = None) -> OrderRequest:
    return OrderRequest(
        instrument=instrument,
        quantity=60.0,
        price=None,
        order_type="MARKET",
        side=side,  # type: ignore[arg-type]
        time_in_force="gtc",
        client_id="qubx_TRXUSDT_1",
        options=options or {},
    )


class TestReduceOnlyStrip:
    @pytest.mark.parametrize("options", [{"reduce_only": True}, {"reduceOnly": True}])
    def test_spot_order_drops_reduce_only(self, options):
        ex = Mock()
        ex.create_order = AsyncMock(return_value={})
        conn, _ = _connector(exchange=ex, quote=Quote(0, 0.3, 0.3001, 1, 1))
        conn.submit_order(_request(_instr("BINANCE"), options=options))
        run(conn._captured[0])  # type: ignore[attr-defined]
        payload = ex.create_order.await_args.kwargs
        assert payload["symbol"] == "TRX/USDT"
        assert "reduceOnly" not in payload["params"]

    def test_perp_order_keeps_reduce_only(self):
        ex = Mock()
        ex.create_order = AsyncMock(return_value={})
        conn, _ = _connector(exchange=ex, quote=Quote(0, 0.3, 0.3001, 1, 1))
        conn.submit_order(_request(_instr("BINANCE.UM"), options={"reduce_only": True}))
        run(conn._captured[0])  # type: ignore[attr-defined]
        assert ex.create_order.await_args.kwargs["params"]["reduceOnly"] is True


def _balance_row(asset: str, **fields) -> dict:
    return {"asset": asset, "updateTime": 1784722758752, **fields}


def _raw_balance(*rows: dict) -> dict:
    return {"info": {"balance": list(rows), "account": {}}}


class TestSpotPositionsFromBalances:
    def test_exact_net_holding_marked(self):
        ex = _pm_exchange()
        try:
            conn, _ = _connector(exchange=ex, quote=Quote(0, 0.30, 0.32, 1, 1))
            raw = _raw_balance(
                _balance_row("USDT", crossMarginAsset="40"),
                _balance_row("USDC", crossMarginAsset="412.57"),  # stable: collateral, not a position
                _balance_row("TRX", crossMarginAsset="60.37", crossMarginBorrowed="0", crossMarginInterest="0"),
            )
            (pos,) = run(conn._balance_positions(raw))
            assert pos.instrument == _instr("BINANCE")
            assert pos.quantity == pytest.approx(60.37)  # as held: 0.07 of it is sub-lot dust
            assert pos.position_avg_price == pytest.approx(0.31)
            assert pos.last_update_price == pytest.approx(0.31)
            assert pos.market_value == pytest.approx(60.37 * 0.31)
        finally:
            run(ex.close())

    def test_borrow_and_interest_reduce_the_holding(self):
        ex = _pm_exchange()
        try:
            conn, _ = _connector(exchange=ex, quote=Quote(0, 0.30, 0.30, 1, 1))
            raw = _raw_balance(
                _balance_row("TRX", crossMarginAsset="10", crossMarginBorrowed="25", crossMarginInterest="0.05")
            )
            (pos,) = run(conn._balance_positions(raw))
            assert pos.quantity == pytest.approx(-15.05)  # net short: borrowed + interest beyond the holding
        finally:
            run(ex.close())

    def test_dust_only_is_a_flat_position_and_disabled_spot_yields_nothing(self):
        ex = _pm_exchange()
        try:
            conn, _ = _connector(exchange=ex, quote=Quote(0, 0.3, 0.3, 1, 1))
            (pos,) = run(conn._balance_positions(_raw_balance(_balance_row("TRX", crossMarginAsset="0.05"))))
            assert pos.quantity == pytest.approx(0.05)
            assert not pos.is_open()
            assert pos.last_update_price == pytest.approx(0.3)

            # - dust with no live quote is left out: no ticker request, no failed snapshot
            bare, _ = _connector(exchange=ex, quote=None)
            ex.publicGetTickerPrice = AsyncMock(return_value=[])
            assert run(bare._balance_positions(_raw_balance(_balance_row("TRX", crossMarginAsset="0.05")))) == []
            ex.publicGetTickerPrice.assert_not_awaited()

            off, _ = _connector(spot=False, exchange=ex, quote=Quote(0, 0.3, 0.3, 1, 1))
            assert run(off._balance_positions(_raw_balance(_balance_row("TRX", crossMarginAsset="60")))) == []
        finally:
            run(ex.close())

    def test_price_falls_back_to_ticker_and_missing_price_raises(self):
        ex = _pm_exchange()
        try:
            conn, _ = _connector(exchange=ex, quote=None)
            ex.publicGetTickerPrice = AsyncMock(return_value=[{"symbol": "TRXUSDT", "price": "0.33"}])
            raw = _raw_balance(_balance_row("TRX", crossMarginAsset="60"))
            (pos,) = run(conn._balance_positions(raw))
            assert pos.position_avg_price == pytest.approx(0.33)
            assert ex.publicGetTickerPrice.await_args.args[0] == {"symbols": '["TRXUSDT"]'}

            ex.publicGetTickerPrice = AsyncMock(return_value=[])
            with pytest.raises(ValueError):
                run(conn._balance_positions(raw))
        finally:
            run(ex.close())

    def test_an_inactive_spot_market_is_left_out_not_failing_the_snapshot(self):
        ex = _pm_exchange()
        try:
            ex.set_markets(
                [TRX_SPOT, TRX_SWAP, USDC_SPOT, {**_market("ADA/USDT", "ADAUSDT", "ADA", "spot"), "active": False}]
            )
            conn, _ = _connector(exchange=ex, quote=None)
            ex.publicGetTickerPrice = AsyncMock(return_value=[{"symbol": "TRXUSDT", "price": "0.33"}])
            raw = _raw_balance(_balance_row("TRX", crossMarginAsset="60"), _balance_row("ADA", crossMarginAsset="100"))
            (pos,) = run(conn._balance_positions(raw))
            assert pos.instrument.symbol == "TRXUSDT"
            assert ex.publicGetTickerPrice.await_args.args[0] == {"symbols": '["TRXUSDT"]'}
        finally:
            run(ex.close())

    def test_a_symbol_the_venue_refuses_to_price_is_left_out_the_rest_priced(self):
        ex = _pm_exchange()
        try:
            ex.set_markets([TRX_SPOT, TRX_SWAP, USDC_SPOT, _market("ADA/USDT", "ADAUSDT", "ADA", "spot")])
            conn, _ = _connector(exchange=ex, quote=None)

            async def tickers(params):
                symbols = json.loads(params["symbols"])
                if "ADAUSDT" in symbols:
                    raise ccxt.BadSymbol('binance {"code":-1121,"msg":"Invalid symbol."}')
                return [{"symbol": "TRXUSDT", "price": "0.33"}]

            ex.publicGetTickerPrice = AsyncMock(side_effect=tickers)
            raw = _raw_balance(_balance_row("TRX", crossMarginAsset="60"), _balance_row("ADA", crossMarginAsset="100"))
            (pos,) = run(conn._balance_positions(raw))
            assert pos.instrument.symbol == "TRXUSDT" and pos.position_avg_price == pytest.approx(0.33)
            assert ex.publicGetTickerPrice.await_count == 3  # - the batch, then one request per symbol
        finally:
            run(ex.close())

    def test_a_network_error_while_pricing_still_fails_the_snapshot(self):
        ex = _pm_exchange()
        try:
            conn, _ = _connector(exchange=ex, quote=None)
            ex.publicGetTickerPrice = AsyncMock(side_effect=ccxt.RequestTimeout("timeout"))
            with pytest.raises(ccxt.NetworkError):
                run(conn._balance_positions(_raw_balance(_balance_row("TRX", crossMarginAsset="60"))))
        finally:
            run(ex.close())


class TestSpotOpenOrders:
    def test_margin_open_orders_join_the_snapshot(self):
        ex = _pm_exchange()
        try:
            conn, _ = _connector(exchange=ex)
            ex.fetch_open_orders = AsyncMock(return_value=[])  # trigger (algo) surface
            ex.papiGetMarginOpenOrders = AsyncMock(return_value=[MARGIN_ORDER])
            (order,) = conn._merge_open_orders([], run(conn._fetch_extra_open_orders()))
            assert order.instrument == _instr("BINANCE")
            assert order.client_order_id == "qubx_TRXUSDT_1"

            off, _ = _connector(spot=False, exchange=ex)
            ex.papiGetMarginOpenOrders.reset_mock()
            assert run(off._fetch_extra_open_orders()) == []
            ex.papiGetMarginOpenOrders.assert_not_awaited()
        finally:
            run(ex.close())


class TestSpotSkips:
    def test_leverage_not_sent_for_spot(self):
        conn, _ = _connector()
        conn.set_instrument_leverage(_instr("BINANCE"), 3)
        assert conn._captured == []  # type: ignore[attr-defined]

    def test_adl_is_futures_only(self):
        conn, _ = _connector()
        conn._adl_levels = {"TRXUSDT": 4}
        conn._em.exchange.market = MagicMock(return_value={"id": "TRXUSDT"})
        assert conn.get_adl_level(_instr("BINANCE")) is None
        assert conn.get_adl_level(_instr("BINANCE.UM")) == 4

    def test_spot_amend_runs_as_cancel_and_recreate(self):
        # - papi has no margin amend: the connector emulates it (see test_amend_replace.py)
        conn, sent = _connector()
        order = Order(
            client_order_id="qubx_TRXUSDT_1",
            type=OrderType.LIMIT,
            instrument=_instr("BINANCE"),
            quantity=60.0,
            side=OrderSide.BUY,
            time_in_force="gtc",
            status=OrderStatus.ACCEPTED,
            venue_order_id="101",
            price=0.3,
        )
        assert not conn._amends_in_place(order.instrument)
        assert conn._amends_in_place(_instr("BINANCE.UM")) == bool(conn._em.exchange.has.get("editOrder", False))
        conn.update_order(order, price=0.31)
        (coro,) = conn._captured  # type: ignore[attr-defined]
        assert coro.cr_code.co_name == "_replace_async"
        coro.close()
        assert sent == []


def test_only_a_pm_account_can_also_trade_binance_spot():
    from qubx.connectors.ccxt.plugin import CcxtPlugin

    plugin = CcxtPlugin()
    assert plugin.extra_markets("BINANCE.PM") == ("BINANCE",)
    assert plugin.extra_markets("BINANCE.UM") == ()
    assert plugin.extra_markets("OKX.F") == ()
