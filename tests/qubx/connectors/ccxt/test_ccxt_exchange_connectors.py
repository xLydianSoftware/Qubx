"""Unit tests for the per-exchange CcxtConnector subclasses (commit 4b).

Covers OKX/Bitfinex split orders/fills streams, the FILLED-promotion-via-last-deal
path (and its AM dedup safety), OKX balance extraction + make_client_id, and the
get_ccxt_connector factory. Mocked ccxt — no credentials or network. Async work is
driven deterministically via the ``_spawn``-capture pattern from the read tests.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.connectors.ccxt.exchanges.bitfinex.connector import BitfinexCcxtConnector
from qubx.connectors.ccxt.exchanges.okx.connector import OkxCcxtConnector
from qubx.connectors.ccxt.factory import get_ccxt_connector
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Instrument, MarketType, Order, OrderOrigin, OrderStatus
from qubx.core.events import (
    OrderAcceptedEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
)
from qubx.core.series import Quote
from tests.qubx.core.utils_test import DummyTimeProvider

CCXT_SYMBOL = "BTC/USDT:USDT"


def _instrument() -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SWAP,
        exchange="OKX.F",
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


def _make_connector(cls: type[CcxtConnector], exchange: Mock | None = None) -> tuple[CcxtConnector, list, Mock]:
    if exchange is None:
        exchange = Mock()
    exchange.name = "okx"

    em = Mock()
    em.exchange = exchange

    dp = Mock()
    dp.get_quote = Mock(return_value=_quote())

    sent: list = []
    channel = Mock()
    channel.send = Mock(side_effect=lambda e: sent.append(e))

    conn = cls(
        exchange_name="OKX.F",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=dp,
    )
    conn._symbol_to_instrument[CCXT_SYMBOL] = _instrument()

    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))
    conn._captured = captured  # type: ignore[attr-defined]
    return conn, sent, exchange


def _ws_order(*, status: str, cid: str = "qubxBTCUSDT1", venue_id: str = "VENUE123") -> dict:
    return {
        "info": {},
        "id": venue_id,
        "clientOrderId": cid,
        "symbol": CCXT_SYMBOL,
        "timestamp": 1700000000000,
        "type": "limit",
        "timeInForce": "GTC",
        "side": "buy",
        "price": 100.0,
        "amount": 1.0,
        "cost": 0.0,
        "status": status,
        "trades": [],
    }


def _ws_trade(trade_id: str = "T1", amount: float = 0.5, price: float = 100.0, venue_order_id: str = "VENUE123") -> dict:
    return {
        "id": trade_id,
        "order": venue_order_id,
        "symbol": CCXT_SYMBOL,
        "timestamp": 1700000000000,
        "side": "buy",
        "amount": amount,
        "price": price,
        "takerOrMaker": "taker",
        "fee": {"cost": 0.01, "currency": "USDT"},
    }


# --------------------------------------------------------------------------- #
# OKX two-stream: open -> accepted; trade -> partial; FILLED -> filled
# --------------------------------------------------------------------------- #
def test_okx_watch_orders_open_emits_accepted() -> None:
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderAcceptedEvent)
    assert ev.client_order_id == "qubxBTCUSDT1"
    assert ev.venue_order_id == "VENUE123"


def test_okx_watch_orders_open_carries_no_fill() -> None:
    # OKX watch_orders never carries a fill even on a partial/filled report; the trade
    # stream owns fills. A PARTIALLY_FILLED status from watch_orders is a no-op here.
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    # Seed the venue ack via the prior open report (the realistic order: open precedes
    # the partial), so the partial status isn't mistaken for a first-seen fill.
    conn._handle_ws_order(_ws_order(status="open"))
    sent.clear()
    raw = _ws_order(status="open")
    raw["info"] = {"status": "PARTIALLY_FILLED"}
    conn._handle_ws_order(raw)
    assert sent == []


def test_okx_watch_my_trades_emits_partially_filled() -> None:
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    # Seed the venue->cid index via the open order so the trade resolves its cid.
    conn._handle_ws_order(_ws_order(status="open"))
    sent.clear()

    conn._handle_ws_trade(_ws_trade("T1", amount=0.5))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderPartiallyFilledEvent)
    assert ev.client_order_id == "qubxBTCUSDT1"
    assert ev.venue_order_id == "VENUE123"
    assert ev.fill.trade_id == "T1"
    assert ev.fill.amount == 0.5


def test_okx_filled_status_promotes_with_last_deal() -> None:
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_trade(_ws_trade("T1", amount=1.0))
    sent.clear()

    # watch_orders now reports the order FILLED (no trade) -> promote carrying T1.
    conn._handle_ws_order(_ws_order(status="closed"))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderFilledEvent)
    assert ev.client_order_id == "qubxBTCUSDT1"
    assert ev.fill.trade_id == "T1"
    assert ev.fill.amount == 1.0


def test_okx_filled_without_remembered_trade_emits_nothing() -> None:
    # No prior trade remembered -> OrderFilledEvent needs a Deal, so skip and rely on
    # snapshot reconcile (do not fabricate a terminal transition without a fill).
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    sent.clear()
    conn._handle_ws_order(_ws_order(status="closed"))
    assert sent == []


@pytest.mark.asyncio
async def test_okx_split_promotion_is_am_dedup_safe() -> None:
    # The same trade_id rides both the partial (trade stream) and the filled
    # (promotion) events. A REAL AccountManager must dedup the fill by trade_id: the
    # position reflects ONE fill, but the order still transitions to FILLED.
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_trade(_ws_trade("T1", amount=1.0, price=100.0))
    conn._handle_ws_order(_ws_order(status="closed"))

    partials = [e for e in sent if isinstance(e, OrderPartiallyFilledEvent)]
    fills = [e for e in sent if isinstance(e, OrderFilledEvent)]
    assert len(partials) == 1
    assert len(fills) == 1
    # Same trade_id across both -> AM's seen_trade_ids dedups the fill amount.
    assert partials[0].fill.trade_id == fills[0].fill.trade_id == "T1"

    strategy = Mock()
    am = SimulatedAccountManager(
        connectors={"OKX.F": object()},
        strategy=strategy,
        time=DummyTimeProvider(),
    )
    # Order must exist for the events to land on it: accepted events are resolve-only
    # (no materialization), so seed the framework order first, then ack it.
    am.add_order(
        Order(
            client_order_id="qubxBTCUSDT1",
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=_instrument(),
            time=DummyTimeProvider().time(),
            quantity=1.0,
            price=100.0,
            side="BUY",
            status=OrderStatus.SUBMITTED,
            time_in_force="gtc",
        )
    )
    am.apply(OrderAcceptedEvent(
        instrument=_instrument(),
        client_order_id="qubxBTCUSDT1",
        venue_order_id="VENUE123",
        accepted_at=DummyTimeProvider().time(),
    ))
    am.apply(partials[0])
    am.apply(fills[0])

    order = am.find_order_by_id("VENUE123")
    assert order is not None
    assert order.status is OrderStatus.FILLED
    # Dedup: filled_quantity reflects ONE fill of 1.0, not 2.0.
    assert order.filled_quantity == pytest.approx(1.0)
    pos = am.get_position(_instrument())
    assert pos.quantity == pytest.approx(1.0)


def test_okx_terminal_evicts_last_deal_map() -> None:
    conn, _sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_trade(_ws_trade("T1", amount=1.0))
    assert "qubxBTCUSDT1" in conn._last_deal_by_cid  # type: ignore[attr-defined]
    conn._handle_ws_order(_ws_order(status="closed"))
    # Terminal eviction dropped both the order cache and the transient last-deal entry.
    assert "qubxBTCUSDT1" not in conn._last_deal_by_cid  # type: ignore[attr-defined]
    assert "qubxBTCUSDT1" not in conn._orders


# --------------------------------------------------------------------------- #
# OKX make_client_id sanitization
# --------------------------------------------------------------------------- #
def test_okx_make_client_id_strips_non_alphanumeric() -> None:
    conn, _sent, _ = _make_connector(OkxCcxtConnector)
    out = conn.make_client_id("qubx_BTC-USDT_1")
    assert out == "qubxBTCUSDT1"
    assert out.isalnum()


def test_okx_make_client_id_truncates_to_32() -> None:
    conn, _sent, _ = _make_connector(OkxCcxtConnector)
    long_suggested = "qubx_" + "A" * 60
    out = conn.make_client_id(long_suggested)
    assert len(out) == 32
    assert out.isalnum()
    assert out.startswith("qubxAAAA")


# --------------------------------------------------------------------------- #
# OKX snapshot balance extraction (cashBal / frozenBal from info.data[0].details)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_okx_snapshot_extracts_cashbal_balances() -> None:
    exchange = Mock()
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.fetch_positions = AsyncMock(return_value=[])
    # ccxt maps OKX eq -> total (1234), but cashBal (1000) is the cash leg we want.
    exchange.fetch_balance = AsyncMock(return_value={
        "total": {"USDT": 1234.0},
        "used": {"USDT": 100.0},
        "info": {"data": [{
            "totalEq": "1234.0",
            "details": [
                {"ccy": "USDT", "cashBal": "1000.0", "frozenBal": "100.0"},
                {"ccy": "BTC", "cashBal": "0", "frozenBal": "0"},  # zero -> skipped
            ],
        }]},
    })
    exchange.markets = {}
    conn, sent, _ = _make_connector(OkxCcxtConnector, exchange=exchange)

    conn.request_snapshot()
    for coro in conn._captured:  # type: ignore[attr-defined]
        await coro

    snap = sent[0].snapshot
    assert len(snap.balances) == 1  # zero-cashBal BTC skipped
    bal = snap.balances[0]
    assert bal.currency == "USDT"
    assert bal.total == 1000.0  # cashBal, NOT ccxt's eq-derived 1234
    assert bal.locked == 100.0
    assert bal.free == 900.0


# --------------------------------------------------------------------------- #
# Bitfinex two-stream: partial + filled
# --------------------------------------------------------------------------- #
def test_bitfinex_two_stream_partial_then_filled() -> None:
    conn, sent, _ = _make_connector(BitfinexCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_trade(_ws_trade("BT1", amount=1.0))
    conn._handle_ws_order(_ws_order(status="closed"))

    partials = [e for e in sent if isinstance(e, OrderPartiallyFilledEvent)]
    fills = [e for e in sent if isinstance(e, OrderFilledEvent)]
    assert len(partials) == 1
    assert len(fills) == 1
    assert partials[0].fill.trade_id == "BT1"
    assert fills[0].fill.trade_id == "BT1"


def test_bitfinex_uses_base_make_client_id_and_balances() -> None:
    # Bitfinex has no clOrdId/balance override beyond the base.
    conn, _sent, _ = _make_connector(BitfinexCcxtConnector)
    assert conn.make_client_id("qubx_BTC-USDT_1") == "qubx_BTC-USDT_1"  # base keeps underscore
    raw_balance = {"total": {"USDT": 50.0}, "used": {"USDT": 5.0}}
    balances = conn._convert_balances(raw_balance)
    assert len(balances) == 1
    assert balances[0].total == 50.0


# --------------------------------------------------------------------------- #
# get_ccxt_connector factory
# --------------------------------------------------------------------------- #
def _factory_kwargs() -> dict:
    em = Mock()
    em.exchange = Mock()
    return dict(
        channel=Mock(),
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=Mock(),
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("okx.f", OkxCcxtConnector),
        ("OKX.F", OkxCcxtConnector),  # case-insensitive
        ("okx", OkxCcxtConnector),  # bare alias
        ("bitfinex.f", BitfinexCcxtConnector),
        ("bitfinex", BitfinexCcxtConnector),  # bare alias
        ("binance.um", CcxtConnector),  # unlisted -> base
        ("hyperliquid", CcxtConnector),  # unlisted -> base
    ],
)
def test_get_ccxt_connector_resolves_subclass(name: str, expected: type) -> None:
    conn = get_ccxt_connector(name, **_factory_kwargs())
    assert type(conn) is expected
    assert conn.exchange_name == name
