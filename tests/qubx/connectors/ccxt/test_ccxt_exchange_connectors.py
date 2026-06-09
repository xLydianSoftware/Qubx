"""Unit tests for the per-exchange CcxtConnector subclasses (commit 4b).

Covers OKX/Bitfinex split orders/fills streams (status events with ``fill=None`` plus
one ``DealEvent`` per trade, with AM-level cross-stream convergence in both orderings),
OKX balance extraction + make_client_id, and the get_ccxt_connector factory. Mocked
ccxt — no credentials or network. Async work is driven deterministically via the
``_spawn``-capture pattern from the read tests.
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
    DealEvent,
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


def _ws_trade(
    trade_id: str = "T1", amount: float = 0.5, price: float = 100.0, venue_order_id: str = "VENUE123"
) -> dict:
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
# OKX two-stream: open -> accepted; trade -> DealEvent; status reports -> fill=None
# --------------------------------------------------------------------------- #
def test_okx_watch_orders_open_emits_accepted() -> None:
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderAcceptedEvent)
    assert ev.client_order_id == "qubxBTCUSDT1"
    assert ev.venue_order_id == "VENUE123"


def test_okx_partial_status_emits_status_only_event() -> None:
    # OKX watch_orders never carries a trade; a PARTIALLY_FILLED report becomes a
    # status-only OrderPartiallyFilledEvent (fill=None) — the deal rides the trade stream.
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    # Seed the venue ack via the prior open report (the realistic order: open precedes
    # the partial), so the partial status isn't mistaken for a first-seen fill.
    conn._handle_ws_order(_ws_order(status="open"))
    sent.clear()
    raw = _ws_order(status="open")
    raw["info"] = {"status": "PARTIALLY_FILLED"}
    conn._handle_ws_order(raw)
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderPartiallyFilledEvent)
    assert ev.client_order_id == "qubxBTCUSDT1"
    assert ev.venue_order_id == "VENUE123"
    assert ev.fill is None


def test_okx_watch_my_trades_emits_deal_event() -> None:
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    # Seed the venue->cid index via the open order so the trade resolves its cid.
    conn._handle_ws_order(_ws_order(status="open"))
    sent.clear()

    conn._handle_ws_trade(_ws_trade("T1", amount=0.5))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, DealEvent)
    assert ev.client_order_id == "qubxBTCUSDT1"
    assert ev.venue_order_id == "VENUE123"
    assert ev.deal.trade_id == "T1"
    assert ev.deal.amount == 0.5


def test_okx_filled_status_emits_status_only_filled() -> None:
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_trade(_ws_trade("T1", amount=1.0))
    sent.clear()

    # watch_orders reports the order FILLED (no trade) -> plain terminal status, no
    # stitching: the deal already rode the trade stream as a DealEvent.
    conn._handle_ws_order(_ws_order(status="closed"))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderFilledEvent)
    assert ev.client_order_id == "qubxBTCUSDT1"
    assert ev.fill is None


def test_okx_filled_status_without_prior_trade_still_emits() -> None:
    # FILLED is emitted even when no trade was seen yet (the trade stream may lag):
    # the terminal transition no longer depends on a remembered deal.
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    sent.clear()
    conn._handle_ws_order(_ws_order(status="closed"))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, OrderFilledEvent)
    assert ev.fill is None


def test_okx_trade_after_terminal_eviction_emits_deal_by_venue_id() -> None:
    # The FILLED status evicts the connector's order cache; a trade landing after that
    # can't resolve its cid but still rides as a DealEvent addressed by venue id — AM
    # resolves it through its own venue-id index.
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_order(_ws_order(status="closed"))
    sent.clear()
    conn._handle_ws_trade(_ws_trade("T1", amount=1.0))
    assert len(sent) == 1
    ev = sent[0]
    assert isinstance(ev, DealEvent)
    assert ev.client_order_id is None
    assert ev.venue_order_id == "VENUE123"
    assert ev.deal.trade_id == "T1"


def _seeded_am() -> SimulatedAccountManager:
    am = SimulatedAccountManager(
        connectors={"OKX.F": object()},
        base_currencies={"OKX.F": "USDT"},
        time=DummyTimeProvider(),
    )
    # Order must exist for the events to land on it: accepted events are resolve-only
    # (no materialization), so seed the framework order first.
    am.add_order(
        Order(
            client_order_id="qubxBTCUSDT1",
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=_instrument(),
            submitted_at=DummyTimeProvider().time(),
            quantity=1.0,
            price=100.0,
            side="BUY",
            status=OrderStatus.SUBMITTED,
            time_in_force="gtc",
        )
    )
    return am


def test_okx_split_streams_converge_in_am_deal_then_status() -> None:
    # Trade stream wins the race: the DealEvent books the fill, the later FILLED status
    # (fill=None) only drives the terminal transition. ONE fill total.
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_trade(_ws_trade("T1", amount=1.0, price=100.0))
    conn._handle_ws_order(_ws_order(status="closed"))

    deals = [e for e in sent if isinstance(e, DealEvent)]
    fills = [e for e in sent if isinstance(e, OrderFilledEvent)]
    assert len(deals) == 1 and len(fills) == 1
    assert fills[0].fill is None

    am = _seeded_am()
    for event in sent:
        am.apply(event)

    order = am.find_order_by_id("VENUE123")
    assert order is not None
    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == pytest.approx(1.0)
    assert am.get_position(_instrument()).quantity == pytest.approx(1.0)


def test_okx_split_streams_converge_in_am_status_then_deal() -> None:
    # Status stream wins the race: FILLED (fill=None) transitions first; the late trade
    # still books against the terminal-but-retained order. ONE fill total.
    conn, sent, _ = _make_connector(OkxCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_order(_ws_order(status="closed"))
    conn._handle_ws_trade(_ws_trade("T1", amount=1.0, price=100.0))

    deals = [e for e in sent if isinstance(e, DealEvent)]
    fills = [e for e in sent if isinstance(e, OrderFilledEvent)]
    assert len(deals) == 1 and len(fills) == 1

    am = _seeded_am()
    for event in sent:
        am.apply(event)

    order = am.find_order_by_id("VENUE123")
    assert order is not None
    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == pytest.approx(1.0)
    assert am.get_position(_instrument()).quantity == pytest.approx(1.0)


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
    exchange.fetch_balance = AsyncMock(
        return_value={
            "total": {"USDT": 1234.0},
            "used": {"USDT": 100.0},
            "info": {
                "data": [
                    {
                        "totalEq": "1234.0",
                        "details": [
                            {"ccy": "USDT", "cashBal": "1000.0", "frozenBal": "100.0"},
                            {"ccy": "BTC", "cashBal": "0", "frozenBal": "0"},  # zero -> skipped
                        ],
                    }
                ]
            },
        }
    )
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
# Bitfinex two-stream: deal + status-only filled
# --------------------------------------------------------------------------- #
def test_bitfinex_two_stream_deal_then_filled() -> None:
    conn, sent, _ = _make_connector(BitfinexCcxtConnector)
    conn._handle_ws_order(_ws_order(status="open"))
    conn._handle_ws_trade(_ws_trade("BT1", amount=1.0))
    conn._handle_ws_order(_ws_order(status="closed"))

    deals = [e for e in sent if isinstance(e, DealEvent)]
    fills = [e for e in sent if isinstance(e, OrderFilledEvent)]
    assert len(deals) == 1
    assert len(fills) == 1
    assert deals[0].deal.trade_id == "BT1"
    assert fills[0].fill is None


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
