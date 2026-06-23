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
from qubx.connectors.ccxt.utils import ccxt_convert_order_info
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Instrument, MarketType, Order, OrderOrigin, OrderStatus, classify_origin
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
# OKX cid origin classification (cid_framework_prefix = venue-sanitized "qubx")
# --------------------------------------------------------------------------- #
def test_okx_cid_framework_prefix_matches_make_client_id_output() -> None:
    # Producer/classifier coherence pin: the prefix is derived with the SAME sanitizing
    # regex make_client_id applies, so whatever cid the producer sends to the venue
    # classifies RECOVERED when echoed back — they can never drift.
    conn, _sent, _ = _make_connector(OkxCcxtConnector)
    produced = conn.make_client_id("qubx_BTCUSDT_1")
    assert produced.startswith(OkxCcxtConnector.cid_framework_prefix)
    assert classify_origin(produced, framework_prefix=conn.cid_framework_prefix) is OrderOrigin.RECOVERED


def test_okx_order_parse_classifies_sanitized_cid_as_recovered() -> None:
    # An OKX-echoed framework cid ("qubx_" with the venue-banned "_" stripped) must read
    # RECOVERED through the connector's venue-aware prefix — the default "qubx_" check
    # would misclassify it as EXTERNAL.
    order = ccxt_convert_order_info(
        _instrument(), _ws_order(status="open"), framework_prefix=OkxCcxtConnector.cid_framework_prefix
    )
    assert order.client_order_id == "qubxBTCUSDT1"
    assert order.origin is OrderOrigin.RECOVERED


# --------------------------------------------------------------------------- #
# OKX snapshot balance extraction (cashBal / frozenBal from info.data[0].details)
# --------------------------------------------------------------------------- #
def _snapshot_exchange(raw_balance: dict | list) -> Mock:
    exchange = Mock()
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.fetch_positions = AsyncMock(return_value=[])
    exchange.fetch_balance = AsyncMock(return_value=raw_balance)
    exchange.markets = {}
    return exchange


async def _snapshot_from(conn: CcxtConnector, sent: list):
    conn.request_snapshot()
    for coro in conn._captured:  # type: ignore[attr-defined]
        await coro
    return sent[0].snapshot


@pytest.mark.asyncio
async def test_okx_snapshot_extracts_cashbal_balances() -> None:
    # ccxt maps OKX eq -> total (1234), but cashBal (1000) is the cash leg we want.
    exchange = _snapshot_exchange(
        {
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
    conn, sent, _ = _make_connector(OkxCcxtConnector, exchange=exchange)

    snap = await _snapshot_from(conn, sent)
    assert len(snap.balances) == 1  # zero-cashBal BTC skipped
    bal = snap.balances[0]
    assert bal.currency == "USDT"
    assert bal.total == 1000.0  # cashBal, NOT ccxt's eq-derived 1234
    assert bal.locked == 100.0
    assert bal.free == 900.0


# --------------------------------------------------------------------------- #
# OKX venue account figures (totalEq / mgnRatio / adjEq − imr from info.data[0])
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_okx_snapshot_extracts_venue_figures_multi_ccy() -> None:
    # Multi-currency margin mode: all account-level figures populated.
    exchange = _snapshot_exchange(
        {
            "total": {"USDT": 40000.0, "BTC": 0.1},
            "used": {"USDT": 1200.0, "BTC": 0.0},
            "info": {
                "data": [
                    {
                        "totalEq": "50000.5",
                        "adjEq": "49000.0",
                        "imr": "1200.0",
                        "mgnRatio": "35.5",
                        "details": [
                            {"ccy": "USDT", "cashBal": "40000.0", "frozenBal": "1200.0"},
                            {"ccy": "BTC", "cashBal": "0.1", "frozenBal": "0"},
                        ],
                    }
                ]
            },
        }
    )
    conn, sent, _ = _make_connector(OkxCcxtConnector, exchange=exchange)

    snap = await _snapshot_from(conn, sent)
    assert snap.equity == 50000.5  # totalEq (USD-denominated, USD~USDT)
    assert snap.available_margin == pytest.approx(47800.0)  # adjEq - imr
    assert snap.margin_ratio == 35.5  # mgnRatio
    assert snap.withdrawable is None  # max-withdrawal lives on a separate OKX endpoint

    # End-to-end: the snapshot lands in AM and the venue figures win over derived metrics.
    am = SimulatedAccountManager(
        connectors={"OKX.F": object()},
        base_currencies={"OKX.F": "USDT"},
        time=DummyTimeProvider(),
    )
    am.apply(sent[0])
    assert am.get_total_capital("OKX.F") == 50000.5
    assert am.get_available_margin("OKX.F") == pytest.approx(47800.0)
    assert am.get_margin_ratio("OKX.F") == 35.5
    # no venue withdrawable -> falls back to the (venue-preferred) available figure
    assert am.get_withdrawable_balance("OKX.F") == pytest.approx(47800.0)


@pytest.mark.asyncio
async def test_okx_snapshot_single_ccy_empty_fields_yield_none() -> None:
    # Outside multi-currency margin mode OKX sends "" for adjEq/imr/mgnRatio:
    # those degrade to None (AM derives), while totalEq still carries equity.
    exchange = _snapshot_exchange(
        {
            "total": {"BTC": 0.049},
            "used": {"BTC": 0.0},
            "info": {
                "data": [
                    {
                        "totalEq": "1918.55678",
                        "adjEq": "",
                        "imr": "",
                        "mgnRatio": "",
                        "details": [{"ccy": "BTC", "cashBal": "0.049", "frozenBal": "0"}],
                    }
                ]
            },
        }
    )
    conn, sent, _ = _make_connector(OkxCcxtConnector, exchange=exchange)

    snap = await _snapshot_from(conn, sent)
    assert snap.equity == 1918.55678
    assert snap.available_margin is None
    assert snap.margin_ratio is None
    assert snap.withdrawable is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "info",
    [
        {"data": []},  # ccxt's safe_dict(data, 0, {}) sanctions this shape
        {},  # data key missing
        None,  # info itself absent/null
        {"data": [None]},  # non-dict first element
    ],
)
async def test_okx_snapshot_survives_malformed_balance_payload(info) -> None:
    # The balance extractors run outside _snapshot_async's per-leg isolation: a raise
    # here would suppress the whole snapshot (orders/positions legs included). Malformed
    # info.data must degrade to empty balances / all-None figures, not IndexError.
    exchange = _snapshot_exchange({"total": {}, "used": {}, "info": info})
    conn, sent, _ = _make_connector(OkxCcxtConnector, exchange=exchange)

    snap = await _snapshot_from(conn, sent)
    assert snap.balances == []
    assert snap.equity is None
    assert snap.available_margin is None
    assert snap.margin_ratio is None
    assert snap.withdrawable is None
    # The good legs still made it into the snapshot.
    assert snap.open_orders == []
    assert snap.positions == []


@pytest.mark.asyncio
async def test_base_snapshot_extracts_binance_figures_incl_withdrawable() -> None:
    # Base extractor maps the Binance fapi account payload (v2 and v3 both carry these
    # top-level): totalMarginBalance -> equity, availableBalance -> available_margin,
    # maxWithdrawAmount -> withdrawable; Binance reports no margin ratio -> None.
    exchange = _snapshot_exchange(
        {
            "total": {"USDT": 111.02},
            "used": {"USDT": 0.0},
            "info": {
                "totalMarginBalance": "111.02007243",
                "availableBalance": "11.39894857",
                "maxWithdrawAmount": "10.50000000",
            },
        }
    )
    conn, sent, _ = _make_connector(CcxtConnector, exchange=exchange)

    snap = await _snapshot_from(conn, sent)
    assert snap.equity == 111.02007243
    assert snap.available_margin == 11.39894857
    assert snap.margin_ratio is None
    assert snap.withdrawable == 10.5


@pytest.mark.asyncio
async def test_okx_snapshot_order_round_trips_as_recovered() -> None:
    # End-to-end: OKX echoes a framework cid with "_" stripped; the snapshot leg
    # classifies it RECOVERED (venue-aware prefix) and AM's reconcile trusts the
    # producer-assigned origin — materializing keep-cid instead of burying the
    # strategy's own order under a synthetic ext:<vid>.
    exchange = _snapshot_exchange({})
    exchange.fetch_open_orders = AsyncMock(return_value=[_ws_order(status="open")])
    conn, sent, _ = _make_connector(OkxCcxtConnector, exchange=exchange)

    snap = await _snapshot_from(conn, sent)
    assert snap.open_orders[0].client_order_id == "qubxBTCUSDT1"
    assert snap.open_orders[0].origin is OrderOrigin.RECOVERED

    am = SimulatedAccountManager(
        connectors={"OKX.F": object()},
        base_currencies={"OKX.F": "USDT"},
        time=DummyTimeProvider(),
    )
    am.apply(sent[0])
    order = am.get_order("qubxBTCUSDT1")
    assert order is not None
    assert order.origin is OrderOrigin.RECOVERED
    assert order.client_order_id == "qubxBTCUSDT1"


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


@pytest.mark.asyncio
async def test_bitfinex_snapshot_venue_figures_pinned_all_none() -> None:
    # Pins the deliberate contract: Bitfinex's fetch_balance info is the raw wallets
    # LIST (no account-level figures exist in this payload) -> all-None venue figures,
    # while the unified total/used maps still parse into balances.
    exchange = _snapshot_exchange(
        {
            "total": {"USDT": 50.0},
            "used": {"USDT": 5.0},
            "free": {"USDT": 45.0},
            "info": [["margin", "UST", 50.0, 0, 45.0]],
        }
    )
    conn, sent, _ = _make_connector(BitfinexCcxtConnector, exchange=exchange)

    snap = await _snapshot_from(conn, sent)
    assert snap.equity is None
    assert snap.available_margin is None
    assert snap.margin_ratio is None
    assert snap.withdrawable is None
    assert len(snap.balances) == 1
    assert snap.balances[0].currency == "USDT"
    assert snap.balances[0].total == 50.0
    assert snap.balances[0].locked == 5.0


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


def test_create_ccxt_connector_binance_pm_reports_canonical_exchange(monkeypatch) -> None:
    """R13: the registered factory keeps the venue name (BINANCE.PM) for credentials and
    the ccxt exchange class, but the connector self-reports the canonical exchange its
    instruments carry (BINANCE.UM) — so the events it stamps route to the right AM state."""
    from qubx.connectors.ccxt import factory as factory_module
    from qubx.connectors.ccxt.factory import create_ccxt_connector

    em = Mock()
    em.exchange = Mock()
    mock_get_em = Mock(return_value=em)
    monkeypatch.setattr(factory_module, "get_ccxt_exchange_manager", mock_get_em)

    credentials = Mock()
    credentials.get_exchange_credentials.return_value = Mock(testnet=False, api_key="k", secret="s", model_extra=None)

    conn = create_ccxt_connector(
        exchange_name="BINANCE.PM",
        time_provider=DummyTimeProvider(),
        channel=Mock(),
        credentials=credentials,
        data_provider=Mock(),
        health_monitor=Mock(),
        loop=Mock(),
    )

    assert conn.exchange_name == "BINANCE.UM"
    credentials.get_exchange_credentials.assert_called_once_with("BINANCE.PM")
    assert mock_get_em.call_args.kwargs["exchange"] == "BINANCE.PM"
