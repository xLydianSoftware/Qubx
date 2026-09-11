"""Bybit connector account surface: wallet-balance figures, margin mode, reject causes.

Offline, mocked ccxt — no credentials or network.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import ccxt
import pytest

from qubx.connectors.ccxt.connector import CcxtConnector
from qubx.connectors.ccxt.exchanges.bybit.connector import BybitCcxtConnector
from qubx.core.basics import CtrlChannel, Instrument, MarketType, Position, RejectCause
from tests.qubx.core.utils_test import DummyTimeProvider

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"


# venue specs, inline: the global lookup ships no bybit symbols, so it resolves only from a
# machine that happens to have a cached bybit.f.json
_SPECS = {"BTCUSDT": (0.1, 0.001), "ETHUSDT": (0.01, 0.01)}


def _instrument(symbol: str = "BTCUSDT") -> Instrument:
    tick, lot = _SPECS[symbol]
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="BYBIT.F",
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=tick,
        lot_size=lot,
        min_size=lot,
        min_notional=5.0,
    )


def _position(symbol: str = "BTCUSDT", quantity: float = 1.0) -> Position:
    return Position(instrument=_instrument(symbol), quantity=quantity, pos_average_price=100.0)


def _make_connector(exchange: Mock | None = None) -> tuple[BybitCcxtConnector, list, Mock]:
    """(connector, sent_events, exchange) with ``_spawn``/``_run_sync`` driven in-test."""
    if exchange is None:
        exchange = Mock()
    exchange.has = {"editOrder": True}

    em = Mock()
    em.exchange = exchange
    em.rate_limiter = None

    sent: list = []
    channel = Mock(spec=CtrlChannel)
    channel.send = Mock(side_effect=lambda e: sent.append(e))

    conn = BybitCcxtConnector(
        exchange_name="BYBIT.F",
        channel=channel,
        time_provider=DummyTimeProvider(),
        exchange_manager=em,
        data_provider=Mock(),
    )
    captured: list = []
    conn._spawn = Mock(side_effect=lambda coro: captured.append(coro))
    conn._captured = captured  # type: ignore[attr-defined]
    conn._run_sync = Mock(side_effect=lambda coro, timeout=None: asyncio.new_event_loop().run_until_complete(coro))
    return conn, sent, exchange


async def _drive(conn: BybitCcxtConnector) -> None:
    for coro in conn._captured:  # type: ignore[attr-defined]
        await coro
    conn._captured.clear()  # type: ignore[attr-defined]


def _position_row(symbol: str = BTC, **info) -> dict:
    return {"symbol": symbol, "contracts": 1.0, "side": "long", "info": info}


def _wallet_balance(**overrides) -> dict:
    account = {
        "totalEquity": "18070.32797922",
        "accountIMRate": "0.0101",
        "totalMarginBalance": "18070.32797922",
        "totalInitialMargin": "182.60183684",
        "accountType": "UNIFIED",
        "totalAvailableBalance": "17887.72614237",
        "accountMMRate": "0.03",
        "totalPerpUPL": "-0.11001349",
        "totalWalletBalance": "18070.43799271",
        "totalMaintenanceMargin": "542.10",
    }
    account.update(overrides)
    return {"info": {"retCode": 0, "retMsg": "OK", "result": {"list": [account]}, "time": 1672125441042}}


def test_get_adl_level_reads_the_rank_off_the_venue_row():
    """No connector-side cache: ``BybitF.parse_position`` puts ``adl`` on the raw row (the
    unified key ``ccxt_convert_position`` already reads) and the base getter takes it there."""
    exchange = Mock()
    exchange.fetch_positions = AsyncMock(return_value=[{"symbol": BTC, "info": {"adl": 1, "adlRankIndicator": "2"}}])
    conn, _, _ = _make_connector(exchange)

    assert conn.get_adl_level(_instrument()) == 1
    exchange.fetch_positions.assert_awaited_once_with([BTC])


def test_a_flat_position_read_reports_no_adl_level():
    exchange = Mock()
    exchange.fetch_positions = AsyncMock(return_value=[])
    conn, _, _ = _make_connector(exchange)

    assert conn.get_adl_level(_instrument()) is None


@pytest.mark.asyncio
async def test_snapshot_stamps_the_account_margin_mode_on_every_position():
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "cross"})
    conn, _, _ = _make_connector(exchange)
    positions = [_position("BTCUSDT"), _position("ETHUSDT")]

    await conn._fill_margin_mode(positions)

    assert [p.margin_mode for p in positions] == ["cross", "cross"]
    exchange.fetch_margin_mode.assert_awaited_once_with(BTC)


@pytest.mark.asyncio
async def test_get_margin_mode_is_a_field_read_once_the_snapshot_has_run():
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "cross"})
    conn, _, _ = _make_connector(exchange)

    await conn._fill_margin_mode([_position()])
    exchange.fetch_margin_mode.reset_mock()

    assert conn.get_margin_mode(_instrument()) == "cross"
    exchange.fetch_margin_mode.assert_not_awaited()


@pytest.mark.parametrize(
    "venue_value,expected", [("cross", "cross"), ("isolated", "isolated"), ("portfolio", None), (None, None)]
)
def test_a_cold_get_margin_mode_still_reads_the_venue(venue_value, expected):
    """Nothing has filled the cache yet; PORTFOLIO_MARGIN has no framework equivalent -> None."""
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": venue_value})
    conn, _, _ = _make_connector(exchange)

    assert conn.get_margin_mode(_instrument()) == expected
    exchange.fetch_margin_mode.assert_awaited_once_with(BTC)


def test_get_margin_mode_survives_a_venue_error():
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(side_effect=ccxt.ExchangeError("boom"))
    conn, _, _ = _make_connector(exchange)

    assert conn.get_margin_mode(_instrument()) is None


def test_get_margin_mode_survives_a_blocking_call_that_never_returns():
    """A venue timeout is raised by _run_sync itself, past _read_margin_mode's own guard —
    it must not reach the strategy thread."""

    def _times_out(coro, timeout=None):
        coro.close()
        raise TimeoutError("venue read timed out")

    conn, _, _ = _make_connector()
    conn._run_sync = Mock(side_effect=_times_out)

    assert conn.get_margin_mode(_instrument()) is None
    assert conn._margin_mode is None


@pytest.mark.asyncio
async def test_a_warm_margin_mode_cache_costs_the_snapshot_no_venue_read():
    """The read shares ccxt's throttle with order placement; the mode is account-wide."""
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "cross"})
    conn, _, _ = _make_connector(exchange)

    await conn._fill_margin_mode([_position("BTCUSDT")])
    exchange.fetch_margin_mode.reset_mock()
    later = [_position("ETHUSDT")]
    await conn._fill_margin_mode(later)

    exchange.fetch_margin_mode.assert_not_awaited()
    assert later[0].margin_mode == "cross"


@pytest.mark.asyncio
async def test_the_snapshot_goes_back_to_the_venue_after_a_margin_mode_write():
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "isolated"})
    conn, _, _ = _make_connector(exchange)
    conn._margin_mode = "cross"
    with patch.object(CcxtConnector, "set_margin_mode", return_value=True):
        conn.set_margin_mode(_instrument(), "isolated")

    await conn._fill_margin_mode([_position()])

    exchange.fetch_margin_mode.assert_awaited_once_with(BTC)
    assert conn._margin_mode == "isolated"


@pytest.mark.asyncio
async def test_a_margin_mode_read_failure_leaves_the_positions_alone():
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(side_effect=ccxt.ExchangeError("boom"))
    conn, _, _ = _make_connector(exchange)
    position = _position()

    await conn._fill_margin_mode([position])

    assert position.margin_mode is None


@pytest.mark.asyncio
async def test_the_snapshot_hook_fills_the_margin_mode():
    """ADL is not filled here: BybitF.parse_position puts it on the row ccxt_convert_position
    already reads (see test_bybit_exchange)."""
    exchange = Mock()
    exchange.has = {"fetchLeverages": False}
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "isolated"})
    conn, _, _ = _make_connector(exchange)
    position = _position()

    await conn._fill_leverage_settings([position])

    assert position.margin_mode == "isolated"


def test_set_margin_mode_invalidates_the_cached_read():
    """A cached mode that outlives the write makes get_margin_mode echo the pre-set value."""
    conn = object.__new__(BybitCcxtConnector)
    conn.exchange_name = "BYBIT.F"
    conn._margin_mode = "isolated"
    with patch.object(CcxtConnector, "set_margin_mode", return_value=True):
        assert conn.set_margin_mode(_instrument(), "cross") is True
    assert conn._margin_mode is None


def test_a_refused_set_margin_mode_keeps_the_cache():
    conn = object.__new__(BybitCcxtConnector)
    conn.exchange_name = "BYBIT.F"
    conn._margin_mode = "isolated"
    with patch.object(CcxtConnector, "set_margin_mode", return_value=False):
        assert conn.set_margin_mode(_instrument(), "cross") is False
    assert conn._margin_mode == "isolated"


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("EC_PostOnlyWillTakeLiquidity", RejectCause.NOT_FILLABLE),
        ("EC_NoImmediateQtyToFill", RejectCause.NOT_FILLABLE),
        ("EC_CancelForNoFullFill", RejectCause.UNKNOWN),
        (None, RejectCause.UNKNOWN),
    ],
)
def test_an_async_rejection_carries_the_venue_reason(reason, expected):
    """Bybit refuses on the read path, so the cause cannot come from a raised ccxt error."""
    conn = object.__new__(BybitCcxtConnector)
    raw = {"info": {"rejectReason": reason} if reason is not None else {}}

    code, cause = conn._reject_details(raw)

    assert code == reason
    assert cause == expected


def _trade_row(**overrides) -> dict:
    """One ccxt-parsed row of /v5/execution/list."""
    row = {
        "id": "cdb53b54-ec42-5d97-923c-5da9ce1c1cdb",
        "order": "f6029fd8-a785-4374-bbdd-9d3cf13b0ef4",
        "timestamp": 1788885297101,
        "symbol": ETH,
        "side": "buy",
        "amount": 0.01,
        "price": 2490.1,
        "takerOrMaker": "taker",
        "fee": {"cost": 0.0249118, "currency": "USDT"},
    }
    row.update(overrides)
    return row
