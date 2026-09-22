"""Bybit connector account surface: wallet-balance figures, margin mode, reject causes.

Offline, mocked ccxt — no credentials or network.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, Mock, patch

import ccxt
import pytest

from qubx.connectors.ccxt.connector import CcxtConnector, _LeverageInfo
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
        # above totalMarginBalance by the collateral discount on non-USDT holdings
        "totalEquity": "19250.5",
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


def test_venue_figures_match_the_base_arity():
    """The snapshot unpacks this tuple positionally, and a short one sinks it inside
    ``_do_request_snapshot``'s except — no snapshot is ever emitted."""
    conn, _, _ = _make_connector()
    expected = len(inspect.signature(CcxtConnector._extract_venue_figures).return_annotation.__args__)

    assert len(conn._extract_venue_figures(_wallet_balance())) == expected


def test_venue_figures_read_the_unified_account_block():
    conn, _, _ = _make_connector()

    equity, available, ratio, withdrawable, maint, initial = conn._extract_venue_figures(_wallet_balance())

    assert equity == 19250.5  # totalEquity, not the discounted totalMarginBalance
    assert available == 17887.72614237
    assert ratio == pytest.approx(1.0 / 0.03)  # accountMMRate is the reciprocal
    assert withdrawable is None
    assert maint == 542.10
    assert initial == 182.60183684


@pytest.mark.parametrize("missing", ["accountMMRate", "totalMaintenanceMargin", "totalInitialMargin"])
def test_venue_figures_survive_a_missing_field(missing: str):
    conn, _, _ = _make_connector()
    balance = _wallet_balance(**{missing: ""})

    assert len(conn._extract_venue_figures(balance)) == 6


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
@pytest.mark.asyncio
async def test_the_venue_mode_is_normalized(venue_value, expected):
    """PORTFOLIO_MARGIN has no framework equivalent -> None."""
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": venue_value})
    conn, _, _ = _make_connector(exchange)

    assert await conn._read_margin_mode(BTC) == expected


def test_a_cold_get_margin_mode_does_not_touch_the_venue():
    """A cold read answers None rather than blocking the strategy thread."""
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "cross"})
    conn, _, _ = _make_connector(exchange)

    assert conn.get_margin_mode(_instrument()) is None
    # asserted on the call, not by raising from it: the old getter swallowed every exception
    conn._run_sync.assert_not_called()


@pytest.mark.asyncio
async def test_a_warm_margin_mode_cache_costs_the_snapshot_no_venue_read():
    """Account-wide on a UTA, so one read serves every position in the snapshot."""
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
async def test_a_margin_mode_write_is_adopted_without_a_venue_read():
    """The getter is cache-only, so the write adopts the value itself."""
    exchange = Mock()
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "cross"})
    conn, _, _ = _make_connector(exchange)
    conn._margin_mode = "cross"
    with patch.object(CcxtConnector, "set_margin_mode", return_value=True):
        conn.set_margin_mode(_instrument(), "isolated")

    assert conn.get_margin_mode(_instrument()) == "isolated"

    await conn._fill_margin_mode([_position()])

    exchange.fetch_margin_mode.assert_not_awaited()
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


def test_set_margin_mode_adopts_the_written_value():
    conn = object.__new__(BybitCcxtConnector)
    conn.exchange_name = "BYBIT.F"
    conn._margin_mode = "isolated"
    with patch.object(CcxtConnector, "set_margin_mode", return_value=True):
        assert conn.set_margin_mode(_instrument(), "cross") is True
    assert conn._margin_mode == "cross"


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


def _ws_trade(order_link_id: str = "qubx_BTCUSDT_17894222596", **info) -> dict:
    """One ccxt parse_ws_trade row: the raw bybit execution frame rides on ``info``."""
    return {
        "info": {"orderLinkId": order_link_id, "execId": "e1", **info},
        "id": "e1",
        "order": "v-596",
        "symbol": BTC,
        "timestamp": 1789399260861,
        "amount": 0.8,
        "price": 11.0,
        "side": "buy",
        "takerOrMaker": "taker",
    }


def test_ws_trade_carries_the_client_order_id():
    """A fill can beat its own submit ack; without the cid it would resolve only by venue id and
    materialize a phantom EXTERNAL order that absorbs the fill."""
    conn, sent, _ = _make_connector()
    conn._instrument_for_symbol = Mock(return_value=_instrument())

    conn._handle_ws_trade(_ws_trade())

    assert len(sent) == 1
    assert sent[0].client_order_id == "qubx_BTCUSDT_17894222596"
    assert sent[0].venue_order_id == "v-596"
    assert sent[0].deal.amount == 0.8


def test_an_externally_placed_order_still_has_no_client_id():
    """bybit leaves orderLinkId empty for an order placed outside the framework — it must keep
    materializing as EXTERNAL rather than claiming a cid."""
    conn, sent, _ = _make_connector()
    conn._instrument_for_symbol = Mock(return_value=_instrument())

    conn._handle_ws_trade(_ws_trade(order_link_id=""))

    assert len(sent) == 1
    assert sent[0].client_order_id is None


# - bybit has no fetch_leverages, so the sweep's `configured` half always fails and must not then
#   speak for it: the write path adopts on every successful send, and the tier read's pagination
#   cap leaves most of the venue out of the sweep.
def _sweeping_exchange(tier_symbols: list[str]) -> Mock:
    exchange = Mock()
    exchange.fetch_leverages = AsyncMock(side_effect=ccxt.NotSupported("bybit fetchLeverages"))
    exchange.fetch_leverage_tiers = AsyncMock(return_value={s: [{"maxLeverage": 50}] for s in tier_symbols})
    return exchange


@pytest.mark.asyncio
async def test_adopted_leverage_survives_the_hourly_rebuild():
    """A wholesale rebuild would reset every symbol to None each hour, so a leverage we set
    successfully would be re-sent on the next universe change, forever."""
    conn, _, exchange = _make_connector(_sweeping_exchange([BTC]))
    # maximum differs from the sweep's value, so this also proves the merge takes fresh tier data
    conn._leverage_cache[BTC] = _LeverageInfo(configured=3.0, maximum=25, max_notional=1.0, margin_mode="cross")

    await conn._refresh_leverage_cache()

    # maximum from the tier read (succeeded) is refreshed; configured/max_notional/margin_mode
    # ride fetch_leverages (failed) and are preserved
    assert conn._leverage_cache[BTC] == _LeverageInfo(configured=3.0, maximum=50, max_notional=1.0, margin_mode="cross")


@pytest.mark.asyncio
async def test_the_hourly_sweep_re_reads_the_margin_mode():
    """Changeable from the venue UI, and nothing else invalidates it."""
    exchange = _sweeping_exchange([BTC])
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "isolated"})
    conn, _, _ = _make_connector(exchange)
    conn._margin_mode = "cross"

    await conn._refresh_leverage_cache()

    assert conn._margin_mode == "isolated"


@pytest.mark.asyncio
async def test_a_failed_margin_mode_re_read_keeps_the_cached_value():
    exchange = _sweeping_exchange([BTC])
    exchange.fetch_margin_mode = AsyncMock(side_effect=ccxt.ExchangeError("boom"))
    conn, _, _ = _make_connector(exchange)
    conn._margin_mode = "cross"

    await conn._refresh_leverage_cache()

    assert conn._margin_mode == "cross"


@pytest.mark.asyncio
async def test_a_switch_to_portfolio_margin_clears_the_cache():
    """A successful read that normalizes to None is not a failed read: keeping the old value
    would report `cross` for the life of the process after the UI switched."""
    exchange = _sweeping_exchange([BTC])
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "portfolio"})
    conn, _, _ = _make_connector(exchange)
    conn._margin_mode = "cross"

    await conn._refresh_leverage_cache()

    assert conn._margin_mode is None


@pytest.mark.asyncio
async def test_the_sweep_re_reads_off_the_universe_when_the_tier_read_gave_nothing():
    exchange = _sweeping_exchange([])
    exchange.fetch_margin_mode = AsyncMock(return_value={"marginMode": "isolated"})
    conn, _, _ = _make_connector(exchange)
    conn._symbol_to_instrument[ETH] = _instrument("ETHUSDT")

    await conn._refresh_leverage_cache()

    assert conn._margin_mode == "isolated"
    exchange.fetch_margin_mode.assert_awaited_once_with(ETH)


@pytest.mark.asyncio
async def test_a_symbol_outside_the_tier_sweep_is_not_evicted():
    """ccxt's risk-limit read stops at 750 symbols (alphabetically, at TQQQUSDT), so TRUMP/XRP/ZEC
    are never swept — a wholesale rebuild drops them."""
    conn, _, exchange = _make_connector(_sweeping_exchange([BTC]))
    conn._leverage_cache[ETH] = _LeverageInfo(configured=3.0, maximum=None, max_notional=None)

    await conn._refresh_leverage_cache()

    assert ETH in conn._leverage_cache
    assert conn._leverage_cache[ETH].configured == 3.0


@pytest.mark.asyncio
async def test_a_venue_that_reports_configured_still_rebuilds_wholesale():
    """The merge is only for venues whose configured read failed — binance must keep dropping a
    symbol the venue stopped reporting."""
    exchange = Mock()
    exchange.fetch_leverages = AsyncMock(return_value={BTC: {"symbol": BTC, "longLeverage": 5, "shortLeverage": 5}})
    exchange.fetch_leverage_tiers = AsyncMock(return_value={})
    conn, _, _ = _make_connector(exchange)
    conn._leverage_cache[ETH] = _LeverageInfo(configured=3.0, maximum=None, max_notional=None)

    await conn._refresh_leverage_cache()

    assert ETH not in conn._leverage_cache  # evicted, as before
    assert conn._leverage_cache[BTC].configured == 5
