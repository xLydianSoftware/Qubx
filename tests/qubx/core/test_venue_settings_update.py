"""``VenueSettingsUpdate`` — the connector announces a venue setting, the AM applies it.

The write path is fire-and-forget and ``Position.leverage`` is otherwise written only by the
snapshot reconcile, so an operator's 5x->3x showed 5x to every reader of the position until a
snapshot happened to carry it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import ccxt
import numpy as np

from qubx import logger
from qubx.connectors.ccxt.connector import CcxtConnector, _LeverageInfo
from qubx.core.account_manager import AccountManager
from qubx.core.basics import (
    VENUE_SETTINGS_EVENT,
    CtrlChannel,
    Instrument,
    MarketType,
    Position,
    VenueSettingsUpdate,
    create_venue_settings_event,
)
from qubx.core.instrument_service import NullInstrumentService
from qubx.core.mixins.universe import UniverseManager
from qubx.core.state_snapshot import position_entry
from tests.qubx.core.conftest import make_pm, real_handler_map


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _T:
    def time(self):
        return np.datetime64("2026-09-16T00:00:00")


def _instrument(symbol: str = "BTCUSDT", exchange: str = "binance") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _am(exchanges=("binance",)) -> AccountManager:
    return AccountManager(
        connectors={ex: MagicMock() for ex in exchanges},
        base_currencies={ex: "USDT" for ex in exchanges},
        time=_T(),
    )


def _held(am: AccountManager, instrument: Instrument, *, leverage: float, max_notional: float | None) -> Position:
    pos = Position(instrument=instrument, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(np.datetime64("2026-09-16T00:00:00"), 50_000.0, 1.0)
    pos.leverage = leverage
    pos.max_notional = max_notional
    am._states[instrument.exchange].set_position(instrument, pos)
    return am.get_position(instrument)


class TestApply:
    def test_the_new_leverage_reaches_the_position(self):
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=1_000_000.0)

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0))

        assert pos.leverage == 3.0

    def test_a_leverage_change_drops_the_notional_cap(self):
        """Tiered venues move the cap with the leverage, so the held one is now wrong — and a
        wrong cap is worse than none. The next snapshot refills it."""
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=1_000_000.0)

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0))

        assert pos.max_notional is None

    def test_the_same_leverage_changes_nothing(self):
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=1_000_000.0)

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=5.0))

        assert pos.leverage == 5.0
        assert pos.max_notional == 1_000_000.0

    def test_the_margin_mode_is_applied_on_its_own(self):
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=1_000_000.0)

        am.apply_venue_settings(VenueSettingsUpdate(instrument, margin_mode="isolated"))

        assert pos.margin_mode == "isolated"
        assert pos.leverage == 5.0
        assert pos.max_notional == 1_000_000.0  # untouched: the leverage did not move

    def test_a_tracked_flat_instrument_is_applied_to(self):
        """Tracked means "has a position entry", flat included — which is every universe
        instrument, so our own writes always land."""
        am = _am()
        instrument = _instrument()
        assert am.get_position(instrument).quantity == 0.0  # materialize, as the universe does

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0))

        assert am.get_position(instrument).leverage == 3.0

    def test_an_untracked_instrument_never_grows_a_position(self):
        """A sweep reads whatever the venue reports, which is the whole venue. Applying one
        would put an instrument this bot does not trade into `ctx.positions` and the 5s
        snapshot."""
        am = _am()

        am.apply_venue_settings(VenueSettingsUpdate(_instrument(), leverage=3.0))

        assert am._states["binance"].get_positions() == {}
        assert am.get_positions() == {}

    def test_an_unmanaged_exchange_is_a_logged_no_op(self):
        """A connector must stamp the exchange key the AM holds its state under; a mismatch is
        warned about and dropped, so a silent one would be invisible to an integrator."""
        am = _am(exchanges=("binance",))
        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(m), level="WARNING")
        try:
            am.apply_venue_settings(VenueSettingsUpdate(_instrument(exchange="bybit"), leverage=3.0))
        finally:
            logger.remove(sink_id)

        assert any(m.record["level"].name == "WARNING" and "no account state" in m for m in messages)
        assert am.get_positions() == {}

    def test_a_held_position_is_applied_to(self):
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=None)

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0))

        assert pos.leverage == 3.0


class TestTrackedMeansUniverse:
    """The tracking guard is only safe because the universe seeds a position for everything it
    adds — driven through the real `UniverseManager`, not assumed."""

    def test_setting_the_universe_makes_its_instruments_tracked(self, mocker):
        am = _am()
        instrument = _instrument()
        market_data_manager = mocker.Mock()
        market_data_manager.is_instrument_listed.return_value = True
        market_data_manager.get_market_data_cache.return_value = mocker.Mock()
        delisting_detector = mocker.Mock()
        delisting_detector.filter_delistings.side_effect = lambda instruments: instruments
        delisting_detector.detect_delistings.return_value = []
        universe = UniverseManager(
            context=mocker.Mock(),
            strategy=mocker.Mock(),
            market_data_manager=market_data_manager,
            logging=mocker.Mock(),
            subscription_manager=mocker.Mock(),
            trading_manager=mocker.Mock(),
            time_provider=_T(),
            account=am,
            position_gathering=mocker.Mock(),
            delisting_detector=delisting_detector,
            instrument_service=NullInstrumentService(),
        )

        universe.set_universe([instrument])

        assert instrument in am._states["binance"].get_positions()
        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0))
        assert am.get_position(instrument).leverage == 3.0


class TestChannelDispatch:
    """The update rides the same (instrument, type, payload, historical) tuple protocol as an
    error, and ``ProcessingManager`` builds its handler map by reflection over ``_handle_*``."""

    @staticmethod
    def _pm(am: AccountManager):
        pm = make_pm(
            _account_manager=am,
            _handlers=real_handler_map(),
            _data_throttler=None,
            _time_provider=MagicMock(),
        )
        pm._context.emitter = None
        return pm

    def test_the_event_type_is_routed(self):
        assert VENUE_SETTINGS_EVENT in real_handler_map()

    def test_the_tuple_carries_no_instrument_and_is_not_historical(self):
        instrument = _instrument()
        event = create_venue_settings_event(VenueSettingsUpdate(instrument, leverage=3.0))

        assert event[0] is None
        assert event[1] == VENUE_SETTINGS_EVENT
        assert event[2].instrument is instrument
        assert event[3] is False

    def test_an_update_off_the_channel_reaches_the_position(self):
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=1_000_000.0)
        pm = self._pm(am)

        pm.process_data(*create_venue_settings_event(VenueSettingsUpdate(instrument, leverage=3.0)))

        assert pos.leverage == 3.0
        assert pos.max_notional is None

    def test_the_strategy_never_hears_about_it(self):
        """An accepted write is not an error — only a refusal is."""
        am = _am()
        pm = self._pm(am)

        pm.process_data(*create_venue_settings_event(VenueSettingsUpdate(_instrument(), leverage=3.0)))

        pm._strategy.on_error.assert_not_called()
        pm._position_gathering.on_error.assert_not_called()
        pm._strategy.on_market_data.assert_not_called()


class TestSweepEndToEnd:
    """A real ccxt connector's sweep into a real AccountManager.

    The connector announces every symbol the venue reports and the AM decides what to keep — the
    filter lives there because it is the side that knows what we track. The connector's own memo
    (`_symbol_to_instrument`) is written only by the order/deal/funding paths, so scoping the
    emit to it would skip exactly the case below.
    """

    @staticmethod
    def _connector(sent: list, *, symbol: str, venue_leverage: int, cached: int):
        exchange = Mock()
        exchange.fetch_leverages = AsyncMock(return_value={symbol: {"symbol": symbol, "longLeverage": venue_leverage}})
        exchange.fetch_leverage_tiers = AsyncMock(side_effect=ccxt.NotSupported("nope"))
        exchange.has = {"fetchLeverages": True}
        em = Mock()
        em.exchange = exchange
        em.rate_limiter = None
        channel = Mock(spec=CtrlChannel)
        channel.send = Mock(side_effect=sent.append)
        conn = CcxtConnector(
            exchange_name="binance",
            channel=channel,
            time_provider=Mock(),
            exchange_manager=em,
            data_provider=Mock(),
        )
        conn._leverage_cache[symbol] = _LeverageInfo(configured=cached, maximum=None)
        return conn

    @staticmethod
    def _pump(sent: list, am: AccountManager) -> None:
        pm = make_pm(
            _account_manager=am,
            _handlers=real_handler_map(),
            _data_throttler=None,
            _time_provider=MagicMock(),
        )
        pm._context.emitter = None
        for event in sent:
            pm.process_data(*event)

    def test_a_held_position_this_session_never_traded_still_gets_the_change(self):
        """The position came from the boot snapshot, so the connector's memo has never seen the
        symbol — and it is precisely the instrument whose leverage we must keep honest."""
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=1_000_000.0)
        sent: list = []
        conn = self._connector(sent, symbol="BTC/USDT:USDT", venue_leverage=3, cached=5)
        assert "BTC/USDT:USDT" not in conn._symbol_to_instrument

        with patch.object(CcxtConnector, "_instrument_for_symbol", return_value=instrument):
            run(conn._refresh_leverage_cache())
        self._pump(sent, am)

        assert pos.leverage == 3.0
        assert pos.max_notional is None

    def test_a_symbol_we_do_not_track_changes_nothing(self):
        """The sweep announces every symbol the venue reports; the AM drops the ones it does not
        track rather than growing a position for them."""
        am = _am()
        other = _instrument("DOGEUSDT")
        sent: list = []
        conn = self._connector(sent, symbol="DOGE/USDT:USDT", venue_leverage=3, cached=5)

        with patch.object(CcxtConnector, "_instrument_for_symbol", return_value=other):
            run(conn._refresh_leverage_cache())
        assert len(sent) == 1  # announced, not suppressed

        self._pump(sent, am)

        assert am.get_positions() == {}


class TestSnapshotEntry:
    def test_the_state_snapshot_reflects_the_new_leverage_immediately(self):
        """What the whole path exists for: the 5s snapshot reads the Position, so the operator
        sees 3x on the next tick rather than at the next snapshot reconcile."""
        am = _am()
        instrument = _instrument()
        _held(am, instrument, leverage=5.0, max_notional=1_000_000.0)
        am._connectors["binance"].get_max_instrument_leverage.return_value = 125.0
        am._connectors["binance"].get_max_instrument_notional.return_value = float("inf")

        assert position_entry(am, instrument, am.get_position(instrument))["instrument_leverage"] == 5.0

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0))

        entry = position_entry(am, instrument, am.get_position(instrument))
        assert entry["instrument_leverage"] == 3.0
        # the cap went with it, and the connector has none to fall back on
        assert entry["max_notional"] is None
