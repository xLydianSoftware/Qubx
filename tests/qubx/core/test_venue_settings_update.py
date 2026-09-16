"""``VenueSettingsUpdate`` — the connector announces a venue setting, the AM applies it.

The write path is fire-and-forget and ``Position.leverage`` is otherwise written only by the
snapshot reconcile, so an operator's 5x->3x showed 5x to every reader of the position until a
snapshot happened to carry it.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx import logger
from qubx.core.account_manager import AccountManager
from qubx.core.basics import (
    VENUE_SETTINGS_EVENT,
    Instrument,
    MarketType,
    Position,
    VenueSettingsUpdate,
    create_venue_settings_event,
)
from qubx.core.state_snapshot import position_entry
from tests.qubx.core.conftest import make_pm, real_handler_map


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

    def test_our_own_ack_may_materialize_the_position(self):
        """The ack can beat the first snapshot, and it is by definition about an instrument we
        asked about — so it is the one source allowed to create state."""
        am = _am()
        instrument = _instrument()

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0, source="ack"))

        assert am.get_position(instrument).leverage == 3.0

    @pytest.mark.parametrize("source", ["sweep", "push", "snapshot"])
    def test_an_observation_never_grows_a_position(self, source):
        """A sweep reads whatever the venue reports, which on a shared account is every other
        bot's instruments. Applying one would put a position this bot does not hold into
        `ctx.positions` and the 5s snapshot."""
        am = _am()

        am.apply_venue_settings(VenueSettingsUpdate(_instrument(), leverage=3.0, source=source))

        assert am._states["binance"].get_positions() == {}

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

    @pytest.mark.parametrize("source", ["ack", "push", "sweep", "snapshot"])
    def test_every_source_applies_to_a_position_we_hold(self, source):
        am = _am()
        instrument = _instrument()
        pos = _held(am, instrument, leverage=5.0, max_notional=None)

        am.apply_venue_settings(VenueSettingsUpdate(instrument, leverage=3.0, source=source))

        assert pos.leverage == 3.0


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
