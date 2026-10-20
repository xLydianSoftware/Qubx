from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from qubx.core.basics import Instrument, InstrumentChanges, InstrumentsLookup, MarketType, Order, Position
from qubx.core.mixins.instrument_updates import INSTRUMENT_CHANGES_EVENT, InstrumentUpdateManager
from tests.qubx.core.conftest import make_pm, real_handler_map


def _instrument(symbol: str = "BTCUSDT", **kw) -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="BINANCE.UM",
        base=symbol.removesuffix("USDT"),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=kw.pop("tick_size", 0.1),
        lot_size=kw.pop("lot_size", 0.001),
        min_size=kw.pop("min_size", 0.001),
        **kw,
    )


class _Lookup(InstrumentsLookup):
    def __init__(self):
        self.listeners = []

    def add_changes_listener(self, listener):
        self.listeners.append(listener)

    def remove_changes_listener(self, listener):
        self.listeners.remove(listener)


def _manager(universe, positions=None, orders=None, simulation=False):
    ctx = SimpleNamespace(
        instruments=universe,
        get_positions=lambda: positions or {},
        get_orders=lambda: orders or {},
        is_simulation=simulation,
        notifier=MagicMock(),
        strategy_name="test",
        register_handler=MagicMock(),
        post_event=MagicMock(),
    )
    strategy = MagicMock()
    provider = MagicMock()
    check_delistings = MagicMock()
    mgr = InstrumentUpdateManager(ctx, strategy, [provider], check_delistings)
    return mgr, ctx, strategy, provider, check_delistings


class TestUpdateFrom:
    def test_copies_venue_facts_and_reports_them(self):
        held = _instrument(tick_size=0.1, min_size=0.001)
        fresh = _instrument(tick_size=0.5, min_size=0.01, listing_id="new")
        assert held.update_from(fresh) == {
            "tick_size": (0.1, 0.5),
            "min_size": (0.001, 0.01),
            "listing_id": (None, "new"),
        }
        assert (held.tick_size, held.min_size, held.listing_id) == (0.5, 0.01, "new")

    def test_nothing_changed(self):
        assert _instrument().update_from(_instrument()) == {}

    def test_resets_cached_precisions(self):
        held = _instrument(tick_size=0.1, lot_size=0.001)
        assert (held.price_precision, held.size_precision) == (1, 3)
        held.update_from(_instrument(tick_size=0.01, lot_size=0.1))
        assert (held.price_precision, held.size_precision) == (2, 1)

    def test_refuses_another_instrument(self):
        with pytest.raises(ValueError):
            _instrument("BTCUSDT").update_from(_instrument("ETHUSDT"))

    def test_identity_is_untouched(self):
        held = _instrument()
        book = {held: 1}
        held.update_from(_instrument(tick_size=0.5, delisted_at=pd.Timestamp("2026-11-01")))
        assert book[_instrument()] == 1


class TestInstrumentUpdateManager:
    def test_updates_every_held_object_in_place(self):
        in_universe = _instrument(tick_size=0.1)
        on_position = _instrument(tick_size=0.1)  # - a connector-built twin
        on_order = _instrument(tick_size=0.1)
        order = MagicMock(spec=Order)
        order.instrument = on_order
        mgr, *_ = _manager([in_universe], {in_universe: Position(on_position)}, {"o1": order})

        [update] = mgr.apply(InstrumentChanges(updated=[_instrument(tick_size=0.5)]))

        assert update.instrument is in_universe
        assert update.changes == {"tick_size": (0.1, 0.5)}
        assert in_universe.tick_size == on_position.tick_size == on_order.tick_size == 0.5

    def test_not_held_instruments_are_ignored(self):
        mgr, _, strategy, provider, check_delistings = _manager([_instrument("BTCUSDT")])
        changes = InstrumentChanges(updated=[_instrument("ETHUSDT", tick_size=0.5)])

        assert mgr.apply(changes) == []

        strategy.on_instrument_update.assert_not_called()
        check_delistings.assert_not_called()
        provider.on_instruments_changed.assert_called_once_with(changes)

    def test_strategy_and_connectors_are_told(self):
        held = _instrument(min_size=0.001)
        mgr, ctx, strategy, provider, check_delistings = _manager([held])
        changes = InstrumentChanges(updated=[_instrument(min_size=0.01)], listed=[_instrument("NEWUSDT")])

        updates = mgr.apply(changes)

        provider.on_instruments_changed.assert_called_once_with(changes)
        strategy.on_instrument_update.assert_called_once_with(ctx, updates)
        ctx.notifier.notify_message.assert_called_once()
        check_delistings.assert_not_called()

    def test_new_delisting_date_runs_the_delisting_check(self):
        held = _instrument()
        mgr, *_, check_delistings = _manager([held])

        mgr.apply(InstrumentChanges(updated=[_instrument(delisted_at=pd.Timestamp("2026-11-01"))]))

        assert held.delisted_at == pd.Timestamp("2026-11-01")
        check_delistings.assert_called_once()

    def test_contract_size_change_resyncs_positions(self):
        held = _instrument(contract_size=1.0)
        position = Position(held)
        mgr, *_ = _manager([held], {held: position})

        mgr.apply(InstrumentChanges(updated=[_instrument(contract_size=10.0)]))

        assert position._qty_multiplier == 10.0

    def test_failing_strategy_callback_is_contained(self):
        mgr, _, strategy, *_ = _manager([_instrument()])
        strategy.on_instrument_update.side_effect = RuntimeError("boom")
        assert mgr.apply(InstrumentChanges(updated=[_instrument(tick_size=0.5)]))

    def test_listener_posts_to_the_strategy_thread_and_is_removed_on_stop(self):
        lookup = _Lookup()
        mgr, ctx, *_ = _manager([_instrument()])
        mgr.start(lookup)
        ctx.register_handler.assert_called_once_with(INSTRUMENT_CHANGES_EVENT, mgr._on_changes_event)

        changes = InstrumentChanges(updated=[_instrument(tick_size=0.5)])
        [listener] = lookup.listeners
        listener(changes)
        ctx.post_event.assert_called_once_with(INSTRUMENT_CHANGES_EVENT, changes)

        mgr.stop()
        assert lookup.listeners == []

    def test_event_name_is_a_valid_handler_name(self):
        pm = make_pm()
        pm._handlers = real_handler_map()
        pm._custom_scheduled_methods = {}
        pm._validate_handler_name(INSTRUMENT_CHANGES_EVENT, lambda ctx, payload: None)
