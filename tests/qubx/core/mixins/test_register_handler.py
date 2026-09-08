from unittest.mock import MagicMock

import pytest

from qubx.core.mixins.processing import ProcessingManager
from tests.qubx.core.conftest import make_pm, real_handler_map


def _pm_with_handlers() -> ProcessingManager:
    """make_pm plus everything the tuple path (__process_data -> _process_custom_event ->
    _run_strategy_pipeline) touches, mirroring _pm_for_tuple_path in test_processing_dispatch.py
    so process_data can drive a registered handler exactly the way post_event (Task 2) will.

    _custom_scheduled_methods carries a mutable class-level default (`{}` on the class body);
    make_pm's __new__ bypass of __init__ never shadows it per-instance, so leaving it alone
    (isinstance(..., dict) is already True) would mutate the SAME dict across every test/pm
    that doesn't set it explicitly -> duplicate-registration bleed between tests. Assign a
    fresh dict unconditionally instead.

    _handlers is the REAL map (real_handler_map), not a synthetic one, so both the dispatch
    fallback and the shadow guard see exactly what production sees.
    """
    pm = make_pm()
    pm._time_provider = MagicMock()
    pm._health_monitor = MagicMock()
    pm._subscription_manager = MagicMock()
    pm._cache = MagicMock()
    pm._strategy_name = "RegisterHandlerTest"
    pm._emitted_signals = []
    pm._data_throttler = None
    pm._handlers = real_handler_map()
    pm._custom_scheduled_methods = {}
    pm._pending_no_quote_signals = {}
    pm._fit_is_running = False
    pm._warmup_finished_is_running = False
    pm._position_tracker.update.return_value = []
    pm._context.instruments = []
    pm._context._strategy_state.is_on_start_called = True
    pm._context._strategy_state.is_warmup_in_progress = False
    pm._context._strategy_state.is_on_warmup_finished_called = True
    pm._context._strategy_state.is_on_fit_called = True
    pm._strategy.on_market_data.return_value = None
    return pm


def test_register_handler_runs_method_on_process_data():
    pm = _pm_with_handlers()
    calls = []
    pm.register_handler("agg.sources", lambda ctx: calls.append(ctx))

    pm.process_data(None, "agg.sources", None, False)  # what post_event enqueues

    assert calls == [pm._context]
    pm._strategy.on_event.assert_not_called()
    pm._strategy.on_market_data.assert_not_called()


def test_register_handler_rejects_duplicate_and_empty_name():
    pm = _pm_with_handlers()
    pm.register_handler("agg.sources", lambda ctx: None)
    with pytest.raises(ValueError):
        pm.register_handler("agg.sources", lambda ctx: None)
    with pytest.raises(ValueError):
        pm.register_handler("", lambda ctx: None)


def test_registered_handler_exception_is_logged_not_raised():
    pm = _pm_with_handlers()

    def boom(ctx):
        raise RuntimeError("x")

    pm.register_handler("agg.sources", boom)
    pm.process_data(None, "agg.sources", None, False)  # must not raise (same as scheduled methods)


@pytest.mark.parametrize("name", ["trade", "fit", "event", "time", "error", "state_snapshot"])
def test_register_handler_rejects_name_that_shadows_builtin_handler(name):
    # The `name in self._handlers` branch. "fit"/"event"/"time"/"error"/"state_snapshot" are
    # NOT data types, so this branch is the only thing guarding them; the real handler map is
    # used so renaming a `_handle_*` method breaks the test instead of quietly unguarding it.
    pm = _pm_with_handlers()
    assert name in pm._handlers  # pins the real key, not a synthetic one
    with pytest.raises(ValueError):
        pm.register_handler(name, lambda ctx: None)


def test_register_handler_rejects_name_whose_datatype_shadows_builtin_handler():
    # "ohlc(1h)" isn't a literal key of _handlers, but DataType.from_str resolves it to
    # DataType.OHLC, whose .value ("ohlc") IS a key -- the same fallback __process_data
    # performs, so a handler registered under this name would still silently never fire.
    pm = _pm_with_handlers()
    assert "ohlc" in pm._handlers
    with pytest.raises(ValueError):
        pm.register_handler("ohlc(1h)", lambda ctx: None)


@pytest.mark.parametrize(
    "name",
    [
        "funding_rate",
        "open_interest",
        "liquidation",
        "aggregated_liquidations",
        "record",
        "fundamental",
        "ohlc_quotes",
    ],
)
def test_register_handler_rejects_datatype_without_a_builtin_handler(name):
    # These data types have NO _handle_* method: they flow through _process_custom_event ->
    # __update_base_data. _process_custom_event consults _custom_scheduled_methods FIRST and
    # returns before __update_base_data, so a handler registered under one of these names
    # would WIN over live market data -- cache never updated, no MarketEvent, no data-arrival
    # health signal -- silently, for the whole run. The guard must reject any name that
    # resolves to a data type at all, not just the ones that happen to have a handler.
    pm = _pm_with_handlers()
    assert name not in pm._handlers  # exactly why the _handlers-key check is not enough
    with pytest.raises(ValueError, match="shadows the built-in data type"):
        pm.register_handler(name, lambda ctx: None)


def test_register_handler_rejects_unparseable_name_with_clear_message():
    # DataType.from_str raises its own ValueError ("unit abbreviation w/o a number") for a
    # name that looks like a parametrized subscription but isn't valid -- register_handler
    # must wrap it so the message points at the real problem (the name), not the parser.
    pm = _pm_with_handlers()
    with pytest.raises(ValueError, match="quote\\(x\\)"):
        pm.register_handler("quote(x)", lambda ctx: None)
