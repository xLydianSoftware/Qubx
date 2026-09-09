import inspect
from unittest.mock import MagicMock, patch

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
    fresh dict unconditionally instead. _event_handlers (the register_handler registry) has
    no class-level default at all, so it can't bleed -- make_pm seeds it per-instance.

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
    pm.register_handler("agg.sources", lambda ctx, payload: calls.append((ctx, payload)))

    pm.process_data(None, "agg.sources", None, False)  # what post_event() enqueues

    assert calls == [(pm._context, None)]
    pm._strategy.on_event.assert_not_called()
    pm._strategy.on_market_data.assert_not_called()


def test_registered_handler_receives_the_posted_payload_object():
    # post_event(name, payload) hands the object over untouched: the handler must see the
    # SAME object (identity, not a copy/repr), which is what makes "the poster must not
    # mutate it afterwards" a meaningful contract.
    pm = _pm_with_handlers()
    payload = {"sources": ["a", "b"]}
    seen = []
    pm.register_handler("agg.sources", lambda ctx, p: seen.append(p))

    pm.process_data(None, "agg.sources", payload, False)  # what post_event(name, payload) enqueues

    assert len(seen) == 1
    assert seen[0] is payload


def test_scheduled_method_is_still_called_with_context_only():
    # The two registries dispatch differently: a scheduled/delayed method keeps the
    # single-argument shape it has always had -- adding the payload arg to register_handler
    # must not leak into it (every existing strategy's scheduled callback would break).
    pm = _pm_with_handlers()
    calls = []
    pm._custom_scheduled_methods["custom_schedule_x"] = lambda ctx: calls.append(ctx)

    pm.process_data(None, "custom_schedule_x", None, False)

    assert calls == [pm._context]


def test_register_handler_rejects_duplicate_and_empty_name():
    pm = _pm_with_handlers()
    pm.register_handler("agg.sources", lambda ctx, payload: None)
    with pytest.raises(ValueError):
        pm.register_handler("agg.sources", lambda ctx, payload: None)
    with pytest.raises(ValueError):
        pm.register_handler("", lambda ctx, payload: None)


def test_register_handler_rejects_a_name_already_used_by_a_scheduled_method():
    # The registries are separate but the DISPATCH KEY is shared (both are matched against
    # the event type in _process_custom_event), so a name may live in only one of them --
    # otherwise the event-handler branch would silently shadow the scheduled method.
    pm = _pm_with_handlers()
    pm._custom_scheduled_methods["rate_limit_metrics"] = lambda ctx: None
    with pytest.raises(ValueError, match="already registered"):
        pm.register_handler("rate_limit_metrics", lambda ctx, payload: None)


def test_has_handler_covers_registered_handlers_only():
    # post_event() gates on has_handler, and only an event handler can consume a payload:
    # a scheduled/delayed id must NOT be postable (it would be called with the wrong arity).
    pm = _pm_with_handlers()
    pm.register_handler("agg.sources", lambda ctx, payload: None)
    pm._custom_scheduled_methods["custom_schedule_x"] = lambda ctx: None

    assert pm.has_handler("agg.sources")
    assert not pm.has_handler("custom_schedule_x")
    assert not pm.has_handler("agg.sorces")


def test_registered_handler_exception_is_logged_not_raised():
    pm = _pm_with_handlers()

    def boom(ctx, payload):
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
        pm.register_handler(name, lambda ctx, payload: None)


def test_register_handler_rejects_name_whose_datatype_shadows_builtin_handler():
    # "ohlc(1h)" isn't a literal key of _handlers, but DataType.from_str resolves it to
    # DataType.OHLC, whose .value ("ohlc") IS a key -- the same fallback __process_data
    # performs, so a handler registered under this name would still silently never fire.
    pm = _pm_with_handlers()
    assert "ohlc" in pm._handlers
    with pytest.raises(ValueError):
        pm.register_handler("ohlc(1h)", lambda ctx, payload: None)


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
    # __update_base_data. _process_custom_event consults the handler registries FIRST and
    # returns before __update_base_data, so a handler registered under one of these names
    # would WIN over live market data -- cache never updated, no MarketEvent, no data-arrival
    # health signal -- silently, for the whole run. The guard must reject any name that
    # resolves to a data type at all, not just the ones that happen to have a handler.
    pm = _pm_with_handlers()
    assert name not in pm._handlers  # exactly why the _handlers-key check is not enough
    with pytest.raises(ValueError, match="shadows the built-in data type"):
        pm.register_handler(name, lambda ctx, payload: None)


class _Strategy:
    """Bound-method handler: inspect.signature already drops `self`, so (ctx, payload) here
    is the same two-parameter shape as a module-level function."""

    def on_wakeup(self, ctx, payload):
        pass

    def on_tick(self, ctx):  # the schedule() shape -- must be rejected as a handler
        pass


def test_register_handler_rejects_a_handler_with_the_scheduled_arity():
    # `def _on_wakeup(self, ctx)` is the shape of the sibling schedule()/delay() API, so it
    # is the easy mistake to make. Dispatch CATCHES the resulting TypeError and logs it, so
    # without this guard the strategy would look alive while every post is a no-op -- the
    # same failure mode validate_account_callback_signatures exists to prevent.
    pm = _pm_with_handlers()
    with pytest.raises(ValueError, match=r"must accept \(ctx, payload\)"):
        pm.register_handler("agg.sources", lambda ctx: None)
    assert "agg.sources" not in pm._event_handlers  # rejected before the registry write

    with pytest.raises(ValueError, match=r"must accept \(ctx, payload\)"):
        pm.register_handler("agg.bound", _Strategy().on_tick)


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda ctx, payload: None, id="two-positional"),
        pytest.param(lambda ctx, payload=None: None, id="payload-defaulted"),
        pytest.param(lambda ctx=None, payload=None: None, id="both-defaulted"),
        pytest.param(lambda *args: None, id="var-positional"),
        pytest.param(lambda ctx, *args, **kw: None, id="one-then-var-positional"),
        pytest.param(_Strategy().on_wakeup, id="bound-method"),
        pytest.param(MagicMock(), id="magicmock"),  # (*args, **kwargs) -- tests must stay writable
    ],
)
def test_register_handler_accepts_every_callable_shape_dispatch_can_call(handler):
    pm = _pm_with_handlers()
    pm.register_handler("agg.sources", handler)
    assert pm._event_handlers["agg.sources"] is handler


def test_register_handler_skips_the_arity_check_for_a_signature_less_callable():
    # Some C callables expose no signature at all; the guard must not turn that into a
    # registration failure (it is a best-effort check, not a gate on exotic callables).
    pm = _pm_with_handlers()
    handler = MagicMock()
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        pm.register_handler("agg.sources", handler)
    assert pm._event_handlers["agg.sources"] is handler


def test_register_handler_rejects_unparseable_name_with_clear_message():
    # DataType.from_str raises its own ValueError ("unit abbreviation w/o a number") for a
    # name that looks like a parametrized subscription but isn't valid -- register_handler
    # must wrap it so the message points at the real problem (the name), not the parser.
    pm = _pm_with_handlers()
    with pytest.raises(ValueError, match="quote\\(x\\)"):
        pm.register_handler("quote(x)", lambda ctx, payload: None)
