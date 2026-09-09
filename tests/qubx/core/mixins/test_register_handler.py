import inspect
from unittest.mock import MagicMock, patch

import pytest

from qubx.core.mixins.processing import ProcessingManager
from tests.qubx.core.conftest import make_pm, real_handler_map


def _pm_with_handlers() -> ProcessingManager:
    """make_pm plus everything the tuple path (__process_data -> _process_custom_event ->
    _run_strategy_pipeline) touches, so process_data can drive a registered handler the way
    post_event does.

    _custom_scheduled_methods has a mutable class-level default that make_pm's __new__ never
    shadows per-instance, so it is reassigned unconditionally — otherwise registrations bleed
    between tests. _handlers is the real map, so the shadow guard sees what production sees.
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
    # handed over, not copied: the handler must see the SAME object
    pm = _pm_with_handlers()
    payload = {"sources": ["a", "b"]}
    seen = []
    pm.register_handler("agg.sources", lambda ctx, p: seen.append(p))

    pm.process_data(None, "agg.sources", payload, False)  # what post_event(name, payload) enqueues

    assert len(seen) == 1
    assert seen[0] is payload


def test_scheduled_method_is_still_called_with_context_only():
    # the payload argument must not leak into the scheduled registry's (ctx) shape
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
    # separate registries, shared dispatch key: one branch would silently shadow the other
    pm = _pm_with_handlers()
    pm._custom_scheduled_methods["rate_limit_metrics"] = lambda ctx: None
    with pytest.raises(ValueError, match="already registered"):
        pm.register_handler("rate_limit_metrics", lambda ctx, payload: None)


def test_has_handler_covers_registered_handlers_only():
    # post_event gates on this, and a scheduled id would be called with the wrong arity
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
    # "fit"/"event"/"time"/"error"/"state_snapshot" are not data types, so the
    # `name in self._handlers` branch is the only thing guarding them
    pm = _pm_with_handlers()
    assert name in pm._handlers  # pins the real key, not a synthetic one
    with pytest.raises(ValueError):
        pm.register_handler(name, lambda ctx, payload: None)


def test_register_handler_rejects_name_whose_datatype_shadows_builtin_handler():
    # not a literal key of _handlers, but DataType.from_str resolves it to one -- the same
    # fallback __process_data performs, so such a handler would silently never fire
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
    # these have no _handle_* method and flow through _process_custom_event, which consults
    # the handler registries BEFORE __update_base_data: such a handler would silently swallow
    # live market data for the whole run
    pm = _pm_with_handlers()
    assert name not in pm._handlers  # exactly why the _handlers-key check is not enough
    with pytest.raises(ValueError, match="shadows the built-in data type"):
        pm.register_handler(name, lambda ctx, payload: None)


class _Strategy:
    # bound methods: inspect.signature already drops `self`
    def on_wakeup(self, ctx, payload):
        pass

    def on_tick(self, ctx):  # the schedule() shape -- must be rejected as a handler
        pass


def test_register_handler_rejects_a_handler_with_the_scheduled_arity():
    # dispatch catches the resulting TypeError and logs it, so without this guard the
    # strategy would look alive while every post is a no-op
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
    # best-effort check: a C callable exposing no signature must still register
    pm = _pm_with_handlers()
    handler = MagicMock()
    with patch.object(inspect, "signature", side_effect=ValueError("no signature")):
        pm.register_handler("agg.sources", handler)
    assert pm._event_handlers["agg.sources"] is handler


def test_register_handler_rejects_unparseable_name_with_clear_message():
    # DataType.from_str's own ValueError must be wrapped so it names the offending event
    pm = _pm_with_handlers()
    with pytest.raises(ValueError, match="quote\\(x\\)"):
        pm.register_handler("quote(x)", lambda ctx, payload: None)
