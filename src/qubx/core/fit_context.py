"""FitContext — the ctx surface handed to strategy.on_fit in thread mode.

When ``live.fit_executor: thread`` is active, ``strategy.on_fit`` receives a
:class:`FitContext` wrapping the real StrategyContext instead of the context itself.
The ENTIRE fit-thread concurrency policy lives here, in four explicit classes of
members (see the classification blocks below):

- **passthrough-read** — delegated to the real context unchanged (scalars, fresh
  lists, services, thread-safe readers).
- **copy-read** — delegated, but the live dict/list views come back as shallow
  copies (dict-size-changed guard), and OHLCV reads come back as locked clones via
  the cache's explicit ``snapshot=True`` path.
- **recorded-mutation** — recorded on the :class:`FitCycleState` and replayed by the
  ProcessorThread when the FitCommit is applied (single-mutator invariant: the fit
  thread computes and records; every ctx mutation happens on the ProcessorThread).
  A replayed universe change's warmup runs on the SubscriptionManager's background
  worker (deferred commit), so the FitCommit apply stays fast and the added
  instruments' subscriptions go live only after their history landed in the cache.
- **DENIED** — raises :class:`NotImplementedError` when called: members that mutate
  live context state with no meaningful deferred semantics (direct trading, venue
  configuration, framework lifecycle/plumbing).

Fail-closed: any member NOT classified above raises
:class:`UnclassifiedFitContextAccess` on access — a future IStrategyContext addition
must be classified here explicitly, it can never silently pass through.

Simulation and inline mode never construct a FitContext.
"""

import uuid
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

from qubx.core.basics import Instrument, ITimeProvider, Signal
from qubx.core.fit_executor import FitCycleState
from qubx.core.helpers import process_schedule_spec

if TYPE_CHECKING:
    from qubx.core.basics import OrderOrigin
    from qubx.core.interfaces import IStrategyContext

_UNCLASSIFIED_MSG = (
    "'{name}' is not available inside a threaded on_fit: it is not classified on FitContext. "
    "If the strategy genuinely needs it during the fit, classify it in FitContext as a "
    "passthrough-read, copy-read, recorded-mutation, or denied member."
)

_DENIED_MSG = (
    "'{name}' is not allowed inside a threaded on_fit{hint}: it mutates live context state "
    "that only the ProcessorThread may touch. Run it outside on_fit, or express the intent "
    "through one of the recorded mutations (set_universe/subscribe/schedule/emit_signal/...)."
)


class UnclassifiedFitContextAccess(NotImplementedError, AttributeError):
    """Unclassified attribute access on FitContext.

    Subclasses NotImplementedError (fail-loudly contract) AND AttributeError so that
    duck-typing probes (``hasattr`` / ``getattr(ctx, name, default)``) degrade to
    "absent" instead of crashing the prober.
    """


def _denied(name: str, hint: str = "") -> Callable[..., Any]:
    _hint = f" ({hint})" if hint else ""

    def _raise(self: "FitContext", *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_DENIED_MSG.format(name=name, hint=_hint))

    _raise.__name__ = name
    _raise.__qualname__ = f"FitContext.{name}"
    return _raise


class _FitAccountView:
    """Fit-thread view of ``ctx.account`` (IAccountViewer): the live dict/list-returning
    accessors return shallow copies; every other member is a read and passes through."""

    def __init__(self, account: Any) -> None:
        self._account = account

    def get_positions(self, exchange: str | None = None) -> dict:
        return dict(self._account.get_positions(exchange))

    @property
    def positions(self) -> dict:
        return dict(self._account.positions)

    def get_orders(self, *args: Any, **kwargs: Any) -> dict:
        return dict(self._account.get_orders(*args, **kwargs))

    def get_balances(self, exchange: str | None = None) -> list:
        return list(self._account.get_balances(exchange))

    def get_leverages(self, exchange: str | None = None) -> dict:
        return dict(self._account.get_leverages(exchange))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._account, name)


class FitContext(ITimeProvider):
    """Proxy over the real StrategyContext for the StrategyFitThread (see module doc).

    Inherits ITimeProvider so ``instrument.signal(ctx, ...)`` / ``instrument.target(ctx,
    ...)`` recognize it as a time source (their ``isinstance(time, ITimeProvider)``
    check); ``time`` itself is a passthrough-read.
    """

    def __init__(self, context: "IStrategyContext", fit_state: FitCycleState) -> None:
        self._context = context
        self._fit_state = fit_state
        # concrete MarketManager (always constructed by StrategyContext) — carries the
        # explicit snapshot=True read paths for ohlc/ohlc_pd/quote
        self._market_manager = context._market_data_provider  # type: ignore[attr-defined]
        self._account_view = _FitAccountView(context.account)

    # ------------------------------------------------------------------
    # recorded mutations — recorded on FitCycleState, replayed at FitCommit
    # ------------------------------------------------------------------

    def set_universe(
        self,
        instruments: list[Instrument],
        skip_callback: bool = False,
        if_has_position_then: Literal["close", "wait_for_close", "wait_for_change"] = "close",
    ) -> None:
        assert if_has_position_then in (
            "close",
            "wait_for_close",
            "wait_for_change",
        ), "Invalid if_has_position_then policy"
        self._fit_state.record(partial(self._context.set_universe, instruments, skip_callback, if_has_position_then))

    def add_instruments(self, instruments: list[Instrument]) -> None:
        self._fit_state.record(partial(self._context.add_instruments, instruments))

    def remove_instruments(
        self,
        instruments: list[Instrument],
        if_has_position_then: Literal["close", "wait_for_close", "wait_for_change"] = "close",
    ) -> None:
        self._fit_state.record(partial(self._context.remove_instruments, instruments, if_has_position_then))

    def subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        self._fit_state.record(partial(self._context.subscribe, subscription_type, instruments))

    def unsubscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        self._fit_state.record(partial(self._context.unsubscribe, subscription_type, instruments))

    def schedule(self, cron_schedule: str, method: Callable[["IStrategyContext"], None]) -> str:
        # - eager validation so bad input still raises into on_fit; the scheduler
        #   mutation itself is deferred. The pre-generated event id stays valid.
        rule = process_schedule_spec(cron_schedule)
        if not rule or rule.get("type") != "cron":
            raise ValueError("Only cron type is supported for custom schedules")
        event_id = f"custom_schedule_{str(uuid.uuid4()).replace('-', '_')}"
        _pm = self._context._processing_manager  # type: ignore[attr-defined]
        self._fit_state.record(partial(_pm._register_schedule, event_id, rule["schedule"], method))
        return event_id

    def delay(self, duration: str, method: Callable[["IStrategyContext"], None]) -> str:
        # - schedule-ish: deferred like schedule(); the delay countdown starts when the
        #   FitCommit is applied
        event_id = f"delay_{str(uuid.uuid4()).replace('-', '_')}"
        _pm = self._context._processing_manager  # type: ignore[attr-defined]
        self._fit_state.record(partial(_pm._register_delay, event_id, duration, method))
        return event_id

    def set_fit_schedule(self, schedule: str) -> None:
        rule = process_schedule_spec(schedule)
        if rule.get("type") != "cron":
            raise ValueError("Only cron type is supported for fit schedule")
        _pm = self._context._processing_manager  # type: ignore[attr-defined]
        self._fit_state.record(partial(_pm._apply_fit_schedule, rule["schedule"], schedule))

    def emit_signal(self, signal: Signal | list[Signal]) -> None:
        # - buffered in the locked fit-cycle state; drained into the normal pipeline
        #   (in emission order) when the FitCommit is applied
        self._fit_state.buffer_signals(signal if isinstance(signal, list) else [signal])

    # ------------------------------------------------------------------
    # copy reads — live views come back as shallow copies / locked clones
    # ------------------------------------------------------------------

    def get_positions(self, exchange: str | None = None) -> dict:
        return dict(self._context.get_positions(exchange))

    @property
    def positions(self) -> dict:
        return dict(self._context.positions)

    def get_orders(
        self,
        instrument: Instrument | None = None,
        exchange: str | None = None,
        origin: "OrderOrigin | None" = None,
    ) -> dict:
        return dict(self._context.get_orders(instrument, exchange, origin))

    def get_balances(self, exchange: str | None = None) -> list:
        return list(self._context.get_balances(exchange))

    def get_leverages(self, exchange: str | None = None) -> dict:
        return dict(self._context.get_leverages(exchange))

    def get_active_targets(self) -> dict:
        return dict(self._context.get_active_targets())

    @property
    def account(self) -> _FitAccountView:
        return self._account_view

    def ohlc(self, instrument: Instrument, timeframe: Any = None, length: int | None = None):
        # - locked snapshot clone; a history fetch runs here on the data provider's normal
        #   loop and merges into the shared cache under the series lock
        return self._market_manager.ohlc(instrument, timeframe, length, snapshot=True)

    def ohlc_pd(
        self, instrument: Instrument, timeframe: Any = None, length: int | None = None, consolidated: bool = True
    ):
        return self._market_manager.ohlc_pd(instrument, timeframe, length, consolidated, snapshot=True)

    def quote(self, instrument: Instrument):
        # - provider quotes are plain reads; only the no-quote OHLC fallback touches the
        #   cache, via the snapshot path
        return self._market_manager.quote(instrument, snapshot=True)

    # ------------------------------------------------------------------
    # denied — no meaningful deferred semantics; fail loudly when called
    # ------------------------------------------------------------------

    # direct trading: on_fit expresses intent via emit_signal; orders would race the
    # account state the ProcessorThread is applying
    trade = _denied("trade", "emit a signal instead — trading applies after the fit commits")
    submit_orders = _denied("submit_orders", "emit signals instead")
    set_target_position = _denied("set_target_position", "emit a signal instead")
    set_target_leverage = _denied("set_target_leverage", "emit a signal instead")
    close_position = _denied("close_position", "emit a zero signal or remove_instruments instead")
    close_positions = _denied("close_positions")
    cancel_order = _denied("cancel_order")
    cancel_orders = _denied("cancel_orders")
    update_order = _denied("update_order")
    settle_position = _denied("settle_position")
    # venue/account configuration writes
    set_max_instrument_leverage = _denied("set_max_instrument_leverage")
    set_margin_mode = _denied("set_margin_mode")
    transfer_funds = _denied("transfer_funds")
    # subscription/schedule plumbing with no deferred story (on_init-time concerns)
    set_base_subscription = _denied("set_base_subscription", "an on_init-time setting")
    set_warmup = _denied("set_warmup", "an on_init-time setting")
    set_event_schedule = _denied("set_event_schedule")
    unschedule = _denied("unschedule")
    trigger_fit = _denied("trigger_fit", "a fit is already running")
    commit = _denied("commit", "pending operations are committed at the FitCommit")
    configure_stale_data_detection = _denied("configure_stale_data_detection")
    update_base_subscription = _denied("update_base_subscription", "framework-internal")
    get_market_data_cache = _denied("get_market_data_cache", "use ctx.ohlc / ctx.get_cached_market_data")
    # framework lifecycle / plumbing
    start = _denied("start")
    stop = _denied("stop")
    process_data = _denied("process_data", "framework-internal")
    process_event = _denied("process_event", "framework-internal")
    on_alter_position = _denied("on_alter_position", "framework-internal")
    set_warmup_positions = _denied("set_warmup_positions", "framework-internal")
    set_warmup_orders = _denied("set_warmup_orders", "framework-internal")
    set_warmup_active_targets = _denied("set_warmup_active_targets", "framework-internal")
    initializer = property(_denied("initializer", "an on_init-time object"))

    # ------------------------------------------------------------------
    # fail-closed default for everything not classified above / below
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        raise UnclassifiedFitContextAccess(_UNCLASSIFIED_MSG.format(name=name))


# ----------------------------------------------------------------------
# passthrough reads — delegated to the real context unchanged. Installed as
# properties so one mechanism serves plain attributes, @property members and
# bound methods (attribute access returns the real ctx's bound method).
# ----------------------------------------------------------------------
_PASSTHROUGH_READS: tuple[str, ...] = (
    # identity / mode
    "strategy",
    "strategy_name",
    "account_id",
    "is_simulation",
    "is_live",
    "is_live_or_warmup",
    "is_paper_trading",
    "is_warmup_in_progress",
    "is_fitting",
    "is_warming_up",
    "is_running",
    "state",
    "exchanges",
    # services (thread-safe by contract)
    "emitter",
    "health",
    "notifier",
    "persistence",
    # time & market data reads (ohlc/ohlc_pd/quote are copy-reads above)
    "time",
    "get_cached_market_data",
    "get_aux_reader",
    "get_aux_data_storage",
    "get_instruments",
    "query_instrument",
    "is_instrument_listed",
    "instruments",
    "get_min_size",
    # account views returning scalars / single objects / fresh reports
    "get_total_capital",
    "get_base_currency",
    "get_balance",
    "get_position",
    "find_order_by_id",
    "find_order_by_client_id",
    "position_report",
    "get_leverage",
    "get_net_leverage",
    "get_gross_leverage",
    "get_fees_calculator",
    "get_max_instrument_leverage",
    "get_max_instrument_notional",
    "get_margin_mode",
    "get_adl_level",
    "get_available_margin",
    "get_total_initial_margin",
    "get_total_maint_margin",
    "get_withdrawable_balance",
    "get_margin_ratio",
    # subscriptions / schedules / universe — read side
    "has_subscription",
    "get_subscriptions",
    "get_base_subscription",
    "get_subscribed_instruments",
    "get_warmup",
    "auto_subscribe",
    "get_event_schedule",
    "is_fitted",
    "is_trading_allowed",
    # blacklist reads
    "is_blacklisted",
    "filter_blacklisted",
    "get_blacklisted_instruments",
    # warmup/restore state reads
    "get_warmup_positions",
    "get_warmup_orders",
    "get_warmup_active_targets",
    "get_restored_state",
    # transfers — read side
    "get_transfer_status",
    "get_transfers",
)


def _install_passthrough_reads() -> None:
    for _name in _PASSTHROUGH_READS:

        def _get(self: FitContext, _n: str = _name) -> Any:
            return getattr(self._context, _n)

        _get.__name__ = _name
        setattr(FitContext, _name, property(_get))


_install_passthrough_reads()
