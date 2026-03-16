import re
import sched
import sys
import time
from inspect import isbuiltin, isclass, isfunction, ismethod, ismethoddescriptor
from threading import Thread
from typing import Any, Callable

import numpy as np
import pandas as pd
from croniter import croniter

from qubx import logger
from qubx.core.basics import CtrlChannel, Timestamped
from qubx.core.series import Bar, OrderBook, Quote, Trade
from qubx.utils.time import convert_seconds_to_str, convert_tf_str_td64, interval_to_cron

SPEC_REGEX = re.compile(
    r"((?P<type>[A-Za-z]+)(\.?(?P<timeframe>[0-9A-Za-z]+))?\s*:)?"
    r"\s*"
    r"((?P<spec>"
    r"(?P<time>((\d+:\d+(:\d+)?)\s*,?\s*)+)?"
    r"((\s*@\s*)(?P<by>([A-Za-z0-9-,\s]+)))?"
    r"(("
    r"((?P<months>[-+]?\d+)(months|month|bm|mo))?"
    r"((?P<weeks>[-+]?\d+)(weeks|week|w))?"
    r"((?P<days>[-+]?\d+)(days|day|d))?"
    r"((?P<hours>[-+]?\d+)(hours|hour|h))?"
    r"((?P<minutes>[-+]?\d+)(mins|min|m))?"
    r"((?P<seconds>[-+]?\d+)(sec|s))?"
    r")(\s*)?)*"
    r".*"
    r"))?",
    re.IGNORECASE,
)


def _mk_cron(time: str, by: list | None) -> str:
    HMS = lambda s: list(map(int, s.split(":") if s.count(":") == 2 else [*s.split(":"), 0]))  # noqa: E731

    h, m, s = HMS(time)
    assert h < 24, f"Wrong value for hour {h}"
    assert m < 60, f"Wrong value for minute {m}"
    assert s < 60, f"Wrong value for seconds {s}"
    b = ",".join(by) if by else "*"
    c = f"{m} {h} * * {b}"
    return c if s == 0 else c + f" {s}"


def _make_shift(_b, _w, _d, _h, _m, _s):
    D0 = pd.Timedelta(0)
    AS_TD = lambda d: pd.Timedelta(d)  # noqa: E731
    P, N = D0, D0

    # return AS_TD(f'{_b*4}W') + AS_TD(f'{_w}W') + AS_TD(f'{_d}D') + AS_TD(f'{_h}h') + AS_TD(f'{_m}Min') + AS_TD(f'{_s}Sec')
    for t in [
        AS_TD(f"{_b * 4}W"),
        AS_TD(f"{_w}W"),
        AS_TD(f"{_d}D"),
        AS_TD(f"{_h}h"),
        AS_TD(f"{_m}Min"),
        AS_TD(f"{_s}Sec"),
    ]:
        if t > D0:
            P += t
        else:
            N += t
    return P, N


def _parse_schedule_spec(schedule: str) -> dict[str, str]:
    m = SPEC_REGEX.match(schedule)
    return {k: v for k, v in m.groupdict().items() if v} if m else {}


def _to_dt_64(time: float) -> np.datetime64:
    return np.datetime64(int(time * 1000000000), "ns")


def process_schedule_spec(spec_str: str | None) -> dict[str, Any]:
    AS_INT = lambda d, k: int(d.get(k, 0))  # noqa: E731
    S = lambda s: [x for x in re.split(r"[, ]", s) if x]  # noqa: E731
    config = {}

    if not spec_str:
        return config

    # - parse schedule spec
    spec = _parse_schedule_spec(spec_str)

    # - check how to run it
    _T, _S = spec.get("type"), spec.get("spec")
    _F = spec.get("timeframe")
    _t, _by = S(spec.get("time", "")), S(spec.get("by", ""))
    _b, _w, _d = AS_INT(spec, "months"), AS_INT(spec, "weeks"), AS_INT(spec, "days")
    _h, _m, _s = AS_INT(spec, "hours"), AS_INT(spec, "minutes"), AS_INT(spec, "seconds")
    _has_intervals = (_b != 0) or (_w != 0) or (_d != 0) or (_h != 0) or (_m != 0) or (_s != 0)
    _s_pos, _s_neg = _make_shift(_b, _w, _d, _h, _m, _s)
    _shift = _s_pos + _s_neg

    match _T:
        case "cron":
            if not _S:
                raise ValueError(f"Empty specification for cron: {spec_str}")

            if not croniter.is_valid(_S):
                _S = interval_to_cron(_S)

                if not croniter.is_valid(_S):
                    raise ValueError(f"Wrong specification for cron: {spec_str}")

            config = dict(type="cron", schedule=_S, spec=_S)

        case "time":
            for t in _t:
                config = dict(type="cron", schedule=_mk_cron(t, _by), spec=_S)

        case None:
            if _t:  # - if time specified
                for t in _t:
                    config = dict(type="cron", schedule=_mk_cron(t, _by), spec=_S)
            else:
                # - check if it's valid cron
                if _S:
                    if croniter.is_valid(_S):
                        config = dict(type="cron", schedule=_S, spec=_S)
                    else:
                        # - try convert to cron
                        _S = interval_to_cron(_S)
                        if croniter.is_valid(_S):
                            config = dict(type="cron", schedule=_S, spec=_S)
                        else:
                            if _has_intervals:
                                _F = (
                                    convert_seconds_to_str(
                                        int(_s_pos.as_unit("s").to_timedelta64().item().total_seconds())
                                    )
                                    if not _F
                                    else _F
                                )
                                config = dict(type="bar", schedule=None, timeframe=_F, delay=_s_neg, spec=_S)
        case _:
            config = dict(type=_T, schedule=None, timeframe=_F, delay=_shift, spec=_S)

    return config


class BasicScheduler:
    """
    Basic scheduler functionality. It helps to create scheduled event task
    """

    _chan: CtrlChannel
    _scdlr: sched.scheduler
    _ns_time_fun: Callable[[], float]
    _crons: dict[str, croniter]
    _is_started: bool
    _next_nearest_time: np.datetime64
    _next_times: dict[str, float]
    _scheduled_events: dict[str, Any]  # Track scheduled event objects for cancellation
    _once_events: set[str]  # Track one-time delayed events

    def __init__(self, channel: CtrlChannel, time_provider_ns: Callable[[], float]):
        self._chan = channel
        self._ns_time_fun = time_provider_ns
        self._scdlr = sched.scheduler(self.time_sec)
        self._crons = dict()
        self._is_started = False
        self._next_nearest_time = np.datetime64(sys.maxsize, "ns")
        self._next_times = dict()
        self._scheduled_events = dict()
        self._once_events = set()

    def time_sec(self) -> float:
        return self._ns_time_fun() / 1000000000.0

    def schedule_event(self, cron_schedule: str, event_name: str):
        if not croniter.is_valid(cron_schedule):
            raise ValueError(f"Specified schedule {cron_schedule} for {event_name} doesn't have valid cron format !")

        # If rescheduling an existing event, cancel any armed event first
        if event_name in self._scheduled_events:
            try:
                self._scdlr.cancel(self._scheduled_events[event_name])
            except ValueError:
                # Event may have already fired or been removed
                pass
            del self._scheduled_events[event_name]

        self._crons[event_name] = croniter(cron_schedule, self.time_sec())

        if self._is_started:
            self._arm_schedule(event_name, self.time_sec())

    def unschedule_event(self, event_name: str) -> bool:
        """
        Remove a scheduled event and cancel any armed events in the scheduler.
        Works for both cron-based and one-time delayed events.

        Args:
            event_name: Name of the event to unschedule

        Returns:
            bool: True if event was found and removed, False otherwise
        """
        is_cron = event_name in self._crons
        is_once = event_name in self._once_events

        if not is_cron and not is_once:
            return False

        # Cancel any armed event from the scheduler
        if event_name in self._scheduled_events:
            try:
                self._scdlr.cancel(self._scheduled_events[event_name])
            except ValueError:
                # Event may have already fired or been removed
                pass
            del self._scheduled_events[event_name]

        # Remove from crons dict if it's a cron event
        if is_cron:
            del self._crons[event_name]

        # Remove from once_events if it's a one-time event
        if is_once:
            self._once_events.discard(event_name)

        # Remove from next_times and recalculate nearest time
        if event_name in self._next_times:
            del self._next_times[event_name]

            # Recalculate next nearest time
            if self._next_times:
                self._next_nearest_time = _to_dt_64(min(self._next_times.values()))
            else:
                self._next_nearest_time = np.datetime64(sys.maxsize, "ns")

        return True

    def delay(self, duration: str, event_name: str) -> None:
        """
        Schedule a one-time event after a duration.

        Args:
            duration: Duration string (e.g., "30s", "1Min", "1h")
            event_name: Name of the event to trigger
        """
        # Parse duration to timedelta
        td = convert_tf_str_td64(duration)
        delay_seconds = td.item().total_seconds()

        # Calculate target time
        target_time = self.time_sec() + delay_seconds

        # Mark as one-time event
        self._once_events.add(event_name)

        # Schedule the event
        scheduled_event = self._scdlr.enterabs(target_time, 1, self._trigger_once, (event_name, target_time))
        self._scheduled_events[event_name] = scheduled_event

        # Update next times tracking
        self._next_times[event_name] = target_time
        self._next_nearest_time = _to_dt_64(min(self._next_times.values()))

    def _trigger_once(self, event: str, trig_time: float):
        """
        Trigger a one-time delayed event.
        Unlike _trigger(), this does NOT re-arm the event.
        """
        # Clean up the scheduled event reference
        self._scheduled_events.pop(event, None)
        self._once_events.discard(event)

        # Send notification via channel
        self._chan.send((None, event, (trig_time, trig_time), False))

        # Clean up from tracking
        self._next_times.pop(event, None)

        # Recalculate nearest time
        if self._next_times:
            self._next_nearest_time = _to_dt_64(min(self._next_times.values()))
        else:
            self._next_nearest_time = np.datetime64(sys.maxsize, "ns")

    def next_expected_event_time(self) -> np.datetime64:
        """
        Returns the next scheduled event time
        """
        return self._next_nearest_time

    def get_schedule_for_event(self, event_name: str) -> str | None:
        if event_name in self._crons:
            return " ".join(self._crons[event_name].expressions)
        return None

    def get_event_last_time(self, event_name: str) -> pd.Timestamp | None:
        if event_name in self._crons:
            _iter = self._crons[event_name]
            _c = _iter.get_current()
            _t = pd.Timestamp(_iter.get_prev(), unit="s")
            _iter.set_current(_c, force=True)
            return _t
        return None

    def get_event_next_time(self, event_name: str) -> pd.Timestamp | None:
        if event_name in self._crons:
            _iter = self._crons[event_name]
            _t = pd.Timestamp(_iter.get_next(start_time=self.time_sec()), unit="s")
            return _t
        return None

    def _arm_schedule(self, event: str, start_time: float) -> bool:
        # Check if event still exists (may have been unscheduled)
        if event not in self._crons:
            return False

        iter = self._crons[event]
        prev_time = iter.get_prev()
        next_time = iter.get_next(start_time=start_time)
        if next_time:
            scheduled_event = self._scdlr.enterabs(
                next_time, 1, self._trigger, (event, _to_dt_64(prev_time), _to_dt_64(next_time))
            )

            # Store the scheduled event object so we can cancel it later if needed
            self._scheduled_events[event] = scheduled_event

            # - update next nearest time
            self._next_times[event] = next_time
            self._next_nearest_time = _to_dt_64(min(self._next_times.values()))
            # logger.debug(f" >>> ({event}) task is scheduled at {self._next_nearest_time}")

            return True
        logger.debug(f"({event}) task is not scheduled")
        return False

    def _trigger(self, event: str, prev_time_sec: float, trig_time: float):
        now = self.time_sec()

        # Clean up the scheduled event reference since it has fired
        self._scheduled_events.pop(event, None)

        # - send notification to channel
        self._chan.send((None, event, (prev_time_sec, trig_time), False))

        # - try to arm this event again
        self._arm_schedule(event, now)

    def check_and_run_tasks(self) -> float | None:
        return self._scdlr.run(blocking=False)

    def run(self):
        if self._is_started:
            logger.warning("Scheduler is already running")
            return

        _has_tasks = False
        for k in self._crons.keys():
            _has_tasks |= self._arm_schedule(k, self.time_sec())

        def _watcher():
            while r := self.check_and_run_tasks():
                if not self._chan.control.is_set():
                    break
                _delay = max(min(r / 5, 5), 0.1)
                time.sleep(_delay)
            logger.debug("Scheduler is stopped ")
            self._is_started = False

        if _has_tasks:
            Thread(target=_watcher).start()
            self._is_started = True


def extract_parameters_from_object(strategy: Any) -> dict[str, Any]:
    """
    Extract default parameters (as defined in class) and their values from object.
    """
    from qubx.core.interfaces import IStrategyContext

    _f_dict = {}
    for o in [*strategy.__class__.mro()[::-1], strategy]:
        if hasattr(o, "__dict__"):  # only objects have __dict__ attribute
            for k, v in o.__dict__.items():
                if not k.startswith("_") and not (
                    # - skip any function, method, built-in, class, method descriptor
                    isinstance(v, IStrategyContext)  # we don't want to have ctx object
                    or isfunction(v)
                    or ismethod(v)
                    or isbuiltin(v)
                    or isclass(v)
                    or ismethoddescriptor(v)
                ):
                    _f_dict[k] = getattr(o, k, v)
    return _f_dict


def set_parameters_to_object(strategy: Any, **kwargs):
    """
    Set given parameters values to object.
    Parameter can be set only if it's declared as attribute of object and it's not starting with underscore (_).
    """
    _log_info = ""
    for k, v in kwargs.items():
        if k.startswith("_"):
            raise ValueError("Internal variable can't be set from external parameter !")
        if hasattr(strategy, k):
            strategy.__dict__[k] = v
            v_str = str(v).replace(">", "").replace("<", "")
            _log_info += f"\n\tset <green>{k}</green> <- <red>{v_str}</red>"

    if _log_info:
        logger.debug(f"<yellow>{strategy.__class__.__name__}</yellow> new parameters:" + _log_info)


def extract_price(update: float | Timestamped) -> float:
    """Extract the price from various types of market data updates.

    Args:
        update: The market data update, which can be a float, Quote, Trade, Bar or OrderBook.

    Returns:
        float: The extracted price.

    Raises:
        ValueError: If the update type is unknown.
    """
    if isinstance(update, float):
        return update
    elif isinstance(update, Quote) or isinstance(update, OrderBook):
        return update.mid_price()
    elif isinstance(update, Trade):
        return update.price
    elif isinstance(update, Bar):
        return update.close
    elif isinstance(update, OrderBook):
        return update.mid_price()
    else:
        raise ValueError(f"Unknown update type: {type(update)}")


def full_qualified_class_name(obj: object):
    """
    Returns full qualified class name of object.
    """
    klass = obj.__class__
    module = klass.__module__
    if module in ["__builtin__", "__main__"]:
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__name__
