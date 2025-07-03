import copy
import re
import sched
import sys
import time
from collections import defaultdict, deque
from inspect import isbuiltin, isclass, isfunction, ismethod, ismethoddescriptor
from threading import Thread
from typing import Any, Callable

import numpy as np
import pandas as pd
from croniter import croniter

from qubx import logger
from qubx.core.basics import SW, CtrlChannel, DataType, Instrument, Timestamped, dt_64, td_64
from qubx.core.series import OHLCV, Bar, OrderBook, Quote, Trade, time_as_nsec
from qubx.utils.time import convert_seconds_to_str, convert_tf_str_td64, interval_to_cron


class CachedMarketDataHolder:
    """
    Collected cached data updates from market
    """

    default_timeframe: np.timedelta64
    _last_bar: dict[Instrument, Bar | None]
    _ohlcvs: dict[Instrument, dict[np.timedelta64, OHLCV]]
    _updates: dict[Instrument, Bar | Quote | Trade]

    _instr_to_sub_to_buffer: dict[Instrument, dict[str, deque]]

    def __init__(self, default_timeframe: str | None = None, max_buffer_size: int = 10_000) -> None:
        self._ohlcvs = dict()
        self._last_bar = defaultdict(lambda: None)
        self._updates = dict()
        self._instr_to_sub_to_buffer = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_buffer_size)))
        if default_timeframe:
            self.update_default_timeframe(default_timeframe)

    def update_default_timeframe(self, default_timeframe: str):
        self.default_timeframe = convert_tf_str_td64(default_timeframe)

    def init_ohlcv(self, instrument: Instrument, max_size=np.inf):
        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {
                self.default_timeframe: OHLCV(instrument.symbol, self.default_timeframe, max_size),
            }

    def remove(self, instrument: Instrument) -> None:
        self._ohlcvs.pop(instrument, None)
        self._last_bar.pop(instrument, None)
        self._updates.pop(instrument, None)
        self._instr_to_sub_to_buffer.pop(instrument, None)

    def set_state_from(self, other: "CachedMarketDataHolder") -> None:
        """
        Set the internal state of this CachedMarketDataHolder to the state of another instance.

        WARNING: This is a shallow copy of the internal state dictionaries.

        Args:
            other: Another CachedMarketDataHolder instance to copy state from
        """
        self.default_timeframe = other.default_timeframe
        self._last_bar = copy.deepcopy(other._last_bar)
        self._ohlcvs = other._ohlcvs
        self._updates = other._updates
        self._instr_to_sub_to_buffer = other._instr_to_sub_to_buffer
        self._last_bar = defaultdict(lambda: None)  # reset the last bar

    @SW.watch("CachedMarketDataHolder")
    def get_ohlcv(
        self, instrument: Instrument, timeframe: str | td_64 | None = None, max_size: float | int = np.inf
    ) -> OHLCV:
        if timeframe is None:
            tf = self.default_timeframe
        elif isinstance(timeframe, str):
            tf = convert_tf_str_td64(timeframe)
        else:  # td_64
            tf = timeframe

        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {}

        if tf not in self._ohlcvs[instrument]:
            # - check requested timeframe
            new_ohlc = OHLCV(instrument.symbol, tf, max_size)
            if tf < self.default_timeframe:
                logger.warning(
                    f"[{instrument.symbol}] Request for timeframe {timeframe} that is smaller then minimal {self.default_timeframe}"
                )
            else:
                # - first try to resample from smaller frame
                if basis := self._ohlcvs[instrument].get(self.default_timeframe):
                    for b in basis[::-1]:
                        new_ohlc.update_by_bar(b.time, b.open, b.high, b.low, b.close, b.volume, b.bought_volume)

            self._ohlcvs[instrument][tf] = new_ohlc

        return self._ohlcvs[instrument][tf]

    def get_data(self, instrument: Instrument, event_type: str) -> list[Any]:
        return list(self._instr_to_sub_to_buffer[instrument][event_type])

    def update(
        self,
        instrument: Instrument,
        event_type: str,
        data: Any,
        update_ohlc: bool = False,
        is_historical: bool = False,
        is_base_data: bool = True,
    ) -> None:
        # - store data in buffer if it's not OHLC
        if event_type != DataType.OHLC:
            self._instr_to_sub_to_buffer[instrument][event_type].append(data)

        if not update_ohlc:
            return

        match event_type:
            case DataType.OHLC:
                self.update_by_bar(instrument, data)
            case DataType.QUOTE:
                self.update_by_quote(instrument, data)
            case DataType.TRADE:
                self.update_by_trade(instrument, data)
            case DataType.ORDERBOOK:
                assert isinstance(data, OrderBook)
                self.update_by_quote(instrument, data.to_quote())
            case _:
                pass

    @SW.watch("CachedMarketDataHolder")
    def update_by_bars(self, instrument: Instrument, timeframe: str | td_64, bars: list[Bar]) -> OHLCV:
        """
        Update or create OHLCV series with the provided historical bars.

        This method:
        1. Creates a new OHLCV series if one doesn't exist for the instrument/timeframe
        2. Updates an existing OHLCV series with the new bars using the OHLCV.update_by_bars method
           which handles:
           - Adding older bars to the back of the series
           - Skipping bars that are already present
           - Adding newer bars to the front
        """
        if instrument not in self._ohlcvs:
            self._ohlcvs[instrument] = {}

        tf = convert_tf_str_td64(timeframe) if isinstance(timeframe, str) else timeframe

        # Get existing OHLCV or create a new one
        if tf in self._ohlcvs[instrument]:
            ohlc = self._ohlcvs[instrument][tf]
            # Update the existing OHLCV with the new bars
            ohlc.update_by_bars(bars)
        else:
            # Create a new OHLCV and add the bars
            ohlc = OHLCV(instrument.symbol, tf)
            ohlc.update_by_bars(bars)
            self._ohlcvs[instrument][tf] = ohlc

        # Update the last update for this instrument
        if bars:
            self._updates[instrument] = bars[-1]  # Use the last bar as the last update

        return ohlc

    @SW.watch("CachedMarketDataHolder")
    def update_by_bar(self, instrument: Instrument, bar: Bar):
        self._updates[instrument] = bar

        _last_bar = self._last_bar[instrument]
        v_tot_inc = bar.volume
        v_buy_inc = bar.bought_volume

        if _last_bar is not None:
            if _last_bar.time == bar.time:  # just current bar updated
                v_tot_inc -= _last_bar.volume
                v_buy_inc -= _last_bar.bought_volume

            if _last_bar.time > bar.time:  # update is too late - skip it
                return

        if instrument in self._ohlcvs:
            self._last_bar[instrument] = bar
            for ser in self._ohlcvs[instrument].values():
                try:
                    ser.update_by_bar(bar.time, bar.open, bar.high, bar.low, bar.close, v_tot_inc, v_buy_inc)
                except ValueError as e:
                    logger.warning(f"Can't update ohlc series for [{instrument.symbol}] ::: {str(e)}")

    @SW.watch("CachedMarketDataHolder")
    def update_by_quote(self, instrument: Instrument, quote: Quote):
        self._updates[instrument] = quote
        series = self._ohlcvs.get(instrument)
        if series:
            for ser in series.values():
                if len(ser) > 0 and ser[0].time > quote.time:
                    continue
                ser.update(quote.time, quote.mid_price(), 0)

    @SW.watch("CachedMarketDataHolder")
    def update_by_trade(self, instrument: Instrument, trade: Trade):
        self._updates[instrument] = trade
        series = self._ohlcvs.get(instrument)
        if series:
            total_vol = trade.size
            bought_vol = total_vol if trade.side == 1 else 0.0
            for ser in series.values():
                if len(ser) > 0 and ser[0].time > trade.time:
                    continue
                ser.update(trade.time, trade.price, total_vol, bought_vol)

    def finalize_ohlc_for_instruments(self, time: dt_64, instruments: list[Instrument]):
        """
        Finalize all OHLCV series at the given time for the given instruments.
        FIXME: (2025-06-17) This is part of urgent live fix and must be removed in future !!!.
        """
        for instrument in instruments:
            # - use most recent update
            if (_u := self._updates.get(instrument)) is not None:
                _px = extract_price(_u)
                self.update_by_bar(instrument, Bar(time_as_nsec(time), _px, _px, _px, _px, 0, 0))


SPEC_REGEX = re.compile(
    r"((?P<type>[A-Za-z]+)(\.?(?P<timeframe>[0-9A-Za-z]+))?\ *:)?"
    r"\ *"
    r"((?P<spec>"
    r"(?P<time>((\d+:\d+(:\d+)?)\ *,?\ *)+)?"
    r"((\ *@\ *)(?P<by>([A-Za-z0-9-,\ ]+)))?"
    r"(("
    r"((?P<months>[-+]?\d+)(months|month|bm|mo))?"
    r"((?P<weeks>[-+]?\d+)(weeks|week|w))?"
    r"((?P<days>[-+]?\d+)(days|day|d))?"
    r"((?P<hours>[-+]?\d+)(hours|hour|h))?"
    r"((?P<minutes>[-+]?\d+)(mins|min|m))?"
    r"((?P<seconds>[-+]?\d+)(sec|s))?"
    r")(\ *)?)*"
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

    def __init__(self, channel: CtrlChannel, time_provider_ns: Callable[[], float]):
        self._chan = channel
        self._ns_time_fun = time_provider_ns
        self._scdlr = sched.scheduler(self.time_sec)
        self._crons = dict()
        self._is_started = False
        self._next_nearest_time = np.datetime64(sys.maxsize, "ns")
        self._next_times = dict()

    def time_sec(self) -> float:
        return self._ns_time_fun() / 1000000000.0

    def schedule_event(self, cron_schedule: str, event_name: str):
        if not croniter.is_valid(cron_schedule):
            raise ValueError(f"Specified schedule {cron_schedule} for {event_name} doesn't have valid cron format !")
        self._crons[event_name] = croniter(cron_schedule, self.time_sec())

        if self._is_started:
            self._arm_schedule(event_name, self.time_sec())

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
        iter = self._crons[event]
        prev_time = iter.get_prev()
        next_time = iter.get_next(start_time=start_time)
        if next_time:
            self._scdlr.enterabs(next_time, 1, self._trigger, (event, _to_dt_64(prev_time), _to_dt_64(next_time)))

            # - update next nearest time
            self._next_times[event] = next_time
            self._next_nearest_time = _to_dt_64(min(self._next_times.values()))
            # logger.debug(f" >>> ({event}) task is scheduled at {self._next_nearest_time}")

            return True
        logger.debug(f"({event}) task is not scheduled")
        return False

    def _trigger(self, event: str, prev_time_sec: float, trig_time: float):
        now = self.time_sec()

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
