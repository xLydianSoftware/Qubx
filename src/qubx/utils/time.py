import re
import time
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

UNIX_T0 = np.datetime64("1970-01-01T00:00:00")


time_to_str = lambda t, u="us": np.datetime_as_string(  # noqa: E731
    t if isinstance(t, np.datetime64) else np.datetime64(t, u), unit=u
)


def convert_tf_str_td64(c_tf: str) -> np.timedelta64:
    """
    Convert string timeframe to timedelta64

    '15Min' -> timedelta64(15, 'm') etc
    """
    _t = re.findall(r"(\d+)([A-Za-z]+)", c_tf)
    _dt = 0
    for g in _t:
        unit = g[1].lower()
        n = int(g[0])
        u1 = unit[0]
        u2 = unit[:2]
        unit = u1

        if u1 in ["d", "w"]:
            unit = u1.upper()

        if u1 in ["y"]:
            n = 356 * n
            unit = "D"

        if u2 in ["ms", "ns", "us", "ps"]:
            unit = u2

        _dt += np.timedelta64(n, unit)

    return _dt


def convert_seconds_to_str(seconds: int, convert_months=False) -> str:
    """
    Convert seconds to string representation: 310 -> '5Min10S' etc
    """
    r = ""

    if convert_months:
        months, seconds = divmod(seconds, 4 * 7 * 86400)
        if months > 0:
            r += "%dmonth" % months

    weeks, seconds = divmod(seconds, 7 * 86400)
    if weeks > 0:
        r += "%dw" % weeks

    days, seconds = divmod(seconds, 86400)
    if days > 0:
        r += "%dd" % days

    hours, seconds = divmod(seconds, 3600)
    if hours > 0:
        r += "%dh" % hours

    minutes, seconds = divmod(seconds, 60)
    if minutes > 0:
        r += "%dmin" % minutes

    if seconds > 0:
        r += "%ds" % seconds
    return r


def timedelta_to_str(td: np.timedelta64 | pd.Timedelta | np.int64 | int) -> str:
    """
    Convert timedelta to string representation
    """
    if isinstance(td, np.timedelta64):
        seconds = td.astype("timedelta64[s]").astype(int)
    elif isinstance(td, pd.Timedelta):
        seconds = int(td.total_seconds())
    elif isinstance(td, (int, np.int64)):
        seconds = int(pd.Timedelta(td).total_seconds())
    else:
        raise ValueError(f"Can't convert {type(td)} to string")

    return convert_seconds_to_str(seconds)


def floor_t64(time: np.datetime64 | datetime, dt: np.timedelta64 | int | str):
    """
    Floor timestamp by dt
    """
    if isinstance(dt, int):
        dt = np.timedelta64(dt, "s")

    if isinstance(dt, str):
        dt = convert_tf_str_td64(dt)

    if isinstance(time, datetime):
        time = np.datetime64(time)

    return time - (time - UNIX_T0) % dt


def infer_series_frequency(series: list | pd.DataFrame | pd.Series | pd.DatetimeIndex) -> np.timedelta64:
    """
    Infer frequency of given timeseries

    :param series: Series, DataFrame, DatetimeIndex or list of timestamps object
    :return: timedelta for found frequency
    """
    if isinstance(series, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
        times_index = (series if isinstance(series, pd.DatetimeIndex) else series.index).to_pydatetime()
    elif isinstance(series, (set, list, tuple)):
        times_index = np.array(series)
    elif isinstance(series, np.ndarray):
        times_index = series
    else:
        raise ValueError("Can't recognize input data")

    if times_index.shape[0] < 2:
        raise ValueError("Series must have at least 2 points to determ frequency")

    values = np.array(
        sorted(
            [
                (
                    x
                    if isinstance(x, (np.timedelta64, int, np.int64))
                    else int(x)
                    if isinstance(x, float)
                    else int(1e9 * x.total_seconds())
                )
                for x in np.abs(np.diff(times_index))
            ]
        )
    )
    diff = np.concatenate(([1], np.diff(values)))
    idx = np.concatenate((np.where(diff)[0], [len(values)]))
    freqs = dict(zip(values[idx[:-1]], np.diff(idx)))
    return np.timedelta64(max(freqs, key=freqs.get))


def handle_start_stop(
    s: str | pd.Timestamp | None, e: str | pd.Timestamp | None, convert: Callable = str
) -> tuple[str | pd.Timestamp | None, str | pd.Timestamp | None]:
    """
    Process start/stop times

    >>>  handle_start_stop('2020-01-01', '2020-02-01') # 2020-01-01, 2020-02-01
    >>>  handle_start_stop('2020-02-01', '2020-01-01') # 2020-01-01, 2020-02-01
    >>>  handle_start_stop('2020-01-01', '1w')         # 2020-01-01, 2020-01-01 + 1week
    >>>  handle_start_stop('1w', '2020-01-01')         # 2020-01-01 - 1week, '2020-01-01'
    >>>  handle_start_stop('2020-01-01', '-1w')        # 2020-01-01 - 1week, 2020-01-01,
    >>>  handle_start_stop(None, '2020-01-01')         # None, '2020-01-01'
    >>>  handle_start_stop('2020-01-01', None)         # '2020-01-01', None
    >>>  handle_start_stop(None, None)                 # None, None

    """

    def _h_time_like(x):
        try:
            return pd.Timestamp(x), False
        except:
            try:
                return pd.Timedelta(x), True
            except:
                pass
        return None, None

    t0, d0 = _h_time_like(s) if s else (None, False)
    t1, d1 = _h_time_like(e) if e else (None, False)

    def _converts(xs):
        return (convert(xs[0]) if xs[0] else None, convert(xs[1]) if xs[1] else None)

    if not t1 and not t0:
        return None, None

    if d0 and d1:
        raise ValueError("Start and stop can't both be deltas !")

    if d0:
        if not t1:
            raise ValueError("First argument is delta but stop time is not defined !")
        return _converts(sorted([t1 - abs(t0), t1]))
    if d1:
        if not t0:
            raise ValueError("Second argument is delta but start time is not defined !")
        return _converts(sorted([t0, t0 + t1]))

    if t0 and t1:
        return _converts(sorted([t0, t1]))

    return _converts([t0, t1])


def timedelta_to_crontab(td: pd.Timedelta) -> str:
    """
    Convert a pandas Timedelta to a crontab specification string.

    Args:
        td (pd.Timedelta): Timedelta to convert to crontab spec

    Returns:
        str: Crontab specification string

    Examples:
        >>> timedelta_to_crontab(pd.Timedelta('4h'))
        '0 */4 * * *'
        >>> timedelta_to_crontab(pd.Timedelta('2d'))
        '59 23 */2 * *'
        >>> timedelta_to_crontab(pd.Timedelta('1d23h50Min10Sec'))
        '50 23 */2 * * 10'
    """
    days = td.days
    hours = td.components.hours
    minutes = td.components.minutes
    seconds = td.components.seconds

    if days > 0:
        if hours == 0 and minutes == 0 and seconds == 0:
            hours, minutes, seconds = 23, 59, 59
        _sched = f"{minutes} {hours} */{days} * *"
        return _sched + f" {seconds}" if seconds > 0 else _sched

    if hours > 0:
        _sched = f"{minutes} */{hours} * * *"
        return _sched + f" {seconds}" if seconds > 0 else _sched

    if minutes > 0:
        _sched = f"*/{minutes} * * * *"
        return _sched + f" {seconds}" if seconds > 0 else _sched

    if seconds > 0:
        return f"* * * * * */{seconds}"

    raise ValueError("Timedelta must specify a non-zero period of days, hours, minutes or seconds")


def _parse_offset_interval(inv: str) -> str | None:
    """
    Parse offset interval patterns like "1h -1min", "1d -5Min".

    This handles expressions where you want to schedule something before the end
    of a regular interval.

    Args:
        inv: Interval string in format "<base_interval> -<offset>"

    Returns:
        Cron expression string, or None if the input doesn't match the offset pattern

    Examples:
        >>> _parse_offset_interval("1h -1min")
        '59 * * * *'
        >>> _parse_offset_interval("1h -1s")
        '59 * * * * 59'
        >>> _parse_offset_interval("1d -5Min")
        '55 23 * * *'
        >>> _parse_offset_interval("2h -30min")
        '30 */2 * * *'
    """
    # - match pattern: <base_interval> -<offset>
    match = re.match(r"^(.+?)\s+(-[0-9]+[A-Za-z]+)$", inv.strip())
    if not match:
        return None  # - not an offset pattern

    base_str = match.group(1)
    offset_str = match.group(2)

    try:
        base_td = pd.Timedelta(base_str)
        offset_td = pd.Timedelta(offset_str)
    except Exception as e:
        raise ValueError(f"Failed to parse interval '{inv}': {e}") from e

    if offset_td >= pd.Timedelta(0):
        raise ValueError(f"Offset must be negative, got: {offset_str}")

    # - calculate position within the period
    position_td = base_td + offset_td

    if position_td <= pd.Timedelta(0):
        raise ValueError(f"Offset {offset_str} is too large for base interval {base_str}")

    # - extract time components from position
    pos_seconds = int(position_td.total_seconds())
    pos_hours = (pos_seconds // 3600) % 24
    pos_minutes = (pos_seconds // 60) % 60
    pos_seconds_only = pos_seconds % 60

    # - determine base interval type and generate cron
    base_total_seconds = base_td.total_seconds()

    if base_td.days >= 1:
        # - day-based interval
        if base_td.days == 1:
            cron = f"{pos_minutes} {pos_hours} * * *"
        else:
            cron = f"{pos_minutes} {pos_hours} */{base_td.days} * *"
        return cron + f" {pos_seconds_only}" if pos_seconds_only > 0 else cron

    elif base_total_seconds >= 3600:
        # - hour-based interval
        base_hours = int(base_total_seconds // 3600)
        if base_hours == 1:
            cron = f"{pos_minutes} * * * *"
        else:
            # - for N>1, */N starts from 0, so we need explicit hour list
            # - example: "2h -30min" should fire at 1:30, 3:30, 5:30, etc.
            # - not at 0:30, 2:30, 4:30 (which is what */2 gives)
            hours_list = ",".join(str((pos_hours + i * base_hours) % 24) for i in range(24 // base_hours))
            cron = f"{pos_minutes} {hours_list} * * *"
        return cron + f" {pos_seconds_only}" if pos_seconds_only > 0 else cron

    elif base_total_seconds >= 60:
        # - minute-based interval
        base_minutes = int(base_total_seconds // 60)
        if base_minutes == 1:
            return f"* * * * * {pos_seconds_only}"
        else:
            # - for N>1, */N starts from 0, so we need explicit minute list
            # - example: "5Min -1s" should fire at 4:59, 9:59, 14:59, etc.
            # - not at 0:59, 5:59, 10:59 (which is what */5 gives)
            minutes_list = ",".join(str((pos_minutes + i * base_minutes) % 60) for i in range(60 // base_minutes))
            return f"{minutes_list} * * * * {pos_seconds_only}"

    else:
        raise ValueError("Second-based intervals with offsets are not supported")


def interval_to_cron(inv: str) -> str:
    """
    Convert a custom schedule format to a cron expression.

    Args:
        inv (str): Custom schedule format string. Can be either:
            - A pandas Timedelta string (e.g. "4h", "2d", "1d12h")
            - An offset interval "<base_interval> -<offset>" (e.g. "1h -1min", "1d -5Min")
            - A custom schedule format "<interval>@<time>" where:
                interval: Optional number + unit (Q=quarter, M=month, Y=year, D=day, SUN=Sunday, MON=Monday, etc.)
                time: HH:MM or HH:MM:SS

    Returns:
        str: Cron expression

    Examples:
        >>> interval_to_cron("4h")  # Pandas Timedelta
        '0 */4 * * *'
        >>> interval_to_cron("2d")  # Pandas Timedelta
        '59 23 */2 * *'
        >>> interval_to_cron("1h -1min")  # 1 minute before every hour
        '59 * * * *'
        >>> interval_to_cron("1h -1s")  # 1 second before every hour
        '59 * * * * 59'
        >>> interval_to_cron("1d -5Min")  # 5 minutes before end of day
        '55 23 * * *'
        >>> interval_to_cron("@10:30")  # Daily at 10:30
        '30 10 * * *'
        >>> interval_to_cron("1M@15:00")  # Monthly at 15:00
        '0 15 1 */1 * *'
        >>> interval_to_cron("2Q@09:30:15")  # Every 2 quarters at 9:30:15
        '30 9 1 */6 * 15'
        >>> interval_to_cron("Y@00:00")  # Annually at midnight
        '0 0 1 1 * *'
        >>> interval_to_cron("TUE @ 23:59")
        '59 23 * * 2'
    """
    # - first check for offset interval pattern (e.g., "1h -1min")
    offset_result = _parse_offset_interval(inv)
    if offset_result is not None:
        return offset_result

    # - next try parsing as pandas Timedelta
    try:
        _td_inv = pd.Timedelta(inv)
        return timedelta_to_crontab(_td_inv)
    except Exception:
        pass

    # - parse custom schedule format
    try:
        # - split into interval and time parts
        interval, time = inv.split("@")
        interval = interval.strip()
        time = time.strip()

        # - parse time
        time_parts = time.split(":")
        if len(time_parts) == 2:
            hour, minute = time_parts
            second = "0"
        elif len(time_parts) == 3:
            hour, minute, second = time_parts
        else:
            raise ValueError("Invalid time format")

        # - parse interval
        if not interval:  # Default to 1 day if no interval specified
            return f"{minute} {hour} * * * {second}"

        match = re.match(r"^(\d+)?([A-Za-z]+)$", interval)
        if not match:
            raise ValueError(f"Invalid interval format: {interval}")
        number = match.group(1) or "1"
        unit = match.group(2).upper()

        dow = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]

        # - convert to cron expression
        match unit:
            case "Q":  # Quarter
                return f"{minute} {hour} 1 */{3 * int(number)} * {second}"
            case "M":  # Month
                return f"{minute} {hour} 1 */{number} * {second}"
            case "Y":  # Year
                return f"{minute} {hour} 1 1 * {second}"
            case "SUN" | "MON" | "TUE" | "WED" | "THU" | "FRI" | "SAT":  # Day of Week
                return f"{minute} {hour} * * {dow.index(unit)} {second}"
            case "D":  # Day
                return f"{minute} {hour} */{number} * * {second}"
            case _:
                raise ValueError(f"Invalid interval unit: {unit}")

    except Exception as e:
        raise ValueError(f"Invalid schedule format: {inv}") from e


def to_utc_naive(timestamp: pd.Timestamp | datetime | str) -> pd.Timestamp:
    """
    Convert a timestamp to UTC and remove timezone info.

    This is safer than using .replace(tzinfo=None) as it properly converts
    timezone-aware timestamps to UTC before removing the timezone info.

    Args:
        timestamp: A pandas Timestamp (timezone-aware or naive) or datetime string

    Returns:
        pd.Timestamp: UTC timestamp without timezone info

    Examples:
        >>> to_utc_naive(pd.Timestamp("2025-07-16T18:00:00+02:00"))
        Timestamp('2025-07-16 16:00:00')
        >>> to_utc_naive(pd.Timestamp("2025-07-16T16:00:00Z"))
        Timestamp('2025-07-16 16:00:00')
    """
    if isinstance(timestamp, (str, datetime)):
        timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        # If already timezone-naive, assume it's already UTC
        return timestamp

    # Convert to UTC and remove timezone info
    return timestamp.tz_convert("UTC").tz_localize(None)


def now_utc() -> pd.Timestamp:
    """
    Get current UTC time as a pandas Timestamp without timezone info.
    """
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def timestamp_to_ms(timestamp: pd.Timestamp) -> int:
    """
    Convert a pandas Timestamp to milliseconds since epoch.
    """
    return int(timestamp.timestamp() * 1000)


def now_ns() -> int:
    """
    Get current UTC time in nanoseconds since epoch.

    This is a high-performance alternative to using pandas for getting
    current time during simulation iterations.

    Returns:
        int: Current UTC time in nanoseconds since epoch

    Examples:
        >>> current_time = now_ns()
        >>> isinstance(current_time, int)
        True
    """
    return int(time.time() * 1_000_000_000)
