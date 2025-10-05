import numpy as np
import pandas as pd

from qubx import logger


def recognize_t(t: int | str, defaultvalue, timeunit: str) -> np.datetime64:
    if isinstance(t, (str, pd.Timestamp)):
        try:
            return np.datetime64(t, timeunit)
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert time {t} to datetime64: {e}")
    return defaultvalue


def find_time_col_idx(column_names: list[str]):
    return find_column_index_in_list(column_names, "time", "timestamp", "datetime", "date", "open_time", "ts")


def find_column_index_in_list(xs: list[str], *args):
    xs = [x.lower() for x in xs]
    for a in args:
        ai = a.lower()
        if ai in xs:
            return xs.index(ai)
    raise IndexError(f"Can't find any specified columns from [{args}] in provided list: {xs}")


def calculate_time_windows_for_chunking(
    start: str | pd.Timestamp | None, end: str | pd.Timestamp | None, timeframe: str, chunksize: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Calculate time windows for efficient chunking based on timeframe and chunksize.

    Args:
        start: Start time string
        end: End time string
        timeframe: Timeframe string (e.g., "1m", "5m", "1h")
        chunksize: Number of candles per chunk

    Returns:
        List of (start_time, end_time) tuples for each chunk
    """
    if not start or not end:
        return []

    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

    try:
        # Calculate time period per chunk based on timeframe and chunksize
        tf_delta = pd.Timedelta(timeframe)
        chunk_duration = tf_delta * chunksize

        windows = []
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + chunk_duration, end_dt)
            windows.append((current_start, current_end))
            current_start = current_end

        # If last window is less than half of the window before it, then merge together
        if len(windows) > 1 and windows[-1][0] - windows[-1][1] < chunk_duration / 2:
            windows[-2] = (windows[-2][0], windows[-1][1])
            windows.pop()

        return windows

    except (ValueError, TypeError):
        # If timeframe can't be parsed, fall back to single window
        return [(start_dt, end_dt)]
