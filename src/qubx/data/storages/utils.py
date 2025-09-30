import numpy as np
import pandas as pd

from qubx import logger


def recognize_t(t: int | str, defaultvalue, timeunit) -> np.datetime64:
    if isinstance(t, (str, pd.Timestamp)):
        try:
            return np.datetime64(t, timeunit)
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to convert time {t} to datetime64: {e}")
    return defaultvalue


def find_time_col_idx(column_names: list[str]):
    return find_column_index_in_list(column_names, "time", "timestamp", "datetime", "date", "open_time", "ts")


def find_column_index_in_list(xs, *args):
    xs = [x.lower() for x in xs]
    for a in args:
        ai = a.lower()
        if ai in xs:
            return xs.index(ai)
    raise IndexError(f"Can't find any specified columns from [{args}] in provided list: {xs}")
