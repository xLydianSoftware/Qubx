import numpy as np
import pandas as pd

dt_type = pd.Timestamp | np.datetime64 | int


def check_interval(t1: dt_type, t2: dt_type, interval: str) -> bool:
    _t1 = pd.Timestamp(t1).floor(interval)
    _t2 = pd.Timestamp(t2).floor(interval)
    return _t1 != _t2
