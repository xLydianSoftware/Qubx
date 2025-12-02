from qubx.utils import convert_tf_str_td64
import numpy as np
cimport numpy as np
import pandas as pd
import datetime

NS = 1_000_000_000 

cpdef recognize_time(time):
    if isinstance(time, str):
        return np.datetime64(time, 'ns')
    elif isinstance(time, np.datetime64):
        return time
    elif isinstance(time, pd.Timestamp):
        return time.asm8.astype('datetime64[ns]')
    elif isinstance(time, datetime.datetime):
        return np.datetime64(time, 'ns')
    elif isinstance(time, int):
        # Heuristic: treat values less than 1990-01-01T00:00:00Z in ns as ms, otherwise as ns (epoch times)
        # 1990-01-01T00:00:00Z in ns since epoch is 631152000000000000
        CUTOFF_1990_NS = 631152000000000000
        if time < CUTOFF_1990_NS:
            # Interpret as milliseconds
            return np.datetime64(time, 'ms').astype('datetime64[ns]')
        else:
            # Interpret as nanoseconds
            return np.datetime64(time, 'ns')
    return np.datetime64(time, 'ns')


cpdef str time_to_str(long long t, str units = 'ns'):
    return str(np.datetime64(t, units)) #.isoformat()


cpdef str time_delta_to_str(long long d):
    """
    Convert timedelta object to pretty print format

    :param d:
    :return:
    """
    days, seconds = divmod(d, 86400*NS)
    hours, seconds = divmod(seconds, 3600*NS)
    minutes, seconds = divmod(seconds, 60*NS)
    seconds, rem  = divmod(seconds, NS)
    r = ''
    if days > 0:
        r += '%dD' % days
    if hours > 0:
        r += '%dh' % hours
    if minutes > 0:
        r += '%dMin' % minutes
    if seconds > 0:
        r += '%dS' % seconds
    if rem > 0:
        r += '%dmS' % (rem // 1000000)
    return r


cpdef recognize_timeframe(timeframe):
    tf = timeframe
    if isinstance(timeframe, str):
        tf = np.int64(convert_tf_str_td64(timeframe).item().total_seconds() * NS)

    elif isinstance(timeframe, (int, float)) and timeframe >= 0:
        tf = timeframe
    
    elif isinstance(timeframe, np.int64):
        tf = timeframe

    elif isinstance(timeframe, np.timedelta64):
        tf = np.int64(timeframe.item().total_seconds() * NS) 

    else:
        raise ValueError(f'Unknown timeframe type: {timeframe} !')
    return tf


cpdef double prec_ceil(double a, int precision):
    return np.sign(a) * np.true_divide(np.ceil(round(abs(a) * 10**precision, precision)), 10**precision)


cpdef double prec_floor(double a, int precision):
    return np.sign(a) * np.true_divide(np.floor(round(abs(a) * 10**precision, precision)), 10**precision)