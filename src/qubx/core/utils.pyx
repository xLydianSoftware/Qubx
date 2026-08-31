from qubx.utils import convert_tf_str_td64
import numpy as np
cimport numpy as np
import pandas as pd
import datetime
from libc.float cimport DBL_EPSILON
from libc.math cimport ceil, copysign, fabs, floor, fmax, pow, round

NS = 1_000_000_000 

cpdef recognize_time(time):
    if isinstance(time, str):
        # Strip timezone suffix (e.g. 'Z', '+00:00') — np.datetime64 expects tz-naive ISO strings.
        # All incoming timestamps are assumed UTC.
        if time.endswith('Z') or time.endswith('z'):
            time = time[:-1]
        elif '+' in time[10:]:
            time = time[:time.rindex('+')]
        return np.datetime64(time, 'ns')
    elif isinstance(time, np.datetime64):
        return time
    elif isinstance(time, pd.Timestamp):
        return time.asm8.astype('datetime64[ns]')
    elif isinstance(time, datetime.datetime):
        return np.datetime64(time, 'ns')
    elif isinstance(time, (int, np.integer)):
        # Heuristic: treat values less than 1990-01-01T00:00:00Z in ns as ms, otherwise as ns (epoch times)
        # 1990-01-01T00:00:00Z in ns since epoch is 631152000000000000
        CUTOFF_1990_NS = 631152000000000000
        if time < CUTOFF_1990_NS:
            # Interpret as milliseconds
            return np.datetime64(int(time), 'ms').astype('datetime64[ns]')
        else:
            # Interpret as nanoseconds
            return np.datetime64(int(time), 'ns')
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


cdef inline double _snap_tick_noise(double x) noexcept:
    """
    Snap x to the nearest integer when it is already within float noise of one. 0.29 * 100 is
    28.999999999999996, which would otherwise floor to 28. The tolerance is relative because
    float error scales with magnitude: 3.6e-15 at x=29, 3.7e-9 at x=2.9e7.
    """
    cdef double nearest = round(x)
    cdef double tolerance = 16.0 * DBL_EPSILON * fmax(1.0, fabs(x))
    if fabs(x - nearest) <= tolerance:
        return nearest
    return x


# - previous implementation rounded the SCALED value to `precision` decimals before the floor.
#   At precision 0 that window is half a lot, so prec_floor(3.7, 0) gave 4.0 and
#   prec_ceil(3.4, 0) gave 3.0 — neither was a floor or a ceil any more.
#   return np.sign(a) * np.true_divide(np.ceil(round(abs(a) * 10**precision, precision)), 10**precision)
#   return np.sign(a) * np.true_divide(np.floor(round(abs(a) * 10**precision, precision)), 10**precision)
cpdef double prec_ceil(double a, int precision):
    cdef double scale = pow(10.0, precision)
    cdef double ticks = _snap_tick_noise(fabs(a) * scale)
    return copysign(ceil(ticks) / scale, a)


cpdef double prec_floor(double a, int precision):
    cdef double scale = pow(10.0, precision)
    cdef double ticks = _snap_tick_noise(fabs(a) * scale)
    return copysign(floor(ticks) / scale, a)


cpdef double add_in_lots(double quantity, double amount, double lot_size) noexcept:
    """
    Sum two lot multiples in whole lots, so a subtraction cannot land a few ulp off the grid.
    """
    return (round(quantity / lot_size) + round(amount / lot_size)) * lot_size


cpdef bint is_lot_multiple(double quantity, double lot_size) noexcept:
    """
    True when quantity sits on the lot grid, within the float noise of one division.
    """
    cdef double raw = quantity / lot_size
    return fabs(raw - round(raw)) <= 16.0 * DBL_EPSILON * fmax(1.0, fabs(raw))
