from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import re

from types import FunctionType
from itertools import product


def _wrap_single_list(param_grid: List | Dict) -> Dict[str, Any] | List:
    """
    Wraps all non list values as single
    :param param_grid:
    :return:
    """
    as_list = lambda x: x if isinstance(x, (tuple, list, dict, np.ndarray)) else [x]
    if isinstance(param_grid, list):
        return [_wrap_single_list(ps) for ps in param_grid]
    return {k: as_list(v) for k, v in param_grid.items()}


def permutate_params(
    parameters: Dict[str, List | Tuple | Any],
    conditions: FunctionType | List | Tuple | None = None,
    wrap_as_list=False,
) -> List[Dict]:
    """
    Generate list of all permutations for given parameters and theirs possible values

    Example:

    >>> def foo(par1, par2):
    >>>     print(par1)
    >>>     print(par2)
    >>>
    >>> # permutate all values and call function for every permutation
    >>> [foo(**z) for z in permutate_params({
    >>>                                       'par1' : [1,2,3],
    >>>                                       'par2' : [True, False]
    >>>                                     }, conditions=lambda par1, par2: par1<=2 and par2==True)]

    1
    True
    2
    True

    :param conditions: list of filtering functions
    :param parameters: dictionary
    :param wrap_as_list: if True (default) it wraps all non list values as single lists (required for sklearn)
    :return: list of permutations
    """
    if conditions is None:
        conditions = []
    elif isinstance(conditions, FunctionType):
        conditions = [conditions]
    elif isinstance(conditions, (tuple, list)):
        if not all([isinstance(e, FunctionType) for e in conditions]):
            raise ValueError("every condition must be a function")
    else:
        raise ValueError("conditions must be of type of function, list or tuple")

    args = []
    vals = []
    for k, v in parameters.items():
        args.append(k)
        # vals.append([v] if not isinstance(v, (list, tuple)) else list(v) if isinstance(v, range) else v)
        match v:
            case list() | tuple():
                vals.append(v)
            case range():
                vals.append(list(v))
            case str():
                vals.append([v])
            case _:
                vals.append(list(v))
        # vals.append(v if isinstance(v, (List, Tuple)) else list(v) if isinstance(v, range) else [v])
    d = [dict(zip(args, p)) for p in product(*vals)]
    result = []
    for params_set in d:
        conditions_met = True
        for cond_func in conditions:
            func_param_args = cond_func.__code__.co_varnames
            func_param_values = [params_set[arg] for arg in func_param_args]
            if not cond_func(*func_param_values):
                conditions_met = False
                break
        if conditions_met:
            result.append(params_set)

    # if we need to follow sklearn rules we should wrap every noniterable as list
    return _wrap_single_list(result) if wrap_as_list else result


def variate(clz, *args, conditions=None, **kwargs) -> Dict[str, Any]:
    """
    Make variations of parameters for simulations (micro optimizer)

    Example:

    >>>    class MomentumStrategy_Ex1_test:
    >>>       def __init__(self, p1, lookback_period=10, filter_type='sma', skip_entries_flag=False):
    >>>            self.p1, self.lookback_period, self.filter_type, self.skip_entries_flag = p1, lookback_period, filter_type, skip_entries_flag
    >>>
    >>>        def __repr__(self):
    >>>            return self.__class__.__name__ + f"({self.p1},{self.lookback_period},{self.filter_type},{self.skip_entries_flag})"
    >>>
    >>>    variate(MomentumStrategy_Ex1_test, 10, lookback_period=[1,2,3], filter_type=['ema', 'sma'], skip_entries_flag=[True, False])

    Output:
    >>>    {
    >>>        'MSE1t_(lp=1,ft=ema,sef=True)':  MomentumStrategy_Ex1_test(10,1,ema,True),
    >>>        'MSE1t_(lp=1,ft=ema,sef=False)': MomentumStrategy_Ex1_test(10,1,ema,False),
    >>>        'MSE1t_(lp=1,ft=sma,sef=True)':  MomentumStrategy_Ex1_test(10,1,sma,True),
    >>>        'MSE1t_(lp=1,ft=sma,sef=False)': MomentumStrategy_Ex1_test(10,1,sma,False),
    >>>        'MSE1t_(lp=2,ft=ema,sef=True)':  MomentumStrategy_Ex1_test(10,2,ema,True),
    >>>        'MSE1t_(lp=2,ft=ema,sef=False)': MomentumStrategy_Ex1_test(10,2,ema,False),
    >>>        'MSE1t_(lp=2,ft=sma,sef=True)':  MomentumStrategy_Ex1_test(10,2,sma,True),
    >>>        'MSE1t_(lp=2,ft=sma,sef=False)': MomentumStrategy_Ex1_test(10,2,sma,False),
    >>>        'MSE1t_(lp=3,ft=ema,sef=True)':  MomentumStrategy_Ex1_test(10,3,ema,True),
    >>>        'MSE1t_(lp=3,ft=ema,sef=False)': MomentumStrategy_Ex1_test(10,3,ema,False),
    >>>        'MSE1t_(lp=3,ft=sma,sef=True)':  MomentumStrategy_Ex1_test(10,3,sma,True),
    >>>        'MSE1t_(lp=3,ft=sma,sef=False)': MomentumStrategy_Ex1_test(10,3,sma,False)
    >>>    }

    and using in simuation:

    >>>    r = simulate(
    >>>             variate(MomentumStrategy_Ex1_test, 10, lookback_period=[1,2,3], filter_type=['ema', 'sma'], skip_entries_flag=[True, False]),
    >>>             data, capital, ["BINANCE.UM:BTCUSDT"], dict(type="ohlc", timeframe="5Min", nback=0), "5Min -1Sec", "vip0_usdt", "2024-01-01", "2024-01-02"
    >>>    )
    """

    def _cmprss(xs: str):
        return "".join([x[0] for x in re.split("((?<!-)(?=[A-Z]))|_|(\d)", xs) if x])

    sfx = _cmprss(clz.__name__)
    to_excl = [s for s, v in kwargs.items() if not isinstance(v, (list, set, tuple, range))]
    dic2str = lambda ds: [_cmprss(k) + "=" + str(v) for k, v in ds.items() if k not in to_excl]

    return {
        f"{sfx}_({ ','.join(dic2str(z)) })": clz(*args, **z) for z in permutate_params(kwargs, conditions=conditions)
    }