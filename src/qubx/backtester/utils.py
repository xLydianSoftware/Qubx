from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeAlias

import numpy as np
import pandas as pd
import stackprinter

from qubx import logger
from qubx.core.basics import (
    CtrlChannel,
    DataType,
    Instrument,
    ITimeProvider,
    Signal,
    TimestampedDict,
    TriggerEvent,
    dt_64,
)
from qubx.core.exceptions import SimulationConfigError, SimulationError
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.core.lookups import lookup
from qubx.core.series import OHLCV, Bar, Quote, Trade
from qubx.core.utils import time_delta_to_str
from qubx.data import TardisMachineReader
from qubx.data.helpers import InMemoryCachedReader, TimeGuardedWrapper
from qubx.data.hft import HftDataReader
from qubx.data.readers import AsDict, DataReader, InMemoryDataFrameReader
from qubx.utils.time import infer_series_frequency, timedelta_to_crontab

SymbolOrInstrument_t: TypeAlias = str | Instrument
ExchangeName_t: TypeAlias = str
SubsType_t: TypeAlias = str | DataType
RawData_t: TypeAlias = pd.DataFrame | OHLCV
DataDecls_t: TypeAlias = DataReader | dict[SubsType_t, DataReader | dict[SymbolOrInstrument_t, RawData_t]]

StrategyOrSignals_t: TypeAlias = IStrategy | pd.DataFrame | pd.Series
DictOfStrats_t: TypeAlias = dict[str, StrategyOrSignals_t]
StrategiesDecls_t: TypeAlias = (
    StrategyOrSignals_t
    | DictOfStrats_t
    | dict[str, DictOfStrats_t]
    | dict[str, StrategyOrSignals_t | list[StrategyOrSignals_t | PositionsTracker]]
    | list[StrategyOrSignals_t | PositionsTracker]
    | tuple[StrategyOrSignals_t | PositionsTracker]
)


class SetupTypes(Enum):
    UKNOWN = "unknown"
    LIST = "list"
    TRACKER = "tracker"
    SIGNAL = "signal"
    STRATEGY = "strategy"
    SIGNAL_AND_TRACKER = "signal_and_tracker"
    STRATEGY_AND_TRACKER = "strategy_and_tracker"


def _type(obj: Any) -> SetupTypes:
    if obj is None:
        t = SetupTypes.UKNOWN
    elif isinstance(obj, (list, tuple)):
        t = SetupTypes.LIST
    elif isinstance(obj, PositionsTracker):
        t = SetupTypes.TRACKER
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        t = SetupTypes.SIGNAL
    elif isinstance(obj, IStrategy):
        t = SetupTypes.STRATEGY
    else:
        t = SetupTypes.UKNOWN
    return t


@dataclass
class SimulationSetup:
    """
    Configuration of setups in the simulation.
    """

    setup_type: SetupTypes
    name: str
    generator: StrategyOrSignals_t
    tracker: PositionsTracker | None
    instruments: list[Instrument]
    exchanges: list[str]
    capital: float | dict[str, float]
    base_currency: str
    commissions: str | dict[str, str | None] | None = None
    signal_timeframe: str = "1Min"
    accurate_stop_orders_execution: bool = False

    def __str__(self) -> str:
        return f"{self.name} {self.setup_type} capital {self.capital} {self.base_currency} for [{','.join(map(lambda x: x.symbol, self.instruments))}] @ {self.exchanges}[{self.commissions}]"


# fmt: off
@dataclass
class SimulationDataConfig:
    """
    Configuration of data passed to the simulator.
    """
    default_trigger_schedule: str                          # default trigger schedule
    default_base_subscription: str                         # the base subscription type
    data_providers: dict[str, DataReader]                  # dictionary of available subscription types with DataReaders
    default_warmups: dict[str, str]                        # default warmups periods
    open_close_time_indent_secs: int                       # open/close ticks shift in seconds
    adjusted_open_close_time_indent_secs: int              # adjusted open/close ticks shift in seconds
    aux_data_provider: InMemoryCachedReader | None = None  # auxiliary data provider

    def get_timeguarded_aux_reader(self, time_provider: ITimeProvider) -> TimeGuardedWrapper | None:
        _aux = None
        if self.aux_data_provider is not None:
            if not isinstance(self.aux_data_provider, InMemoryCachedReader):
                logger.warning("Aux data provider should be an instance of InMemoryCachedReader ! Otherwise it can lead to unnecessary effects !")
            _aux = TimeGuardedWrapper(self.aux_data_provider, time_guard=time_provider)
        return _aux
# fmt: on


class SimulatedLogFormatter:
    def __init__(self, time_provider: ITimeProvider):
        self.time_provider = time_provider

    def formatter(self, record):
        end = record["extra"].get("end", "\n")
        fmt = "<lvl>{message}</lvl>%s" % end
        if record["level"].name in {"WARNING", "SNAKY"}:
            fmt = "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - %s" % fmt

        dt = self.time_provider.time()
        if isinstance(dt, int):
            now = pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            now = self.time_provider.time().astype("datetime64[us]").item().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # prefix = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [ <level>%s</level> ] " % record["level"].icon
        prefix = f"<lc>{now}</lc> [<level>{record['level'].icon}</level>] "

        if record["exception"] is not None:
            record["extra"]["stack"] = stackprinter.format(record["exception"], style="darkbg3")
            fmt += "\n{extra[stack]}\n"

        if record["level"].name in {"TEXT"}:
            prefix = ""

        return prefix + fmt


class SimulatedScheduler(BasicScheduler):
    def run(self):
        self._is_started = True
        _has_tasks = False
        _time = self.time_sec()
        for k in self._crons.keys():
            _has_tasks |= self._arm_schedule(k, _time)


class SimulatedCtrlChannel(CtrlChannel):
    """
    Simulated communication channel. Here we don't use queue but it invokes callback directly
    """

    _callback: Callable[[tuple], bool]

    def register(self, callback):
        self._callback = callback

    def send(self, data):
        # - when data is sent, invoke callback
        return self._callback.process_data(*data)

    def receive(self, timeout: int | None = None) -> Any:
        raise SimulationError("Method SimulatedCtrlChannel::receive() should not be called in a simulated environment.")

    def stop(self):
        self.control.clear()

    def start(self):
        self.control.set()


class SimulatedTimeProvider(ITimeProvider):
    _current_time: dt_64

    def __init__(self, initial_time: dt_64 | str):
        self._current_time = np.datetime64(initial_time, "ns") if isinstance(initial_time, str) else initial_time

    def time(self) -> dt_64:
        return self._current_time

    def set_time(self, time: dt_64):
        self._current_time = max(time, self._current_time)


class SignalsProxy(IStrategy):
    """
    Proxy strategy for generated signals.
    """

    timeframe: str = "1m"

    def on_init(self, ctx: IStrategyContext):
        ctx.set_base_subscription(DataType.OHLC[self.timeframe])

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | None:
        if event.data and event.type == "event":
            signal = event.data.get("order")
            # - TODO: also need to think about how to pass stop/take here
            if signal is not None and event.instrument:
                return [event.instrument.signal(ctx, signal)]
        return None


def _process_single_symbol_or_instrument(
    symbol_or_instrument: SymbolOrInstrument_t,
    default_exchange: ExchangeName_t | None,
    requested_exchange: ExchangeName_t | None,
) -> tuple[Instrument | None, str | None]:
    """
    Process a single symbol or instrument and return the resolved instrument and exchange.

    Returns:
        tuple[Instrument | None, str | None]: (instrument, exchange) or (None, None) if processing failed
    """
    match symbol_or_instrument:
        case str():
            _e, _s = (
                symbol_or_instrument.split(":")
                if ":" in symbol_or_instrument
                else (default_exchange, symbol_or_instrument)
            )

            if _e is None:
                logger.warning(
                    f"Can't extract exchange name from symbol's spec ({symbol_or_instrument}) and exact exchange name is not provided - skip this symbol !"
                )
                return None, None

            if (
                requested_exchange is not None
                and isinstance(requested_exchange, str)
                and _e.lower() != requested_exchange.lower()
            ):
                logger.warning(
                    f"Exchange from symbol's spec ({_e}) is different from requested: {requested_exchange} !"
                )

            if (instrument := lookup.find_symbol(_e, _s)) is not None:
                return instrument, _e.upper()
            else:
                logger.warning(f"Can't find instrument for specified symbol ({symbol_or_instrument}) - ignoring !")
                return None, None

        case Instrument():
            return symbol_or_instrument, symbol_or_instrument.exchange

        case _:
            raise SimulationConfigError(
                f"Unsupported type for {symbol_or_instrument} only str or Instrument instances are allowed!"
            )


def find_instruments_and_exchanges(
    instruments: list[SymbolOrInstrument_t] | dict[ExchangeName_t, list[SymbolOrInstrument_t]],
    exchange: ExchangeName_t | list[ExchangeName_t] | None,
) -> tuple[list[Instrument], list[ExchangeName_t]]:
    _instrs: list[Instrument] = []
    _exchanges = [] if exchange is None else [exchange] if isinstance(exchange, str) else exchange

    # Handle dictionary case where instruments is {exchange: [symbols]}
    if isinstance(instruments, dict):
        for exchange_name, symbol_list in instruments.items():
            if exchange_name not in _exchanges:
                _exchanges.append(exchange_name)

            for symbol in symbol_list:
                instrument, resolved_exchange = _process_single_symbol_or_instrument(symbol, exchange_name, exchange)
                if instrument is not None and resolved_exchange is not None:
                    _instrs.append(instrument)
                    _exchanges.append(resolved_exchange)

    # Handle list case
    else:
        for symbol in instruments:
            instrument, resolved_exchange = _process_single_symbol_or_instrument(symbol, exchange, exchange)
            if instrument is not None and resolved_exchange is not None:
                _instrs.append(instrument)
                _exchanges.append(resolved_exchange)

    return _instrs, list(set(_exchanges))


class _StructureSniffer:
    _probe_size: int

    def __init__(self, _probe_size: int = 50) -> None:
        self._probe_size = _probe_size

    def _is_strategy(self, obj) -> bool:
        return _type(obj) == SetupTypes.STRATEGY

    def _is_tracker(self, obj) -> bool:
        return _type(obj) == SetupTypes.TRACKER

    def _is_signal(self, obj) -> bool:
        return _type(obj) == SetupTypes.SIGNAL

    def _is_signal_or_strategy(self, obj) -> bool:
        return self._is_signal(obj) or self._is_strategy(obj)

    def _possible_instruments_ids(self, i: Instrument) -> set[str]:
        return set((i.symbol, str(i), f"{i.exchange}:{i.symbol}"))

    def _pick_instruments(self, instruments: list[Instrument], s: pd.Series | pd.DataFrame) -> list[Instrument]:
        if isinstance(s, pd.Series):
            _instrs = [i for i in instruments if s.name in self._possible_instruments_ids(i)]

        elif isinstance(s, pd.DataFrame):
            _s_cols = set(s.columns)
            _instrs = [i for i in instruments if self._possible_instruments_ids(i) & _s_cols]

        else:
            raise SimulationConfigError("Invalid signals or strategy configuration")

        return list(set(_instrs))

    def _name_in_instruments(self, n, instrs: list[Instrument]) -> bool:
        return any([n in self._possible_instruments_ids(i) for i in instrs])

    def _check_signals_structure(
        self, instruments: list[Instrument], s: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        if isinstance(s, pd.Series):
            # - it's possible to put anything to series name, so we convert it to string
            s.name = str(s.name)
            if not self._name_in_instruments(s.name, instruments):
                raise SimulationConfigError(f"Can't find instrument for signal's name: '{s.name}'")

        if isinstance(s, pd.DataFrame):
            s.columns = s.columns.map(lambda x: str(x))
            for col in s.columns:
                if not self._name_in_instruments(col, instruments):
                    raise SimulationConfigError(f"Can't find instrument for signal's name: '{col}'")
        return s

    def _has_columns(self, v: pd.DataFrame, columns: list[str]) -> bool:
        return all([c in v.columns for c in columns])

    def _has_keys(self, v: dict[str, Any], keys: list[str]) -> bool:
        return all([c in v.keys() for c in keys])

    def _sniff_list(self, v: list[Any]) -> str:
        match v[0]:
            case Bar():
                _tf = time_delta_to_str(infer_series_frequency([x.time for x in v[: self._probe_size]]).item())
                return DataType.OHLC[_tf]

            case dict():
                return self._sniff_dicts(v)

            case Quote():
                return DataType.QUOTE

            case TimestampedDict():
                _t = self._sniff_dicts(v[0].data)
                if _t in [DataType.OHLC, DataType.OHLC_TRADES, DataType.OHLC_QUOTES]:
                    _tf = time_delta_to_str(infer_series_frequency([x.time for x in v[: self._probe_size]]).item())
                    return DataType(_t)[_tf]
                return _t

            case Trade():
                return DataType.TRADE

        return DataType.RECORD

    def _sniff_dicts(self, v: dict[str, Any] | list[dict[str, Any]]) -> str:
        v, vs = (v[0], v) if isinstance(v, list) else (v, None)

        if self._has_keys(v, ["open", "high", "low", "close"]):
            if vs:
                _tf = time_delta_to_str(infer_series_frequency([x.get("time") for x in vs[: self._probe_size]]).item())
                return DataType.OHLC[_tf]
            return DataType.OHLC

        if self._has_keys(v, ["bid", "ask"]):
            return DataType.QUOTE

        if self._has_keys(v, ["price", "size"]):
            return DataType.TRADE

        if self._has_keys(v, ["top_bid", "top_ask", "bids", "asks"]):
            return DataType.ORDERBOOK

        return DataType.RECORD

    def _sniff_pandas(self, v: pd.DataFrame) -> str:
        if self._has_columns(v, ["open", "high", "low", "close"]):
            _tf = time_delta_to_str(infer_series_frequency(v[: self._probe_size]).item())
            return DataType.OHLC[_tf]

        if self._has_columns(v, ["bid", "ask"]):
            return DataType.QUOTE

        if self._has_columns(v, ["price", "size"]):
            return DataType.TRADE

        return DataType.RECORD

    def _pre_read(self, symbol: str, reader: DataReader, time: str, data_type: str) -> list[Any]:
        for dt in ["2h", "12h", "2d", "28d", "60d", "720d"]:
            try:
                _it = reader.read(
                    symbol,
                    transform=AsDict(),
                    start=time,
                    stop=pd.Timestamp(time) + pd.Timedelta(dt),  # type: ignore
                    timeframe=None,
                    chunksize=self._probe_size,
                    data_type=data_type,
                )
                if len(data := next(_it)) >= 2:  # type: ignore
                    return data
            except Exception:
                pass
        return []

    def _sniff_reader(self, symbol: str, reader: DataReader, preferred_data_type: str | None) -> str:
        _probing_types = [DataType.OHLC, DataType.QUOTE, DataType.TRADE]
        _probing_types = ([preferred_data_type] + _probing_types) if preferred_data_type is not None else _probing_types
        _found_type = None
        for _type in _probing_types:
            _t1, _t2 = reader.get_time_ranges(symbol, str(_type))
            if _t1 is not None:
                time = str(_t1 + (_t2 - _t1) / 2)
                _found_type = _type
                break
        else:
            logger.warning(f"Failed to find data start time and supported type for symbol: {symbol}")
            return DataType.NONE

        if _found_type is None:
            logger.warning(f"Failed to detect data type for symbol: {symbol}")
            return DataType.NONE

        data = self._pre_read(symbol, reader, time, _found_type)
        if data:
            return self._sniff_list(data)

        logger.warning(f"Failed to read probe data for symbol: {symbol}")
        return DataType.NONE


def recognize_simulation_configuration(
    name: str,
    configs: StrategiesDecls_t,
    instruments: list[Instrument],
    exchanges: list[str],
    capital: float | dict[str, float],
    basic_currency: str,
    commissions: str | dict[str, str | None] | None,
    signal_timeframe: str,
    accurate_stop_orders_execution: bool,
    run_separate_instruments: bool = False,
) -> list[SimulationSetup]:
    """
    Recognize and create setups based on the provided simulation configuration.

    This function processes the given configuration and creates a list of SimulationSetup
    objects that represent different simulation scenarios. It handles various types of
    configurations including dictionaries, lists, signals, and strategies.

    Parameters:
    - name (str): The name of the simulation setup.
    - configs (VariableStrategyConfig): The configuration for the simulation. Can be a
        strategy, signals, or a nested structure of these.
    - instruments (list[Instrument]): List of available instruments for the simulation.
    - exchange (str): The name of the exchange to be used.
    - capital (float): The initial capital for the simulation.
    - basic_currency (str): The base currency for the simulation.
    - commissions (str): The commission structure to be applied.
    - signal_timeframe (str): Timeframe for generated signals.
    - accurate_stop_orders_execution (bool): If True, enables more accurate stop order execution simulation.
    - run_separate_instruments (bool): If True, creates separate setups for each instrument.

    Returns:
    - list[SimulationSetup]: A list of SimulationSetup objects, each representing a
        distinct simulation configuration based on the input parameters.

    Raises:
    - SimulationConfigError: If the signal structure is invalid or if an instrument cannot be found
        for a given signal.
    """

    r = list()
    _sniffer = _StructureSniffer()

    # fmt: off
    if isinstance(configs, dict):
        for n, v in configs.items():
            _n = (name + "/") if name else ""
            r.extend(
                recognize_simulation_configuration(
                    _n + n, v, instruments, exchanges, capital, basic_currency, commissions, 
                    signal_timeframe, accurate_stop_orders_execution, run_separate_instruments
                )
            )

    elif isinstance(configs, (list, tuple)):
        if len(configs) == 2 and _sniffer._is_signal_or_strategy(configs[0]) and _sniffer._is_tracker(configs[1]):
            c0, c1 = configs[0], configs[1]
            _s = _sniffer._check_signals_structure(instruments, c0)   # type: ignore

            if _sniffer._is_signal(c0):
                _t = SetupTypes.SIGNAL_AND_TRACKER

            if _sniffer._is_strategy(c0):
                _t = SetupTypes.STRATEGY_AND_TRACKER

            # - extract actual symbols that have signals
            setup_instruments = _sniffer._pick_instruments(instruments, _s) if _sniffer._is_signal(c0) else instruments
            
            if run_separate_instruments:
                # Create separate setups for each instrument
                for instrument in setup_instruments:
                    r.append(
                        SimulationSetup(
                            _t, f"{name}/{instrument.symbol}", _s, c1,   # type: ignore
                            [instrument],
                            exchanges, capital, basic_currency, commissions, 
                            signal_timeframe, accurate_stop_orders_execution
                        )
                    )
            else:
                r.append(
                    SimulationSetup(
                        _t, name, _s, c1,   # type: ignore
                        setup_instruments,
                        exchanges, capital, basic_currency, commissions, 
                        signal_timeframe, accurate_stop_orders_execution
                    )
                )
        else:
            for j, s in enumerate(configs):
                r.extend(
                    recognize_simulation_configuration(
                        # name + "/" + str(j), s, instruments, exchange, capital, basic_currency, commissions
                        name, s, instruments, exchanges, capital, basic_currency, commissions,  # type: ignore
                        signal_timeframe, accurate_stop_orders_execution, run_separate_instruments
                    )
                )

    elif _sniffer._is_strategy(configs):
        if run_separate_instruments:
            # Create separate setups for each instrument
            for instrument in instruments:
                r.append(
                    SimulationSetup(
                        SetupTypes.STRATEGY,
                        f"{name}/{instrument.symbol}", configs, None, [instrument],
                        exchanges, capital, basic_currency, commissions, 
                        signal_timeframe, accurate_stop_orders_execution
                    )
                )
        else:
            r.append(
                SimulationSetup(
                    SetupTypes.STRATEGY,
                    name, configs, None, instruments,
                    exchanges, capital, basic_currency, commissions, 
                    signal_timeframe, accurate_stop_orders_execution
                )
            )

    elif _sniffer._is_signal(configs):
        # - check structure of signals
        c1 = _sniffer._check_signals_structure(instruments, configs)  # type: ignore
        setup_instruments = _sniffer._pick_instruments(instruments, c1)
        
        if run_separate_instruments:
            # Create separate setups for each instrument
            for instrument in setup_instruments:
                r.append(
                    SimulationSetup(
                        SetupTypes.SIGNAL,
                        f"{name}/{instrument.symbol}", c1, None, [instrument],
                        exchanges, capital, basic_currency, commissions, 
                        signal_timeframe, accurate_stop_orders_execution
                    )
                )
        else:
            r.append(
                SimulationSetup(
                    SetupTypes.SIGNAL,
                    name, c1, None, setup_instruments,
                    exchanges, capital, basic_currency, commissions, 
                    signal_timeframe, accurate_stop_orders_execution
                )
            )

    # fmt: on
    return r


def _get_default_warmup_period(base_subscription: str, in_timeframe: pd.Timedelta | None) -> pd.Timedelta:
    if in_timeframe is None or base_subscription in [DataType.QUOTE, DataType.TRADE, DataType.ORDERBOOK]:
        return pd.Timedelta("1Min")

    if in_timeframe < pd.Timedelta("1h"):
        return 5 * in_timeframe

    return 2 * in_timeframe


def _detect_defaults_from_subscriptions(
    requests: dict[str, tuple[str, DataReader]], open_close_time_indent_secs: int
) -> SimulationDataConfig:
    def _tf(x):
        _p = DataType.from_str(x)[1]
        return pd.Timedelta(_p["timeframe"]) if "timeframe" in _p else None

    _base_subscr = None
    _t_readers = {}
    _in_base_tf = None
    _out_tf = None

    _has_in_qts = False
    _has_in_trd = False
    _has_in_ohlc = False
    _has_in_ob = False
    _has_out_trd = False
    _has_out_qts = False
    _has_out_ob = False

    for _t, (_src, _r) in requests.items():
        _has_in_ohlc |= _src == DataType.OHLC
        _has_in_qts |= _src == DataType.QUOTE
        _has_in_trd |= _src == DataType.TRADE
        _has_in_ob |= _src == DataType.ORDERBOOK
        _has_out_trd |= _t == DataType.TRADE
        _has_out_qts |= _t == DataType.QUOTE
        _has_out_ob |= _t == DataType.ORDERBOOK

        match _t, _src:
            case (DataType.OHLC, DataType.OHLC):
                _t_readers[DataType.OHLC] = _r
                _out_tf = _tf(_t)
                _in_base_tf = _tf(_src)

                if not _in_base_tf:
                    SimulationConfigError(f"ohlc data specified for {_src} but it's timeframe was not detected")

                if not _out_tf:
                    _out_tf = _in_base_tf

                assert _out_tf and _in_base_tf
                if _in_base_tf > _out_tf:
                    logger.warning(
                        f"Can't produce OHLC {_out_tf} data from provided {_in_base_tf} timeframe, reduce to {_in_base_tf}"
                    )
                    _out_tf = _in_base_tf

                _base_subscr = _src

            case (DataType.OHLC, DataType.QUOTE) | (DataType.OHLC, DataType.TRADE):
                _t_readers[DataType.OHLC] = _r
                _out_tf = _tf(_t)
                _base_subscr = _src
                if _out_tf is None:
                    raise SimulationConfigError(f"ohlc output data timeframe is not specified for {_t}")

            case (DataType.QUOTE, DataType.OHLC):
                _t_readers[DataType.OHLC_QUOTES] = _r
                _in_base_tf = _tf(_src)

            case (DataType.TRADE, DataType.OHLC):
                _t_readers[DataType.OHLC_TRADES] = _r
                _in_base_tf = _tf(_src)

            case (_, _):
                _t_readers[_t] = _r

    if not _base_subscr:
        if _has_in_qts:  # it has input quotes - so base subscription is quotes
            _base_subscr = DataType.QUOTE

        elif _has_in_trd:  # it has input trades - so base subscription is trades
            _base_subscr = DataType.TRADE

        elif _has_in_ob:  # it has input orderbooks - so base subscription is orderbooks
            _base_subscr = DataType.ORDERBOOK

        elif _has_in_ohlc:  # it has input ohlc - let's generate quotes from this ohlc
            _out_tf = _in_base_tf

            if _has_out_trd:
                _base_subscr = DataType.OHLC_TRADES

            if _has_out_qts:
                _base_subscr = DataType.OHLC_QUOTES

    if not _base_subscr:
        raise SimulationConfigError("Can't detect base subscription in provided data specification")

    _default_trigger_schedule = ""  # default trigger on every event
    adj_open_close_time_indent_secs = open_close_time_indent_secs
    if _out_tf:
        _default_trigger_schedule = timedelta_to_crontab(_out_tf_tdelta := pd.Timedelta(_out_tf))

        # - if strategy doesn't set it's own schedule then this default trigger schedule would be used for triggering on_event() method.
        # - In this case we want that last price update was arrived before this trigger's time to have
        # - most recent market data
        adj_open_close_time_indent_secs = _adjust_open_close_time_indent_secs(
            _out_tf_tdelta, open_close_time_indent_secs
        )

    # - default warmups
    _warmups = {str(_base_subscr): time_delta_to_str(_get_default_warmup_period(_base_subscr, _in_base_tf).asm8.item())}

    return SimulationDataConfig(
        _default_trigger_schedule,
        _base_subscr,
        _t_readers,
        _warmups,
        open_close_time_indent_secs,
        adj_open_close_time_indent_secs,
    )


def _adjust_open_close_time_indent_secs(timeframe: pd.Timedelta, original_indent_secs: int) -> int:
    # - if it triggers at daily+ bar let's assume this bar is 'closed' 5 min before exact closing time
    if timeframe >= pd.Timedelta("1d"):
        return max(original_indent_secs, 5 * 60)

    # - if it triggers at 1Min+ bar let's assume this bar is 'closed' 5 sec before exact closing time
    if timeframe >= pd.Timedelta("1min"):
        return max(original_indent_secs, 5)

    # - for all sub-minute timeframes just use 1 sec shift
    if timeframe > pd.Timedelta("1s"):
        return max(original_indent_secs, 1)

    # - for rest just keep original indent
    return original_indent_secs


def _is_transformable(_dest: str, _src: str) -> bool:
    match _dest:
        case DataType.OHLC:
            return _src in [DataType.OHLC, DataType.QUOTE, DataType.TRADE]

        case DataType.QUOTE:
            return _src in [DataType.OHLC, DataType.QUOTE, DataType.ORDERBOOK]

        case DataType.TRADE:
            return _src in [DataType.OHLC, DataType.TRADE]

    return True


def recognize_simulation_data_config(
    decls: DataDecls_t,
    instruments: list[Instrument],
    open_close_time_indent_secs: int = 1,
    aux_data: DataReader | None = None,
) -> SimulationDataConfig:
    """
    Recognizes and configures simulation data based on the provided declarations.

    This function processes the given data declarations and determines the appropriate
    data readers and configurations for simulation. It supports various data types and
    structures, including DataReaders, pandas DataFrames, and dictionaries.

    Parameters:
    - decls (DataDecls_t): The data declarations for the simulation. Can be a DataReader,
        pandas DataFrame, or a dictionary of these.
    - instruments (list[Instrument]): List of available instruments for the simulation.

    Returns:
    - tuple[str, str, dict[str, DataReader]]: A tuple containing the default trigger schedule,
        the base subscription type, and a dictionary of available subscription types with
        their corresponding DataReaders.

    Raises:
    - SimulationConfigError: If the data provider type is unsupported or if a requested data type
        cannot be produced from the supported data type.
    """
    # TODO: in the future we should be able to start without initial instruments
    if not instruments:
        raise SimulationConfigError("No initial instruments provided")

    sniffer = _StructureSniffer()
    _requested_types = []
    _requests = {}
    _first_exchange = instruments[0].exchange
    _first_symbol = instruments[0].symbol
    _data_id = f"{_first_exchange}:{_first_symbol}"

    match decls:
        case HftDataReader():
            # For HftDataReader, we need to check which data types are enabled
            _supported_types = []
            if decls.enable_orderbook:
                _supported_types.append(DataType.ORDERBOOK)
            if decls.enable_quote:
                _supported_types.append(DataType.QUOTE)
            if decls.enable_trade:
                _supported_types.append(DataType.TRADE)

            if not _supported_types:
                raise SimulationConfigError("No data types are enabled in HftDataReader")

            # Add each enabled data type to requests
            for _type in _supported_types:
                _requests[_type] = (_type, decls)

        case TardisMachineReader():
            _supported_types = [DataType.ORDERBOOK, DataType.TRADE]
            for _type in _supported_types:
                _requests[_type] = (_type, decls)

        case DataReader():
            _supported_data_type = sniffer._sniff_reader(_data_id, decls, None)
            _requests[_supported_data_type] = (_supported_data_type, decls)

        case pd.DataFrame():
            _supported_data_type = sniffer._sniff_pandas(decls)
            _reader = InMemoryDataFrameReader(decls, _first_exchange)
            _requests[_supported_data_type] = (_supported_data_type, _reader)

        case dict():
            _is_dict_of_pandas = False

            for _requested_type, _provider in decls.items():
                # - if we already have this type declared, skip it#-
                # - it prevents to have duplicated ohlc (and potentially other data types with parametrization)#-
                _t = DataType.from_str(_requested_type)[0]
                if _t != DataType.NONE and _t in _requested_types:
                    raise SimulationConfigError(f"Type {_t} already declared")

                _requested_types.append(_t)

                match _provider:
                    case HftDataReader():
                        # For HftDataReader, check enabled data types
                        if _requested_type == DataType.ORDERBOOK and not _provider.enable_orderbook:
                            raise SimulationConfigError("Orderbook data is not enabled in HftDataReader")
                        elif _requested_type == DataType.QUOTE and not _provider.enable_quote:
                            raise SimulationConfigError("Quote data is not enabled in HftDataReader")
                        elif _requested_type == DataType.TRADE and not _provider.enable_trade:
                            raise SimulationConfigError("Trade data is not enabled in HftDataReader")

                        _supported_data_type = _requested_type  # HftDataReader directly supports the requested types
                        _requests[_requested_type] = (_supported_data_type, _provider)

                    case TardisMachineReader():
                        _supported_data_type = _requested_type
                        _requests[_requested_type] = (_supported_data_type, _provider)

                    case DataReader():
                        _supported_data_type = sniffer._sniff_reader(_data_id, _provider, _requested_type)
                        _requests[_requested_type] = (_supported_data_type, _provider)
                        if not _is_transformable(_requested_type, _supported_data_type):
                            raise SimulationConfigError(f"Can't produce {_requested_type} from {_supported_data_type}")

                    case dict():
                        try:
                            _reader = InMemoryDataFrameReader(_provider, _first_exchange)  # type: ignore
                            _supported_data_type = sniffer._sniff_reader(_data_id, _reader, _requested_type)
                            _requests[_requested_type] = (_supported_data_type, _reader)
                            if not _is_transformable(_requested_type, _supported_data_type):
                                raise SimulationConfigError(
                                    f"Can't produce {_requested_type} from {_supported_data_type}"
                                )

                        except Exception as e:
                            raise SimulationConfigError(
                                f"Error in declared data provider for: {_requested_type} -> {type(_provider)} ({str(e)})"
                            )

                    case pd.DataFrame():
                        _is_dict_of_pandas = True
                        break

                    case _:
                        raise SimulationConfigError(f"Unsupported data provider type: {type(_provider)}")

            if _is_dict_of_pandas:
                try:
                    _reader = InMemoryDataFrameReader(decls, _first_exchange)  # type: ignore
                    _supported_data_type = sniffer._sniff_reader(_data_id, _reader, _requested_type)
                    _requests[DataType.OHLC] = (_supported_data_type, _reader)
                    if not _is_transformable(_requested_type, _supported_data_type):
                        raise SimulationConfigError(f"Can't produce {_requested_type} from {_supported_data_type}")

                except Exception as e:
                    raise SimulationConfigError(
                        f"Error in declared data provider for: {_requested_type} -> {type(_provider)} ({str(e)})"
                    )

        case _:
            raise SimulationConfigError(f"Can't recognize declared data provider: {type(decls)}")

    # detect setup's defaults from declared data
    _setup_defaults = _detect_defaults_from_subscriptions(_requests, open_close_time_indent_secs)

    # - just pass it to config, TODO: we need to think how to handle auxiliary data provider better
    _setup_defaults.aux_data_provider = aux_data  # type: ignore

    return _setup_defaults
