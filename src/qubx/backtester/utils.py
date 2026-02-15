from collections.abc import Mapping, Sequence
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
    TriggerEvent,
    dt_64,
)
from qubx.core.exceptions import SimulationConfigError, SimulationError
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IStrategy, IStrategyContext, PositionsTracker
from qubx.core.lookups import lookup
from qubx.core.utils import time_delta_to_str
from qubx.data.helpers import TimeGuardedWrapper
from qubx.data.storage import IStorage
from qubx.utils.misc import safe_dtype_timeframe
from qubx.utils.runner.configs import PrefetchConfig

SymbolOrInstrument_t: TypeAlias = str | Instrument
ExchangeName_t: TypeAlias = str
SubsType_t: TypeAlias = str | DataType
DataDecls_t: TypeAlias = IStorage | dict[SubsType_t, IStorage | dict[SymbolOrInstrument_t, IStorage]]


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
    enable_funding: bool = False

    def __str__(self) -> str:
        return f"{self.name} {self.setup_type} capital {self.capital} {self.base_currency} for [{','.join(map(lambda x: x.symbol, self.instruments))}] @ {self.exchanges}[{self.commissions}]"


# fmt: off
@dataclass
class SimulationDataConfig:
    """
    Configuration of data passed to the simulator.
    """
    data_storage: IStorage                                 # main data provider (storage)
    customized_data_storages: dict[str, IStorage]          # may have custom storages for subscription types (like {"quote": stor2, "features": stor3} etc)
    # default_warmups: dict[str, str]                      # default warmups periods

    # - TO REMOVE -
    aux_providers: dict[str, IStorage] | None = None       # aux data providers
    prefetch_config: PrefetchConfig | None = None          # prefetch configuration


    def get_timeguarded_aux_reader(self, time_provider: ITimeProvider) -> TimeGuardedWrapper | None:
        _aux = None
        # TODO: do we still need that in such form ?
        # if self.customized_data_storages is not None:
        #     aux_reader = self.customized_data_storages
            
        #     # Wrap with CachedPrefetchReader if not already wrapped
        #     if not isinstance(aux_reader, CachedPrefetchReader):
        #         prefetch_period = "1w"
        #         cache_size_mb = 100
                
        #         # Get prefetch configuration if available
        #         if self.prefetch_config:
        #             prefetch_period = self.prefetch_config.prefetch_period
        #             cache_size_mb = self.prefetch_config.cache_size_mb
                
        #         aux_reader = CachedPrefetchReader(
        #             aux_reader, 
        #             prefetch_period=prefetch_period,
        #             cache_size_mb=cache_size_mb
        #         )
            
        #     _aux = TimeGuardedWrapper(aux_reader, time_guard=time_provider)
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
        prefix = f"<lc>{now}</lc> [<level>{record['level'].icon}</level>] <cyan>({{module}})</cyan> "

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

    Supports notation formats:
        "BTCUSDT"                    - plain symbol (uses default_exchange)
        "BINANCE.UM:BTCUSDT"        - 2-part (exchange:symbol)
        "BINANCE.UM:SWAP:BTCUSDT"   - 3-part (exchange:market_type:symbol)

    Returns:
        tuple[Instrument | None, str | None]: (instrument, exchange) or (None, None) if processing failed
    """
    match symbol_or_instrument:
        case str():
            _e, _mt, _s = Instrument.parse_notation(symbol_or_instrument)

            # - fall back to default exchange if not specified
            if _e is None:
                _e = default_exchange

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

            if (instrument := lookup.find_symbol(_e, _s, market_type=_mt)) is not None:
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
    instruments: Sequence[SymbolOrInstrument_t] | Mapping[ExchangeName_t, Sequence[SymbolOrInstrument_t]],
    exchange: ExchangeName_t | list[ExchangeName_t] | None,
) -> tuple[list[Instrument], list[ExchangeName_t]]:
    _instrs: list[Instrument] = []
    _exchanges = [] if exchange is None else [exchange] if isinstance(exchange, str) else exchange

    # - single exchange string for symbol resolution, None if multiple provided
    _single_exchange: ExchangeName_t | None = exchange if isinstance(exchange, str) else None

    # Handle dictionary case where instruments is {exchange: [symbols]}
    if isinstance(instruments, dict):
        for exchange_name, symbol_list in instruments.items():
            if exchange_name not in _exchanges:
                _exchanges.append(exchange_name)

            for symbol in symbol_list:
                instrument, resolved_exchange = _process_single_symbol_or_instrument(
                    symbol, exchange_name, _single_exchange
                )
                if instrument is not None and resolved_exchange is not None:
                    _instrs.append(instrument)
                    _exchanges.append(resolved_exchange)

    # Handle list case
    else:
        for symbol in instruments:
            instrument, resolved_exchange = _process_single_symbol_or_instrument(
                symbol, _single_exchange, _single_exchange
            )
            if instrument is not None and resolved_exchange is not None:
                _instrs.append(instrument)
                _exchanges.append(resolved_exchange)

    return _instrs, list(set(_exchanges))


class _StructureSniffer:
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
    enable_funding: bool = False,
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
    - enable_funding (bool): If True, enables funding rate simulation, default is False.

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
                    signal_timeframe, accurate_stop_orders_execution, run_separate_instruments,
                    enable_funding
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
                    _s1 = c1[instrument.symbol] if isinstance(_s, pd.DataFrame) else _s
                    r.append(
                        SimulationSetup(
                            _t, f"{name}/{instrument.symbol}", _s1, c1,   # type: ignore
                            [instrument],
                            exchanges, capital, basic_currency, commissions, 
                            signal_timeframe, accurate_stop_orders_execution,
                            enable_funding
                        )
                    )
            else:
                r.append(
                    SimulationSetup(
                        _t, name, _s, c1,   # type: ignore
                        setup_instruments,
                        exchanges, capital, basic_currency, commissions, 
                        signal_timeframe, accurate_stop_orders_execution,
                        enable_funding
                    )
                )
        else:
            for j, s in enumerate(configs):
                r.extend(
                    recognize_simulation_configuration(
                        # name + "/" + str(j), s, instruments, exchange, capital, basic_currency, commissions
                        name, s, instruments, exchanges, capital, basic_currency, commissions,  # type: ignore
                        signal_timeframe, accurate_stop_orders_execution, run_separate_instruments,
                        enable_funding
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
                        signal_timeframe, accurate_stop_orders_execution,
                        enable_funding
                    )
                )
        else:
            r.append(
                SimulationSetup(
                    SetupTypes.STRATEGY,
                    name, configs, None, instruments,
                    exchanges, capital, basic_currency, commissions, 
                    signal_timeframe, accurate_stop_orders_execution,
                    enable_funding
                )
            )

    elif _sniffer._is_signal(configs):
        # - check structure of signals
        c1 = _sniffer._check_signals_structure(instruments, configs)  # type: ignore
        setup_instruments = _sniffer._pick_instruments(instruments, c1)
        
        if run_separate_instruments:
            # Create separate setups for each instrument
            for instrument in setup_instruments:
                _c1 = c1[instrument.symbol] if isinstance(c1, pd.DataFrame) else c1
                r.append(
                    SimulationSetup(
                        SetupTypes.SIGNAL,
                        f"{name}/{instrument.symbol}", _c1, None, [instrument],
                        exchanges, capital, basic_currency, commissions, 
                        signal_timeframe, accurate_stop_orders_execution,
                        enable_funding
                    )
                )
        else:
            r.append(
                SimulationSetup(
                    SetupTypes.SIGNAL,
                    name, c1, None, setup_instruments,
                    exchanges, capital, basic_currency, commissions, 
                    signal_timeframe, accurate_stop_orders_execution,
                    enable_funding
                )
            )

    # fmt: on
    return r


def _adjust_open_close_time_indent_secs(timeframe: pd.Timedelta | None, original_indent_secs: int) -> int:
    if timeframe is None:
        return original_indent_secs

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


def _get_default_warmup_period(base_subscription: str, in_timeframe: pd.Timedelta | None) -> pd.Timedelta:
    if in_timeframe is None or base_subscription in [DataType.QUOTE, DataType.TRADE, DataType.ORDERBOOK]:
        return pd.Timedelta("1Min")

    if in_timeframe < pd.Timedelta("1h"):
        return 5 * in_timeframe

    return 2 * in_timeframe


def get_default_warmup(base_subscription: str) -> dict[str, str]:
    assert (tf := safe_dtype_timeframe(base_subscription)) is not None

    # - Apply warmup periods before the start
    #   merge default warmups with strategy warmups (strategy warmups take precedence)
    return {
        str(base_subscription): time_delta_to_str(_get_default_warmup_period(str(base_subscription), tf).asm8.item())
    }


def find_open_close_time_indent_secs_from_subscription(
    subscription: str, original_indent_secs: int
) -> tuple[int, pd.Timedelta | None]:
    """
    Try to detect what time indeent to use in simulated data for given subscription.
    This only applies when OHLC or OHLC_* data is provided and we need to emulated updates from it.
    On raw data (like quotes, trades, etc) we don't need any indents.

    :param subscription: subscription (ohlc must have timeframe like: ohlc(1h) etc)
    :param original_indent_secs: default indent
    :return: derived adjusted time indent in seconds
    """
    _tf = safe_dtype_timeframe(subscription)
    return (_adjust_open_close_time_indent_secs(_tf, original_indent_secs), _tf)


def recognize_simulation_data_config(
    data_provider: IStorage, aux_data: dict[str, IStorage] | None = None, prefetch_config: PrefetchConfig | None = None
) -> SimulationDataConfig:
    """
    Recognizes and configures simulation data based on the provided config.

    This function processes the given data declarations and determines the appropriate
    data provieders and configurations for simulation.

    Parameters:
    - data_provider (IStorage): The data storage provider for the simulation.
    - aux_data (dict[str, IStorage]): auxiliary data providers (like for fundamental data or as substitution of main data etc)

    Returns:
    - instance of SimulationDataConfig class

    Raises:
    - SimulationConfigError: If the data provider type is unsupported or if a requested data type
        cannot be produced from the supported data type.
    """
    _aux_providers: dict[str, IStorage] = {}

    # - quick validation of aux data
    if aux_data and isinstance(aux_data, dict):
        _aux_providers = {}
        for _requested_type, _provider in aux_data.items():
            if not isinstance(_provider, IStorage):
                logger.warning(
                    f"Incorrect data provider type for '{_requested_type}'. Received '{type(_requested_type)}' but must be instance of IStorage class."
                )
                continue

            _aux_providers[_requested_type] = _provider

    return SimulationDataConfig(data_provider, _aux_providers, prefetch_config=prefetch_config)
