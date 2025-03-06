from typing import Literal

import numpy as np
import pandas as pd
from joblib import delayed

from qubx import QubxLogConfig, logger
from qubx.core.basics import SW, DataType
from qubx.core.context import StrategyContext
from qubx.core.exceptions import SimulationConfigError, SimulationError
from qubx.core.helpers import extract_parameters_from_object, full_qualified_class_name
from qubx.core.interfaces import IStrategy
from qubx.core.loggers import InMemoryLogsWriter, StrategyLogging
from qubx.core.lookups import lookup
from qubx.core.metrics import TradingSessionResult
from qubx.data.readers import DataReader
from qubx.pandaz.utils import _frame_to_str
from qubx.utils.misc import ProgressParallel, Stopwatch, get_current_user
from qubx.utils.time import handle_start_stop

from .account import SimulatedAccountProcessor
from .broker import SimulatedBroker
from .data import SimulatedDataProvider
from .runner import BacktestContextRunner
from .utils import (
    DataDecls_t,
    ExchangeName_t,
    SetupTypes,
    SignalsProxy,
    SimulatedCtrlChannel,
    SimulatedLogFormatter,
    SimulatedScheduler,
    SimulatedTimeProvider,
    SimulationDataConfig,
    SimulationSetup,
    StrategiesDecls_t,
    SymbolOrInstrument_t,
    find_instruments_and_exchanges,
    recognize_simulation_configuration,
    recognize_simulation_data_config,
)


def simulate(
    strategies: StrategiesDecls_t,
    data: DataDecls_t,
    capital: float,
    instruments: list[SymbolOrInstrument_t] | dict[ExchangeName_t, list[SymbolOrInstrument_t]],
    commissions: str | None,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp | None = None,
    exchange: ExchangeName_t | None = None,
    base_currency: str = "USDT",
    n_jobs: int = 1,
    silent: bool = False,
    aux_data: DataReader | None = None,
    accurate_stop_orders_execution: bool = False,
    signal_timeframe: str = "1Min",
    open_close_time_indent_secs=1,
    debug: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = "WARNING",
    show_latency_report: bool = False,
    portfolio_log_freq: str = "5Min",
    parallel_backend: Literal["loky", "multiprocessing"] = "multiprocessing",
) -> list[TradingSessionResult]:
    """
    Backtest utility for trading strategies or signals using historical data.

    Args:
        - strategies (StrategiesDecls_t): Trading strategy or signals configuration.
        - data (DataDecls_t): Historical data for simulation, either as a dictionary of DataFrames or a DataReader object.
        - capital (float): Initial capital for the simulation.
        - instruments (list[SymbolOrInstrument_t] | dict[ExchangeName_t, list[SymbolOrInstrument_t]]): List of trading instruments or a dictionary mapping exchanges to instrument lists.
        - commissions (str): Commission structure for trades.
        - start (str | pd.Timestamp): Start time of the simulation.
        - stop (str | pd.Timestamp | None): End time of the simulation. If None, simulates until the last accessible data.
        - exchange (ExchangeName_t | None): Exchange name if not specified in the instruments list.
        - base_currency (str): Base currency for the simulation, default is "USDT".
        - n_jobs (int): Number of parallel jobs for simulation, default is 1.
        - silent (bool): If True, suppresses output during simulation.
        - aux_data (DataReader | None): Auxiliary data provider (default is None).
        - accurate_stop_orders_execution (bool): If True, enables more accurate stop order execution simulation.
        - signal_timeframe (str): Timeframe for signals, default is "1Min".
        - open_close_time_indent_secs (int): Time indent in seconds for open/close times, default is 1.
        - debug (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None): Logging level for debugging.
        - show_latency_report: If True, shows simulator's latency report.

    Returns:
        - list[TradingSessionResult]: A list of TradingSessionResult objects containing the results of each simulation setup.
    """

    # - setup logging
    QubxLogConfig.set_log_level(debug.upper() if debug else "WARNING")

    # - we need to reset stopwatch
    Stopwatch().reset()

    # - process instruments:
    _instruments, _exchanges = find_instruments_and_exchanges(instruments, exchange)

    # - check we have exchange
    if not _exchanges:
        logger.error(
            _msg
            := "No exchange information provided - you can specify it by exchange parameter or use <yellow>EXCHANGE:SYMBOL</yellow> format for symbols"
        )
        raise SimulationError(_msg)

    # - check if instruments are from the same exchange (mmulti-exchanges is not supported yet)
    if len(_exchanges) > 1:
        logger.error(
            _msg := f"Multiple exchanges found: {', '.join(_exchanges)} - this mode is not supported yet in Qubx !"
        )
        raise SimulationError(_msg)

    exchange = _exchanges[0]

    # - recognize provided data
    data_setup = recognize_simulation_data_config(data, _instruments, exchange, open_close_time_indent_secs, aux_data)

    # - recognize setup: it can be either a strategy or set of signals
    simulation_setups = recognize_simulation_configuration(
        "",
        strategies,
        _instruments,
        exchange,
        capital,
        base_currency,
        commissions,
        signal_timeframe,
        accurate_stop_orders_execution,
    )
    if not simulation_setups:
        logger.error(
            _msg
            := "Can't recognize setup - it should be a strategy, a set of signals or list of signals/strategies + tracker !"
        )
        raise SimulationError(_msg)

    # - preprocess start and stop and convert to datetime if necessary
    if stop is None:
        # - check stop time : here we try to backtest till now (may be we need to get max available time from data reader ?)
        stop = pd.Timestamp.now(tz="UTC").astimezone(None)

    _start, _stop = handle_start_stop(start, stop, convert=pd.Timestamp)
    assert isinstance(_start, pd.Timestamp) and isinstance(_stop, pd.Timestamp), "Invalid start and stop times"

    # - run simulations
    return _run_setups(
        simulation_setups,
        data_setup,
        _start,
        _stop,
        n_jobs=n_jobs,
        silent=silent,
        show_latency_report=show_latency_report,
        portfolio_log_freq=portfolio_log_freq,
        parallel_backend=parallel_backend,
    )


def _run_setups(
    strategies_setups: list[SimulationSetup],
    data_setup: SimulationDataConfig,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    n_jobs: int = -1,
    silent: bool = False,
    show_latency_report: bool = False,
    portfolio_log_freq: str = "5Min",
    parallel_backend: Literal["loky", "multiprocessing"] = "multiprocessing",
) -> list[TradingSessionResult]:
    # loggers don't work well with joblib and multiprocessing in general because they contain
    # open file handlers that cannot be pickled. I found a solution which requires the usage of enqueue=True
    # in the logger configuration and specifying backtest "multiprocessing" instead of the default "loky"
    # for joblib. But it works now.
    # See: https://stackoverflow.com/questions/59433146/multiprocessing-logging-how-to-use-loguru-with-joblib-parallel
    _main_loop_silent = len(strategies_setups) == 1
    n_jobs = 1 if _main_loop_silent else n_jobs

    reports = ProgressParallel(
        n_jobs=n_jobs, total=len(strategies_setups), silent=_main_loop_silent, backend=parallel_backend
    )(
        delayed(_run_setup)(
            id, f"Simulated-{id}", setup, data_setup, start, stop, silent, show_latency_report, portfolio_log_freq
        )
        for id, setup in enumerate(strategies_setups)
    )
    return reports  # type: ignore


def _run_setup(
    setup_id: int,
    account_id: str,
    setup: SimulationSetup,
    data_setup: SimulationDataConfig,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    silent: bool,
    show_latency_report: bool,
    portfolio_log_freq: str,
) -> TradingSessionResult:
    _stop = pd.Timestamp(stop)

    # - fees for this exchange
    tcc = lookup.fees.find(setup.exchange.lower(), setup.commissions)
    if tcc is None:
        raise SimulationConfigError(
            f"Can't find transaction costs calculator for '{setup.exchange}' for specification '{setup.commissions}' !"
        )

    channel = SimulatedCtrlChannel("databus", sentinel=(None, None, None, None))
    simulated_clock = SimulatedTimeProvider(np.datetime64(start, "ns"))

    # - we want to see simulate time in log messages
    QubxLogConfig.setup_logger(QubxLogConfig.get_log_level(), SimulatedLogFormatter(simulated_clock).formatter)

    logger.debug(
        f"[<y>simulator</y>] :: Preparing simulated trading on <g>{setup.exchange.upper()}</g> for {setup.capital} {setup.base_currency}..."
    )

    account = SimulatedAccountProcessor(
        account_id=account_id,
        channel=channel,
        base_currency=setup.base_currency,
        initial_capital=setup.capital,
        time_provider=simulated_clock,
        tcc=tcc,
        accurate_stop_orders_execution=setup.accurate_stop_orders_execution,
    )
    scheduler = SimulatedScheduler(channel, lambda: simulated_clock.time().item())
    broker = SimulatedBroker(channel, account, setup.exchange)
    data_provider = SimulatedDataProvider(
        exchange_id=setup.exchange,
        channel=channel,
        scheduler=scheduler,
        time_provider=simulated_clock,
        account=account,
        readers=data_setup.data_providers,
        open_close_time_indent_secs=data_setup.adjusted_open_close_time_indent_secs,
    )

    # - it will store simulation results into memory
    logs_writer = InMemoryLogsWriter(account_id, setup.name, "0")
    strat: IStrategy | None = None

    match setup.setup_type:
        case SetupTypes.STRATEGY:
            strat = setup.generator  # type: ignore

        case SetupTypes.STRATEGY_AND_TRACKER:
            strat = setup.generator  # type: ignore
            strat.tracker = lambda ctx: setup.tracker  # type: ignore

        case SetupTypes.SIGNAL:
            strat = SignalsProxy(timeframe=setup.signal_timeframe)
            data_provider.set_generated_signals(setup.generator)  # type: ignore

            # - we don't need any unexpected triggerings
            _stop = min(setup.generator.index[-1], _stop)  # type: ignore

        case SetupTypes.SIGNAL_AND_TRACKER:
            strat = SignalsProxy(timeframe=setup.signal_timeframe)
            strat.tracker = lambda ctx: setup.tracker
            data_provider.set_generated_signals(setup.generator)  # type: ignore

            # - we don't need any unexpected triggerings
            _stop = min(setup.generator.index[-1], _stop)  # type: ignore

        case _:
            raise SimulationError(f"Unsupported setup type: {setup.setup_type} !")

    if not isinstance(strat, IStrategy):
        raise SimulationConfigError(f"Strategy should be an instance of IStrategy, but got {strat} !")

    # - get aux data provider
    _aux_data = data_setup.get_timeguarded_aux_reader(simulated_clock)

    ctx = StrategyContext(
        strategy=strat,
        broker=broker,
        data_provider=data_provider,
        account=account,
        scheduler=scheduler,
        time_provider=simulated_clock,
        instruments=setup.instruments,
        logging=StrategyLogging(logs_writer, portfolio_log_freq=portfolio_log_freq),
        aux_data_provider=_aux_data,
    )

    # - setup base subscription from spec
    if ctx.get_base_subscription() == DataType.NONE:
        logger.debug(
            f"[<y>simulator</y>] :: Setting up default base subscription: {data_setup.default_base_subscription}"
        )
        ctx.set_base_subscription(data_setup.default_base_subscription)

    # - set default on_event schedule if detected and strategy didn't set it's own schedule
    if not ctx.get_event_schedule("time") and data_setup.default_trigger_schedule:
        logger.debug(f"[<y>simulator</y>] :: Setting default schedule: {data_setup.default_trigger_schedule}")
        ctx.set_event_schedule(data_setup.default_trigger_schedule)

    # - get strategy parameters BEFORE simulation start
    #   potentially strategy may change it's parameters during simulation
    _s_class, _s_params = "", None
    if setup.setup_type in [SetupTypes.STRATEGY, SetupTypes.STRATEGY_AND_TRACKER]:
        _s_params = extract_parameters_from_object(setup.generator)
        _s_class = full_qualified_class_name(setup.generator)

    # Use the BacktestContextRunner to run the simulation
    assert isinstance(start, pd.Timestamp) and isinstance(_stop, pd.Timestamp), "Invalid start and stop times"
    runner = BacktestContextRunner(ctx, data_provider, data_setup)
    runner.run(start, _stop, silent=silent)

    # - service latency report
    if show_latency_report:
        _l_r = SW.latency_report()
        if _l_r is not None:
            logger.info(
                "<BLUE>   Time spent in simulation report   </BLUE>\n<r>"
                + _frame_to_str(
                    _l_r.sort_values("latency", ascending=False).reset_index(drop=True), "simulation", -1, -1, False
                )
                + "</r>"
            )

    return TradingSessionResult(
        setup_id,
        setup.name,
        start,
        stop,
        setup.exchange,
        setup.instruments,
        setup.capital,
        setup.base_currency,
        setup.commissions,
        logs_writer.get_portfolio(as_plain_dataframe=True),
        logs_writer.get_executions(),
        logs_writer.get_signals(),
        strategy_class=_s_class,
        parameters=_s_params,
        is_simulation=True,
        author=get_current_user(),
    )
