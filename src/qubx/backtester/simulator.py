from typing import Literal

import pandas as pd
from joblib import delayed

from qubx import QubxLogConfig, logger
from qubx.core.exceptions import SimulationError
from qubx.core.metrics import TradingSessionResult
from qubx.data.readers import DataReader
from qubx.utils.misc import ProgressParallel, Stopwatch, get_current_user
from qubx.utils.runner.configs import EmissionConfig
from qubx.utils.runner.factory import create_metric_emitters
from qubx.utils.time import handle_start_stop

from .runner import SimulationRunner
from .utils import (
    DataDecls_t,
    ExchangeName_t,
    SimulatedLogFormatter,
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
    emission: EmissionConfig | None = None,
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
        - portfolio_log_freq (str): Frequency for portfolio logging, default is "5Min".
        - parallel_backend (Literal["loky", "multiprocessing"]): Backend for parallel processing, default is "multiprocessing".
        - emission (EmissionConfig | None): Configuration for metric emitters, default is None.

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
        emission=emission,
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
    emission: EmissionConfig | None = None,
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
            id,
            f"Simulated-{id}",
            setup,
            data_setup,
            start,
            stop,
            silent,
            show_latency_report,
            portfolio_log_freq,
            emission,
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
    emission: EmissionConfig | None = None,
) -> TradingSessionResult:
    # Create metric emitter if configured
    emitter = None
    if emission is not None:
        emitter = create_metric_emitters(emission, setup.name)

    runner = SimulationRunner(
        setup=setup,
        data_config=data_setup,
        start=start,
        stop=stop,
        account_id=account_id,
        portfolio_log_freq=portfolio_log_freq,
        emitter=emitter,
    )

    # - we want to see simulate time in log messages
    QubxLogConfig.setup_logger(
        level=QubxLogConfig.get_log_level(), custom_formatter=SimulatedLogFormatter(runner.ctx).formatter
    )

    runner.run(silent=silent)

    # - service latency report
    if show_latency_report:
        runner.print_latency_report()

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
        runner.logs_writer.get_portfolio(as_plain_dataframe=True),
        runner.logs_writer.get_executions(),
        runner.logs_writer.get_signals(),
        strategy_class=runner.strategy_class,
        parameters=runner.strategy_params,
        is_simulation=True,
        author=get_current_user(),
    )
