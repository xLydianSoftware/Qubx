from typing import Literal

import pandas as pd
from joblib import delayed

from qubx import QubxLogConfig, logger
from qubx.core.basics import Instrument
from qubx.core.exceptions import SimulationError
from qubx.core.metrics import TradingSessionResult
from qubx.data.readers import DataReader
from qubx.emitters.inmemory import InMemoryMetricEmitter
from qubx.utils.misc import ProgressParallel, Stopwatch, get_current_user
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
    capital: float | dict[str, float],
    instruments: list[str] | list[Instrument] | dict[ExchangeName_t, list[SymbolOrInstrument_t]],
    commissions: str | dict[str, str | None] | None,
    start: str | pd.Timestamp,
    stop: str | pd.Timestamp | None = None,
    exchange: ExchangeName_t | list[ExchangeName_t] | None = None,
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
    enable_inmemory_emitter: bool = False,
    emitter_stats_interval: str = "1h",
    run_separate_instruments: bool = False,
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
        - enable_inmemory_emitter (bool): If True, attaches an in-memory metric emitter and returns its dataframe in TradingSessionResult.emitter_data.
        - emitter_stats_interval (str): Interval for emitting stats in the in-memory emitter (default: "1h").
        - run_separate_instruments (bool): If True, creates separate simulation setups for each instrument, default is False.

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

    # - recognize provided data
    data_setup = recognize_simulation_data_config(data, _instruments, open_close_time_indent_secs, aux_data)

    # - recognize setup: it can be either a strategy or set of signals
    simulation_setups = recognize_simulation_configuration(
        name="",
        configs=strategies,
        instruments=_instruments,
        exchanges=_exchanges,
        capital=capital,
        basic_currency=base_currency,
        commissions=commissions,
        signal_timeframe=signal_timeframe,
        accurate_stop_orders_execution=accurate_stop_orders_execution,
        run_separate_instruments=run_separate_instruments,
    )
    if not simulation_setups:
        logger.error(
            _msg
            := "Can't recognize setup - it should be a strategy, a set of signals or list of signals/strategies + tracker !"
        )
        raise SimulationError(_msg)

    # - inform about separate instruments mode
    if run_separate_instruments and len(simulation_setups) > 1:
        logger.info(f"Running separate simulations for each instrument. Total simulations: {len(simulation_setups)}")

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
        enable_inmemory_emitter=enable_inmemory_emitter,
        emitter_stats_interval=emitter_stats_interval,
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
    enable_inmemory_emitter: bool = False,
    emitter_stats_interval: str = "1h",
) -> list[TradingSessionResult]:
    # loggers don't work well with joblib and multiprocessing in general because they contain
    # open file handlers that cannot be pickled. I found a solution which requires the usage of enqueue=True
    # in the logger configuration and specifying backtest "multiprocessing" instead of the default "loky"
    # for joblib. But it works now.
    # See: https://stackoverflow.com/questions/59433146/multiprocessing-logging-how-to-use-loguru-with-joblib-parallel
    _main_loop_silent = len(strategies_setups) == 1
    n_jobs = 1 if _main_loop_silent else n_jobs

    if n_jobs == 1:
        reports = [
            _run_setup(
                id,
                f"Simulated-{id}",
                setup,
                data_setup,
                start,
                stop,
                silent,
                show_latency_report,
                portfolio_log_freq,
                enable_inmemory_emitter,
                emitter_stats_interval,
            )
            for id, setup in enumerate(strategies_setups)
        ]
    else:
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
                enable_inmemory_emitter,
                emitter_stats_interval,
            )
            for id, setup in enumerate(strategies_setups)
        )

    # Filter out None results and log warnings for failed simulations
    successful_reports = []
    for i, report in enumerate(reports):
        if report is None:
            logger.warning(f"Simulation setup {i} failed - skipping from results")
        else:
            successful_reports.append(report)

    return successful_reports


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
    enable_inmemory_emitter: bool = False,
    emitter_stats_interval: str = "1h",
) -> TradingSessionResult | None:
    try:
        emitter = None
        emitter_data = None
        if enable_inmemory_emitter:
            emitter = InMemoryMetricEmitter(stats_interval=emitter_stats_interval)
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

        # Convert commissions to the expected type for TradingSessionResult
        commissions_for_result = setup.commissions
        if isinstance(commissions_for_result, dict):
            # Filter out None values to match TradingSessionResult expected type
            commissions_for_result = {k: v for k, v in commissions_for_result.items() if v is not None}

        if enable_inmemory_emitter and emitter is not None:
            emitter_data = emitter.get_dataframe()

        return TradingSessionResult(
            setup_id,
            setup.name,
            start,
            stop,
            setup.exchanges,
            setup.instruments,
            capital=setup.capital,
            base_currency=setup.base_currency,
            commissions=commissions_for_result,
            portfolio_log=runner.logs_writer.get_portfolio(as_plain_dataframe=True),
            executions_log=runner.logs_writer.get_executions(),
            signals_log=runner.logs_writer.get_signals(),
            targets_log=runner.logs_writer.get_targets(),
            strategy_class=runner.strategy_class,
            parameters=runner.strategy_params,
            is_simulation=True,
            author=get_current_user(),
            emitter_data=emitter_data,
        )
    except Exception as e:
        logger.error(f"Simulation setup {setup_id} failed with error: {e}")
        return None
