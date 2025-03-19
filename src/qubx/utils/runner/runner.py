import inspect
import os
import socket
import time
import traceback
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Optional

import pandas as pd

from qubx import QubxLogConfig, logger
from qubx.backtester import simulate
from qubx.backtester.account import SimulatedAccountProcessor
from qubx.backtester.broker import SimulatedBroker
from qubx.backtester.optimization import variate
from qubx.backtester.runner import SimulationRunner
from qubx.backtester.utils import (
    SetupTypes,
    SimulatedLogFormatter,
    SimulationConfigError,
    SimulationSetup,
    recognize_simulation_data_config,
)
from qubx.connectors.ccxt.account import CcxtAccountProcessor
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import get_ccxt_exchange
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    LiveTimeProvider,
    RestoredState,
    TransactionCostsCalculator,
)
from qubx.core.context import CachedMarketDataHolder, StrategyContext
from qubx.core.helpers import BasicScheduler
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import (
    IAccountProcessor,
    IBroker,
    IDataProvider,
    IMetricEmitter,
    IStrategyContext,
    IStrategyLifecycleNotifier,
    ITimeProvider,
    ITradeDataExport,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.lookups import lookup
from qubx.data.composite import CompositeReader
from qubx.data.readers import DataReader
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.restarts.state_resolvers import StateResolver
from qubx.restarts.time_finders import TimeFinder
from qubx.restorers import create_state_restorer
from qubx.utils.misc import class_import, makedirs, red
from qubx.utils.runner.configs import (
    ExchangeConfig,
    LoggingConfig,
    ReaderConfig,
    RestorerConfig,
    StrategyConfig,
    WarmupConfig,
    load_simulation_config_from_yaml,
    load_strategy_config_from_yaml,
)

from .accounts import AccountConfigurationManager


def run_strategy_yaml(
    config_file: Path,
    account_file: Path | None = None,
    paper: bool = False,
    restore: bool = False,
    blocking: bool = False,
) -> IStrategyContext:
    """
    Run the strategy with the given configuration file.

    Args:
        config_file (Path): The path to the configuration file.
        account_file (Path, optional): The path to the account configuration file. Defaults to None.
        paper (bool, optional): Whether to run in paper trading mode. Defaults to False.
        jupyter (bool, optional): Whether to run in a Jupyter console. Defaults to False.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    if account_file is not None and not account_file.exists():
        raise FileNotFoundError(f"Account configuration file not found: {account_file}")

    acc_manager = AccountConfigurationManager(account_file, config_file.parent, search_qubx_dir=True)
    stg_config = load_strategy_config_from_yaml(config_file)
    return run_strategy(stg_config, acc_manager, paper=paper, restore=restore, blocking=blocking)


def run_strategy_yaml_in_jupyter(
    config_file: Path, account_file: Path | None = None, paper: bool = False, restore: bool = False
) -> None:
    """
    Run a strategy in a Jupyter notebook.

    Args:
        config_file: Path to the strategy configuration file
        account_file: Path to the account configuration file
        paper: Whether to run in paper trading mode
        restore: Whether to restore the strategy state
    """
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        logger.error("Can't find <r>nest_asyncio</r> module - try to install it first")
        return

    try:
        from jupyter_console.app import ZMQTerminalIPythonApp
    except ImportError:
        logger.error(
            "Can't find <r>ZMQTerminalIPythonApp</r> module - try to install <g>jupyter-console</g> package first"
        )
        return

    class TerminalRunner(ZMQTerminalIPythonApp):
        def __init__(self, **kwargs) -> None:
            self.init_code = kwargs.pop("init_code")
            super().__init__(**kwargs)

        def init_banner(self):
            pass

        def initialize(self, argv=None):
            super().initialize(argv=[])
            self.shell.run_cell(self.init_code)

    _base = Path(__file__).parent.absolute()
    with open(_base / "_jupyter_runner.pyt", "r") as f:
        content = f.read()

    content_with_values = content.format_map(
        {"config_file": config_file, "account_file": account_file, "paper": paper, "restore": restore}
    )
    logger.info("Running in Jupyter console")
    TerminalRunner.launch_instance(init_code=content_with_values)


def run_strategy(
    config: StrategyConfig,
    account_manager: AccountConfigurationManager,
    paper: bool = False,
    restore: bool = False,
    blocking: bool = False,
) -> IStrategyContext:
    """
    Run a strategy with the given configuration.

    Args:
        config (StrategyConfig): The configuration of the strategy.
        account_manager (AccountManager): The account manager to use.
        paper (bool, optional): Whether to run in paper trading mode. Defaults to False.
        blocking (bool, optional): Whether to block the main thread. Defaults to False.

    Returns:
        IStrategyContext: The strategy context.
    """
    QubxLogConfig.setup_logger(
        level=QubxLogConfig.get_log_level(),
        custom_formatter=(simulated_formatter := SimulatedLogFormatter(LiveTimeProvider())).formatter,
    )

    # Restore state if configured
    restored_state = _restore_state(config.warmup.restorer if config.warmup else None) if restore else None

    # Create the strategy context
    ctx = create_strategy_context(
        config=config,
        account_manager=account_manager,
        paper=paper,
        restored_state=restored_state,
        simulated_formatter=simulated_formatter,
    )

    try:
        _run_warmup(
            ctx=ctx,
            restored_state=restored_state,
            exchanges=config.exchanges,
            warmup=config.warmup,
            aux_config=config.aux,
            simulated_formatter=simulated_formatter,
        )
    except KeyboardInterrupt:
        logger.info("Warmup interrupted by user")
        return ctx
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise e

    # Start the strategy context
    if blocking:
        try:
            ctx.start(blocking=True)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            ctx.stop()
    else:
        ctx.start()

    return ctx


def _restore_state(restorer_config: RestorerConfig | None) -> RestoredState | None:
    if restorer_config is None:
        restorer_config = RestorerConfig(type="CsvStateRestorer", parameters={"base_dir": "logs"})

    state_restorer = create_state_restorer(
        restorer_config.type,
        restorer_config.parameters,
    )

    state = state_restorer.restore_state()
    logger.info(
        f"<yellow>Restored state with {len(state.positions)} positions "
        f"and {sum(len(s) for s in state.instrument_to_target_positions.values())} signals</yellow>"
    )
    logger.info("<yellow> - Positions:</yellow>")
    for position in state.positions.values():
        logger.info(f"<yellow>   - {position}</yellow>")
    return state


def _resolve_env_vars(value: str) -> str:
    """
    Resolve environment variables in a value.
    If the value is a string and starts with 'env:', the rest is treated as an environment variable name.
    """
    if isinstance(value, str) and value.startswith("env:"):
        env_var = value[4:].strip()
        _value = os.environ.get(env_var)
        if _value is None:
            raise ValueError(f"Environment variable {env_var} not found")
        return _value
    return value


def _create_exporters(config: StrategyConfig, strategy_name: str) -> Optional[ITradeDataExport]:
    """
    Create exporters from the configuration.

    Args:
        config: Strategy configuration
        strategy_name: Name of the strategy

    Returns:
        ITradeDataExport or None if no exporters are configured
    """
    if not config.exporters:
        return None

    exporters = []

    for exporter_config in config.exporters:
        exporter_class_name = exporter_config.exporter
        if "." not in exporter_class_name:
            exporter_class_name = f"qubx.exporters.{exporter_class_name}"

        try:
            exporter_class = class_import(exporter_class_name)

            # Process parameters and resolve environment variables
            params = {}
            for key, value in exporter_config.parameters.items():
                resolved_value = _resolve_env_vars(value)

                # Handle formatter if specified
                if key == "formatter" and isinstance(resolved_value, dict):
                    formatter_class_name = resolved_value.get("class")
                    formatter_args = resolved_value.get("args", {})

                    # Resolve env vars in formatter args
                    for fmt_key, fmt_value in formatter_args.items():
                        formatter_args[fmt_key] = _resolve_env_vars(fmt_value)

                    if formatter_class_name:
                        if "." not in formatter_class_name:
                            formatter_class_name = f"qubx.exporters.formatters.{formatter_class_name}"
                        formatter_class = class_import(formatter_class_name)
                        params[key] = formatter_class(**formatter_args)
                else:
                    params[key] = resolved_value

            # Add strategy_name if the exporter requires it and it's not already provided
            if "strategy_name" in inspect.signature(exporter_class).parameters and "strategy_name" not in params:
                params["strategy_name"] = strategy_name

            # Create the exporter instance
            exporter = exporter_class(**params)
            exporters.append(exporter)
            logger.info(f"Created exporter: {exporter_class_name}")

        except Exception as e:
            logger.error(f"Failed to create exporter {exporter_class_name}: {e}")
            logger.opt(colors=False).error(f"Exporter parameters: {exporter_config.parameters}")

    if not exporters:
        return None

    # If there's only one exporter, return it directly
    if len(exporters) == 1:
        return exporters[0]

    # If there are multiple exporters, create a composite exporter
    from qubx.exporters.composite import CompositeExporter

    return CompositeExporter(exporters)


def _create_metric_emitters(config: StrategyConfig, strategy_name: str) -> Optional[IMetricEmitter]:
    """
    Create metric emitters from the configuration.

    Args:
        config: Strategy configuration
        strategy_name: Name of the strategy

    Returns:
        IMetricEmitter or None if no metric emitters are configured
    """
    if not hasattr(config, "emission") or not config.emission or not config.emission.emitters:
        return None

    emitters = []
    stats_to_emit = config.emission.stats_to_emit
    stats_interval = config.emission.stats_interval

    for metric_config in config.emission.emitters:
        emitter_class_name = metric_config.emitter
        if "." not in emitter_class_name:
            emitter_class_name = f"qubx.emitters.{emitter_class_name}"

        try:
            emitter_class = class_import(emitter_class_name)

            # Process parameters and resolve environment variables
            params = {}
            for key, value in metric_config.parameters.items():
                params[key] = _resolve_env_vars(value)

            # Add strategy_name if the emitter requires it and it's not already provided
            if "strategy_name" in inspect.signature(emitter_class).parameters and "strategy_name" not in params:
                params["strategy_name"] = strategy_name

            # Add stats_to_emit if the emitter supports it and it's not already provided
            if (
                "stats_to_emit" in inspect.signature(emitter_class).parameters
                and "stats_to_emit" not in params
                and stats_to_emit
            ):
                params["stats_to_emit"] = stats_to_emit

            # Add stats_interval if the emitter supports it and it's not already provided
            if "stats_interval" in inspect.signature(emitter_class).parameters and "stats_interval" not in params:
                params["stats_interval"] = stats_interval

            # Process tags and add strategy_name as a tag
            tags = dict(metric_config.tags)
            for k, v in tags.items():
                tags[k] = _resolve_env_vars(v)

            tags["strategy"] = strategy_name

            # Add tags if the emitter supports it
            if "tags" in inspect.signature(emitter_class).parameters:
                params["tags"] = tags

            # Create the emitter instance
            emitter = emitter_class(**params)
            emitters.append(emitter)
            logger.info(f"Created metric emitter: {emitter_class_name}")
        except Exception as e:
            logger.error(f"Failed to create metric emitter {metric_config.emitter}: {e}")
            logger.opt(colors=False).error(traceback.format_exc())

    if not emitters:
        return None
    elif len(emitters) == 1:
        return emitters[0]
    else:
        return CompositeMetricEmitter(emitters, stats_interval=stats_interval)


def _create_lifecycle_notifiers(config: StrategyConfig, strategy_name: str) -> Optional[IStrategyLifecycleNotifier]:
    """
    Create lifecycle notifiers from the configuration.

    Args:
        config: Strategy configuration
        strategy_name: Name of the strategy

    Returns:
        IStrategyLifecycleNotifier or None if no lifecycle notifiers are configured
    """
    if not config.notifiers:
        return None

    notifiers = []

    for notifier_config in config.notifiers:
        notifier_class_name = notifier_config.notifier
        if "." not in notifier_class_name:
            notifier_class_name = f"qubx.notifications.{notifier_class_name}"

        try:
            notifier_class = class_import(notifier_class_name)

            # Process parameters and resolve environment variables
            params = {}
            for key, value in notifier_config.parameters.items():
                params[key] = _resolve_env_vars(value)

            # Create the notifier instance
            notifier = notifier_class(**params)
            notifiers.append(notifier)
            logger.info(f"Created lifecycle notifier: {notifier_class_name}")

        except Exception as e:
            logger.error(f"Failed to create lifecycle notifier {notifier_class_name}: {e}")
            logger.opt(colors=False).error(f"Lifecycle notifier parameters: {notifier_config.parameters}")

    if not notifiers:
        return None

    # If there's only one notifier, return it directly
    if len(notifiers) == 1:
        return notifiers[0]

    # If there are multiple notifiers, create a composite notifier
    from qubx.notifications.composite import CompositeLifecycleNotifier

    return CompositeLifecycleNotifier(notifiers)


def create_strategy_context(
    config: StrategyConfig,
    account_manager: AccountConfigurationManager,
    paper: bool,
    restored_state: RestoredState | None,
    simulated_formatter: SimulatedLogFormatter,
) -> IStrategyContext:
    """
    Create a strategy context from the given configuration.
    """
    stg_name = _get_strategy_name(config)
    _run_mode = "paper" if paper else "live"

    if isinstance(config.strategy, list):
        _strategy_class = reduce(lambda x, y: x + y, [class_import(x) for x in config.strategy])
    else:
        _strategy_class = class_import(config.strategy)

    _logging = _setup_strategy_logging(stg_name, config.logging, simulated_formatter)

    _aux_reader = _construct_reader(config.aux) if config.aux else None

    # Create exporters if configured
    _exporter = _create_exporters(config, stg_name)

    # Create metric emitters
    _metric_emitter = _create_metric_emitters(config, stg_name)

    # Create lifecycle notifiers
    _lifecycle_notifier = _create_lifecycle_notifiers(config, stg_name)

    # Create strategy initializer
    _initializer = BasicStrategyInitializer()

    _time = LiveTimeProvider()
    _chan = CtrlChannel("databus", sentinel=(None, None, None, None))
    _sched = BasicScheduler(_chan, lambda: _time.time().item())

    # Set time provider for metric emitters
    if _metric_emitter is not None:
        _metric_emitter.set_time_provider(_time)

    exchanges = list(config.exchanges.keys())
    if len(exchanges) > 1:
        raise ValueError("Multiple exchanges are not supported yet !")

    _exchange_to_tcc = {}
    _exchange_to_broker = {}
    _exchange_to_data_provider = {}
    _exchange_to_account = {}
    _instruments = []
    for exchange_name, exchange_config in config.exchanges.items():
        _exchange_to_tcc[exchange_name] = (tcc := _create_tcc(exchange_name, account_manager))
        _exchange_to_data_provider[exchange_name] = _create_data_provider(
            exchange_name,
            exchange_config,
            time_provider=_time,
            channel=_chan,
            account_manager=account_manager,
        )
        _exchange_to_account[exchange_name] = (
            account := _create_account_processor(
                exchange_name,
                exchange_config,
                channel=_chan,
                time_provider=_time,
                account_manager=account_manager,
                tcc=tcc,
                paper=paper,
                restored_state=restored_state,
            )
        )
        _exchange_to_broker[exchange_name] = _create_broker(
            exchange_name,
            exchange_config,
            _chan,
            time_provider=_time,
            account=account,
            account_manager=account_manager,
            paper=paper,
        )
        _instruments.extend(_create_instruments_for_exchange(exchange_name, exchange_config))

    # TODO: rework strategy context to support multiple exchanges
    _broker = _exchange_to_broker[exchanges[0]]
    _data_provider = _exchange_to_data_provider[exchanges[0]]
    _account = _exchange_to_account[exchanges[0]]
    _initializer = BasicStrategyInitializer(simulation=_data_provider.is_simulation)

    logger.info(f"- Strategy: <blue>{stg_name}</blue>\n- Mode: {_run_mode}\n- Parameters: {config.parameters}")
    ctx = StrategyContext(
        strategy=_strategy_class,
        broker=_broker,
        data_provider=_data_provider,
        account=_account,
        scheduler=_sched,
        time_provider=_time,
        instruments=_instruments,
        logging=_logging,
        config=config.parameters,
        aux_data_provider=_aux_reader,
        exporter=_exporter,
        emitter=_metric_emitter,
        lifecycle_notifier=_lifecycle_notifier,
        initializer=_initializer,
        strategy_name=stg_name,
    )

    return ctx


def _get_strategy_name(cfg: StrategyConfig) -> str:
    if cfg.name is not None:
        return cfg.name
    if isinstance(cfg.strategy, list):
        return "_".join(map(lambda x: x.split(".")[-1], cfg.strategy))
    return cfg.strategy.split(".")[-1]


def _setup_strategy_logging(
    stg_name: str, log_config: LoggingConfig, simulated_formatter: SimulatedLogFormatter
) -> StrategyLogging:
    log_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    run_folder = f"logs/run_{log_id}"
    logger.add(
        f"{run_folder}/strategy/{stg_name}_{{time}}.log",
        format=simulated_formatter.formatter,
        rotation="100 MB",
        colorize=False,
    )

    run_id = f"{socket.gethostname()}-{str(int(time.time() * 10**9))}"

    _log_writer_name = log_config.logger
    if "." not in _log_writer_name:
        _log_writer_name = f"qubx.core.loggers.{_log_writer_name}"

    logger.debug(f"Setup <g>{_log_writer_name}</g> logger...")
    _log_writer_class = class_import(_log_writer_name)
    _log_writer_params = {
        "account_id": "account",
        "strategy_id": stg_name,
        "run_id": run_id,
        "log_folder": run_folder,
    }
    _log_writer_sig_params = inspect.signature(_log_writer_class).parameters
    _log_writer_params = {k: v for k, v in _log_writer_params.items() if k in _log_writer_sig_params}
    _log_writer = _log_writer_class(**_log_writer_params)
    stg_logging = StrategyLogging(
        logs_writer=_log_writer,
        positions_log_freq=log_config.position_interval,
        portfolio_log_freq=log_config.portfolio_interval,
        heartbeat_freq=log_config.heartbeat_interval,
    )
    return stg_logging


def _construct_reader(reader_config: ReaderConfig | None) -> DataReader | None:
    if reader_config is None:
        return None

    from qubx.data.registry import ReaderRegistry

    try:
        # Use the ReaderRegistry.get method to construct the reader directly
        return ReaderRegistry.get(reader_config.reader, **reader_config.args)
    except ValueError as e:
        # Log the error and re-raise
        logger.error(f"Failed to construct reader: {e}")
        raise


def _create_tcc(exchange_name: str, account_manager: AccountConfigurationManager) -> TransactionCostsCalculator:
    if exchange_name == "BINANCE.PM":
        # TODO: clean this up
        exchange_name = "BINANCE.UM"
    settings = account_manager.get_exchange_settings(exchange_name)
    tcc = lookup.fees.find(exchange_name, settings.commissions)
    assert tcc is not None, f"Can't find fees calculator for {exchange_name} exchange"
    return tcc


def _create_data_provider(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    time_provider: ITimeProvider,
    channel: CtrlChannel,
    account_manager: AccountConfigurationManager,
) -> IDataProvider:
    settings = account_manager.get_exchange_settings(exchange_name)
    match exchange_config.connector.lower():
        case "ccxt":
            exchange = get_ccxt_exchange(exchange_name, use_testnet=settings.testnet)
            return CcxtDataProvider(exchange, time_provider, channel)
        case _:
            raise ValueError(f"Connector {exchange_config.connector} is not supported yet !")


def _create_account_processor(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account_manager: AccountConfigurationManager,
    tcc: TransactionCostsCalculator,
    paper: bool,
    restored_state: RestoredState | None = None,
) -> IAccountProcessor:
    if paper:
        settings = account_manager.get_exchange_settings(exchange_name)
        return SimulatedAccountProcessor(
            account_id=exchange_name,
            channel=channel,
            base_currency=settings.base_currency,
            time_provider=time_provider,
            tcc=tcc,
            initial_capital=settings.initial_capital,
            restored_state=restored_state,
        )

    creds = account_manager.get_exchange_credentials(exchange_name)
    match exchange_config.connector.lower():
        case "ccxt":
            exchange = get_ccxt_exchange(
                exchange_name, use_testnet=creds.testnet, api_key=creds.api_key, secret=creds.secret
            )
            return CcxtAccountProcessor(
                exchange_name,
                exchange,
                channel,
                time_provider,
                base_currency=creds.base_currency,
                tcc=tcc,
            )
        case _:
            raise ValueError(f"Connector {exchange_config.connector} is not supported yet !")


def _create_broker(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    account_manager: AccountConfigurationManager,
    paper: bool,
) -> IBroker:
    if paper:
        assert isinstance(account, SimulatedAccountProcessor)
        return SimulatedBroker(channel=channel, account=account, exchange_id=exchange_name)

    creds = account_manager.get_exchange_credentials(exchange_name)

    match exchange_config.connector.lower():
        case "ccxt":
            exchange = get_ccxt_exchange(
                exchange_name, use_testnet=creds.testnet, api_key=creds.api_key, secret=creds.secret
            )
            return CcxtBroker(exchange, channel, time_provider, account)
        case _:
            raise ValueError(f"Connector {exchange_config.connector} is not supported yet !")


def _create_instruments_for_exchange(exchange_name: str, exchange_config: ExchangeConfig) -> list[Instrument]:
    exchange_name = exchange_name.upper()
    if exchange_name == "BINANCE.PM":
        # TODO: clean this up
        exchange_name = "BINANCE.UM"
    symbols = exchange_config.universe
    instruments = [lookup.find_symbol(exchange_name, symbol.upper()) for symbol in symbols]
    instruments = [i for i in instruments if i is not None]
    return instruments


def _create_data_type_readers(warmup: WarmupConfig | None) -> dict[str, DataReader]:
    """
    Create a dictionary mapping data types to readers based on the warmup configuration.

    This function ensures that identical reader configurations are only instantiated once,
    and multiple data types can share the same reader instance if they have identical configurations.

    Args:
        warmup: The warmup configuration containing reader definitions.

    Returns:
        A dictionary mapping data types to reader instances.
    """
    if warmup is None:
        return {}

    # First, create unique readers to avoid duplicate instantiation
    unique_readers = {}  # Maps reader config hash to reader instance
    data_type_to_reader = {}  # Maps data type to reader instance

    for typed_reader_config in warmup.readers:
        data_types = typed_reader_config.data_type
        if isinstance(data_types, str):
            data_types = [data_types]
        readers_for_types = []

        for reader_config in typed_reader_config.readers:
            # Create a hashable representation of the reader config
            # Create a hashable key from reader name and stringified args
            if reader_config.args:
                args_str = str(reader_config.args)
                reader_key = f"{reader_config.reader}:{args_str}"
            else:
                reader_key = reader_config.reader

            # Check if we've already created this reader
            if reader_key not in unique_readers:
                try:
                    reader = _construct_reader(reader_config)
                    if reader is None:
                        raise ValueError(f"Reader {reader_config.reader} could not be created")
                    unique_readers[reader_key] = reader
                except Exception as e:
                    logger.error(f"Reader {reader_config.reader} could not be created: {e}")
                    raise

            # Add the reader to the list for these data types
            readers_for_types.append(unique_readers[reader_key])

        # Create a composite reader if needed, or use the single reader
        if len(readers_for_types) > 1:
            composite_reader = CompositeReader(readers_for_types)
            for data_type in data_types:
                data_type_to_reader[data_type] = composite_reader
        elif len(readers_for_types) == 1:
            single_reader = readers_for_types[0]
            for data_type in data_types:
                data_type_to_reader[data_type] = single_reader

    return data_type_to_reader


def _run_warmup(
    ctx: IStrategyContext,
    restored_state: RestoredState | None,
    exchanges: dict[str, ExchangeConfig],
    warmup: WarmupConfig | None,
    aux_config: ReaderConfig | None,
    simulated_formatter: SimulatedLogFormatter,
) -> None:
    """
    Run the warmup period for the strategy.
    """
    if warmup is None:
        return

    initializer = ctx.initializer
    warmup_period = initializer.get_warmup()

    # - find start time for warmup
    if (start_time_finder := initializer.get_start_time_finder()) is None:
        initializer.set_start_time_finder(start_time_finder := TimeFinder.LAST_SIGNAL)

    if initializer.get_state_resolver() is None:
        initializer.set_state_resolver(StateResolver.REDUCE_ONLY)

    current_time = ctx.time()
    warmup_start_time = current_time
    if restored_state is not None:
        warmup_start_time = start_time_finder(current_time, restored_state)
        time_delta = pd.Timedelta(current_time - warmup_start_time)
        if time_delta.total_seconds() > 0:
            logger.info(f"<yellow>Start time finder estimated to go back in time by {time_delta}</yellow>")

    if warmup_period is not None:
        logger.info(f"<yellow>Warmup period is set to {pd.Timedelta(warmup_period)}</yellow>")
        warmup_start_time -= warmup_period

    if warmup_start_time == current_time:
        # if start time is the same as current time, we don't need to run warmup
        return

    logger.info(f"<yellow>Warmup start time: {warmup_start_time}</yellow>")

    # - construct warmup readers
    data_type_to_reader = _create_data_type_readers(warmup)

    if not data_type_to_reader:
        logger.warning("<yellow>No readers were created for warmup</yellow>")
        return

    _aux_reader = _construct_reader(aux_config) if aux_config else None

    # - create instruments
    instruments = []
    for exchange_name, exchange_config in exchanges.items():
        instruments.extend(_create_instruments_for_exchange(exchange_name, exchange_config))
    if restored_state is not None:
        instruments.extend(restored_state.instrument_to_target_positions.keys())

    assert isinstance(ctx.initializer, BasicStrategyInitializer)
    ctx.initializer.simulation = True

    logger.info(f"<yellow>Running warmup from {warmup_start_time} to {current_time}</yellow>")
    warmup_runner = SimulationRunner(
        setup=SimulationSetup(
            setup_type=SetupTypes.STRATEGY,
            name=getattr(ctx, "_strategy_name", ctx.strategy.__class__.__name__),
            generator=ctx.strategy,
            tracker=None,
            instruments=instruments,
            exchange=ctx.broker.exchange(),
            capital=ctx.account.get_capital(),
            base_currency=ctx.account.get_base_currency(),
            commissions=None,  # TODO: get commissions from somewhere
        ),
        data_config=recognize_simulation_data_config(
            decls=data_type_to_reader,  # type: ignore
            instruments=instruments,
            exchange=ctx.broker.exchange(),
            aux_data=_aux_reader,
        ),
        start=pd.Timestamp(warmup_start_time),
        stop=pd.Timestamp(current_time),
        emitter=ctx.emitter,
        strategy_state=ctx._strategy_state,
        initializer=ctx.initializer,
    )

    # - set the time provider to the simulated runner
    _live_time_provider = simulated_formatter.time_provider
    simulated_formatter.time_provider = warmup_runner.ctx

    # Set the time provider for metric emitters to use simulation time
    if ctx.emitter is not None:
        ctx.emitter.set_time_provider(warmup_runner.ctx)

    ctx._strategy_state.is_warmup_in_progress = True

    try:
        warmup_runner.run(catch_keyboard_interrupt=False, close_data_readers=True)
    finally:
        # Restore the live time provider
        simulated_formatter.time_provider = _live_time_provider
        if ctx.emitter is not None:
            ctx.emitter.set_time_provider(_live_time_provider)
        ctx._strategy_state.is_warmup_in_progress = False
        ctx.initializer.simulation = False

    logger.info("<yellow>Warmup completed</yellow>")

    # - reset the strategy ctx to point back to live context
    if hasattr(ctx.strategy, "ctx"):
        setattr(ctx.strategy, "ctx", ctx)

    # - create a restored state based on warmup runner context
    warmup_account = warmup_runner.ctx.account

    _positions = warmup_account.get_positions()
    _orders = warmup_account.get_orders()
    instrument_to_orders = defaultdict(list)
    for o in _orders.values():
        instrument_to_orders[o.instrument].append(o)

    # - set the warmup positions and orders
    ctx.set_warmup_positions(_positions)
    ctx.set_warmup_orders(instrument_to_orders)

    # - subscribe to new subscriptions that could have been added during warmup
    live_subscriptions = ctx.get_subscriptions()
    warmup_subscriptions = warmup_runner.ctx.get_subscriptions()
    new_subscriptions = set(warmup_subscriptions) - set(live_subscriptions)
    for sub in new_subscriptions:
        ctx.subscribe(sub)

    # - update cache in the original context
    if (
        hasattr(ctx, "_cache")
        and isinstance((live_cache := getattr(ctx, "_cache")), CachedMarketDataHolder)
        and hasattr(warmup_runner.ctx, "_cache")
        and isinstance((warmup_cache := getattr(warmup_runner.ctx, "_cache")), CachedMarketDataHolder)
    ):
        # Only select the instruments from cache that are in the positions
        warmup_cache._ohlcvs = {k: v for k, v in warmup_cache._ohlcvs.items() if k in _positions}
        live_cache.set_state_from(warmup_cache)


def simulate_strategy(
    config_file: Path, save_path: str | None = None, start: str | None = None, stop: str | None = None
):
    """
    Simulate a strategy.

    Args:
        config_file: Path to the strategy configuration file
        save_path: Path to save the simulation results
        start: Start time for the simulation
        stop: Stop time for the simulation
    """
    # - this import is needed to register the loader functions
    # We don't need to import loader explicitly anymore since the registry handles it

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file for simualtion not found: {config_file}")

    cfg = load_simulation_config_from_yaml(config_file)
    stg = cfg.strategy
    simulation_name = config_file.stem
    _v_id = pd.Timestamp("now").strftime("%Y%m%d%H%M%S")

    match stg:
        case list():
            stg_cls = reduce(lambda x, y: x + y, [class_import(x) for x in stg])
        case str():
            stg_cls = class_import(stg)
        case _:
            raise SimulationConfigError(f"Invalid strategy type: {stg}")

    # - create simulation setup
    if cfg.variate:
        # - get conditions for variations if exists
        cond = cfg.variate.pop("with", None)
        conditions = []
        dict2lambda = lambda a, d: eval(f"lambda {a}: {d}")  # noqa: E731
        if cond:
            for a, c in cond.items():
                conditions.append(dict2lambda(a, c))

        experiments = variate(stg_cls, **(cfg.parameters | cfg.variate), conditions=conditions)
        experiments = {f"{simulation_name}.{_v_id}.[{k}]": v for k, v in experiments.items()}
        print(f"Parameters variation is configured. There are {len(experiments)} simulations to run.")
        _n_jobs = -1
    else:
        strategy = stg_cls(**cfg.parameters)
        experiments = {simulation_name: strategy}
        _n_jobs = 1

    data_i = {}

    for k, v in cfg.data.items():
        data_i[k] = eval(v)

    sim_params = cfg.simulation
    for mp in ["instruments", "capital", "commissions", "start", "stop"]:
        if mp not in sim_params:
            raise ValueError(f"Simulation parameter {mp} is required")

    if start is not None:
        sim_params["start"] = start
        logger.info(f"Start date set to {start}")

    if stop is not None:
        sim_params["stop"] = stop
        logger.info(f"Stop date set to {stop}")

    # - check for aux_data parameter
    if "aux_data" in sim_params:
        aux_data = sim_params.pop("aux_data")
        if aux_data is not None:
            try:
                sim_params["aux_data"] = eval(aux_data)
            except Exception as e:
                raise ValueError(f"Invalid aux_data parameter: {aux_data}") from e

    # - run simulation
    print(f" > Run simulation for [{red(simulation_name)}] ::: {sim_params['start']} - {sim_params['stop']}")
    sim_params["n_jobs"] = sim_params.get("n_jobs", _n_jobs)
    test_res = simulate(experiments, data=data_i, **sim_params)

    _where_to_save = save_path if save_path is not None else Path("results/")
    s_path = Path(makedirs(str(_where_to_save))) / simulation_name

    # logger.info(f"Saving simulation results to <g>{s_path}</g> ...")
    if cfg.description is not None:
        _descr = cfg.description
        if isinstance(cfg.description, list):
            _descr = "\n".join(cfg.description)
        else:
            _descr = str(cfg.description)

    if len(test_res) > 1:
        # - TODO: think how to deal with variations !
        s_path = s_path / f"variations.{_v_id}"
        print(f" > Saving variations results to <g>{s_path}</g> ...")
        for k, t in enumerate(test_res):
            # - set variation name
            t.variation_name = f"{simulation_name}.{_v_id}"
            t.to_file(str(s_path), description=_descr, suffix=f".{k}", attachments=[str(config_file)])
    else:
        print(f" > Saving simulation results to <g>{s_path}</g> ...")
        test_res[0].to_file(str(s_path), description=_descr, attachments=[str(config_file)])

    return test_res
