import asyncio
import socket
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from threading import Thread

from qubx import QubxLogConfig, file_formatter, logger
from qubx.backtester.optimization import variate
from qubx.backtester.runner import SimulationRunner
from qubx.backtester.simulator import simulate
from qubx.backtester.utils import (
    SetupTypes,
    SimulationConfigError,
    SimulationSetup,
    recognize_simulation_data_config,
)
from qubx.connectors.registry import ConnectorRegistry
from qubx.core.account import CompositeAccountProcessor
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    LiveTimeProvider,
    Position,
    RestoredState,
    TransactionCostsCalculator,
)
from qubx.core.context import StrategyContext
from qubx.core.exceptions import WarmupValidationError
from qubx.core.helpers import BasicScheduler
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import (
    IAccountProcessor,
    IBroker,
    IDataProvider,
    IHealthMonitor,
    IStrategyContext,
    ITimeProvider,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.lookups import lookup, register_accounts
from qubx.core.mixins.utils import EXCHANGE_MAPPINGS
from qubx.data.cache import CachedStorage, MemoryCache
from qubx.data.storages.stub import NoConfiguredStorage
from qubx.health import BaseHealthMonitor
from qubx.loggers import create_logs_writer
from qubx.restarts.state_resolvers import StateResolver
from qubx.restarts.time_finders import TimeFinder
from qubx.restorers import create_state_restorer
from qubx.utils.misc import class_import, green, install_uvloop, makedirs, red
from qubx.utils.results import SimulationResultsSaver, normalize_tags
from qubx.utils.runner.configs import (
    ExchangeConfig,
    LiveConfig,
    LoggingConfig,
    PrefetchConfig,
    RestorerConfig,
    StorageConfig,
    StrategyConfig,
    WarmupConfig,
    load_strategy_config_from_yaml,
    resolve_aux_config,
)
from qubx.utils.runner.factory import (
    construct_multi_storage,
    construct_storage,
    create_data_type_storages,
    create_exporters,
    create_metric_emitters,
    create_notifiers,
    create_state_persistence,
)
from qubx.utils.s3 import S3Client, is_account_uri, is_cloud_path
from qubx.utils.time import convert_seconds_to_str, to_timedelta, to_timestamp

from .accounts import AccountConfigurationManager

INVERSE_EXCHANGE_MAPPINGS = {mapping: exchange for exchange, mapping in EXCHANGE_MAPPINGS.items()}


def _cleanup_event_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """
    Cleanup the shared event loop.

    Args:
        loop: The event loop to cleanup. If None, does nothing.
    """
    if loop is None:
        return

    try:
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
            logger.debug("Shared event loop stopped")
    except Exception as e:
        logger.warning(f"Failed to cleanup event loop: {e}")


def run_strategy_yaml(
    config_file: Path,
    account_file: Path | None = None,
    paper: bool = False,
    restore: bool = False,
    blocking: bool = False,
    no_color: bool = False,
    no_emission: bool = False,
    no_notifiers: bool = False,
    no_exporters: bool = False,
    config_overrides: Path | None = None,
) -> IStrategyContext:
    """
    Run the strategy with the given configuration file.

    Args:
        config_file (Path): The path to the configuration file.
        account_file (Path, optional): The path to the account configuration file. Defaults to None.
        paper (bool, optional): Whether to run in paper trading mode. Defaults to False.
        no_color (bool, optional): Whether to disable colored logging. Defaults to False.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    if account_file is not None and not account_file.exists():
        raise FileNotFoundError(f"Account configuration file not found: {account_file}")

    # Register built-in connectors and load plugins
    import qubx.connectors  # noqa: F401, I001 - registers ccxt/tardis/xlighter connectors
    from qubx.plugins import load_plugins  # noqa: I001

    acc_manager = AccountConfigurationManager(account_file, config_file.parent, search_qubx_dir=True)
    stg_config = load_strategy_config_from_yaml(config_file, overrides_path=config_overrides)

    # Load plugins from configuration
    load_plugins(stg_config.plugins)

    return run_strategy(
        stg_config,
        acc_manager,
        paper=paper,
        restore=restore,
        blocking=blocking,
        no_color=no_color,
        no_emission=no_emission,
        no_notifiers=no_notifiers,
        no_exporters=no_exporters,
    )


def run_strategy_yaml_in_jupyter(
    config_file: Path,
    account_file: Path | None = None,
    paper: bool = False,
    restore: bool = False,
    no_emission: bool = False,
    no_notifiers: bool = False,
    no_exporters: bool = False,
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
        {
            "config_file": config_file,
            "account_file": account_file,
            "paper": paper,
            "restore": restore,
            "no_emission": no_emission,
            "no_notifiers": no_notifiers,
            "no_exporters": no_exporters,
        }
    )
    logger.info("Running in Jupyter console")
    TerminalRunner.launch_instance(init_code=content_with_values)


def run_strategy(
    config: StrategyConfig,
    account_manager: AccountConfigurationManager,
    paper: bool = False,
    restore: bool = False,
    blocking: bool = False,
    no_color: bool = False,
    no_emission: bool = False,
    no_notifiers: bool = False,
    no_exporters: bool = False,
) -> IStrategyContext:
    """
    Run a strategy with the given configuration.

    Args:
        config (StrategyConfig): The configuration of the strategy.
        account_manager (AccountManager): The account manager to use.
        paper (bool, optional): Whether to run in paper trading mode. Defaults to False.
        blocking (bool, optional): Whether to block the main thread. Defaults to False.
        no_color (bool, optional): Whether to disable colored logging. Defaults to False.

    Returns:
        IStrategyContext: The strategy context.
    """
    # Validate that live configuration exists
    if not config.live:
        raise ValueError("Live configuration is required for strategy execution")

    # Install uvloop and create shared event loop
    install_uvloop()
    loop = asyncio.new_event_loop()
    loop_thread = Thread(target=loop.run_forever, daemon=True, name="SharedEventLoop")
    loop_thread.start()
    logger.debug("Shared event loop started in background thread")

    register_accounts(account_manager)

    _live_time_provider = LiveTimeProvider()
    QubxLogConfig.bind_time_provider(_live_time_provider)
    QubxLogConfig.setup_logger(level=QubxLogConfig.get_log_level(), colorize=not no_color)

    # Start health server early so liveness probe works during init/warmup
    from qubx.config import settings as _qubx_settings

    _health_server = None
    _health_ctx_ref: list[IStrategyContext | None] = [None]  # mutable ref for closure

    if _qubx_settings.health_port:
        from qubx.health import HealthServer

        def _ready_check() -> bool:
            c = _health_ctx_ref[0]
            return c is not None and c._strategy_state.is_on_warmup_finished_called

        _health_server = HealthServer(_qubx_settings.health_port, ready_check=_ready_check)
        _health_server.start()

    # Resolve strategy identity once — BOT_ID takes precedence over config name.
    # This identity is used for state restoration, logging, metric emission, and persistence.
    stg_name = _qubx_settings.bot_id or _get_strategy_name(config)

    # Restore state if configured
    restored_state = (
        _restore_state(
            restorer_config=config.live.warmup.restorer if config.live.warmup else None,
            logging_config=config.live.logging if config.live.logging else None,
            strategy_name=stg_name,
        )
        if restore
        else None
    )

    # Resolve aux config with live section override (needed for both warmup and live context)
    aux_configs = resolve_aux_config(config.aux, getattr(config.live, "aux", None))

    # Create the strategy context
    ctx = create_strategy_context(
        config=config,
        account_manager=account_manager,
        paper=paper,
        restored_state=restored_state,
        stg_name=stg_name,
        no_color=no_color,
        aux_configs=aux_configs,
        loop=loop,
        no_emission=no_emission,
        no_notifiers=no_notifiers,
        no_exporters=no_exporters,
    )
    _health_ctx_ref[0] = ctx  # expose to health server ready_check

    try:
        _run_warmup(
            ctx=ctx,
            restored_state=restored_state,
            exchanges=config.live.exchanges,
            warmup=config.live.warmup,
            prefetch_config=config.live.prefetch,
            live_time_provider=_live_time_provider,
            account_manager=account_manager,
            enable_funding=config.live.warmup.enable_funding if config.live.warmup else False,
            aux_configs=aux_configs,
        )
    except KeyboardInterrupt:
        logger.info("Warmup interrupted by user")
        _cleanup_event_loop(loop)
        return ctx
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        _cleanup_event_loop(loop)
        raise e

    _apply_base_live_subscription(ctx)

    # Start the strategy context
    if blocking:
        try:
            ctx.start(blocking=True)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            ctx.stop()
            if _health_server:
                _health_server.stop()
            _cleanup_event_loop(loop)
    else:
        ctx.start()

    return ctx


def _infer_restorer_from_logger(logging_config: LoggingConfig, strategy_name: str) -> RestorerConfig | None:
    """Infer a matching state restorer config from the logger config."""
    logger_type = logging_config.logger
    args = logging_config.args

    if logger_type == "PostgresLogsWriter":
        return RestorerConfig(
            type="PostgresStateRestorer",
            parameters={
                "strategy_name": strategy_name,
                "postgres_uri": args.get("postgres_uri", "postgresql://localhost:5432/qubx_logs"),
                "table_prefix": args.get("table_prefix", "qubx_logs"),
            },
        )
    elif logger_type == "MongoDBLogsWriter":
        return RestorerConfig(
            type="MongoDBStateRestorer",
            parameters={
                "strategy_name": strategy_name,
                "mongo_uri": args.get("mongo_uri", "mongodb://localhost:27017/"),
                "db_name": args.get("db_name", "default_logs_db"),
                "collection_name_prefix": args.get("collection_name_prefix", "qubx_logs"),
            },
        )
    elif logger_type == "CsvFileLogsWriter":
        return RestorerConfig(
            type="CsvStateRestorer",
            parameters={
                "strategy_name": strategy_name,
                "base_dir": args.get("log_folder", "logs"),
            },
        )

    return None


def _restore_state(
    restorer_config: RestorerConfig | None,
    logging_config: LoggingConfig | None = None,
    strategy_name: str = "",
) -> RestoredState | None:
    if restorer_config is None and logging_config is not None:
        restorer_config = _infer_restorer_from_logger(logging_config, strategy_name)

    if restorer_config is None:
        restorer_config = RestorerConfig(type="CsvStateRestorer", parameters={"base_dir": "logs"})

    state_restorer = create_state_restorer(
        restorer_config.type,
        restorer_config.parameters,
    )

    state = state_restorer.restore_state()
    logger.info(
        f"<yellow>Restored state with {len(state.positions)} positions "
        f"and {sum(len(s) for s in state.instrument_to_signal_positions.values())} signals</yellow>"
    )
    logger.info("<yellow> - Positions:</yellow>")
    for position in state.positions.values():
        logger.info(f"<yellow>   - {position}</yellow>")
    return state


def create_strategy_context(
    config: StrategyConfig,
    account_manager: AccountConfigurationManager,
    paper: bool,
    restored_state: RestoredState | None,
    stg_name: str,
    no_color: bool = False,
    aux_configs: list[StorageConfig] | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    no_emission: bool = False,
    no_notifiers: bool = False,
    no_exporters: bool = False,
) -> IStrategyContext:
    """
    Create a strategy context from the given configuration.

    Args:
        stg_name: Strategy identity — BOT_ID when running on the platform,
                  otherwise derived from the config name. Resolved once by the
                  caller and used consistently for logging, metrics, state
                  persistence, and state restoration.
    """
    # Validate that live configuration exists
    if not config.live:
        raise ValueError("Live configuration is required for strategy execution")

    # --- Platform identity from unified settings ---
    from qubx.config import settings as qubx_settings

    _bot_id = qubx_settings.bot_id
    _run_mode = "paper" if paper else "live"

    # Generate run_id once to be shared between logging and metric emissions
    run_id = f"{socket.gethostname()}-{str(int(time.time() * 10**9))}"

    if isinstance(config.strategy, list):
        _strategy_class = reduce(lambda x, y: x + y, [class_import(x) for x in config.strategy])
    elif isinstance(config.strategy, str):
        _strategy_class = class_import(config.strategy)
    else:
        _strategy_class = config.strategy

    _logging = _setup_strategy_logging(stg_name, config.live.logging, run_id)
    _instance_id = qubx_settings.instance_id or socket.gethostname()
    _platform_tags: dict[str, str] = {}
    if _bot_id:
        _platform_tags["bot_id"] = _bot_id
    _platform_tags["instance_id"] = _instance_id

    # Bind platform identity to all log messages
    QubxLogConfig.bind_platform_identity(_bot_id, _instance_id)

    # Create metric emitters with run_id as a tag
    if no_emission:
        logger.info("Metric emission disabled via CLI flag")
        _metric_emitter = None
    else:
        _metric_emitter = (
            create_metric_emitters(config.live.emission, stg_name, run_id, extra_tags=_platform_tags)
            if config.live.emission
            else None
        )

    # Create lifecycle notifiers
    if no_notifiers:
        logger.info("Lifecycle notifiers disabled via CLI flag")
        _notifier = None
    else:
        _notifier = create_notifiers(config.live.notifiers, stg_name) if config.live.notifiers else None

    # Create strategy initializer
    _initializer = BasicStrategyInitializer()

    _time = LiveTimeProvider()
    _chan = CtrlChannel("databus", sentinel=(None, None, None, None))
    _sched = BasicScheduler(_chan, lambda: _time.time().item())

    # Create health metrics monitor with emitter
    _health_monitor = BaseHealthMonitor(
        _time, emitter=_metric_emitter, channel=_chan, **config.live.health.model_dump()
    )

    exchanges = list(config.live.exchanges.keys())

    _exchange_to_tcc = {}
    _exchange_to_broker = {}
    _exchange_to_data_provider = {}
    _exchange_to_account = {}
    _instruments = []

    for exchange_name, exchange_config in config.live.exchanges.items():
        _exchange_to_tcc[exchange_name] = (tcc := _create_tcc(exchange_name, account_manager))
        _exchange_to_data_provider[exchange_name] = (
            data_provider := _create_data_provider(
                exchange_name,
                exchange_config,
                time_provider=_time,
                channel=_chan,
                account_manager=account_manager,
                health_monitor=_health_monitor,
                loop=loop,
            )
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
                health_monitor=_health_monitor,
                live_config=config.live,
                data_provider=data_provider,
                restored_state=restored_state.filter_by_exchange(exchange_name) if restored_state else None,
                read_only=config.live.read_only,
                loop=loop,
            )
        )
        _exchange_to_broker[exchange_name] = _create_broker(
            exchange_name,
            exchange_config,
            _chan,
            time_provider=_time,
            account=account,
            data_provider=data_provider,
            account_manager=account_manager,
            health_monitor=_health_monitor,
            paper=paper,
            loop=loop,
        )
        _instruments.extend(_create_instruments_for_exchange(exchange_name, exchange_config))

    # Use provided aux_configs or resolve if not provided (for backwards compatibility)
    if aux_configs is None:
        aux_configs = resolve_aux_config(config.aux, getattr(config.live, "aux", None))

    # - create aux storage from config
    _aux_storage = construct_multi_storage(aux_configs)

    # - construct CachedStorage(CachedReader) here instead of CachedPrefetchReader.
    if _aux_storage is not None and config.live.prefetch:
        prefetch_config = config.live.prefetch
        if prefetch_config.enabled:
            _aux_storage = CachedStorage(
                _aux_storage,
                prefetch_period=prefetch_config.prefetch_period,
                cache_factory=lambda: MemoryCache(max_size_mb=prefetch_config.cache_size_mb),
            )

    # - when no any aux storage is configured let's use empty one
    # - we don't raise an exception here because some strategies may not require aux data
    # - but if strategy asks for aux data it would rise an error with a clear message about missing aux storage configuration
    if _aux_storage is None:
        _aux_storage = NoConfiguredStorage(
            f"Strategy {config.name or ''} is trying to access aux data bit no auxiliary storage configured for live mode"
        )

    _account = (
        CompositeAccountProcessor(_time, _exchange_to_account)
        if len(exchanges) > 1
        else _exchange_to_account[exchanges[0]]
    )
    _initializer = BasicStrategyInitializer(simulation=_exchange_to_data_provider[exchanges[0]].is_simulation)

    # Create exporters if configured
    if no_exporters:
        logger.info("Trade exporters disabled via CLI flag")
        _exporter = None
    else:
        _exporter = create_exporters(config.live.exporters, stg_name, _account) if config.live.exporters else None

    # Create data throttler from config
    _data_throttler = _create_data_throttler(config.live.throttling) if config.live.throttling else None

    # Create state persistence if configured
    _state_persistence = create_state_persistence(config.live.state, stg_name)
    _state_snapshot_interval = config.live.state.snapshot_interval if config.live.state else None

    logger.info(f"- Strategy: <blue>{stg_name}</blue>\n- Mode: {_run_mode}\n- Parameters: {config.parameters}")

    ctx = StrategyContext(
        strategy=_strategy_class,  # type: ignore
        brokers=list(_exchange_to_broker.values()),
        data_providers=list(_exchange_to_data_provider.values()),
        account=_account,
        scheduler=_sched,
        time_provider=_time,
        instruments=_instruments,
        logging=_logging,
        config=config.parameters,
        aux_data_storage=_aux_storage,
        exporter=_exporter,
        emitter=_metric_emitter,
        notifier=_notifier,
        initializer=_initializer,
        strategy_name=stg_name,
        health_monitor=_health_monitor,
        restored_state=restored_state,
        data_throttler=_data_throttler,
        state_persistence=_state_persistence,
        state_snapshot_interval=_state_snapshot_interval,
    )

    # Store the shared event loop reference for cleanup
    if loop is not None:
        ctx._shared_event_loop = loop  # type: ignore

    # Set context for metric emitters to enable is_live tag and time access
    if _metric_emitter is not None:
        _metric_emitter.set_context(ctx)

    return ctx


def _get_strategy_name(cfg: StrategyConfig) -> str:
    if cfg.name is not None:
        return cfg.name
    if isinstance(cfg.strategy, list):
        return "_".join(map(lambda x: x.split(".")[-1], cfg.strategy))
    elif isinstance(cfg.strategy, str):
        return cfg.strategy.split(".")[-1]
    else:
        return cfg.strategy.__class__.__name__


def _setup_strategy_logging(
    stg_name: str,
    log_config: LoggingConfig,
    run_id: str,
) -> StrategyLogging:
    if not hasattr(log_config, "args") or not isinstance(log_config.args, dict):
        log_config.args = {}
    log_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    log_folder = log_config.args.get("log_folder", "logs")
    run_folder = f"{log_folder}/run_{log_id}"
    logger.add(
        f"{run_folder}/strategy/{stg_name}_{{time}}.log",
        format=file_formatter,
        rotation="100 MB",
        colorize=False,
        level=QubxLogConfig.get_log_level(),
    )

    _log_writer_name = log_config.logger

    logger.debug(f"Setup <g>{_log_writer_name}</g> logger...")

    override_params = {
        "account_id": "account",
        "strategy_id": stg_name,
        "run_id": run_id,
        "log_folder": run_folder,
    }
    _log_writer_params = {**override_params, **log_config.args}

    _log_writer = create_logs_writer(_log_writer_name, _log_writer_params)
    stg_logging = StrategyLogging(
        logs_writer=_log_writer,
        positions_log_freq=log_config.position_interval,
        portfolio_log_freq=log_config.portfolio_interval,
        heartbeat_freq=log_config.heartbeat_interval,
    )
    return stg_logging


def _create_tcc(exchange_name: str, account_manager: AccountConfigurationManager) -> TransactionCostsCalculator:
    if exchange_name == "BINANCE.PM":
        # TODO: clean this up
        exchange_name = "BINANCE.UM"
    settings = account_manager.get_exchange_settings(exchange_name)
    tcc = lookup.find_fees(exchange_name, settings.commissions)
    assert tcc is not None, f"Can't find fees calculator for {exchange_name} exchange"
    return tcc


def _create_data_provider(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    time_provider: ITimeProvider,
    channel: CtrlChannel,
    account_manager: AccountConfigurationManager,
    health_monitor: IHealthMonitor,
    loop: asyncio.AbstractEventLoop | None = None,
) -> IDataProvider:
    connector_name = exchange_config.connector.lower()

    return ConnectorRegistry.get_data_provider(
        connector_name,
        exchange_name=exchange_name,
        time_provider=time_provider,
        channel=channel,
        health_monitor=health_monitor,
        account_manager=account_manager,
        loop=loop,
        **exchange_config.params,
    )


def _create_account_processor(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account_manager: AccountConfigurationManager,
    tcc: TransactionCostsCalculator,
    paper: bool,
    health_monitor: IHealthMonitor,
    live_config: LiveConfig,
    data_provider: IDataProvider | None = None,
    restored_state: RestoredState | None = None,
    read_only: bool = False,
    loop: asyncio.AbstractEventLoop | None = None,
) -> IAccountProcessor:
    # Resolve base_currency with priority: per-exchange YAML > global YAML > accounts.toml
    if exchange_config.base_currency is not None:
        base_currency = exchange_config.base_currency
    elif live_config.base_currency is not None:
        base_currency = live_config.base_currency
    else:
        base_currency = account_manager.get_exchange_settings(exchange_name).base_currency

    if paper:
        # Paper trading: create SimulatedAccountProcessor directly (not registered with registry)
        from qubx.backtester.account import SimulatedAccountProcessor
        from qubx.backtester.simulated_exchange import get_simulated_exchange

        settings = account_manager.get_exchange_settings(exchange_name)
        simulated_exchange = get_simulated_exchange(exchange_name, time_provider, tcc)

        return SimulatedAccountProcessor(
            account_id=exchange_name,
            exchange=simulated_exchange,
            channel=channel,
            health_monitor=health_monitor,
            base_currency=base_currency,
            exchange_name=exchange_name,
            initial_capital=settings.initial_capital,
            restored_state=restored_state,
        )

    if exchange_config.account is not None:
        connector = exchange_config.account.connector
    else:
        connector = exchange_config.connector

    connector_name = connector.lower()

    return ConnectorRegistry.get_account_processor(
        connector_name,
        exchange_name=exchange_name,
        channel=channel,
        time_provider=time_provider,
        account_manager=account_manager,
        tcc=tcc,
        health_monitor=health_monitor,
        data_provider=data_provider,
        restored_state=restored_state,
        read_only=read_only,
        loop=loop,
        base_currency=base_currency,
    )


def _create_broker(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    data_provider: IDataProvider,
    account_manager: AccountConfigurationManager,
    health_monitor: IHealthMonitor,
    paper: bool,
    loop: asyncio.AbstractEventLoop | None = None,
) -> IBroker:
    if paper:
        # Paper trading: create SimulatedBroker directly (not registered with registry)
        from qubx.backtester.account import SimulatedAccountProcessor
        from qubx.backtester.broker import SimulatedBroker

        assert isinstance(account, SimulatedAccountProcessor), (
            "Account must be SimulatedAccountProcessor for paper mode"
        )

        return SimulatedBroker(
            channel=channel,
            account=account,
            simulated_exchange=account._exchange,
        )

    if exchange_config.broker is not None:
        connector = exchange_config.broker.connector
        params = dict(exchange_config.broker.params)
    else:
        connector = exchange_config.connector
        params = {}

    connector_name = connector.lower()

    return ConnectorRegistry.get_broker(
        connector_name,
        exchange_name=exchange_name,
        channel=channel,
        time_provider=time_provider,
        account=account,
        data_provider=data_provider,
        account_manager=account_manager,
        health_monitor=health_monitor,
        loop=loop,
        **params,
    )


def _create_instruments_for_exchange(exchange_name: str, exchange_config: ExchangeConfig) -> list[Instrument]:
    exchange_name = exchange_name.upper()
    if exchange_name == "BINANCE.PM":
        # TODO: clean this up
        exchange_name = "BINANCE.UM"
    symbols = exchange_config.universe
    instruments = []
    for symbol in symbols:
        _e, _mt, _s = Instrument.parse_notation(symbol)
        # - use exchange from notation if provided, otherwise use section exchange
        _exch = _e.upper() if _e else exchange_name
        instr = lookup.find_symbol(_exch, _s.upper(), market_type=_mt)
        if instr is not None:
            instruments.append(instr)
    return instruments


def _create_data_throttler(throttling_config):
    """
    Create data throttler from throttling configuration.

    Args:
        throttling_config: ThrottlingConfig object with throttle settings

    Returns:
        InstrumentThrottler configured with per-data-type frequency limits, or None if disabled
    """
    from qubx.utils.throttler import InstrumentThrottler

    if not throttling_config or not throttling_config.enabled:
        return None

    # Build config dict: {data_type: max_frequency_hz}
    throttle_cfg_dict = {}
    for throttle_cfg in throttling_config.throttles:
        if throttle_cfg.enabled:
            throttle_cfg_dict[throttle_cfg.data_type] = throttle_cfg.max_frequency_hz
            logger.info(
                f"Throttling <y>{throttle_cfg.data_type}</y>: max {throttle_cfg.max_frequency_hz} Hz per instrument"
            )

    if not throttle_cfg_dict:
        return None

    return InstrumentThrottler(throttle_cfg_dict)


def _apply_inverse_exchange_mapping(exchanges: list[str]) -> list[str]:
    """
    Apply inverse exchange mapping to the list of exchanges.

    This converts mapped exchanges (like BINANCE.PM) back to their original form (like BINANCE.UM)
    so that SimulationRunner doesn't need to handle EXCHANGE_MAPPINGS.
    """
    mapped_exchanges = []
    for exchange in exchanges:
        if exchange in INVERSE_EXCHANGE_MAPPINGS:
            mapped_exchanges.append(INVERSE_EXCHANGE_MAPPINGS[exchange])
        else:
            mapped_exchanges.append(exchange)
    return mapped_exchanges


@dataclass(frozen=True)
class WarmupResult:
    """Captures the outcome of a warmup simulation for validation."""

    requested_start_ns: int
    requested_stop_ns: int
    data_start_ns: int | None
    data_stop_ns: int | None
    initial_capital: float
    final_capital: float

    @property
    def requested_duration_ns(self) -> int:
        return self.requested_stop_ns - self.requested_start_ns

    @property
    def data_duration_ns(self) -> int:
        if self.data_start_ns is not None and self.data_stop_ns is not None:
            return self.data_stop_ns - self.data_start_ns
        return 0

    @property
    def coverage_ratio(self) -> float:
        if self.requested_duration_ns <= 0:
            return 1.0
        if self.data_start_ns is None:
            return 0.0
        return max(0.0, self.data_duration_ns / self.requested_duration_ns)

    @property
    def has_data(self) -> bool:
        return self.data_start_ns is not None


def _validate_warmup_result(result: WarmupResult, min_coverage_ratio: float = 0.5) -> None:
    """Validate that warmup simulation produced usable state for live trading."""
    if not result.has_data:
        raise WarmupValidationError(
            f"Warmup received no data at all. "
            f"Requested window: {to_timedelta(result.requested_duration_ns, unit='ns')}. "
            f"Check your warmup data source configuration."
        )

    if result.coverage_ratio < min_coverage_ratio:
        data_start_gap = to_timedelta(result.data_start_ns - result.requested_start_ns, unit="ns")
        data_end_gap = to_timedelta(result.requested_stop_ns - result.data_stop_ns, unit="ns")
        raise WarmupValidationError(
            f"Warmup data coverage insufficient: "
            f"received {to_timedelta(result.data_duration_ns, unit='ns')} "
            f"of requested {to_timedelta(result.requested_duration_ns, unit='ns')} "
            f"({result.coverage_ratio:.1%} coverage, minimum required: {min_coverage_ratio:.0%}). "
            f"Data started {data_start_gap} late, ended {data_end_gap} early. "
            f"Check your warmup data source — it may have a max_history limit "
            f"that is shorter than the warmup period."
        )

    if result.final_capital <= 0:
        raise WarmupValidationError(
            f"Warmup resulted in non-positive capital: {result.final_capital:.2f} "
            f"(started with {result.initial_capital:.2f}). "
            f"This typically means the warmup simulation traded with insufficient data, "
            f"causing unrealistic losses. Check warmup data coverage and indicator warmup periods."
        )

    if result.initial_capital > 0 and result.final_capital < result.initial_capital * 0.5:
        logger.warning(
            f"<red>Warmup capital dropped significantly: "
            f"{result.initial_capital:.2f} -> {result.final_capital:.2f} "
            f"({(1 - result.final_capital / result.initial_capital):.1%} loss). "
            f"Review warmup data coverage and strategy behavior during warmup.</red>"
        )


def _run_warmup(
    ctx: IStrategyContext,
    restored_state: RestoredState | None,
    exchanges: dict[str, ExchangeConfig],
    warmup: WarmupConfig | None,
    prefetch_config: PrefetchConfig,
    live_time_provider,
    account_manager: AccountConfigurationManager | None = None,
    enable_funding: bool = False,
    aux_configs: list[StorageConfig] | None = None,
    trading_sessions_time: str | None = None,
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
        time_delta = to_timedelta(current_time - warmup_start_time)
        if time_delta.total_seconds() > 0:
            logger.info(f"<yellow>Start time finder estimated to go back in time by {time_delta}</yellow>")

    if warmup_period is not None:
        logger.info(f"<yellow>Warmup period is set to {to_timedelta(warmup_period)}</yellow>")
        warmup_start_time -= warmup_period

    if warmup_start_time == current_time:
        # if start time is the same as current time, we don't need to run warmup
        return

    logger.info(f"<yellow>Warmup start time: {warmup_start_time}</yellow>")

    # - resolve warmup data sources (mirrors SimulationConfig layout)
    # - warmup.data is required (StorageConfig, not None) so construct_storage always returns IStorage
    _warmup_data_storage = construct_storage(warmup.data)
    assert _warmup_data_storage is not None, f"Failed to construct warmup data storage from: {warmup.data}"
    _warmup_custom_data = create_data_type_storages(warmup.custom_data) if warmup.custom_data else None
    # - use resolved live/global aux configs (passed from caller)
    _warmup_aux_storage = construct_multi_storage(aux_configs) if aux_configs else None

    # - create instruments
    instruments = []
    for exchange_name, exchange_config in exchanges.items():
        instruments.extend(_create_instruments_for_exchange(exchange_name, exchange_config))
    if restored_state is not None:
        instruments.extend(restored_state.instrument_to_signal_positions.keys())

    assert isinstance(ctx.initializer, BasicStrategyInitializer)
    ctx.initializer.simulation = True

    logger.info(f"<yellow>Running warmup from {warmup_start_time} to {current_time}</yellow>")
    warmup_runner = SimulationRunner(
        setup=SimulationSetup(
            setup_type=SetupTypes.STRATEGY,
            name=ctx.strategy_name,
            generator=ctx.strategy,
            tracker=None,
            instruments=instruments,
            # Apply inverse exchange mapping so SimulationRunner doesn't need EXCHANGE_MAPPINGS
            exchanges=_apply_inverse_exchange_mapping(ctx.exchanges),
            capital=ctx.account.get_total_capital(),
            base_currency=ctx.account.get_base_currency(),
            commissions=None,  # TODO: get commissions from somewhere
            enable_funding=enable_funding,
        ),
        data_config=recognize_simulation_data_config(
            data_storage=_warmup_data_storage,
            custom_data=_warmup_custom_data,
            aux_data_storage=_warmup_aux_storage,
            prefetch_config=prefetch_config,
            trading_sessions_time=trading_sessions_time,
        ),
        start=to_timestamp(warmup_start_time),
        stop=to_timestamp(current_time),
        emitter=ctx.emitter,
        notifier=ctx.notifier,
        strategy_state=ctx._strategy_state,
        initializer=ctx.initializer,
        warmup_mode=True,
    )
    _initial_capital = ctx.account.get_total_capital()

    ctx._strategy_state.is_warmup_in_progress = True
    QubxLogConfig.bind_phase("warmup")
    QubxLogConfig.bind_time_provider(warmup_runner.ctx)

    try:
        warmup_runner.run(catch_keyboard_interrupt=False, close_data_readers=True)
    finally:
        # Restore the live time provider
        QubxLogConfig.bind_time_provider(live_time_provider)
        # Set back the context for metric emitters to use live context
        if ctx.emitter is not None:
            ctx.emitter.set_context(ctx)
        ctx._strategy_state.is_warmup_in_progress = False
        ctx.initializer.simulation = False
        QubxLogConfig.bind_phase("live")

    logger.info("<yellow>Warmup completed</yellow>")

    data_range = warmup_runner.data_time_range
    warmup_result = WarmupResult(
        requested_start_ns=to_timestamp(warmup_start_time).value,
        requested_stop_ns=to_timestamp(current_time).value,
        data_start_ns=data_range[0] if data_range else None,
        data_stop_ns=data_range[1] if data_range else None,
        initial_capital=_initial_capital,
        final_capital=warmup_runner.ctx.account.get_total_capital(),
    )
    _validate_warmup_result(warmup_result)

    # - reset the strategy ctx to point back to live context
    if hasattr(ctx.strategy, "ctx"):
        setattr(ctx.strategy, "ctx", ctx)

    # - create a restored state based on warmup runner context
    warmup_account = warmup_runner.ctx.account

    # - get the instruments from the warmup runner context
    _instruments = warmup_runner.ctx.instruments
    _positions = warmup_account.get_positions()
    _positions = {k: v for k, v in _positions.items() if k in _instruments and v is not None and v.quantity is not None}
    _orders = warmup_account.get_orders()
    instrument_to_orders = defaultdict(list)
    for o in _orders.values():
        if o.instrument in _instruments:
            instrument_to_orders[o.instrument].append(o)

    # - find instruments with nonzero positions from restored state and add them to the context
    if restored_state is not None:
        restored_positions = {k: p for k, p in restored_state.positions.items() if p.is_open()}
        # - if there is no warmup position for a restored position, then create a new zero position
        for pos in restored_positions.values():
            if pos.instrument not in _positions:
                _positions[pos.instrument] = Position(pos.instrument)

    # - set the warmup positions and orders
    ctx.set_warmup_positions(_positions)
    ctx.set_warmup_orders(instrument_to_orders)
    ctx.set_warmup_active_targets(warmup_runner.ctx.get_active_targets())

    # - subscribe to new subscriptions that could have been added during warmup
    live_subscriptions = ctx.get_subscriptions()
    warmup_subscriptions = warmup_runner.ctx.get_subscriptions()
    new_subscriptions = set(warmup_subscriptions) - set(live_subscriptions)

    for sub in new_subscriptions:
        ctx.subscribe(sub)

    # - update cache in the original context (only for instruments with positions)
    live_cache = ctx.get_market_data_cache()
    warmup_cache = warmup_runner.ctx.get_market_data_cache()
    live_cache.set_state_from(warmup_cache, instruments=list(_positions.keys()))


def _apply_base_live_subscription(ctx: IStrategyContext) -> None:
    """Apply base_live_subscription if it differs from the base subscription."""
    base_subscription = ctx.initializer.get_base_subscription()
    base_live_subscription = ctx.initializer.get_base_live_subscription()
    if base_live_subscription is not None and base_live_subscription != base_subscription:
        logger.info(f"Setting base live subscription from {base_subscription} to {base_live_subscription}")
        ctx.set_base_subscription(base_live_subscription)
        ctx.subscribe(base_live_subscription)


def _import_strategy_class(stg: str | list[str]):
    """Import and return the strategy class from a dotted path string (or list of strings)."""
    match stg:
        case list():
            stg_cls = reduce(lambda x, y: x + y, [class_import(x) for x in stg])
            return stg_cls, stg
        case str():
            return class_import(stg), [stg]
        case _:
            raise SimulationConfigError(f"Invalid strategy type: {stg}")


def _build_sim_params(
    cfg: StrategyConfig,
    start: str | None = None,
    stop: str | None = None,
) -> tuple[dict, dict[str, object]]:
    """
    Build the simulate() kwargs from a StrategyConfig.

    Returns:
        (data_kwargs, sim_params) where data_kwargs contains 'data' and 'custom_data'
        keys, and sim_params contains all other simulate() keyword arguments.
    """
    sim = cfg.simulation
    assert sim is not None

    # - resolve storages
    data = construct_storage(sim.data)
    data_i = create_data_type_storages(sim.custom_data) if sim.custom_data else {}

    sim_params: dict[str, object] = {
        "instruments": sim.instruments,
        "capital": sim.capital,
        "commissions": sim.commissions,
        "start": start or sim.start,
        "stop": stop or sim.stop,
        "enable_funding": sim.enable_funding,
        "enable_inmemory_emitter": sim.enable_inmemory_emitter,
    }

    if sim.base_currency is not None:
        sim_params["base_currency"] = sim.base_currency
    if sim.debug is not None:
        sim_params["debug"] = sim.debug
    if sim.portfolio_log_freq is not None:
        sim_params["portfolio_log_freq"] = sim.portfolio_log_freq
    if sim.n_jobs is not None:
        sim_params["n_jobs"] = sim.n_jobs
    if sim.run_separate_instruments:
        sim_params["run_separate_instruments"] = True
    if sim.prefetch is not None:
        sim_params["prefetch_config"] = sim.prefetch
    if sim.trading_session is not None:
        sim_params["trading_sessions_time"] = sim.trading_session

    # - resolve aux_data
    aux_configs = resolve_aux_config(cfg.aux, getattr(sim, "aux", None))
    if aux_configs:
        sim_params["aux_data"] = construct_multi_storage(aux_configs)

    return {"data": data, "custom_data": data_i}, sim_params


def _safe_store_results(
    results_saver: "SimulationResultsSaver",
    test_res: list,
    sim_time_sec: int,
    cloud_log_file: str | None,
    save_path: str | None,
    yaml_name: str,
    strategy_full_classes: str | list[str],
    simulation_name: str,
    sim_start: str,
    sim_stop: str,
    v_id: str,
    config_file: Path,
    tags: list[str],
    descr: str,
    is_variation: bool,
) -> None:
    """
    Store simulation results, falling back to a local temp directory if cloud storage is unavailable.

    If ``save_path`` is a cloud URI and writing fails (network issue, credentials, etc.),
    results are saved to ``{tempdir}/backtests/`` so completed simulations are never lost.
    Any cloud log file is copied to the fallback run directory before the temp file is removed.
    For non-cloud failures the exception is re-raised as usual.
    """
    try:
        results_saver.store_simulation_results(test_res, sim_time_sec, log_file=cloud_log_file)
    except Exception as _store_err:
        if save_path is not None and is_cloud_path(save_path):
            import os
            import shutil
            import tempfile

            _fallback_base = str(Path(tempfile.gettempdir()) / "backtests")
            logger.warning(
                f"[simulate_strategy] Failed to save results to cloud storage ({save_path}): {_store_err}"
                f"\n  → Falling back to local storage: {_fallback_base}"
            )
            _fallback_saver = SimulationResultsSaver(
                name=yaml_name,
                strategy_class=strategy_full_classes,
                config_name=simulation_name,
                sim_start=sim_start,
                sim_stop=sim_stop,
                save_path=_fallback_base,
                run_id=v_id,
                config_file=str(config_file),
                tags=tags,
                description=descr,
                is_variation=is_variation,
                storage_options=None,  # - local, no cloud credentials needed
            )
            # - cloud log file: copy to fallback run_dir instead of uploading
            _fallback_log: str | None = None
            if cloud_log_file is not None:
                try:
                    makedirs(_fallback_saver.run_dir)
                    _dst = Path(_fallback_saver.run_dir) / f"{simulation_name}.log"
                    shutil.copy2(cloud_log_file, str(_dst))
                    os.unlink(cloud_log_file)
                    logger.info(f"[simulate_strategy] Log saved locally: {_dst}")
                except Exception as _log_err:
                    logger.warning(f"[simulate_strategy] Could not copy log to fallback dir: {_log_err}")
            _fallback_saver.store_simulation_results(test_res, sim_time_sec, log_file=_fallback_log)
            print(f" > Results saved locally (cloud fallback): {green(_fallback_saver.run_dir)}")
        else:
            raise


def simulate_strategy(
    config_file: Path,
    save_path: str | None = None,
    start: str | None = None,
    stop: str | None = None,
    report: str | None = None,
    storage_options: dict | None = None,
    name: str | None = None,
    log_file: str | None = None,
):
    """
    Simulate a strategy from the CLI with result saving, logging, and variation support.

    Args:
        config_file: Path to the strategy configuration file
        save_path: Path to save the simulation results. Always uses parquet-based storage.
                   Supports local paths and cloud URIs (s3://, gs://, az://).
                   When set, simulation logs are automatically written alongside results.
        start: Start time for the simulation
        stop: Stop time for the simulation
        report: Ignored (kept for backward compatibility — parquet storage has no separate report)
        storage_options: Cloud storage credentials dict. None = uses default_s3_account from settings.
        name: Override the run name used for output folder construction. When provided, takes
              precedence over the 'name:' field in the config file. Useful for distinguishing
              runs (e.g. 'smoketest', '01_reference') without editing the config.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file for simualtion not found: {config_file}")

    # - resolve account URI (e.g. r2:backtests) → s3:// + credentials before any path use
    if save_path is not None and is_account_uri(save_path) and storage_options is None:
        _client, _s3_key = S3Client.from_uri(save_path)
        storage_options = _client.storage_options
        save_path = f"s3://{_s3_key}"

    cfg = load_strategy_config_from_yaml(config_file)

    # Load plugins from configuration
    from qubx.plugins import load_plugins

    load_plugins(cfg.plugins)

    if cfg.simulation is None:
        raise ValueError("Simulation configuration is required")

    if cfg.simulation.run_separate_instruments and cfg.simulation.variate:
        raise ValueError("Run separate instruments is not supported with variate")

    stg_cls, _strategy_full_classes = _import_strategy_class(cfg.strategy)

    simulation_name = config_file.stem
    _v_id = to_timestamp("now").strftime("%Y%m%d_%H%M%S")

    # - resolve run name: CLI --name > config 'name:' field > config filename stem
    _yaml_name = name or cfg.name or simulation_name
    if name:
        logger.info(f"Run name overridden via CLI: <g>{name}</g>")

    # - create simulation setup
    if cfg.simulation.variate:
        # - get conditions for variations if exists
        cond = cfg.simulation.variate.pop("with", None)
        conditions = []
        dict2lambda = lambda a, d: eval(f"lambda {a}: {d}")  # noqa: E731
        if cond:
            for a, c in cond.items():
                conditions.append(dict2lambda(a, c))

        # - if a parameter is of type list, then transform it to list of lists to avoid invalid variation
        for k, v in cfg.parameters.items():
            if isinstance(v, list):
                cfg.parameters[k] = [v]

        experiments = variate(stg_cls, **(cfg.parameters | cfg.simulation.variate), conditions=conditions)
        experiments = {f"{_yaml_name}.{_v_id}.[{k}]": v for k, v in experiments.items()}
        print(f"Parameters variation is configured. There are {len(experiments)} simulations to run.")
        _n_jobs = -1
    else:
        strategy = stg_cls(**cfg.parameters)
        experiments = {_yaml_name: strategy}
        _n_jobs = 1

    data_kwargs, sim_params = _build_sim_params(cfg, start=start, stop=stop)

    # - normalize description and tags from config
    _descr: str = ""
    if cfg.description is not None:
        _descr = "\n".join(cfg.description) if isinstance(cfg.description, list) else str(cfg.description)

    _tags = normalize_tags(cfg.tags)

    # - always use parquet-based storage; S3 creds resolved inside SimulationResultsSaver
    _use_storage = save_path is not None

    # - run simulation
    print(f" > Run simulation for [{red(simulation_name)}] ::: {sim_params['start']} - {sim_params['stop']}")
    sim_params["n_jobs"] = sim_params.get("n_jobs", _n_jobs)

    # - create simulation results saver first (needed for run_dir to set up local log path)
    _results_saver: SimulationResultsSaver | None = None
    if _use_storage:
        _results_saver = SimulationResultsSaver(
            name=_yaml_name,
            strategy_class=_strategy_full_classes,
            config_name=simulation_name,
            sim_start=cfg.simulation.start,
            sim_stop=cfg.simulation.stop,
            save_path=save_path,
            run_id=_v_id,
            config_file=str(config_file),
            tags=_tags,
            description=_descr,
            is_variation=bool(cfg.simulation.variate),
            storage_options=storage_options,
        )
        _results_saver.write_pending()

    # - resolve log file path (after saver so we can use run_dir for the local case)
    _log_file: str | None = None
    _cloud_log_file: str | None = None  # - set only for cloud; uploaded by saver, deleted on failure
    if log_file:
        # - explicit log file path from CLI --log-file
        _log_file = log_file
        print(f" > Logging to file {green(_log_file)} ...")
    elif save_path is not None:
        _is_cloud = is_cloud_path(save_path)

        if _is_cloud:
            import tempfile

            _tmp = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
            _tmp.close()
            _log_file = _cloud_log_file = _tmp.name
            print(f" > Logging to temp file (will upload to {green(save_path)} after simulation) ...")
        elif _results_saver is not None:
            # - local with storage: write directly into run_dir (already the right destination)
            makedirs(_results_saver.run_dir)
            _log_file = str(Path(_results_saver.run_dir) / f"{simulation_name}.log")
            print(f" > Logging to file {green(_log_file)} ...")
        else:
            # - local without storage: fall back to results/simulation_name/
            _log_dir = Path(makedirs("results/")) / simulation_name
            makedirs(str(_log_dir))
            _log_file = str(_log_dir / f"{simulation_name}.log")
            print(f" > Logging to file {green(_log_file)} ...")

    _t_sim_start = time.monotonic()
    try:
        test_res = simulate(experiments, **data_kwargs, log_file=_log_file, **sim_params)  # type: ignore
    except Exception as e:
        if _results_saver is not None:
            _results_saver.write_failed(e, log_file=_cloud_log_file)
        raise

    _sim_time_sec = int(round(time.monotonic() - _t_sim_start))

    # - attach simulation wall-clock time (raw seconds) to every result
    for _r in test_res:
        _r.simulation_time_sec = _sim_time_sec

    if _results_saver is not None:
        # - for cloud: pass temp log file so saver uploads it; for local it's already in run_dir
        _safe_store_results(
            results_saver=_results_saver,
            test_res=test_res,
            sim_time_sec=_sim_time_sec,
            cloud_log_file=_cloud_log_file,
            save_path=save_path,
            yaml_name=_yaml_name,
            strategy_full_classes=_strategy_full_classes,
            simulation_name=simulation_name,
            sim_start=cfg.simulation.start,
            sim_stop=cfg.simulation.stop,
            v_id=_v_id,
            config_file=config_file,
            tags=_tags,
            descr=_descr,
            is_variation=bool(cfg.simulation.variate),
        )
    elif _use_storage is False and len(test_res) > 0:
        # - no saver (no output path) — just log timing
        print(f" > Simulation finished in {green(convert_seconds_to_str(_sim_time_sec))}")

    return test_res
