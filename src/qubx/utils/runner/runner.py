import socket
import time
from collections import defaultdict
from functools import reduce
from pathlib import Path

import pandas as pd

from qubx import QubxLogConfig, logger
from qubx.backtester import simulate
from qubx.backtester.account import SimulatedAccountProcessor
from qubx.backtester.broker import SimulatedBroker
from qubx.backtester.optimization import variate
from qubx.backtester.runner import SimulationRunner
from qubx.backtester.simulated_exchange import get_simulated_exchange
from qubx.backtester.utils import (
    SetupTypes,
    SimulatedLogFormatter,
    SimulationConfigError,
    SimulationSetup,
    recognize_simulation_data_config,
)
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import get_ccxt_account, get_ccxt_broker, get_ccxt_exchange
from qubx.connectors.tardis.data import TardisDataProvider
from qubx.core.account import CompositeAccountProcessor
from qubx.core.basics import (
    CtrlChannel,
    Instrument,
    LiveTimeProvider,
    Position,
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
    IHealthMonitor,
    IStrategyContext,
    ITimeProvider,
)
from qubx.core.loggers import StrategyLogging
from qubx.core.lookups import lookup
from qubx.health import BaseHealthMonitor
from qubx.loggers import create_logs_writer
from qubx.restarts.state_resolvers import StateResolver
from qubx.restarts.time_finders import TimeFinder
from qubx.restorers import create_state_restorer
from qubx.utils.misc import class_import, green, makedirs, red
from qubx.utils.runner.configs import (
    ExchangeConfig,
    LoggingConfig,
    ReaderConfig,
    RestorerConfig,
    StrategyConfig,
    WarmupConfig,
    load_strategy_config_from_yaml,
)
from qubx.utils.runner.factory import (
    construct_reader,
    create_data_type_readers,
    create_exporters,
    create_lifecycle_notifiers,
    create_metric_emitters,
)

from .accounts import AccountConfigurationManager


def run_strategy_yaml(
    config_file: Path,
    account_file: Path | None = None,
    paper: bool = False,
    restore: bool = False,
    blocking: bool = False,
    no_color: bool = False,
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

    acc_manager = AccountConfigurationManager(account_file, config_file.parent, search_qubx_dir=True)
    stg_config = load_strategy_config_from_yaml(config_file)
    return run_strategy(stg_config, acc_manager, paper=paper, restore=restore, blocking=blocking, no_color=no_color)


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
    no_color: bool = False,
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

    QubxLogConfig.setup_logger(
        level=QubxLogConfig.get_log_level(),
        custom_formatter=(simulated_formatter := SimulatedLogFormatter(LiveTimeProvider())).formatter,
        colorize=not no_color,
    )

    # Restore state if configured
    restored_state = _restore_state(config.live.warmup.restorer if config.live.warmup else None) if restore else None

    # Create the strategy context
    ctx = create_strategy_context(
        config=config,
        account_manager=account_manager,
        paper=paper,
        restored_state=restored_state,
        simulated_formatter=simulated_formatter,
        no_color=no_color,
    )

    try:
        _run_warmup(
            ctx=ctx,
            restored_state=restored_state,
            exchanges=config.live.exchanges,
            warmup=config.live.warmup,
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
    simulated_formatter: SimulatedLogFormatter,
    no_color: bool = False,
) -> IStrategyContext:
    """
    Create a strategy context from the given configuration.
    """
    # Validate that live configuration exists
    if not config.live:
        raise ValueError("Live configuration is required for strategy execution")

    stg_name = _get_strategy_name(config)
    _run_mode = "paper" if paper else "live"

    # Generate run_id once to be shared between logging and metric emissions
    run_id = f"{socket.gethostname()}-{str(int(time.time() * 10**9))}"

    if isinstance(config.strategy, list):
        _strategy_class = reduce(lambda x, y: x + y, [class_import(x) for x in config.strategy])
    elif isinstance(config.strategy, str):
        _strategy_class = class_import(config.strategy)
    else:
        _strategy_class = config.strategy

    _logging = _setup_strategy_logging(stg_name, config.live.logging, simulated_formatter, run_id)

    _aux_reader = construct_reader(config.aux) if config.aux else None

    # Create metric emitters with run_id as a tag
    _metric_emitter = create_metric_emitters(config.live.emission, stg_name, run_id) if config.live.emission else None

    # Create lifecycle notifiers
    _lifecycle_notifier = create_lifecycle_notifiers(config.live.notifiers, stg_name) if config.live.notifiers else None

    # Create strategy initializer
    _initializer = BasicStrategyInitializer()

    _time = LiveTimeProvider()
    _chan = CtrlChannel("databus", sentinel=(None, None, None, None))
    _sched = BasicScheduler(_chan, lambda: _time.time().item())

    # Create time provider for metric emitters
    if _metric_emitter is not None:
        _metric_emitter.set_time_provider(_time)

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
                restored_state=restored_state,
                read_only=config.live.read_only,
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
            paper=paper,
        )
        _instruments.extend(_create_instruments_for_exchange(exchange_name, exchange_config))

    _account = (
        CompositeAccountProcessor(_time, _exchange_to_account)
        if len(exchanges) > 1
        else _exchange_to_account[exchanges[0]]
    )
    _initializer = BasicStrategyInitializer(simulation=_exchange_to_data_provider[exchanges[0]].is_simulation)

    # Create exporters if configured
    _exporter = create_exporters(config.live.exporters, stg_name, _account) if config.live.exporters else None

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
        aux_data_provider=_aux_reader,
        exporter=_exporter,
        emitter=_metric_emitter,
        lifecycle_notifier=_lifecycle_notifier,
        initializer=_initializer,
        strategy_name=stg_name,
        health_monitor=_health_monitor,
    )

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
    simulated_formatter: SimulatedLogFormatter,
    run_id: str,
) -> StrategyLogging:
    if not hasattr(log_config, "args") or not isinstance(log_config.args, dict):
        log_config.args = {}
    log_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    log_folder = log_config.args.get("log_folder", "logs")
    run_folder = f"{log_folder}/run_{log_id}"
    logger.add(
        f"{run_folder}/strategy/{stg_name}_{{time}}.log",
        format=simulated_formatter.formatter,
        rotation="100 MB",
        colorize=False,  # File logs should never have colors
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
    health_monitor: IHealthMonitor | None = None,
) -> IDataProvider:
    settings = account_manager.get_exchange_settings(exchange_name)
    match exchange_config.connector.lower():
        case "ccxt":
            exchange = get_ccxt_exchange(exchange_name, use_testnet=settings.testnet)
            return CcxtDataProvider(
                exchange=exchange,
                time_provider=time_provider,
                channel=channel,
                health_monitor=health_monitor,
            )
        case "tardis":
            return TardisDataProvider(
                host=exchange_config.params.get("host", "localhost"),
                port=exchange_config.params.get("port", 8011),
                exchange=exchange_name,
                time_provider=time_provider,
                channel=channel,
                health_monitor=health_monitor,
            )
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
    read_only: bool = False,
) -> IAccountProcessor:
    if paper:
        connector = "paper"
    elif exchange_config.account is not None:
        connector = exchange_config.account.connector
    else:
        connector = exchange_config.connector

    match connector.lower():
        case "ccxt":
            creds = account_manager.get_exchange_credentials(exchange_name)
            exchange = get_ccxt_exchange(
                exchange_name, use_testnet=creds.testnet, api_key=creds.api_key, secret=creds.secret
            )
            return get_ccxt_account(
                exchange_name,
                account_id=exchange_name,
                exchange=exchange,
                channel=channel,
                time_provider=time_provider,
                base_currency=creds.base_currency,
                tcc=tcc,
                read_only=read_only,
            )
        case "paper":
            settings = account_manager.get_exchange_settings(exchange_name)

            # - TODO: here we can create  different types of simulated exchanges based on it's name etc
            simulated_exchange = get_simulated_exchange(exchange_name, time_provider, tcc)

            return SimulatedAccountProcessor(
                account_id=exchange_name,
                exchange=simulated_exchange,
                channel=channel,
                base_currency=settings.base_currency,
                initial_capital=settings.initial_capital,
                restored_state=restored_state,
            )
        case _:
            raise ValueError(f"Connector {exchange_config.connector} is not supported yet !")


def _create_broker(
    exchange_name: str,
    exchange_config: ExchangeConfig,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    data_provider: IDataProvider,
    account_manager: AccountConfigurationManager,
    paper: bool,
) -> IBroker:
    if paper:
        connector = "paper"
        params = {}
    elif exchange_config.broker is not None:
        connector = exchange_config.broker.connector
        params = exchange_config.broker.params
    else:
        connector = exchange_config.connector
        params = exchange_config.params

    match connector.lower():
        case "ccxt":
            creds = account_manager.get_exchange_credentials(exchange_name)
            _enable_mm = params.pop("enable_mm", False)
            exchange = get_ccxt_exchange(
                exchange_name,
                use_testnet=creds.testnet,
                api_key=creds.api_key,
                secret=creds.secret,
                enable_mm=_enable_mm,
            )
            return get_ccxt_broker(exchange_name, exchange, channel, time_provider, account, data_provider, **params)
        case "paper":
            assert isinstance(account, SimulatedAccountProcessor)
            return SimulatedBroker(channel=channel, account=account, simulated_exchange=account._exchange)
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
    data_type_to_reader = create_data_type_readers(warmup.readers) if warmup else {}

    if not data_type_to_reader:
        logger.warning("<yellow>No readers were created for warmup</yellow>")
        return

    _aux_reader = construct_reader(aux_config) if aux_config else None

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
            name=getattr(ctx, "_strategy_name", ctx.strategy.__class__.__name__),
            generator=ctx.strategy,
            tracker=None,
            instruments=instruments,
            exchanges=ctx.exchanges,
            capital=ctx.account.get_capital(),
            base_currency=ctx.account.get_base_currency(),
            commissions=None,  # TODO: get commissions from somewhere
        ),
        data_config=recognize_simulation_data_config(
            decls=data_type_to_reader,  # type: ignore
            instruments=instruments,
            aux_data=_aux_reader,
            prefetch_config=warmup.prefetch,
        ),
        start=pd.Timestamp(warmup_start_time),
        stop=pd.Timestamp(current_time),
        emitter=ctx.emitter,
        strategy_state=ctx._strategy_state,
        initializer=ctx.initializer,
        warmup_mode=True,
    )

    # - set the time provider to the simulated runner
    _live_time_provider = simulated_formatter.time_provider
    simulated_formatter.time_provider = warmup_runner.ctx

    ctx._strategy_state.is_warmup_in_progress = True

    try:
        warmup_runner.run(catch_keyboard_interrupt=False, close_data_readers=True)
    finally:
        # Restore the live time provider
        simulated_formatter.time_provider = _live_time_provider
        # Set back the time provider for metric emitters to use live time provider
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

    cfg = load_strategy_config_from_yaml(config_file)

    if cfg.simulation is None:
        raise ValueError("Simulation configuration is required")

    if cfg.simulation.run_separate_instruments and cfg.simulation.variate:
        raise ValueError("Run separate instruments is not supported with variate")

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
        experiments = {f"{simulation_name}.{_v_id}.[{k}]": v for k, v in experiments.items()}
        print(f"Parameters variation is configured. There are {len(experiments)} simulations to run.")
        _n_jobs = -1
    else:
        strategy = stg_cls(**cfg.parameters)
        experiments = {simulation_name: strategy}
        _n_jobs = 1

    # - resolve data readers
    data_i = create_data_type_readers(cfg.simulation.data) if cfg.simulation.data else {}

    sim_params = {
        "instruments": cfg.simulation.instruments,
        "capital": cfg.simulation.capital,
        "commissions": cfg.simulation.commissions,
        "start": cfg.simulation.start,
        "stop": cfg.simulation.stop,
        "enable_funding": cfg.simulation.enable_funding,
        "enable_inmemory_emitter": cfg.simulation.enable_inmemory_emitter,
    }

    if cfg.simulation.debug is not None:
        sim_params["debug"] = cfg.simulation.debug

    if start is not None:
        sim_params["start"] = start
        logger.info(f"Start date set to {start}")

    if stop is not None:
        sim_params["stop"] = stop
        logger.info(f"Stop date set to {stop}")

    if cfg.simulation.n_jobs is not None:
        sim_params["n_jobs"] = cfg.simulation.n_jobs

    # - check for aux_data parameter
    if cfg.aux is not None:
        sim_params["aux_data"] = construct_reader(cfg.aux)

    # - add run_separate_instruments parameter
    if cfg.simulation.run_separate_instruments:
        sim_params["run_separate_instruments"] = True

    # - add prefetch_config parameter
    if cfg.simulation.prefetch is not None:
        sim_params["prefetch_config"] = cfg.simulation.prefetch

    # - run simulation
    print(f" > Run simulation for [{red(simulation_name)}] ::: {sim_params['start']} - {sim_params['stop']}")
    sim_params["n_jobs"] = sim_params.get("n_jobs", _n_jobs)
    test_res = simulate(experiments, data=data_i, **sim_params)  # type: ignore

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
        print(f" > Saving variations results to {green(s_path)} ...")
        for k, t in enumerate(test_res):
            # - set variation name
            t.variation_name = f"{simulation_name}.{_v_id}"
            t.to_file(str(s_path), description=_descr, suffix=f".{k}", attachments=[str(config_file)])
    else:
        print(f" > Saving simulation results to {green(s_path)} ...")
        test_res[0].to_file(str(s_path), description=_descr, attachments=[str(config_file)])

    return test_res
