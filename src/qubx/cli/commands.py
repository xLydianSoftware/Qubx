import os
from importlib.metadata import version
from pathlib import Path

import click
from dotenv import load_dotenv

from qubx import QubxLogConfig, logger


@click.group()
@click.version_option(version=version("qubx"), prog_name="qubx")
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Enable debug mode.",
)
@click.option(
    "--debug-port",
    "-p",
    type=int,
    help="Debug port.",
    default=5678,
)
@click.option(
    "--log-level",
    "-l",
    type=str,
    help="Log level.",
    default=os.getenv("LOG_LEVEL", "INFO"),
)
def main(debug: bool, debug_port: int, log_level: str):
    """
    Qubx CLI.
    """
    # Suppress syntax warnings from AST parsing during import resolution
    import warnings

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    log_level = log_level.upper() if not debug else "DEBUG"

    env_file = Path.cwd().joinpath(".env")
    if env_file.exists():
        logger.info(f"Loading environment variables from {env_file}")
        load_dotenv(env_file)
        log_level = os.getenv("QUBX_LOG_LEVEL", log_level)

    QubxLogConfig.set_log_level(log_level)

    if debug:
        import debugpy

        logger.info(f"Waiting for debugger to attach (port {debug_port})")

        debugpy.listen(debug_port)
        debugpy.wait_for_client()


@main.command()
@click.argument("config-file", type=Path, required=True)
@click.option(
    "--account-file",
    "-a",
    type=Path,
    help="Account configuration file path.",
    required=False,
)
@click.option("--paper", "-p", is_flag=True, default=False, help="Use paper trading mode.", show_default=True)
@click.option(
    "--jupyter", "-j", is_flag=True, default=False, help="Run strategy in jupyter console.", show_default=True
)
@click.option("--textual", "-t", is_flag=True, default=False, help="Run strategy in textual TUI.", show_default=True)
@click.option(
    "--textual-dev",
    is_flag=True,
    default=False,
    help="Enable Textual dev mode (use with 'textual console').",
    show_default=True,
)
@click.option("--textual-web", is_flag=True, default=False, help="Serve Textual app in web browser.", show_default=True)
@click.option(
    "--textual-port",
    type=int,
    default=None,
    help="Port for Textual (web server: 8000, devtools: 8081).",
    show_default=False,
)
@click.option("--textual-host", type=str, default="0.0.0.0", help="Host for Textual web server.", show_default=True)
@click.option(
    "--kernel-only",
    is_flag=True,
    default=False,
    help="Start kernel without UI (returns connection file).",
    show_default=True,
)
@click.option(
    "--connect", type=Path, default=None, help="Connect to existing kernel via connection file.", show_default=False
)
@click.option(
    "--restore", "-r", is_flag=True, default=False, help="Restore strategy state from previous run.", show_default=True
)
@click.option("--no-color", is_flag=True, default=False, help="Disable colored logging output.", show_default=True)
@click.option(
    "--dev", is_flag=True, default=False, help="Enable dev mode (adds ~/projects to path).", show_default=True
)
@click.option("--no-emission", is_flag=True, default=False, help="Disable metric emission.", show_default=True)
@click.option("--no-notifiers", is_flag=True, default=False, help="Disable lifecycle notifiers.", show_default=True)
@click.option("--no-exporters", is_flag=True, default=False, help="Disable trade exporters.", show_default=True)
@click.option("--override", type=Path, default=None, help="Sparse YAML file to deep-merge on top of config.", show_default=False)
def run(
    config_file: Path,
    account_file: Path | None,
    paper: bool,
    jupyter: bool,
    textual: bool,
    textual_dev: bool,
    textual_web: bool,
    textual_port: int | None,
    textual_host: str,
    kernel_only: bool,
    connect: Path | None,
    restore: bool,
    no_color: bool,
    dev: bool,
    no_emission: bool,
    no_notifiers: bool,
    no_exporters: bool,
    override: Path | None,
):
    """
    Starts the strategy with the given configuration file. If paper mode is enabled, account is not required.

    Account configurations are searched in the following priority:\n
    - If provided, the account file is searched first.\n
    - If exists, accounts.toml located in the same folder with the config searched.\n
    - If neither of the above are provided, the accounts.toml in the ~/qubx/accounts.toml path is searched.
    """
    from qubx.utils.misc import add_project_to_system_path, logo
    from qubx.utils.runner.runner import run_strategy_yaml, run_strategy_yaml_in_jupyter
    from qubx.utils.runner.textual import run_strategy_yaml_in_textual

    # Ensure jupyter and textual are mutually exclusive
    if jupyter and textual:
        click.echo("Error: --jupyter and --textual cannot be used together.", err=True)
        raise click.Abort()

    if dev:
        add_project_to_system_path()  # Adds ~/projects in dev mode
    add_project_to_system_path(config_file.parent)

    # Handle --kernel-only mode
    if kernel_only:
        import asyncio

        from qubx.utils.runner.kernel_service import KernelService

        click.echo("Starting persistent kernel...")
        connection_file = asyncio.run(KernelService.start(config_file, account_file, paper, restore))
        click.echo(click.style("✓ Kernel started successfully!", fg="green", bold=True))
        click.echo(click.style(f"Connection file: {connection_file}", fg="cyan"))
        click.echo()
        click.echo("To connect a UI to this kernel:")
        click.echo(f"  qubx run --textual --connect {connection_file}")
        click.echo()
        click.echo("To stop this kernel:")
        click.echo(f"  qubx kernel stop {connection_file}")
        click.echo()
        click.echo("Press Ctrl+C to stop the kernel and exit...")

        # Keep the process alive until interrupted
        try:
            import signal

            signal.pause()
        except KeyboardInterrupt:
            click.echo("\nShutting down kernel...")
            asyncio.run(KernelService.stop(connection_file))
            click.echo("Kernel stopped.")
        return

    if jupyter:
        run_strategy_yaml_in_jupyter(config_file, account_file, paper, restore, no_emission, no_notifiers, no_exporters)
    elif textual:
        run_strategy_yaml_in_textual(
            config_file,
            account_file,
            paper,
            restore,
            textual_dev,
            textual_web,
            textual_port,
            textual_host,
            connect,
            dev,
            no_emission,
            no_notifiers,
            no_exporters,
        )
    else:
        logo()
        logger.info(f"Process PID: {os.getpid()}")
        run_strategy_yaml(
            config_file,
            account_file,
            paper=paper,
            restore=restore,
            blocking=True,
            no_color=no_color,
            no_emission=no_emission,
            no_notifiers=no_notifiers,
            no_exporters=no_exporters,
            config_overrides=override,
        )


@main.command()
@click.argument("config-file", type=Path, required=True)
@click.option(
    "--start", "-s", default=None, type=str, help="Override simulation start date from config.", show_default=True
)
@click.option(
    "--end", "-e", default=None, type=str, help="Override simulation end date from config.", show_default=True
)
@click.option(
    "--output", "-o", default="results", type=str, help="Output directory for simulation results.", show_default=True
)
@click.option(
    "--report", "-r", default=None, type=str, help="Output directory for simulation reports.", show_default=True
)
@click.option(
    "--log",
    "-L",
    "log_to_file",
    is_flag=True,
    default=False,
    help="Write simulation logs to a file in the output directory.",
    show_default=True,
)
@click.option(
    "--name",
    "-n",
    default=None,
    type=str,
    help="Override the run name used for output folder (e.g. 'smoketest', '01_reference'). "
    "Overrides the 'name:' field from the config file.",
    show_default=True,
)
def simulate(
    config_file: Path,
    start: str | None,
    end: str | None,
    output: str | None,
    report: str | None,
    log_to_file: bool,
    name: str | None,
):
    """
    Simulates the strategy with the given configuration file.
    """
    from qubx.utils.misc import add_project_to_system_path, logo
    from qubx.utils.runner.runner import simulate_strategy

    add_project_to_system_path()
    add_project_to_system_path(str(config_file.parent))
    logo()
    logger.info(f"Process PID: <g>{os.getpid()}</g>")
    simulate_strategy(config_file, output, start, end, report, log_to_file, name=name)


@main.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, resolve_path=True),
    default=".",
    callback=lambda ctx, param, value: os.path.abspath(os.path.expanduser(value)),
)
def ls(directory: str):
    """
    Lists all strategies in the given directory.

    Strategies are identified by the inheritance from IStrategy interface.
    """
    from .release import ls_strats

    ls_strats(directory)


@main.command()
@click.argument("config-file", type=Path, required=True)
@click.option(
    "--no-check-imports",
    is_flag=True,
    default=False,
    help="Skip checking if strategy class can be imported",
    show_default=True,
)
def validate(config_file: Path, no_check_imports: bool):
    """
    Validates a strategy configuration file without running it.

    Checks for:
    - Valid YAML syntax
    - Required configuration fields
    - Strategy class exists and can be imported (unless --no-check-imports)
    - Exchange configurations are valid
    - Simulation parameters are valid (if present)

    Returns exit code 0 if valid, 1 if invalid.
    """
    from qubx.utils.runner.configs import validate_strategy_config

    result = validate_strategy_config(config_file, check_imports=not no_check_imports)

    if result.valid:
        click.echo(click.style("✓ Configuration is valid", fg="green", bold=True))
        if result.warnings:
            click.echo(click.style("\nWarnings:", fg="yellow", bold=True))
            for warning in result.warnings:
                click.echo(click.style(f"  - {warning}", fg="yellow"))
        raise SystemExit(0)
    else:
        click.echo(click.style("✗ Configuration is invalid", fg="red", bold=True))
        click.echo(click.style("\nErrors:", fg="red", bold=True))
        for error in result.errors:
            click.echo(click.style(f"  - {error}", fg="red"))
        if result.warnings:
            click.echo(click.style("\nWarnings:", fg="yellow", bold=True))
            for warning in result.warnings:
                click.echo(click.style(f"  - {warning}", fg="yellow"))
        raise SystemExit(1)


@main.command()
@click.argument(
    "directory",
    type=click.Path(exists=False),
    default=".",
    callback=lambda ctx, param, value: os.path.abspath(os.path.expanduser(value)),
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to a config YAML file",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=False),
    help="Output directory to put zip file.",
    default=".releases",
    show_default=True,
)
@click.option(
    "--tag",
    "-t",
    type=click.STRING,
    help="Additional tag for this release (e.g. 'v1.0.0')",
    required=False,
)
@click.option(
    "--message",
    "-m",
    type=click.STRING,
    help="Release message (added to the info yaml file).",
    required=False,
    default=None,
    show_default=True,
)
@click.option(
    "--commit",
    is_flag=True,
    default=False,
    help="Commit changes and create tag in repo (default: False)",
    show_default=True,
)
def release(
    directory: str,
    config: str,
    tag: str | None,
    message: str | None,
    commit: bool,
    output_dir: str,
) -> None:
    """
    Releases the strategy to a zip file.

    The strategy is specified by a path to a config YAML file containing the strategy configuration in StrategyConfig format.

    The config file must follow the StrategyConfig structure with:
    - strategy: The strategy name or path
    - parameters: Dictionary of strategy parameters
    - exchanges: Dictionary of exchange configurations
    - aux: Auxiliary configuration
    - logging: Logging configuration

    All of the dependencies are included in the zip file.
    """
    from .release import release_strategy

    release_strategy(
        directory=directory,
        config_file=config,
        tag=tag,
        message=message,
        commit=commit,
        output_dir=output_dir,
    )


@main.command()
@click.argument(
    "zip-file",
    type=click.Path(exists=True, resolve_path=True),
    callback=lambda ctx, param, value: os.path.abspath(os.path.expanduser(value)),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=False),
    help="Output directory to unpack the zip file. Defaults to the directory containing the zip file.",
    default=None,
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force overwrite if the output directory already exists.",
    show_default=True,
)
@click.option(
    "--system",
    is_flag=True,
    default=False,
    help="Install wheels into system site-packages (Docker mode, no venv/uv needed).",
    show_default=True,
)
def deploy(zip_file: str, output_dir: str | None, force: bool, system: bool):
    """
    Deploys a strategy from a zip file created by the release command.

    This command:
    1. Unpacks the zip file to the specified output directory
    2. Auto-detects the package manager (uv or legacy poetry) from the lock file
    3. Creates a virtual environment and installs dependencies

    With --system flag (Docker mode):
    - Installs wheels directly into system site-packages via pip
    - No virtual environment or uv needed

    Supports both uv-based releases (uv.lock) and legacy Poetry-based releases (poetry.lock).

    If no output directory is specified, the zip file is unpacked in the same directory
    as the zip file, in a folder with the same name as the zip file (without the .zip extension).
    """
    from .deploy import deploy_strategy

    deploy_strategy(zip_file, output_dir, force, system=system)


@main.command()
@click.argument(
    "results-path",
    type=str,
    default="results",
)
def browse(results_path: str):
    """
    Browse backtest results using an interactive TUI.

    Opens a text-based user interface for exploring backtest results.
    Supports local directories and cloud storage (S3, GCS, Azure).

    The browser provides:
    - Tree view of results organized by strategy
    - Table view with sortable metrics (Sharpe, CAGR, drawdown...)
    - Tearsheet viewer with one-click browser open

    Examples:

        qubx browse /backtests/momentum/

        qubx browse s3://my-bucket/backtests/
    """
    from qubx.utils.results import is_cloud_path

    from .tui import run_backtest_browser

    # - only resolve local paths; cloud URIs (s3://, gs://, az://) are passed as-is
    if not is_cloud_path(results_path):
        results_path = os.path.abspath(os.path.expanduser(results_path))

    run_backtest_browser(results_path)


@main.command()
@click.option(
    "--template",
    "-t",
    type=str,
    default="simple",
    help="Built-in template to use (default: simple)",
    show_default=True,
)
@click.option(
    "--template-path",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to custom template directory",
)
@click.option(
    "--name",
    "-n",
    type=str,
    default="my_strategy",
    help="Name of the strategy to create",
    show_default=True,
)
@click.option(
    "--exchange",
    "-e",
    type=str,
    default="BINANCE.UM",
    help="Exchange to configure for the strategy",
    show_default=True,
)
@click.option(
    "--symbols",
    "-s",
    type=str,
    default="BTCUSDT",
    help="Comma-separated list of symbols to trade",
    show_default=True,
)
@click.option(
    "--timeframe",
    type=str,
    default="1h",
    help="Timeframe for market data",
    show_default=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(resolve_path=True),
    default=".",
    help="Directory to create the strategy in",
    show_default=True,
)
@click.option(
    "--list-templates",
    is_flag=True,
    help="List all available built-in templates",
)
def init(
    template: str,
    template_path: str | None,
    name: str,
    exchange: str,
    symbols: str,
    timeframe: str,
    output_dir: str,
    list_templates: bool,
):
    """
    Create a new strategy from a template.

    This command generates a complete strategy project structure with:
    - Strategy class implementing IStrategy interface
    - Configuration file for qubx run command
    - Package structure for proper imports

    The generated strategy can be run immediately with:
    uv run qubx run --config config.yml --paper
    """
    from qubx.templates import TemplateError, TemplateManager

    try:
        manager = TemplateManager()

        if list_templates:
            templates = manager.list_templates()
            if not templates:
                click.echo("No templates available.")
                return

            click.echo("Available templates:")
            for template_name, metadata in templates.items():
                description = metadata.get("description", "No description")
                click.echo(f"  {template_name:<15} - {description}")
            return

        # Generate strategy
        strategy_path = manager.generate_strategy(
            template_name=template if not template_path else None,
            template_path=template_path,
            output_dir=output_dir,
            name=name,
            exchange=exchange,
            symbols=symbols,
            timeframe=timeframe,
        )

        click.echo(f"✅ Strategy '{name}' created successfully!")
        click.echo(f"📁 Location: {strategy_path}")
        click.echo()
        click.echo("To run your strategy:")
        click.echo(f"  cd {strategy_path}")
        click.echo("  uv run qubx run config.yml --paper")
        click.echo()
        click.echo("To run in Jupyter mode:")
        click.echo("  ./jpaper.sh")

    except TemplateError as e:
        click.echo(f"❌ Template error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        raise click.Abort()


@main.group()
def kernel():
    """
    Manage persistent Jupyter kernels for strategy execution.

    Kernels can be started independently of the UI, allowing multiple
    UI instances to connect to the same running strategy.
    """
    pass


@kernel.command("list")
def kernel_list():
    """
    List all active kernel sessions.

    Shows connection files and associated strategy configurations
    for all currently running kernels.
    """
    from qubx.utils.runner.kernel_service import KernelService

    active = KernelService.list_active()

    if not active:
        click.echo("No active kernels found.")
        return

    import datetime

    click.echo(click.style("Active Kernels:", fg="cyan", bold=True))
    click.echo()
    for i, kernel_info in enumerate(active, 1):
        # Format timestamp
        ts = datetime.datetime.fromtimestamp(kernel_info["timestamp"])
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        click.echo(f"{i}. {click.style('Strategy:', fg='yellow')} {kernel_info['strategy_name']}")
        click.echo(f"   {click.style('Started:', fg='yellow')} {time_str}")
        click.echo(f"   {click.style('Connection:', fg='yellow')} {kernel_info['connection_file']}")
        click.echo()


@kernel.command("stop")
@click.argument("connection-file", type=Path, required=True)
def kernel_stop(connection_file: Path):
    """
    Stop a running kernel by its connection file.

    This will gracefully shutdown the kernel and clean up
    the connection file.
    """
    import asyncio

    from qubx.utils.runner.kernel_service import KernelService

    if not connection_file.exists():
        click.echo(click.style(f"✗ Connection file not found: {connection_file}", fg="red"))
        raise click.Abort()

    click.echo(f"Stopping kernel: {connection_file}")
    asyncio.run(KernelService.stop(str(connection_file)))
    click.echo(click.style("✓ Kernel stopped successfully", fg="green"))


def _resolve_storage_path(ctx: click.Context, param: click.Parameter, value: str) -> str:
    # - S3 and other cloud paths are returned as-is
    if value and not (value.startswith("s3://") or value.startswith("gs://") or value.startswith("az://")):
        return os.path.abspath(os.path.expanduser(value))
    return value


@main.command()
@click.argument(
    "storage-path",
    type=str,
    default="results",
    callback=_resolve_storage_path,
)
@click.option(
    "--where",
    "-w",
    default=None,
    type=str,
    help='SQL WHERE clause to filter results (e.g. "sharpe > 1.5").',
    show_default=False,
)
@click.option(
    "--order-by",
    "-O",
    default="creation_time DESC",
    type=str,
    help="SQL ORDER BY clause for sorting results.",
    show_default=True,
)
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Limit number of results shown.",
    show_default=False,
)
@click.option(
    "--params",
    "-p",
    is_flag=True,
    default=False,
    help="Show strategy parameters for each result.",
    show_default=True,
)
def backtests(storage_path: str, where: str | None, order_by: str, limit: int | None, params: bool):
    """
    List backtest results stored in the given path.

    Reads parquet-based storage created by qubx simulate and displays
    a formatted summary of all simulation results with performance metrics.

    Supports local directories and cloud paths (s3://, gs://, az://).

    Examples:\n
      qubx backtests /data/storage/backtests/\n
      qubx backtests /data/storage/backtests/ --where "sharpe > 1.5"\n
      qubx backtests /data/storage/backtests/ --limit 10 --order-by "sharpe DESC"\n
      qubx backtests s3://my-bucket/backtests/ --params
    """
    from qubx.backtester.management import BacktestStorage

    logger.info(f"Loading backtest results from: <g>{storage_path}</g> ...")
    bs = BacktestStorage(storage_path)
    bs.print(where=where, order_by=order_by, limit=limit, params=params)


@main.group()
def s3():
    """
    Operate on S3-compatible storage using named accounts.

    Paths use the format: account:bucket/path

    Examples:\n
      qubx s3 ls hetzner:frab/spreads/\n
      qubx s3 rm hetzner:frab/spreads/ --recursive\n
      qubx s3 cp hetzner:frab/spreads/data.parquet ./local_copy.parquet
    """
    pass


@s3.command("accounts")
def s3_accounts_cmd():
    """List configured S3 accounts."""
    from .s3 import s3_accounts

    s3_accounts()


@s3.command("ls")
@click.argument("path", type=str)
@click.option("--recursive", "-r", is_flag=True, help="List files recursively.")
@click.option("--long", "-l", is_flag=True, help="Show detailed info (size, type).")
def s3_ls_cmd(path: str, recursive: bool, long: bool):
    """
    List files in an S3 path.

    Examples:\n
      qubx s3 ls hetzner:frab/\n
      qubx s3 ls hetzner:frab/spreads/ -r -l
    """
    from .s3 import s3_ls

    s3_ls(path, recursive=recursive, long=long)


@s3.command("rm")
@click.argument("path", type=str)
@click.option("--recursive", "-r", is_flag=True, help="Delete recursively.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def s3_rm_cmd(path: str, recursive: bool, yes: bool):
    """
    Delete files from S3.

    Examples:\n
      qubx s3 rm hetzner:frab/spreads/long_exchange=BINANCE.UM/asset=BTC/data.parquet\n
      qubx s3 rm hetzner:frab/spreads/ -r
    """
    if not yes:
        label = f"{path} (recursive)" if recursive else path
        click.confirm(f"Delete {label}?", abort=True)

    from .s3 import s3_rm

    s3_rm(path, recursive=recursive)


@s3.command("cp")
@click.argument("src", type=str)
@click.argument("dst", type=str)
@click.option("--recursive", "-r", is_flag=True, help="Copy recursively.")
def s3_cp_cmd(src: str, dst: str, recursive: bool):
    """
    Copy files to/from S3.

    At least one of SRC or DST must be an S3 path (account:bucket/path).
    The other can be a local path for upload/download.

    Examples:\n
      qubx s3 cp hetzner:frab/data.parquet ./local.parquet\n
      qubx s3 cp ./local.parquet hetzner:frab/data.parquet\n
      qubx s3 cp hetzner:frab/spreads/ ./spreads/ -r
    """
    from .s3 import s3_cp

    s3_cp(src, dst, recursive=recursive)


if __name__ == "__main__":
    main()
