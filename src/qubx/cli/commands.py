import os
from pathlib import Path

import click
from dotenv import load_dotenv

from qubx import QubxLogConfig, logger


@click.group()
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
@click.option(
    "--restore", "-r", is_flag=True, default=False, help="Restore strategy state from previous run.", show_default=True
)
@click.option("--no-color", is_flag=True, default=False, help="Disable colored logging output.", show_default=True)
def run(config_file: Path, account_file: Path | None, paper: bool, jupyter: bool, restore: bool, no_color: bool):
    """
    Starts the strategy with the given configuration file. If paper mode is enabled, account is not required.

    Account configurations are searched in the following priority:\n
    - If provided, the account file is searched first.\n
    - If exists, accounts.toml located in the same folder with the config searched.\n
    - If neither of the above are provided, the accounts.toml in the ~/qubx/accounts.toml path is searched.
    """
    from qubx.utils.misc import add_project_to_system_path, logo
    from qubx.utils.runner.runner import run_strategy_yaml, run_strategy_yaml_in_jupyter

    add_project_to_system_path()
    add_project_to_system_path(str(config_file.parent.parent))
    add_project_to_system_path(str(config_file.parent))
    if jupyter:
        run_strategy_yaml_in_jupyter(config_file, account_file, paper, restore)
    else:
        logo()
        run_strategy_yaml(config_file, account_file, paper=paper, restore=restore, blocking=True, no_color=no_color)


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
def simulate(config_file: Path, start: str | None, end: str | None, output: str | None):
    """
    Simulates the strategy with the given configuration file.
    """
    from qubx.utils.misc import add_project_to_system_path, logo
    from qubx.utils.runner.runner import simulate_strategy

    add_project_to_system_path()
    add_project_to_system_path(str(config_file.parent))
    logo()
    simulate_strategy(config_file, output, start, end)


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
def deploy(zip_file: str, output_dir: str | None, force: bool):
    """
    Deploys a strategy from a zip file created by the release command.

    This command:
    1. Unpacks the zip file to the specified output directory
    2. Creates a Poetry virtual environment in the .venv folder
    3. Installs dependencies from the poetry.lock file

    If no output directory is specified, the zip file is unpacked in the same directory
    as the zip file, in a folder with the same name as the zip file (without the .zip extension).
    """
    from .deploy import deploy_strategy

    deploy_strategy(zip_file, output_dir, force)


@main.command()
@click.argument(
    "results-path",
    type=click.Path(exists=True, resolve_path=True),
    default="results",
    callback=lambda ctx, param, value: os.path.abspath(os.path.expanduser(value)),
)
def browse(results_path: str):
    """
    Browse backtest results using an interactive TUI.

    Opens a text-based user interface for exploring backtest results stored in ZIP files.
    The browser provides:
    - Tree view of results organized by strategy
    - Table view with sortable metrics
    - Equity chart view for comparing performance

    Results are loaded from the specified directory containing .zip files
    created by qubx simulate or result.to_file() methods.
    """
    from .tui import run_backtest_browser

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
    poetry run qubx run --config config.yml --paper
    """
    from qubx.templates import TemplateManager, TemplateError
    
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
        click.echo("  poetry run qubx run config.yml --paper")
        click.echo()
        click.echo("To run in Jupyter mode:")
        click.echo("  ./jpaper.sh")
        
    except TemplateError as e:
        click.echo(f"❌ Template error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
