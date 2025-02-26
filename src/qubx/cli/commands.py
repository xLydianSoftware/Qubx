import os
import sys
from pathlib import Path

import click

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
    default="INFO",
)
def main(debug: bool, debug_port: int, log_level: str):
    """
    Qubx CLI.
    """
    log_level = log_level.upper() if not debug else "DEBUG"

    QubxLogConfig.set_log_level(log_level)

    if debug:
        os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

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
def run(config_file: Path, account_file: Path | None, paper: bool, jupyter: bool):
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
    add_project_to_system_path(str(config_file.parent))
    if jupyter:
        run_strategy_yaml_in_jupyter(config_file, account_file, paper)
    else:
        logo()
        run_strategy_yaml(config_file, account_file, paper, blocking=True)


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
    "--strategy",
    "-s",
    type=click.STRING,
    help="Strategy name to release (should match the strategy class name) or path to a config YAML file",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.STRING,
    help="Output directory to put zip file.",
    default="releases",
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
    "-c",
    is_flag=True,
    default=False,
    help="Commit changes and create tag in repo (default: False)",
    show_default=True,
)
@click.option(
    "--default-exchange",
    type=click.STRING,
    help="Default exchange to use in the generated config.",
    default="BINANCE.UM",
    show_default=True,
)
@click.option(
    "--default-connector",
    type=click.STRING,
    help="Default connector to use in the generated config.",
    default="ccxt",
    show_default=True,
)
@click.option(
    "--default-instruments",
    type=click.STRING,
    help="Default instruments to use in the generated config (comma-separated).",
    default="BTCUSDT",
    show_default=True,
)
def release(
    directory: str,
    strategy: str,
    tag: str | None,
    message: str | None,
    commit: bool,
    output_dir: str,
    default_exchange: str,
    default_connector: str,
    default_instruments: str,
) -> None:
    """
    Releases the strategy to a zip file.

    The strategy can be specified in two ways:
    1. As a strategy name (class name) - strategies are scanned in the given directory
    2. As a path to a config YAML file containing the strategy configuration in StrategyConfig format

    If a strategy name is provided, a default configuration will be generated with:
    - The strategy parameters from the strategy class
    - Default exchange, connector, and instruments from the command options
    - Standard logging configuration

    If a config file is provided, it must follow the StrategyConfig structure with:
    - strategy: The strategy name or path
    - parameters: Dictionary of strategy parameters
    - exchanges: Dictionary of exchange configurations
    - aux: Auxiliary configuration
    - logging: Logging configuration

    All of the dependencies are included in the zip file.
    """
    from .release import release_strategy

    # Parse default instruments
    instruments = [instr.strip() for instr in default_instruments.split(",")]

    release_strategy(
        directory=directory,
        strategy_name=strategy,
        tag=tag,
        message=message,
        commit=commit,
        output_dir=output_dir,
        default_exchange=default_exchange,
        default_connector=default_connector,
        default_instruments=instruments,
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


if __name__ == "__main__":
    main()
