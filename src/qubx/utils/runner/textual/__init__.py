"""
Textual-based TUI for running qubx strategies with Jupyter kernel integration.

Features:
- Live REPL output panel for kernel interaction
- Application logs panel for qubx logger output
- Input bar for executing Python code in the kernel
- Real-time strategy monitoring
- Positions table with live updates
"""

from pathlib import Path

from qubx import logger

from .app import TextualStrategyApp

__all__ = ["run_strategy_yaml_in_textual"]


def run_strategy_yaml_in_textual(
    config_file: Path, account_file: Path | None = None, paper: bool = False, restore: bool = False
) -> None:
    """
    Run a strategy in a Textual TUI with Jupyter kernel integration.

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
        from jupyter_client import AsyncKernelManager  # noqa: F401
    except ImportError:
        logger.error("Can't find <r>jupyter_client</r> module - try to install it first")
        return

    logger.info("Running strategy in Textual TUI mode")

    # Create and run the app
    app = TextualStrategyApp(config_file, account_file, paper, restore)

    try:
        app.run()
    except Exception as e:
        logger.error(f"Textual app failed: {e}")
        raise
