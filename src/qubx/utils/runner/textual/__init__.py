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
    config_file: Path,
    account_file: Path | None = None,
    paper: bool = False,
    restore: bool = False,
    dev_mode: bool = False,
    web_mode: bool = False,
    port: int | None = None,
    host: str = "0.0.0.0",
) -> None:
    """
    Run a strategy in a Textual TUI with Jupyter kernel integration.

    Args:
        config_file: Path to the strategy configuration file
        account_file: Path to the account configuration file
        paper: Whether to run in paper trading mode
        restore: Whether to restore the strategy state
        dev_mode: Whether to enable Textual dev mode (use with 'textual console')
        web_mode: Whether to serve the app in a web browser
        port: Port for Textual (web server: 8000 default, devtools: 8081 default)
        host: Host for Textual web server (default: 0.0.0.0 for all interfaces)
    """
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return

    # Handle web serving mode
    if web_mode:
        try:
            from textual_serve.server import Server
        except ImportError:
            logger.error("Can't find <r>textual-serve</r> module - try to install it first")
            logger.error("Run: poetry add textual-serve")
            return

        # Set default port for web server
        if port is None:
            port = 8000

        # Build the command to run the app
        import shlex
        import sys

        cmd_parts = [sys.executable, "-m", "qubx.cli.commands", "run", str(config_file)]
        if account_file:
            cmd_parts.extend(["--account-file", str(account_file)])
        if paper:
            cmd_parts.append("--paper")
        if restore:
            cmd_parts.append("--restore")
        cmd_parts.append("--textual")
        if dev_mode:
            cmd_parts.append("--textual-dev")

        command = " ".join(shlex.quote(str(part)) for part in cmd_parts)

        logger.info(f"Starting Textual web server on http://{host}:{port}")
        logger.info(f"Command: {command}")

        # Create and start the server
        server = Server(command, host=host, port=port, title="Qubx Strategy Runner")
        server.serve()
        return

    # Normal terminal mode
    import os

    # Set default port for devtools
    if port is None:
        port = 8081

    # Set Textual dev mode environment variables if requested
    if dev_mode:
        os.environ["TEXTUAL"] = "devtools"
        os.environ["TEXTUAL_DEVTOOLS_PORT"] = str(port)
        logger.info(f"Textual dev mode enabled on port {port}")
        logger.info(f"Run 'textual console --port {port}' in another terminal to see debug output")

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
