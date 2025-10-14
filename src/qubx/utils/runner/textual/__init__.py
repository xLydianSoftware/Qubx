"""
Textual-based TUI for running qubx strategies with Jupyter kernel integration.

Features:
- Live REPL output panel for kernel interaction
- Application logs panel for qubx logger output
- Input bar for executing Python code in the kernel
- Real-time strategy monitoring
- Positions table with live updates
"""

import asyncio
import os
import shlex
import sys
from pathlib import Path

from qubx import logger

from ..kernel_service import KernelService
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
    connection_file: str | None = None,
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
        connection_file: Optional path to existing kernel connection file (for web mode subprocesses)
    """
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return

    # Handle web serving mode
    if web_mode:
        try:
            from textual_serve.server import Server  # type: ignore
        except ImportError:
            logger.error("Can't find <r>textual-serve</r> module - try to install it first")
            logger.error("Run: poetry add textual-serve")
            return

        # Set default port for web server
        if port is None:
            port = 8000

        # Create a persistent event loop for web mode
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Start persistent kernel once in main process
        logger.info("Starting persistent kernel for web mode...")
        kernel_connection_file = loop.run_until_complete(KernelService.start(config_file, account_file, paper, restore))
        logger.info(f"Kernel started: {kernel_connection_file}")

        # Build the command to run the app (subprocesses will connect to existing kernel)
        cmd_parts = [sys.executable, "-m", "qubx.cli.commands", "run", str(Path(config_file).absolute())]
        if account_file:
            cmd_parts.extend(["--account-file", str(account_file)])
        if paper:
            cmd_parts.append("--paper")
        if restore:
            cmd_parts.append("--restore")
        cmd_parts.append("--textual")
        if dev_mode:
            cmd_parts.append("--textual-dev")
        # Pass connection file to subprocesses
        cmd_parts.extend(["--textual-connect", kernel_connection_file])

        command = " ".join(shlex.quote(str(part)) for part in cmd_parts)

        logger.info(f"Starting Textual web server on http://{host}:{port}")
        logger.info(f"Command: {command}")
        logger.info("All browser tabs will connect to the same kernel instance")

        try:
            # Create and start the server
            server = Server(command, host=host, port=port, title="Qubx Strategy Runner")
            server.serve()
        finally:
            # Stop the persistent kernel when server exits
            logger.info("Stopping persistent kernel...")
            loop.run_until_complete(KernelService.stop(kernel_connection_file))
            logger.info("Kernel stopped")
            loop.close()
        return

    # Normal terminal mode

    # Set default port for devtools
    if port is None:
        port = 8081

    # Set Textual dev mode environment variables if requested
    if dev_mode:
        os.environ["TEXTUAL"] = "devtools"
        os.environ["TEXTUAL_DEVTOOLS_PORT"] = str(port)
        logger.info(f"Textual dev mode enabled on port {port}")
        logger.info(f"Run 'textual console --port {port}' in another terminal to see debug output")

    logger.info("Running strategy in Textual TUI mode")

    # Add file logging for debugging (especially for web subprocess)
    import tempfile

    debug_log_file = Path(tempfile.gettempdir()) / f"qubx_textual_debug_{os.getpid()}.log"
    log_handler_id = logger.add(debug_log_file, level="DEBUG", format="{time} {level} {message}")
    logger.info(f"Debug logging to: {debug_log_file}")

    try:
        logger.debug(f"Process ID: {os.getpid()}")
        logger.debug(f"Config file: {config_file}")
        logger.debug(f"Connection file: {connection_file}")
        logger.debug(f"Paper mode: {paper}")

        # Determine if we need to start a kernel or connect to existing one
        kernel_started = False
        loop = None

        if connection_file is None:
            # Terminal mode: Need nest_asyncio and custom loop
            logger.debug("connection_file is None, starting new kernel (terminal mode)")

            # Apply nest_asyncio for terminal mode (required for custom event loop)
            try:
                import nest_asyncio

                nest_asyncio.apply()
                logger.debug("nest_asyncio applied for terminal mode")
            except ImportError:
                logger.error("Can't find nest_asyncio module - try to install it first")
                return

            # Create event loop for kernel startup
            loop = asyncio.new_event_loop()
            logger.debug(f"Created event loop: {loop}")

            # Start persistent kernel before creating UI (terminal mode)
            logger.info("Starting persistent kernel...")
            connection_file = loop.run_until_complete(KernelService.start(config_file, account_file, paper, restore))
            logger.info(f"Kernel started: {connection_file}")
            kernel_started = True
        else:
            logger.debug(f"connection_file provided: {connection_file} (web subprocess mode)")
            # Web subprocess mode: No nest_asyncio, no custom loop
            # Let Textual manage everything to work with textual-serve
            logger.info(f"Connecting to existing kernel: {connection_file}")

        logger.debug("About to create TextualStrategyApp")
        # Now start Textual app (it manages its own event loop)
        try:
            # Create and run the app connected to the persistent kernel
            logger.debug("Creating app instance")
            app = TextualStrategyApp(
                config_file,
                account_file,
                paper,
                restore,
                connection_file=Path(connection_file),
            )
            logger.debug(f"App instance created: {app}")

            # For web subprocess mode, let Textual use its default event loop
            # For terminal mode, pass the custom loop we created
            if loop:
                logger.debug("Running in terminal mode - passing custom event loop to Textual")
                app.run(loop=loop)
            else:
                logger.debug("Running in web subprocess mode - using default Textual event loop")
                app.run()

            logger.debug("app.run() completed normally")
        except Exception as e:
            logger.error(f"Textual app failed: {e}", exc_info=True)
            raise
        finally:
            logger.debug("Entered finally block")
            # Only stop the kernel if we started it (not if we connected to existing)
            if kernel_started and loop:
                logger.info("Stopping persistent kernel...")
                loop.run_until_complete(KernelService.stop(connection_file))
                logger.info("Kernel stopped")
            logger.debug("Exiting finally block")
    finally:
        # Remove debug log handler
        logger.remove(log_handler_id)
        logger.info(f"Debug log saved to: {debug_log_file}")
