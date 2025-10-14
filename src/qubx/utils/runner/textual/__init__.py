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
    connection_file: Path | None = None,
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
        connection_file: Optional path to existing kernel connection file
    """
    # Apply nest_asyncio early to ensure event loop is patched before any async operations
    # This is critical for Jupyter kernel connections to work in subprocess contexts (web mode)
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        logger.error("Can't find <r>nest_asyncio</r> module - try to install it first")
        return

    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_file}")
        return

    if connection_file and not connection_file.exists():
        connection_file = connection_file.absolute()

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

        # Start or use existing kernel for web mode
        kernel_conn_file = None
        if connection_file:
            # Use existing kernel
            kernel_conn_file = str(connection_file)
            logger.info(f"Using existing kernel: {kernel_conn_file}")
        else:
            # Start a persistent kernel for web mode
            import asyncio

            from qubx.utils.runner.kernel_service import KernelService

            logger.info("Starting persistent kernel for web mode...")
            kernel_conn_file = asyncio.run(KernelService.start(config_file, account_file, paper, restore))
            logger.info(f"Kernel started: {kernel_conn_file}")

        # Build the command to run the app with --connect
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
        cmd_parts.extend(["--connect", kernel_conn_file])
        if dev_mode:
            cmd_parts.append("--textual-dev")

        command = " ".join(shlex.quote(str(part)) for part in cmd_parts)

        logger.info(f"Starting Textual web server on http://{host}:{port}")
        logger.info("All browser connections will share the same kernel")
        logger.info(f"Command: {command}")

        # Create and start the server
        server = Server(command, host=host, port=port, title="Qubx Strategy Runner")

        try:
            server.serve()
        finally:
            # Clean up kernel if we started it
            if not connection_file:
                import asyncio

                from qubx.utils.runner.kernel_service import KernelService

                logger.info("Shutting down kernel...")
                asyncio.run(KernelService.stop(kernel_conn_file))
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

    # Verify jupyter_client is available
    try:
        from jupyter_client import AsyncKernelManager  # noqa: F401
    except ImportError:
        logger.error("Can't find <r>jupyter_client</r> module - try to install it first")
        return

    logger.info("Running strategy in Textual TUI mode")

    # If connecting to existing kernel, do it before creating the app to avoid event loop conflicts
    # Note: We don't start the iopub listener here because we're in a temporary event loop
    # It will be started later in the app's event loop
    kernel = None
    if connection_file:
        logger.info(f"Pre-connecting to kernel: {connection_file}")
        import asyncio

        from .kernel import IPyKernel

        kernel = IPyKernel()
        try:
            asyncio.run(kernel.connect_to_existing(str(connection_file), start_iopub=False))
            logger.info("Successfully pre-connected to kernel (iopub will start in app event loop)")
        except Exception as e:
            logger.error(f"Failed to pre-connect to kernel: {e}")
            raise

    # Create and run the app
    app = TextualStrategyApp(config_file, account_file, paper, restore, connection_file, kernel=kernel, watch_css=dev_mode)

    try:
        app.run()
    except Exception as e:
        logger.error(f"Textual app failed: {e}")
        raise
