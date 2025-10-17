"""
Kernel service manager for persistent strategy kernels.

Manages the lifecycle of Jupyter kernels that run independently of the UI,
allowing multiple UI instances to connect to the same running strategy.
"""

import asyncio
import time
from pathlib import Path

from jupyter_client.manager import AsyncKernelManager

from qubx import logger


class KernelService:
    """Manages persistent kernel subprocesses for strategy execution."""

    _active_kernels: dict[str, tuple[AsyncKernelManager, Path]] = {}

    @classmethod
    async def start(
        cls,
        config_file: Path,
        account_file: Path | None,
        paper: bool,
        restore: bool,
    ) -> str:
        """
        Start a persistent kernel with strategy initialization.

        Args:
            config_file: Path to strategy configuration
            account_file: Path to account configuration (optional)
            paper: Whether to run in paper trading mode
            restore: Whether to restore strategy state

        Returns:
            Path to the kernel connection file
        """
        # Create ~/.qubx/kernels directory for connection files
        kernels_dir = Path.home() / ".qubx" / "kernels"
        kernels_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique connection file name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        strategy_name = config_file.stem
        conn_file = kernels_dir / f"{strategy_name}_{timestamp}.json"

        # Create kernel manager with custom connection file
        km = AsyncKernelManager(kernel_name="python3", connection_file=str(conn_file))
        await km.start_kernel()

        # Write the connection file (now it will use our custom path)
        km.write_connection_file()

        # Wait for kernel to be ready
        kc = km.client()
        kc.start_channels()
        await kc.wait_for_ready()

        logger.info(f"Kernel started with connection file: {conn_file}")

        # Initialize the strategy in the kernel
        from .textual.init_code import generate_init_code

        init_code = generate_init_code(config_file, account_file, paper, restore)
        msg_id = kc.execute(init_code, silent=False)

        # Wait for initialization to complete before allowing UI connections
        logger.info("Waiting for strategy initialization to complete...")
        timeout = 30.0  # Give it 30 seconds to initialize
        start_time = asyncio.get_event_loop().time()

        init_complete = False
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                msg = await asyncio.wait_for(kc.get_iopub_msg(timeout=1.0), timeout=1.0)

                if msg["parent_header"].get("msg_id") == msg_id:
                    msg_type = msg["header"]["msg_type"]

                    if msg_type == "status":
                        execution_state = msg["content"].get("execution_state")
                        if execution_state == "idle":
                            # Init code finished executing
                            init_complete = True
                            break
                    elif msg_type == "error":
                        # Init code had an error
                        error_content = msg["content"]
                        logger.error(f"Strategy initialization failed: {error_content.get('ename')}: {error_content.get('evalue')}")
                        raise RuntimeError(f"Strategy initialization failed: {error_content.get('ename')}")
            except asyncio.TimeoutError:
                continue

        if not init_complete:
            logger.warning("Strategy initialization did not complete within timeout")

        # Stop the client channels (we don't need them, clients will reconnect)
        kc.stop_channels()

        # Store kernel manager reference
        cls._active_kernels[str(conn_file)] = (km, config_file)

        return str(conn_file)

    @classmethod
    async def stop(cls, connection_file: str) -> None:
        """
        Stop a kernel identified by its connection file.

        Args:
            connection_file: Path to the kernel connection file
        """
        if connection_file not in cls._active_kernels:
            logger.warning(f"Kernel not found for connection file: {connection_file}")
            return

        km, _ = cls._active_kernels[connection_file]

        try:
            await km.shutdown_kernel(now=False)
            logger.info(f"Kernel stopped: {connection_file}")
        except Exception as e:
            logger.error(f"Error stopping kernel: {e}")
            # Force kill if graceful shutdown fails
            try:
                await km.shutdown_kernel(now=True)
            except Exception:
                pass

        # Remove from active kernels
        del cls._active_kernels[connection_file]

        # Clean up connection file
        try:
            Path(connection_file).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not delete connection file: {e}")

    @classmethod
    async def is_alive(cls, connection_file: str) -> bool:
        """
        Check if a kernel is still alive.

        Args:
            connection_file: Path to the kernel connection file

        Returns:
            True if kernel is alive, False otherwise
        """
        if connection_file not in cls._active_kernels:
            return False

        km, _ = cls._active_kernels[connection_file]

        try:
            return await km.is_alive()
        except Exception:
            return False

    @classmethod
    def list_active(cls) -> list[dict[str, str]]:
        """
        List all active kernel sessions by scanning the kernels directory.

        Returns:
            List of dicts with connection_file and config_file info
        """
        kernels_dir = Path.home() / ".qubx" / "kernels"
        if not kernels_dir.exists():
            return []

        result = []
        for conn_file in kernels_dir.glob("*.json"):
            # Parse the filename to extract config info
            # Format: {strategy_name}_{timestamp}.json
            strategy_name = conn_file.stem.rsplit("_", 2)[0] if "_" in conn_file.stem else conn_file.stem
            result.append({
                "connection_file": str(conn_file),
                "strategy_name": strategy_name,
                "timestamp": conn_file.stat().st_mtime,
            })

        # Sort by timestamp (newest first)
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        return result

    @classmethod
    async def cleanup_all(cls) -> None:
        """Stop all active kernels (cleanup on exit)."""
        for conn_file in list(cls._active_kernels.keys()):
            await cls.stop(conn_file)