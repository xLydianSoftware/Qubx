"""
Jupyter kernel wrapper for Textual TUI integration.

Manages kernel lifecycle, code execution, and message processing.
"""

import asyncio
import traceback
from typing import Any, Callable

from jupyter_client.manager import AsyncKernelManager

from qubx import logger


class IPyKernel:
    """Wrapper around AsyncKernelManager for managing Jupyter kernel lifecycle."""

    def __init__(self, km: AsyncKernelManager | None = None, kc=None) -> None:
        self.km = km  # AsyncKernelManager
        self.kc = kc  # AsyncKernelClient
        self.iopub_task: asyncio.Task | None = None
        self.callbacks: list[Callable[[str, Any], None]] = []
        self.owns_kernel = False  # Track if we started the kernel or just connected
        self._is_connected = False  # Track if already connected

    async def start(self) -> None:
        """Start the Jupyter kernel and its channels."""
        if self._is_connected:
            return  # Already connected
        self.km = AsyncKernelManager(kernel_name="python3")
        await self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        # Ensure kernel is ready
        await self.kc.wait_for_ready()
        self.iopub_task = asyncio.create_task(self._drain_iopub())
        self.owns_kernel = True  # We started this kernel
        self._is_connected = True

    async def connect_to_existing(self, connection_file: str, start_iopub: bool = True) -> None:
        """
        Connect to an existing Jupyter kernel via its connection file.

        Args:
            connection_file: Path to the kernel connection file (.json)
            start_iopub: Whether to start the iopub draining task (default True)

        Raises:
            FileNotFoundError: If connection file doesn't exist
            RuntimeError: If kernel is not alive or connection fails
        """
        if self._is_connected:
            return  # Already connected

        from pathlib import Path

        conn_path = Path(connection_file)
        if not conn_path.exists():
            raise FileNotFoundError(f"Connection file not found: {connection_file}")

        self.km = AsyncKernelManager()
        self.km.load_connection_file(connection_file)
        self.kc = self.km.client()
        self.kc.start_channels()

        # Ensure kernel is ready with timeout
        try:
            await asyncio.wait_for(self.kc.wait_for_ready(), timeout=5.0)
        except asyncio.TimeoutError:
            self.kc.stop_channels()
            raise RuntimeError(
                f"Timeout connecting to kernel. The kernel may have died.\n"
                f"Connection file: {connection_file}\n"
                f"Try starting a new kernel or check if the kernel process is still running."
            )
        except Exception as e:
            self.kc.stop_channels()
            raise RuntimeError(f"Failed to connect to kernel: {e}")

        if start_iopub:
            self.iopub_task = asyncio.create_task(self._drain_iopub())
        self.owns_kernel = False  # We're connecting to someone else's kernel
        self._is_connected = True
        logger.info(f"Connected to existing kernel: {connection_file}")

    def is_connected(self) -> bool:
        """Check if kernel is already connected."""
        return self._is_connected

    def start_iopub_listener(self) -> None:
        """Start the iopub message draining task on the current event loop."""
        if self.iopub_task is None and self.kc is not None:
            self.iopub_task = asyncio.create_task(self._drain_iopub())

    async def stop(self) -> None:
        """Stop the kernel and its channels."""
        if self.kc:
            self.kc.stop_channels()
        # Only shutdown kernel if we own it (started it ourselves)
        if self.km and self.owns_kernel:
            await self.km.shutdown_kernel(now=False)
        if self.iopub_task:
            self.iopub_task.cancel()

    def register(self, cb: Callable[[str, Any], None]) -> None:
        """Register a callback to receive kernel events."""
        self.callbacks.append(cb)

    def execute(self, code: str, *, silent: bool = False) -> str:
        """Execute code in the kernel and return the message ID."""
        if not self.kc:
            raise RuntimeError("Kernel client not ready")
        msg_id = self.kc.execute(code, allow_stdin=False, silent=silent)
        return msg_id

    async def complete(self, code: str, cursor_pos: int) -> list[str]:
        """
        Request completions from the kernel.

        Args:
            code: The code context for completion
            cursor_pos: Cursor position in the code (in unicode codepoints)

        Returns:
            List of completion matches
        """
        if not self.kc:
            raise RuntimeError("Kernel client not ready")

        # Send completion request
        msg_id = self.kc.complete(code, cursor_pos)

        # Wait for completion reply on shell channel (up to 1 second total)
        timeout = 1.0
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return []

            try:
                # Wait for a message with remaining time
                remaining = timeout - elapsed
                msg = await asyncio.wait_for(self.kc.get_shell_msg(timeout=remaining), timeout=remaining)

                if msg["parent_header"].get("msg_id") == msg_id:
                    if msg["msg_type"] == "complete_reply":
                        content = msg["content"]
                        status = content.get("status")
                        matches = content.get("matches", [])
                        if status == "ok":
                            return matches
                        else:
                            return []
            except asyncio.TimeoutError:
                return []
            except Exception:
                # Handle Empty exception from jupyter_client
                await asyncio.sleep(0.01)  # Small delay before retry
                continue

    async def interrupt(self) -> None:
        """Send interrupt signal to the kernel."""
        if self.km:
            await self.km.interrupt_kernel()

    async def get_output_history(self) -> list[dict[str, Any]]:
        """
        Retrieve output history from the kernel.

        Returns:
            List of history entries with timestamp, type, and content
        """
        if not self.kc:
            return []

        try:
            # Request the history from kernel globals
            msg_id = self.kc.execute(
                "import json; json.dumps(globals().get('_qubx_output_history', []))",
                silent=True,
                store_history=False,
            )

            # Wait for the result
            timeout = 2.0
            start_time = asyncio.get_event_loop().time()

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    return []

                try:
                    remaining = timeout - elapsed
                    msg = await asyncio.wait_for(self.kc.get_iopub_msg(timeout=remaining), timeout=remaining)

                    if msg["parent_header"].get("msg_id") == msg_id:
                        if msg["msg_type"] == "execute_result":
                            content = msg["content"]
                            data = content.get("data", {})
                            history_json = data.get("text/plain", "[]")
                            # Strip quotes if present
                            history_json = history_json.strip("'\"")
                            import json

                            return json.loads(history_json)
                except asyncio.TimeoutError:
                    return []
                except Exception:
                    await asyncio.sleep(0.01)
                    continue
        except Exception as e:
            logger.warning(f"Could not retrieve output history: {e}")
            return []

    async def _drain_iopub(self):
        """Continuously drain iopub messages from the kernel."""
        try:
            while True:
                msg = await self.kc.get_iopub_msg()
                msg_type = msg["header"]["msg_type"]
                content = msg["content"]

                # Skip certain message types
                if msg_type in ("status", "comm_open", "comm_msg", "comm_close", "execute_input"):
                    continue

                # Handle different message types
                if msg_type == "stream":
                    self._emit("stream", {"name": content.get("name"), "text": content.get("text", "")})
                elif msg_type in ("display_data", "execute_result"):
                    data = content.get("data", {})
                    # Check for custom MIME type for dashboard
                    if "application/x-qubx-dashboard+json" in data:
                        self._emit("qubx_dashboard", data["application/x-qubx-dashboard+json"])
                    elif "text/markdown" in data:
                        self._emit("markdown", data["text/markdown"])
                    elif "text/plain" in data:
                        self._emit("text", data["text/plain"])
                    else:
                        self._emit("text", str(data))
                elif msg_type == "error":
                    self._emit(
                        "error",
                        {
                            "ename": content.get("ename"),
                            "evalue": content.get("evalue"),
                            "traceback": "\n".join(content.get("traceback", [])),
                        },
                    )
                elif msg_type == "clear_output":
                    self._emit("clear", {})
                else:
                    self._emit("debug", {"msg_type": msg_type, "content": content})
        except asyncio.CancelledError:
            return
        except Exception:
            self._emit("fatal", traceback.format_exc())

    def _emit(self, kind: str, payload: Any) -> None:
        """Emit an event to all registered callbacks."""
        for cb in list(self.callbacks):
            try:
                cb(kind, payload)
            except Exception:
                logger.exception("Callback failed")
