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

    def __init__(self) -> None:
        self.km = None  # AsyncKernelManager
        self.kc = None  # AsyncKernelClient
        self.iopub_task: asyncio.Task | None = None
        self.callbacks: list[Callable[[str, Any], None]] = []

    async def start(self) -> None:
        """Start the Jupyter kernel and its channels."""
        self.km = AsyncKernelManager(kernel_name="python3")
        await self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        # Ensure kernel is ready
        await self.kc.wait_for_ready()
        self.iopub_task = asyncio.create_task(self._drain_iopub())

    async def stop(self) -> None:
        """Stop the kernel and its channels."""
        if self.kc:
            self.kc.stop_channels()
        if self.km:
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
