import asyncio
import contextlib
import json
import random
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import Any, Awaitable, Callable, Optional, cast

import orjson
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from qubx import logger


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class ReconnectionConfig:
    enabled: bool = True
    max_retries: int = 10
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.5  # add up to +/- 0.5s jitter


@dataclass
class ChannelSubscription:
    channel: str
    handler: Callable[[dict], Awaitable[None]]
    params: dict[str, Any]


class MessageRateTracker:
    """
    Time-bucketed message rate tracker.
    """

    def __init__(self, max_window_seconds: int = 900, bucket_size: float = 1.0):
        self.bucket_size = bucket_size
        self.max_window = max_window_seconds
        self._buckets: dict[int, int] = defaultdict(int)  # {bucket_id: count}
        self._bucket_times: deque[int] = deque()  # Ordered bucket IDs for cleanup
        self._total_messages = 0

    def record_message(self) -> None:
        bucket_id = int(monotonic() / self.bucket_size)
        if self._buckets[bucket_id] == 0:
            self._bucket_times.append(bucket_id)
        self._buckets[bucket_id] += 1
        self._total_messages += 1
        self._cleanup()

    def get_count(self, window_seconds: float) -> int:
        current_bucket = int(monotonic() / self.bucket_size)
        oldest_bucket = current_bucket - int(window_seconds / self.bucket_size)
        return sum(count for bucket_id, count in self._buckets.items() if bucket_id > oldest_bucket)

    def get_rate(self, window_seconds: float) -> float:
        count = self.get_count(window_seconds)
        return count / window_seconds if window_seconds > 0 else 0.0

    @property
    def total_messages(self) -> int:
        return self._total_messages

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    def _cleanup(self) -> None:
        cutoff = int(monotonic() / self.bucket_size) - int(self.max_window / self.bucket_size)
        while self._bucket_times and self._bucket_times[0] < cutoff:
            old_bucket = self._bucket_times.popleft()
            del self._buckets[old_bucket]


class BaseWebSocketManager:
    """
    Clean base WS manager with:
      - decoupled reader (recv-only) and worker(s) (parse+dispatch)
      - bounded inbox with drop-oldest on overflow (keeps latest)
      - optional app-level heartbeat hook
      - automatic reconnect + resubscribe
    """

    def __init__(
        self,
        url: str,
        *,
        reconnection: Optional[ReconnectionConfig] = None,
        # Disable protocol pings by default; subclasses may override or add app pings
        ping_interval: Optional[float] = None,
        ping_timeout: Optional[float] = None,
        max_size: Optional[int] = 16 * 1024 * 1024,
        # Let websockets keep reading; we enforce our own bounded inbox
        max_queue: Optional[int] = None,
        compression: Optional[str] = None,
        # Internal queue & workers
        inbox_size: int = 5000,
        workers: int = 1,
        # Optional app-level heartbeat
        app_ping_interval: Optional[float] = None,  # seconds; if None, no app pings
        no_rx_reconnect_after: Optional[float] = 60.0,  # if no inbound traffic for N seconds, reconnect
    ) -> None:
        self.url = url
        self.recon = reconnection or ReconnectionConfig()
        self.ws_opts = dict(
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            max_size=max_size,
            max_queue=max_queue,
            compression=compression,
        )
        self._state = ConnectionState.DISCONNECTED
        self._ws: Optional[ClientConnection] = None
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

        # subscriptions
        self._subs: dict[str, ChannelSubscription] = {}
        self._subs_lock = asyncio.Lock()

        # tasks & infra
        self._inbox: asyncio.Queue[bytes | str] = asyncio.Queue(maxsize=inbox_size)
        self._reader_task: Optional[asyncio.Task] = None
        self._worker_tasks: list[asyncio.Task] = []
        self._workers_n = max(1, workers)
        self._hb_task: Optional[asyncio.Task] = None
        self._lag_task: Optional[asyncio.Task] = None
        self._on_reconnected: list[Callable[[], Awaitable[None]]] = []

        # metrics
        self._total_messages = 0
        self._last_rx = monotonic()

        # heartbeat config
        self._app_ping_interval = app_ping_interval
        self._no_rx_reconnect_after = no_rx_reconnect_after

        # message rate tracking
        self._message_tracker = MessageRateTracker(max_window_seconds=900, bucket_size=1.0)

        self.log = logger

    # ---------- public props ----------
    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED and self._ws is not None

    @property
    def subscriptions(self) -> dict[str, ChannelSubscription]:
        """Read-only access to current subscriptions"""
        return self._subs.copy()

    @property
    def total_messages(self) -> int:
        return self._total_messages

    def get_message_count(self, window_seconds: float) -> int:
        return self._message_tracker.get_count(window_seconds)

    def get_message_rate(self, window_seconds: float) -> float:
        return self._message_tracker.get_rate(window_seconds)

    def on_reconnected(self, cb: Callable[[], Awaitable[None]]) -> None:
        self._on_reconnected.append(cb)

    # ---------- lifecycle ----------
    async def connect(self) -> None:
        async with self._lock:
            if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
                return
            self._state = ConnectionState.CONNECTING
            self._stop_event.clear()
            try:
                self._ws = await websockets.connect(self.url, **self.ws_opts)  # type: ignore
                self._state = ConnectionState.CONNECTED
                self._start_tasks()
                self.log.info(f"WS connected: {self.url}")
            except Exception as e:
                self._state = ConnectionState.DISCONNECTED
                self.log.error(f"WS connect failed: {e}")
                raise

    async def disconnect(self) -> None:
        async with self._lock:
            if self._state in (ConnectionState.DISCONNECTED, ConnectionState.CLOSED):
                return
            self._state = ConnectionState.CLOSING
            self._stop_event.set()
            await self._stop_tasks()
            if self._ws:
                with contextlib.suppress(Exception):
                    await self._ws.close()
                self._ws = None
            self._state = ConnectionState.CLOSED
            self.log.info("WS disconnected")

    # ---------- subscriptions ----------
    async def subscribe(self, channel: str, handler: Callable[[dict], Awaitable[None]], **params) -> None:
        if not self.is_connected:
            raise ConnectionError("Not connected")
        async with self._subs_lock:
            self._subs[channel] = ChannelSubscription(channel, handler, params)
        await self._send_subscription_message(channel, params)

    async def unsubscribe(self, channel: str) -> None:
        async with self._subs_lock:
            if channel not in self._subs:
                return
            await self._send_unsubscription_message(channel)
            del self._subs[channel]

    # ---------- messaging ----------
    async def send(self, message: dict) -> None:
        if not self._ws:
            raise ConnectionError("Not connected")
        await self._ws.send(json.dumps(message))

    async def wait_until_closed(self) -> None:
        await self._stop_event.wait()

    # ---------- internals ----------
    def _start_tasks(self) -> None:
        # reader
        self._reader_task = asyncio.create_task(self._reader_loop(), name="ws-reader")
        # workers
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(), name=f"ws-worker-{i}") for i in range(self._workers_n)
        ]
        # heartbeat
        if self._app_ping_interval:
            self._hb_task = asyncio.create_task(self._heartbeat_loop(), name="ws-heartbeat")
        # loop lag
        self._lag_task = asyncio.create_task(self._loop_lag_monitor(), name="ws-looplag")

    async def _stop_tasks(self) -> None:
        logger.debug("Canceling tasks...")
        victims = [self._reader_task, self._hb_task, self._lag_task, *self._worker_tasks]
        # Filter out None values
        valid_tasks = [t for t in victims if t is not None]

        for t in valid_tasks:
            if not t.done():
                t.cancel()

        logger.debug("Waiting for tasks to complete...")
        if valid_tasks:
            results = await asyncio.gather(*valid_tasks, return_exceptions=True)
            for t, r in zip(valid_tasks, results):
                if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                    self.log.debug(f"task {t.get_name() if hasattr(t, 'get_name') else t} finished with {r}")

        logger.debug("Tasks canceled and completed")
        self._reader_task = None
        self._hb_task = None
        self._lag_task = None
        self._worker_tasks = []

    async def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self._ws:
                    await asyncio.sleep(0.1)
                    continue
                raw = await self._ws.recv()
                self._last_rx = monotonic()
                try:
                    self._inbox.put_nowait(raw)
                except asyncio.QueueFull:
                    # Drop oldest to keep the stream fresh
                    with contextlib.suppress(Exception):
                        _ = self._inbox.get_nowait()
                        self._inbox.task_done()
                    await self._inbox.put(raw)
            except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed) as e:
                self.log.warning(f"WS closed: code={getattr(e, 'code', None)} reason={getattr(e, 'reason', None)}")
                if not self._stop_event.is_set():
                    asyncio.create_task(self._handle_reconnection())
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error(f"Reader error: {e}")
                if not self._stop_event.is_set():
                    asyncio.create_task(self._handle_reconnection())
                break

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = await self._inbox.get()
                self._message_tracker.record_message()

                try:
                    if isinstance(raw, bytes):
                        msg = orjson.loads(raw)
                    else:
                        msg = orjson.loads(raw.encode("utf-8"))
                except orjson.JSONDecodeError as e:
                    self.log.error(f"JSON parse error: {e}")
                    continue

                self._total_messages += 1
                channel = self._extract_channel(msg)
                handled = False
                if channel:
                    sub = self._subs.get(channel)
                    if sub:
                        try:
                            await sub.handler(msg)
                            handled = True
                        except Exception as e:
                            self.log.error(f"Handler error for {channel}: {e}")
                if not handled:
                    await self._handle_unknown_message(msg)
            except Exception as e:
                tb = traceback.format_exc()
                self.log.warning(f"Worker error: {e}\n{tb}")
            finally:
                with contextlib.suppress(Exception):
                    self._inbox.task_done()

    async def _handle_reconnection(self) -> None:
        self.log.debug("Handling reconnection")
        if not self.recon.enabled:
            logger.debug("Reconnection disabled, stopping")
            self._stop_event.set()
            self._state = ConnectionState.DISCONNECTED
            return

        # stop tasks and close old connection
        try:
            logger.debug("Stopping tasks")
            await self._stop_tasks()
        except Exception as e:
            self.log.error(f"Error stopping tasks: {e}")

        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        self.log.debug("Stopped tasks and closed old connection")

        # backoff
        for attempt in range(1, self.recon.max_retries + 1):
            self._state = ConnectionState.RECONNECTING
            delay = min(self.recon.initial_delay * (self.recon.exponential_base ** (attempt - 1)), self.recon.max_delay)
            delay += random.uniform(-self.recon.jitter, self.recon.jitter)
            delay = max(0.0, delay)
            self.log.info(f"Reconnecting in {delay:.2f}s (attempt {attempt}/{self.recon.max_retries})")
            await asyncio.sleep(delay)
            try:
                self._ws = await websockets.connect(self.url, **self.ws_opts)  # type: ignore
                self._state = ConnectionState.CONNECTED
                self._start_tasks()
                await self._resubscribe_all()
                for cb in self._on_reconnected:
                    with contextlib.suppress(Exception):
                        await cb()
                self.log.info("Reconnected")
                return
            except Exception as e:
                self.log.warning(f"Reconnect attempt {attempt} failed: {e}")
                continue

        self.log.error("Max reconnect attempts reached; giving up.")
        self._stop_event.set()

    async def _resubscribe_all(self) -> None:
        async with self._subs_lock:
            items = list(self._subs.items())
        for channel, sub in items:
            with contextlib.suppress(Exception):
                await self._send_subscription_message(channel, sub.params)
            await asyncio.sleep(0.05)  # gentle throttle

    # ----- hooks for subclasses -----
    async def _send_subscription_message(self, channel: str, params: dict[str, Any]) -> None:
        # Default: {"type":"subscribe","channel":channel,...params}
        await self.send({"type": "subscribe", "channel": channel, **params})

    async def _send_unsubscription_message(self, channel: str) -> None:
        await self.send({"type": "unsubscribe", "channel": channel})

    def _extract_channel(self, message: dict) -> Optional[str]:
        try:
            channel = message.get("channel", None)
            return channel
        except Exception:
            return None

    async def _handle_unknown_message(self, message: dict) -> None:
        # Override for heartbeats/system messages
        pass

    # ----- optional app heartbeat in base -----
    def _app_ping_payload(self) -> Optional[dict]:
        # Subclasses can return e.g. {"type":"ping"}
        return None

    async def _heartbeat_loop(self) -> None:
        interval = float(cast(float, self._app_ping_interval))
        no_rx = float(cast(float, self._no_rx_reconnect_after)) if self._no_rx_reconnect_after else None
        while not self._stop_event.is_set():
            try:
                if self._ws and interval:
                    payload = self._app_ping_payload()
                    if payload:
                        await self.send(payload)
                if no_rx and (monotonic() - self._last_rx) > no_rx:
                    self.log.warning(f"No inbound traffic for {no_rx:.1f}s -> reconnect")
                    asyncio.create_task(self._handle_reconnection())
                    return
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.log.warning(f"heartbeat error: {e}")
            await asyncio.sleep(interval or 1.0)

    async def _loop_lag_monitor(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            t = loop.time()
            await asyncio.sleep(1)
            lag_ms = max(0.0, (loop.time() - t - 1.0) * 1000.0)
            if lag_ms > 1000.0:  # 1 second
                self.log.debug(f"Event-loop lag: {lag_ms:.1f} ms")
