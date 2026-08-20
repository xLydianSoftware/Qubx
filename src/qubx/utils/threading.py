"""Bounded background-work primitives (spec: 2026-08-19-safe-state-persistence).

Absolute imports mean ``import threading`` below resolves to the stdlib even
though this module shares its name.
"""

import threading
import time
from collections import deque
from typing import Any, Callable

from qubx import logger


class BoundedWorker:
    """Single daemon worker thread over a drop-oldest bounded queue.

    ``submit`` never blocks and never raises: when the queue is full the OLDEST
    pending item is dropped (counted, warned at most once per ``warn_every_s``).
    One worker per instance is deliberate — it preserves FIFO ordering, which
    redis-stream exports rely on.
    """

    def __init__(self, name: str, maxlen: int, warn_every_s: float = 30.0) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._name = name
        self._maxlen = maxlen
        self._warn_every_s = warn_every_s
        self._queue: deque[tuple[Callable[..., Any], tuple, dict]] = deque()
        self._cond = threading.Condition()
        self._dropped = 0
        self._dropped_unreported = 0
        self._last_warn = 0.0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=f"BoundedWorker-{name}", daemon=True)
        self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def queued(self) -> int:
        with self._cond:
            return len(self._queue)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        with self._cond:
            if self._stopped:
                return
            if len(self._queue) >= self._maxlen:
                self._queue.popleft()
                self._dropped += 1
                self._dropped_unreported += 1
                now = time.monotonic()
                if now - self._last_warn >= self._warn_every_s:
                    logger.warning(
                        f"[BoundedWorker:{self._name}] queue full (maxlen={self._maxlen}) — "
                        f"dropped {self._dropped_unreported} oldest items since last report"
                    )
                    self._last_warn = now
                    self._dropped_unreported = 0
            self._queue.append((fn, args, kwargs))
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopped:
                    self._cond.wait()
                if not self._queue and self._stopped:
                    return
                fn, args, kwargs = self._queue.popleft()
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[BoundedWorker:{self._name}] task failed: {e}")

    def stop(self, flush_timeout_s: float = 5.0) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=flush_timeout_s)
