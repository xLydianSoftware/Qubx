"""Async wrapper making any IStatePersistence backend safe for the event loop.

Contract (spec 2026-08-19-safe-state-persistence, D2): all writes are buffered
per-key (latest wins) and flushed by ONE background writer thread; ``load``
consults the pending buffer first (read-your-writes); no caller ever blocks on
the network and network errors never raise from ``save``/``delete``.
"""

import json
import threading
import time
from typing import Any, Callable

from qubx import logger
from qubx.core.interfaces import IStatePersistence

_TOMBSTONE = object()


class SafeStatePersistence(IStatePersistence):
    def __init__(
        self,
        backend: IStatePersistence,
        *,
        staleness_threshold_s: float = 60.0,
        flush_timeout_s: float = 5.0,
        retry_backoff_s: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0),
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._backend = backend
        self.staleness_threshold_s = staleness_threshold_s
        self._flush_timeout_s = flush_timeout_s
        self._retry_backoff_s = retry_backoff_s
        self._sleep = sleep_fn

        self._pending: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._stopped = False
        self._last_success: float | None = None
        self._consecutive_failures = 0
        self._last_fail_log = 0.0
        self._writer = threading.Thread(target=self._run, name="StatePersistenceWriter", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------------ writes
    def save(self, key: str, value: Any) -> None:
        json.dumps(value)  # - eager dry-run: programming errors (TypeError/ValueError) stay loud at the call site
        with self._cond:
            self._pending[key] = value
            self._cond.notify()

    def delete(self, key: str) -> bool:
        with self._cond:
            self._pending[key] = _TOMBSTONE
            self._cond.notify()
        return True  # optimistic: actual backend result is async

    # ------------------------------------------------------------------- reads
    def load(self, key: str, default: Any = None) -> Any:
        with self._cond:
            if key in self._pending:
                v = self._pending[key]
                return default if v is _TOMBSTONE else v
        return self._backend.load(key, default)

    def exists(self, key: str) -> bool:
        with self._cond:
            if key in self._pending:
                return self._pending[key] is not _TOMBSTONE
        return self._backend.exists(key)

    # ------------------------------------------------------------------ health
    def last_success_age(self) -> float | None:
        return None if self._last_success is None else time.monotonic() - self._last_success

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    # ------------------------------------------------------------------ writer
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopped:
                    self._cond.wait()
                if not self._pending and self._stopped:
                    return
                batch, self._pending = self._pending, {}

            failed: dict[str, Any] = {}
            for key, value in batch.items():
                try:
                    if value is _TOMBSTONE:
                        self._backend.delete(key)
                    else:
                        self._backend.save(key, value)
                    self._last_success = time.monotonic()
                    self._consecutive_failures = 0
                except (TypeError, ValueError):
                    logger.error(f"[SafeStatePersistence] unserializable value for '{key}' reached writer; dropped")
                except Exception as e:
                    failed[key] = value
                    self._consecutive_failures += 1
                    now = time.monotonic()
                    if now - self._last_fail_log >= 30.0:
                        logger.warning(
                            f"[SafeStatePersistence] backend write failed ({self._consecutive_failures} in a row): {e}"
                        )
                        self._last_fail_log = now

            if failed:
                with self._cond:
                    for key, value in failed.items():
                        self._pending.setdefault(key, value)  # - newer pending values win over the failed batch
                    if self._stopped:
                        return  # - do not backoff-sleep during shutdown
                backoff = self._retry_backoff_s[min(self._consecutive_failures, len(self._retry_backoff_s)) - 1]
                self._sleep(backoff)

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._writer.join(timeout=self._flush_timeout_s)
