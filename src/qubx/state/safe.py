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
from qubx.core.exceptions import StatePersistenceUnavailable
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
        # keys currently being flushed by the writer (dequeued from _pending, not yet resolved) -
        # consulted by load()/exists() so read-your-writes holds for the whole batch-write duration,
        # not just while a key sits in _pending (fixes the in-flight read gap, spec review F2).
        self._inflight: dict[str, Any] = {}
        self._cond = threading.Condition()
        self._stopped = False
        self._stop_event = threading.Event()  # lets stop() interrupt an in-progress backoff wait (fixes F3)
        self._warned_after_stop = False
        self._last_success: float | None = None
        self._consecutive_failures = 0
        self._last_fail_log = 0.0
        self._writer = threading.Thread(target=self._run, name="StatePersistenceWriter", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------------ writes
    def save(self, key: str, value: Any) -> None:
        json.dumps(value)  # - eager dry-run: programming errors (TypeError/ValueError) stay loud at the call site
        with self._cond:
            if self._stopped:
                self._warn_write_after_stop()
                return
            self._pending[key] = value
            self._cond.notify()

    def delete(self, key: str) -> bool:
        with self._cond:
            if self._stopped:
                self._warn_write_after_stop()
                return True  # optimistic per contract; nothing will actually be persisted
            self._pending[key] = _TOMBSTONE
            self._cond.notify()
        return True  # optimistic: actual backend result is async

    def _warn_write_after_stop(self) -> None:
        # called under self._cond; log once so a busy shutdown path can't spam the log
        if not self._warned_after_stop:
            logger.warning("[SafeStatePersistence] save after stop; value will not be persisted")
            self._warned_after_stop = True

    # ------------------------------------------------------------------- reads
    def load(self, key: str, default: Any = None) -> Any:
        with self._cond:
            if key in self._pending:
                v = self._pending[key]
                return default if v is _TOMBSTONE else v
            if key in self._inflight:
                v = self._inflight[key]
                return default if v is _TOMBSTONE else v
        return self._backend.load(key, default)

    def exists(self, key: str) -> bool:
        with self._cond:
            if key in self._pending:
                return self._pending[key] is not _TOMBSTONE
            if key in self._inflight:
                return self._inflight[key] is not _TOMBSTONE
        return self._backend.exists(key)

    # ------------------------------------------------------------------ health
    def last_success_age(self) -> float | None:
        return None if self._last_success is None else time.monotonic() - self._last_success

    @property
    def consecutive_failures(self) -> int:
        """Consecutive *rounds* (batch attempts) with at least one failing key, not failing keys."""
        return self._consecutive_failures

    def validate_startup(
        self,
        deadline_s: float = 60.0,
        probe_backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 10.0),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Probe the backend with backoff for up to ``deadline_s``; raise on exhaustion (spec D3)."""
        start = clock()
        attempt = 0
        last_err: Exception | None = None
        while True:
            try:
                self._backend.exists("__qubx_probe__")
                # - the probe proves backend liveness: seed last-success now so a backend that
                #   dies after startup but before the first write is still caught by staleness
                #   monitoring instead of leaving last_success_age() = None forever
                self._last_success = time.monotonic()
                logger.info(f"[SafeStatePersistence] backend validated after {attempt + 1} attempt(s)")
                return
            except Exception as e:
                last_err = e
                elapsed = clock() - start
                if elapsed >= deadline_s:
                    raise StatePersistenceUnavailable(
                        f"state persistence unreachable after {elapsed:.0f}s ({attempt + 1} attempts): {last_err}"
                    ) from last_err
                backoff = probe_backoff_s[min(attempt, len(probe_backoff_s) - 1)]
                logger.warning(
                    f"[SafeStatePersistence] startup probe failed (attempt {attempt + 1}): {e}; retrying in {backoff}s"
                )
                self._sleep(backoff)
                attempt += 1

    # ------------------------------------------------------------------ writer
    def _attempt_batch(self, batch: dict[str, Any]) -> tuple[dict[str, Any], bool, Exception | None]:
        """Flush one batch to the backend. Returns (still-failed subset, round had any failure, last error)."""
        failed: dict[str, Any] = {}
        round_failed = False
        last_error: Exception | None = None
        for key, value in batch.items():
            try:
                if value is _TOMBSTONE:
                    self._backend.delete(key)
                else:
                    self._backend.save(key, value)
                self._last_success = time.monotonic()
            except (TypeError, ValueError):
                logger.error(f"[SafeStatePersistence] unserializable value for '{key}' reached writer; dropped")
            except Exception as e:
                failed[key] = value
                round_failed = True
                last_error = e
        return failed, round_failed, last_error

    def _wait_backoff(self, backoff: float) -> None:
        if self._sleep is time.sleep:
            # default sleep: wait on the event so stop() can interrupt a long backoff immediately
            self._stop_event.wait(backoff)
        else:
            # an injected sleep_fn (tests) is honored as given; _stopped is re-checked right after
            self._sleep(backoff)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stopped:
                    self._cond.wait()
                if not self._pending and self._stopped:
                    return
                batch, self._pending = self._pending, {}
                self._inflight = batch

            failed, round_failed, last_error = self._attempt_batch(batch)

            with self._cond:
                # merge failed keys back into _pending BEFORE clearing _inflight so a key is never
                # visible in neither map (read-your-writes gap, spec review F2)
                for key, value in failed.items():
                    self._pending.setdefault(key, value)  # - newer pending values win over the failed batch
                self._inflight = {}
                is_stopped = self._stopped

            # count consecutive FAILING ROUNDS, not failing keys (spec review F1: a batch with one
            # failing key and one succeeding key must not reset to 0 and then backoff-index to -1)
            self._consecutive_failures = self._consecutive_failures + 1 if round_failed else 0

            if not failed:
                continue

            if is_stopped:
                # bounded flush: this attempt (or the one that follows an interrupted backoff wait)
                # is the single final drain attempt - no more retries, no more backend calls after this.
                logger.warning(
                    f"[SafeStatePersistence] stop(): abandoning {len(failed)} unflushed key(s) after "
                    f"final drain attempt: {sorted(failed)} (last error: {last_error})"
                )
                return

            now = time.monotonic()
            if now - self._last_fail_log >= 30.0:
                logger.warning(
                    f"[SafeStatePersistence] backend write failed for {len(failed)} key(s) "
                    f"({self._consecutive_failures} consecutive failing round(s)): {last_error}"
                )
                self._last_fail_log = now

            backoff = self._retry_backoff_s[min(self._consecutive_failures - 1, len(self._retry_backoff_s) - 1)]
            self._wait_backoff(backoff)

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._stop_event.set()
        self._writer.join(timeout=self._flush_timeout_s)
        if self._writer.is_alive():
            with self._cond:
                n_unflushed = len(self._pending) + len(self._inflight)
            logger.warning(
                f"[SafeStatePersistence] stop(): flush timed out after {self._flush_timeout_s}s with "
                f"{n_unflushed} key(s) still unflushed"
            )
