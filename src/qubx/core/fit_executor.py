"""Threaded on_fit execution.

Live-only machinery for running ``strategy.on_fit`` on a dedicated worker thread
("StrategyFitThread") so the ProcessorThread keeps draining the channel (order/account
applies, market-cache updates, AM ticks, schedules) while a slow fit computes — the
2026-07 incident was a 7-minute on_fit freezing all event processing.

Single-mutator invariant: the fit thread only *computes and records*. Intercepted ctx
calls (set_universe / add_instruments / remove_instruments / subscribe / schedule /
set_fit_schedule) are recorded on the :class:`FitCycleState`; ``emit_signal`` is buffered
there too. When the fit body returns, a single :class:`FitCommitData` is posted on the
CtrlChannel and the ProcessorThread applies everything atomically — every ctx mutation
still happens on the ProcessorThread.

Simulation never constructs a :class:`SingleThreadWorker` and never calls
:meth:`FitCycleState.begin`, so backtests keep today's inline path exactly.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from threading import Thread

from qubx import logger
from qubx.core.basics import Signal

# Channel d_type under which a FitCommitData rides the CtrlChannel as a
# (None, FIT_COMMIT_EVENT, FitCommitData, False) tuple; dispatched to
# ProcessingManager._handle_fit_commit via the auto-registered handler map.
FIT_COMMIT_EVENT = "fit_commit"


@dataclass(frozen=True, slots=True)
class FitCommitData:
    """Outcome of one threaded fit, applied atomically by the ProcessorThread.

    ``ops`` are zero-arg callables replaying the intercepted ctx mutations in call
    order (they run on the ProcessorThread, where the fit-thread interception no
    longer triggers, so each takes today's normal path). ``signals`` are the
    fit-emitted signals, drained into the normal pipeline in emission order.
    """

    ops: tuple[Callable[[], None], ...] = ()
    signals: tuple[Signal, ...] = ()
    duration_s: float = 0.0


class FitCycleState:
    """Shared token for the currently-running threaded fit cycle.

    Managers (universe / subscription / processing) and the context consult
    :meth:`is_fit_thread` to decide between today's direct path and the
    record-for-commit path. Inert unless :meth:`begin` was called from the fit
    worker — in simulation and inline mode the check is a single ``None`` test.
    """

    __slots__ = ("_lock", "_thread_ident", "_ops", "_signals")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread_ident: int | None = None
        self._ops: list[Callable[[], None]] = []
        self._signals: list[Signal] = []

    def begin(self, thread_ident: int) -> None:
        """Arm the cycle: called by the fit worker at the start of the fit body."""
        with self._lock:
            self._thread_ident = thread_ident
            self._ops.clear()
            self._signals.clear()

    def end(self) -> tuple[tuple[Callable[[], None], ...], tuple[Signal, ...]]:
        """Disarm and capture everything recorded during the cycle."""
        with self._lock:
            ops, signals = tuple(self._ops), tuple(self._signals)
            self._thread_ident = None
            self._ops.clear()
            self._signals.clear()
            return ops, signals

    def is_fit_thread(self) -> bool:
        """True only when called from the fit worker while a threaded fit is in flight."""
        ident = self._thread_ident
        return ident is not None and ident == threading.get_ident()

    def record(self, op: Callable[[], None]) -> None:
        """Record a deferred ctx mutation for ProcessorThread replay at the commit."""
        with self._lock:
            self._ops.append(op)

    def buffer_signals(self, signals: list[Signal]) -> None:
        """Buffer fit-emitted signals; drained into the normal pipeline at the commit."""
        with self._lock:
            self._signals.extend(signals)


class SingleThreadWorker:
    """Single-thread FIFO background worker (daemon thread + queue), live-only.

    Two owners: the processing layer's "StrategyFitThread" (threaded on_fit bodies) and
    the SubscriptionManager's "WarmupThread" (deferred warmup fetches). A plain daemon
    thread + queue rather than a ThreadPoolExecutor: pool threads are non-daemon and
    joined at interpreter exit, so a wedged task would hang shutdown. Never constructed
    in simulation.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._queue: Queue[Callable[[], None] | None] = Queue()
        self._thread = Thread(target=self._loop, daemon=True, name=name)
        self._started = False

    def submit(self, fn: Callable[[], None]) -> None:
        """Enqueue a task. Called only from the ProcessorThread (lazy thread start is
        not thread-safe and task order must follow submission order)."""
        if not self._started:
            self._thread.start()
            self._started = True
        self._queue.put(fn)

    def stop(self) -> None:
        self._queue.put(None)

    def _loop(self) -> None:
        while True:
            fn = self._queue.get()
            if fn is None:
                return
            try:
                fn()
            except Exception:
                # Tasks carry their own error handling (the fit body posts its commit
                # from a finally; the warmup task posts its swap likewise) — anything
                # reaching here is a framework bug: log loudly, keep serving.
                logger.exception(f"[{self._name}] :: background task raised outside its own error handling")
