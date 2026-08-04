"""
Context status — whether the framework is able to trade normally, and if not, why.

The status is owned by the StrategyContext and read through ``ctx.status``. Writers
(the health monitor, connectors) run on their own threads and call
:meth:`ContextStatus.add` / :meth:`ContextStatus.clear`; readers get an immutable
:class:`QubxStatusInfo` snapshot, so a read is a single attribute load and never
blocks a write.

Degradations accumulate: a queue backlog and an exchange in maintenance can hold at
the same time and clear independently. ``scope`` names the exchange a degradation
applies to; ``None`` means the whole context.
"""

import threading
from dataclasses import dataclass, replace
from enum import StrEnum

from qubx.core.basics import dt_64


class QubxStatus(StrEnum):
    NORMAL = "normal"
    DEGRADED = "degraded"


class DegradeReason(StrEnum):
    INTERNAL_QUEUE_OVERFLOW = "internal_queue_overflow"
    EXCHANGE_MAINTENANCE = "exchange_maintenance"


@dataclass(frozen=True, slots=True)
class Degradation:
    reason: DegradeReason
    since: dt_64
    scope: str | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class QubxStatusInfo:
    status: QubxStatus
    degradations: tuple[Degradation, ...] = ()


_NORMAL = QubxStatusInfo(QubxStatus.NORMAL)


class ContextStatus:
    """
    Current context status, writable from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._info = _NORMAL

    @property
    def info(self) -> QubxStatusInfo:
        return self._info

    def add(self, reason: DegradeReason, since: dt_64, scope: str | None = None, message: str = "") -> None:
        """
        Record a degradation, or refresh the message of one already held for this
        (reason, scope). A refresh keeps the original ``since`` — a repeatedly
        re-asserted degradation must not look like it just started.
        """
        with self._lock:
            held = list(self._info.degradations)
            fresh = Degradation(reason, since, scope, message)
            for i, d in enumerate(held):
                if d.reason is reason and d.scope == scope:
                    held[i] = replace(fresh, since=d.since)
                    break
            else:
                held.append(fresh)
            self._info = QubxStatusInfo(QubxStatus.DEGRADED, tuple(held))

    def clear(self, reason: DegradeReason, scope: str | None = None) -> None:
        """
        Drop the degradation held for this (reason, scope); back to NORMAL once none
        are left. A no-op when none is held.
        """
        with self._lock:
            kept = tuple(d for d in self._info.degradations if not (d.reason is reason and d.scope == scope))
            if len(kept) == len(self._info.degradations):
                return
            self._info = QubxStatusInfo(QubxStatus.DEGRADED if kept else QubxStatus.NORMAL, kept)
