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
from dataclasses import dataclass, field
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

    @property
    def label(self) -> str:
        """``reason@exchange`` when scoped, plain ``reason`` when context-wide."""
        return f"{self.reason}@{self.scope}" if self.scope else str(self.reason)


@dataclass(frozen=True, slots=True)
class QubxStatusInfo:
    status: QubxStatus
    degradations: tuple[Degradation, ...] = ()
    # - scopes held, precomputed once per transition so readers on the order path do not
    #   iterate; None is the context-wide scope and matches every exchange
    _scopes: frozenset = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_scopes", frozenset(d.scope for d in self.degradations))

    def is_degraded_for(self, exchange: str) -> bool:
        """True when a degradation affects this exchange: either context-wide (no scope)
        or scoped to it. Two set lookups, no iteration — this is read per order."""
        return None in self._scopes or exchange in self._scopes

    def degradations_for(self, exchange: str) -> tuple[Degradation, ...]:
        """The degradations behind :meth:`is_degraded_for` — for messages, not the hot path."""
        return tuple(d for d in self.degradations if d.scope is None or d.scope == exchange)


NORMAL_STATUS = QubxStatusInfo(QubxStatus.NORMAL)
"""Shared NORMAL snapshot: the default for any context that never degrades."""


class ContextStatus:
    """
    Current context status, writable from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # - keyed by (reason, scope): dict order is insertion order, so a re-asserted
        #   degradation keeps its place and the snapshot ordering stays stable
        self._held: dict[tuple[DegradeReason, str | None], Degradation] = {}
        self._info = NORMAL_STATUS

    @property
    def info(self) -> QubxStatusInfo:
        return self._info

    def add(self, reason: DegradeReason, since: dt_64, scope: str | None = None, message: str = "") -> None:
        """
        Record a degradation, or refresh the message of one already held for this
        (reason, scope). A refresh keeps the original ``since`` — a repeatedly
        re-asserted degradation must not look like it just started.
        """
        key = (reason, scope)
        with self._lock:
            previous = self._held.get(key)
            self._held[key] = Degradation(reason, previous.since if previous else since, scope, message)
            self._info = QubxStatusInfo(QubxStatus.DEGRADED, tuple(self._held.values()))

    def clear(self, reason: DegradeReason, scope: str | None = None) -> None:
        """
        Drop the degradation held for this (reason, scope); back to NORMAL once none
        are left. A no-op when none is held.
        """
        with self._lock:
            if self._held.pop((reason, scope), None) is None:
                return
            held = tuple(self._held.values())
            self._info = QubxStatusInfo(QubxStatus.DEGRADED if held else QubxStatus.NORMAL, held)
