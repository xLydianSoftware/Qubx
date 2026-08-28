from enum import IntEnum

from qubx import logger
from qubx.core.basics import dt_64, td_64


class BootPhase(IntEnum):
    BLOCKED = -1
    WAIT_READY = 0
    ON_START = 1
    RESOLVE = 2
    RESTORE = 3
    WARMUP_FINISHED = 4
    BOOT_FIT = 5
    TRADING = 6


class BootStateMachine:
    """Live-boot phase bookkeeping: current phase, boot-fit retry policy, health gauges.

    Pure state — the ProcessingManager drives transitions and executes side effects
    (see _advance_boot). BLOCKED is sticky for trading; a later successful fit releases
    it (record_fit_success), a late account snapshot releases WAIT_READY (caller-side).
    """

    def __init__(self, health_monitor, *, fit_max_attempts: int = 3, fit_retry_delay: td_64 = td_64(60, "s")) -> None:
        self.phase = BootPhase.WAIT_READY
        self.blocked_reason: str | None = None
        self.fit_attempts = 0
        self._health = health_monitor
        self._fit_max_attempts = fit_max_attempts
        self._fit_retry_delay = fit_retry_delay
        self._next_fit_attempt: dt_64 | None = None
        self._sync_alerted = False

    @property
    def is_trading(self) -> bool:
        return self.phase == BootPhase.TRADING

    @property
    def is_blocked(self) -> bool:
        return self.phase == BootPhase.BLOCKED

    def advance(self, phase: BootPhase) -> None:
        if phase == self.phase:
            return
        logger.info(f"<yellow>boot</yellow> :: {self.phase.name} -> <green>{phase.name}</green>")
        self.phase = phase
        self._gauge("boot.state", float(phase))

    def account_sync_alert(self) -> None:
        if self._sync_alerted:
            return
        self._sync_alerted = True
        self._gauge("boot.account_sync_blocked", 1.0)
        logger.warning("<yellow>Boot blocked: initial account snapshot not applied — holding until synced</yellow>")

    def account_synced(self) -> None:
        if not self._sync_alerted:
            return
        self._sync_alerted = False
        self._gauge("boot.account_sync_blocked", 0.0)

    def fit_attempt_allowed(self, now: dt_64) -> bool:
        if self.phase != BootPhase.BOOT_FIT:
            return False
        return self._next_fit_attempt is None or now >= self._next_fit_attempt

    def record_fit_attempt(self) -> None:
        self.fit_attempts += 1
        self._gauge("boot.fit_attempts", float(self.fit_attempts))

    def record_fit_failure(self, now: dt_64) -> None:
        if self.is_blocked:
            return
        if self.fit_attempts >= self._fit_max_attempts:
            self.blocked_reason = "boot fit failed"
            self.advance(BootPhase.BLOCKED)
            self._gauge("boot.fit_failed", 1.0)
            logger.error(
                f"<red>Boot fit failed after {self.fit_attempts} attempts — trading blocked, "
                "book unreconciled; a later successful fit will unblock</red>"
            )
        else:
            self._next_fit_attempt = now + self._fit_retry_delay
            logger.warning(
                f"<yellow>Boot fit failed (attempt {self.fit_attempts}/{self._fit_max_attempts}) — retrying</yellow>"
            )

    def record_fit_success(self) -> None:
        if self.is_blocked:
            self._gauge("boot.fit_failed", 0.0)
            self.blocked_reason = None
        self.advance(BootPhase.TRADING)

    def record_warmup_finished_failure(self) -> None:
        self._gauge("boot.warmup_finished_failed", 1.0)

    def _gauge(self, name: str, value: float) -> None:
        try:
            self._health.record_gauge(name, value)
        except Exception:
            logger.exception(f"failed to record {name} gauge")
