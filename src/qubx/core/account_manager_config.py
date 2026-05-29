from dataclasses import dataclass


@dataclass
class AccountManagerConfig:
    inflight_check_interval_ms: int = 2_000
    inflight_check_threshold_ms: int = 5_000
    inflight_check_retries: int = 5

    snapshot_check_interval_ms: int = 30_000
    snapshot_check_threshold_ms: int = 5_000

    liveness_check_interval_ms: int = 5_000
    liveness_check_threshold_ms: int = 30_000

    terminal_order_retention_ms: int = 30_000
    terminal_order_history_size: int = 10_000


def _ms_to_cron(interval_ms: int) -> str:
    """Translate ms into the 6-field 'seconds-at-end' croniter format.

    Single source of truth for the cron-position convention so other
    modules never hand-write 6-field cron strings.
    """
    if interval_ms % 1000 != 0:
        raise ValueError(
            f"pm.schedule resolution is 1s; {interval_ms}ms not representable"
        )
    s = interval_ms // 1000
    if s < 60:
        return f"* * * * * */{s}"
    if s % 60 == 0 and s < 3600:
        return f"*/{s // 60} * * * *"
    raise ValueError(
        f"interval {s}s not expressible as cron; "
        f"use a sub-minute or whole-minute value"
    )
