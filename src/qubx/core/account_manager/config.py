from dataclasses import dataclass

import pandas as pd

from qubx.utils.time import timedelta_to_crontab


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
    """Adapt a millisecond interval to a pm.schedule cron string via the shared
    qubx.utils.time.timedelta_to_crontab (no hand-rolled cron building here)."""
    if interval_ms <= 0:
        raise ValueError(f"interval must be positive; got {interval_ms}ms")
    return timedelta_to_crontab(pd.Timedelta(interval_ms, "ms"))
