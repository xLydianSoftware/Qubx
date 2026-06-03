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
