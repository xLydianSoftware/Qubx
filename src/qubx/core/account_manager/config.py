from dataclasses import dataclass


@dataclass
class AccountManagerConfig:
    reconcile_tick_interval_ms: int = 2_000  # - single reconcile heartbeat cadence
    snapshot_interval_ms: int = 30_000  # - snapshot due-timer (Reconciler requests one when due)
    snapshot_grace_ms: int = 5_000  # - grace before an order missing from a snapshot counts as drift

    missing_order_wait_ms: int = 5_000  # - ResolveMissingOrder: wait before fetching status
    missing_order_retries: int = 5  # - ResolveMissingOrder: status fetches before LOST

    order_confirm_wait_ms: int = 5_000  # - AwaitOrderConfirm: wait before fetching a sent order's status
    order_confirm_retries: int = 5  # - AwaitOrderConfirm: status fetches before LOST

    position_confirm_wait_ms: int = 2_000  # - ConfirmPositionBySnapshot: coverage window before hist-deals

    liveness_check_interval_ms: int = 5_000
    liveness_check_threshold_ms: int = 30_000

    terminal_order_retention_ms: int = 30_000
    terminal_order_history_size: int = 10_000
