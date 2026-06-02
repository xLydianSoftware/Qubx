"""
``AccountState`` — the pure-data account snapshot the ``AccountManager`` owns and mutates.

``AccountState`` holds no behaviour; all transitions go through ``AccountManager`` so the
state machine stays the single source of truth. It is mutated only on the dispatch thread
(see ``ACCOUNT_MANAGEMENT_PLAN.md`` — single-writer invariant).

Scope note (Q1): orders are modelled in full (the bug-magnet); positions/balances reuse the
existing ``qubx.core.basics`` ``Position`` / ``AssetBalance`` and their fill maths, wired in a
later PR — they are intentionally not duplicated here.
"""

from dataclasses import dataclass, field

from qubx.core.account_manager.state_machine import OrderState
from qubx.core.basics import Instrument, OrderSide, OrderType, dt_64


@dataclass
class ManagedOrder:
    """An order plus the state-machine bookkeeping the AccountManager needs."""

    client_id: str  # cid — stable, framework-generated, never mutated
    instrument: Instrument
    side: OrderSide
    quantity: float
    price: float | None = None
    order_type: OrderType = "LIMIT"
    time_in_force: str = "gtc"

    status: OrderState = OrderState.SUBMITTED
    venue_id: str | None = None
    filled_quantity: float = 0.0

    # status captured on entering a PENDING_* state, restored on revert (cancel/update reject)
    pre_pending_status: OrderState | None = None
    # number of in-flight status queries issued by the stuck-order sweep
    retry_count: int = 0

    last_updated_at: dt_64 | None = None
    created_at: dt_64 | None = None

    # trade ids already applied, for fill dedup
    fill_trade_ids: set[str] = field(default_factory=set)


@dataclass
class AccountState:
    """Pure-data container for orders and the indexes the manager maintains."""

    # cid -> order
    active_orders: dict[str, ManagedOrder] = field(default_factory=dict)
    # venue_order_id -> cid
    _venue_id_index: dict[str, str] = field(default_factory=dict)
    # cids awaiting a venue verdict (SUBMITTED / PENDING_CANCEL / PENDING_UPDATE) — swept for staleness
    _inflight_index: set[str] = field(default_factory=set)
    # terminal cid -> time after which it may be evicted (grace window vs late events / snapshots)
    _pending_evict_index: dict[str, dt_64] = field(default_factory=dict)
