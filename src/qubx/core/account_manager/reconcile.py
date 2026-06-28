"""Order-transition chokepoint + small leaf helpers shared by the reducer and manager.

Snapshot reconciliation now lives entirely in the Reconciler (``reconciler.py`` +
``diffs.py``); this module is reduced to the pieces the reducer and manager still share.

Import rule: this module must never import the reducer (or the manager) — that is the
cycle-relevant rule. The validate+transition helper (``transition``) lives HERE for that
reason: the reducer and the manager import it from here, never the other way around.
"""

import numpy as np

from qubx.core.account_manager.state import AccountState
from qubx.core.account_manager.state_machine import validate_transition
from qubx.core.basics import Instrument, Order, OrderStatus


def transition(
    state: AccountState, cid: str, new_status: OrderStatus, now: np.datetime64, *, update_time: np.datetime64 | None = None
) -> Order:
    """Validate-then-apply for an active order's status — the single legality chokepoint
    (the reducer and the manager delegate here; see the import rule above)."""
    order = state.get_active_order(cid)
    if order is None:
        raise KeyError(f"order {cid} not found in {state.exchange}")
    validate_transition(cid, order.status, new_status)
    return state.transition_order(cid, new_status, now, update_time=update_time)


def fill_qty_epsilon(instrument: Instrument | None) -> float:
    """Float-dust tolerance for fill-quantity comparisons.

    Venue quantities arrive via float subtraction (e.g. 0.6 - 0.4 -> 0.19999999999999998),
    so exact compares against 0.0 read ~1e-16 dust as a genuine excess. Any real quantity
    difference is at least one lot, so half a lot cleanly separates dust from real fills.
    0.0 (exact compare) for an order without an instrument.
    """
    return instrument.lot_size * 0.5 if instrument is not None else 0.0


def liveness_overdue(unready_since: np.datetime64, now: np.datetime64, threshold: np.timedelta64) -> bool:
    return (now - unready_since) >= threshold
