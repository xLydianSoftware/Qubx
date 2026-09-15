"""Per-position entry shared by the 5s state snapshot and the ``get_state`` control action.

One builder so the two never drift: the snapshot writes it verbatim, the action rounds it
for display.
"""

import math
from typing import Any

from qubx.core.basics import Instrument, Position
from qubx.core.interfaces import IAccountViewer


def finite(value: float | None) -> float | None:
    """None for None, NaN and ±inf.

    The snapshot is plain ``json.dumps`` and the platform's Go decoder rejects Infinity/NaN,
    which would blank the whole bot state.
    """
    return float(value) if value is not None and math.isfinite(value) else None


def position_entry(account: IAccountViewer, instrument: Instrument, position: Position) -> dict[str, Any]:
    """One position's snapshot entry: what we hold, plus the venue's settings for it.

    ``notional`` carries the quantity's sign. ``max_notional`` is ``float('inf')`` from the
    account manager when the venue publishes no cap, and None here.
    """
    return {
        "quantity": position.quantity,
        "avg_price": position.position_avg_price,
        "market_value": position.market_value_funds,
        "unrealized_pnl": position.unrealized_pnl(),
        "current_price": position.last_update_price,
        "leverage": account.get_leverage(instrument),
        "notional": finite(position.notional_value),
        "instrument_leverage": finite(account.get_instrument_leverage(instrument)),
        "max_instrument_leverage": finite(account.get_max_instrument_leverage(instrument)),
        "max_notional": finite(account.get_max_instrument_notional(instrument)),
    }
