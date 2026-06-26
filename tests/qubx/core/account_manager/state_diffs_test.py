"""Unit tests for the snapshot Differ (stage 1 of the reconciliator redesign).

The Differ compares local AccountState against a venue AccountSnapshot and returns a
flat list of fine-grained Diff atoms — a deterministic, clock-free value (modulo the
as_of-based grace gate) so every live-trading scenario is a plain assertion.

See docs/account-management/reconciliator-redesign.md.
"""

import numpy as np

from qubx.core.account_manager.diffs import (
    BalanceMismatch,
    Differ,
    LocalBalanceMissing,
    LocalOrderMissing,
    LocalPositionMissing,
    OrderAvgFillPriceMismatch,
    OrderFilledQtyMismatch,
    OrderPriceMismatch,
    OrderQuantityMismatch,
    OrderStatusMismatch,
    OrderVenueIdMismatch,
    OriginalBalanceMissing,
    OriginalOrderMissing,
    OriginalPositionMissing,
    PositionAvgPriceMismatch,
    PositionMarginMismatch,
    PositionSizeMismatch,
    VenueFiguresMismatch,
)
from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.basics import (
    Balance,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from qubx.core.events import AccountSnapshot
from qubx.core.lookups import lookup

EXCHANGE = "BINANCE.UM"

# as_of is the snapshot request time; SETTLED is well before it (past grace), FRESH is
# inside the 5s grace window. Default grace in tests is 5s.
AS_OF = np.datetime64("2026-05-28T00:00:30", "ns")
SETTLED = np.datetime64("2026-05-28T00:00:00", "ns")  # as_of - 30s -> past grace
FRESH = np.datetime64("2026-05-28T00:00:27", "ns")  # as_of - 3s  -> within grace
AFTER = np.datetime64("2026-05-28T00:00:31", "ns")  # as_of + 1s  -> after request


def _inst(symbol: str = "BTCUSDT"):
    inst = lookup.find_symbol(EXCHANGE, symbol)
    assert inst is not None, f"fixture instrument {symbol} not found"
    return inst


def _order(
    cid: str = "qubx_1",
    *,
    instrument=None,
    venue_id: str | None = "v1",
    status: OrderStatus = OrderStatus.ACCEPTED,
    side: OrderSide = OrderSide.BUY,
    quantity: float = 1.0,
    price: float = 50_000.0,
    filled_quantity: float = 0.0,
    avg_fill_price: float = 0.0,
    last_update_time: np.datetime64 | None = SETTLED,
    submitted_at: np.datetime64 | None = SETTLED,
    origin: OrderOrigin = OrderOrigin.FRAMEWORK,
) -> Order:
    return Order(
        client_order_id=cid,
        type=OrderType.LIMIT,
        instrument=instrument if instrument is not None else _inst(),
        quantity=quantity,
        side=side,
        time_in_force="gtc",
        status=status,
        venue_order_id=venue_id,
        price=price,
        filled_quantity=filled_quantity,
        avg_fill_price=avg_fill_price,
        submitted_at=submitted_at,
        last_update_time=last_update_time,
        origin=origin,
    )


def _pos(instrument=None, *, quantity: float = 1.0, avg: float = 50_000.0, maint_margin: float = 100.0) -> Position:
    p = Position(instrument if instrument is not None else _inst(), quantity=quantity, pos_average_price=avg)
    p.maint_margin = maint_margin
    return p


def _state(*, orders=(), positions=(), balances=(), figures=None) -> AccountState:
    st = AccountState(EXCHANGE, "USDT")
    for o in orders:
        st.add_order(o)
    for p in positions:
        st.set_position(p.instrument, p)
    for b in balances:
        st.apply_balance_snapshot(b)
    if figures is not None:
        st.set_venue_figures(figures)
    return st


def _snap(*, as_of=AS_OF, open_orders=None, positions=None, balances=None, **figures) -> AccountSnapshot:
    return AccountSnapshot(
        exchange=EXCHANGE, as_of=as_of, open_orders=open_orders, positions=positions, balances=balances, **figures
    )


def _differ(grace: str = "5s") -> Differ:
    return Differ(grace=grace)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


def test_exchange_mismatch_raises():
    local = AccountState(EXCHANGE, "USDT")
    origin = AccountSnapshot(exchange="OKX", as_of=AS_OF)
    import pytest

    with pytest.raises(ValueError):
        _differ().diff(local, origin)


# --------------------------------------------------------------------------- #
# orders — in sync / single-field mismatch
# --------------------------------------------------------------------------- #


def test_in_sync_order_yields_no_diff():
    o = _order()
    s = _order()  # identical
    assert _differ().diff(_state(orders=[o]), _snap(open_orders=[s])) == []


def test_price_drift_emits_price_mismatch():
    local = _state(orders=[_order(price=50_000.0)])
    origin = _snap(open_orders=[_order(price=50_100.0)])
    diffs = _differ().diff(local, origin)
    assert len(diffs) == 1
    assert isinstance(diffs[0], OrderPriceMismatch)
    assert diffs[0].local.price == 50_000.0
    assert diffs[0].origin.price == 50_100.0


def test_filled_qty_drift_emits_filled_qty_mismatch():
    local = _state(orders=[_order(filled_quantity=0.4)])
    origin = _snap(open_orders=[_order(filled_quantity=0.6)])
    diffs = _differ().diff(local, origin)
    assert [type(d) for d in diffs] == [OrderFilledQtyMismatch]


def test_status_drift_emits_status_mismatch():
    local = _state(orders=[_order(status=OrderStatus.ACCEPTED)])
    origin = _snap(open_orders=[_order(status=OrderStatus.PARTIALLY_FILLED)])
    diffs = _differ().diff(local, origin)
    assert [type(d) for d in diffs] == [OrderStatusMismatch]


def test_quantity_amend_emits_quantity_mismatch():
    local = _state(orders=[_order(quantity=1.0)])
    origin = _snap(open_orders=[_order(quantity=2.0)])
    diffs = _differ().diff(local, origin)
    assert [type(d) for d in diffs] == [OrderQuantityMismatch]


def test_avg_fill_price_drift_emits_avg_fill_mismatch():
    local = _state(orders=[_order(filled_quantity=0.5, avg_fill_price=50_000.0)])
    origin = _snap(open_orders=[_order(filled_quantity=0.5, avg_fill_price=50_010.0)])
    diffs = _differ().diff(local, origin)
    assert [type(d) for d in diffs] == [OrderAvgFillPriceMismatch]


def test_multi_field_drift_emits_one_atom_per_field():
    local = _state(orders=[_order(price=50_000.0, filled_quantity=0.4)])
    origin = _snap(open_orders=[_order(price=50_100.0, filled_quantity=0.6)])
    kinds = {type(d) for d in _differ().diff(local, origin)}
    assert kinds == {OrderPriceMismatch, OrderFilledQtyMismatch}


# --------------------------------------------------------------------------- #
# orders — missing / grace / tolerance
# --------------------------------------------------------------------------- #


def test_local_only_past_grace_emits_local_missing():
    local = _state(orders=[_order(cid="qubx_z", last_update_time=SETTLED)])
    diffs = _differ().diff(local, _snap(open_orders=[]))
    assert [type(d) for d in diffs] == [LocalOrderMissing]
    assert diffs[0].order.client_order_id == "qubx_z"


def test_local_only_within_grace_emits_nothing():
    local = _state(orders=[_order(last_update_time=FRESH)])  # as_of - 3s < 5s
    assert _differ().diff(local, _snap(open_orders=[])) == []


def test_local_only_changed_after_request_emits_nothing():
    local = _state(orders=[_order(last_update_time=AFTER)])  # seen_at > as_of
    assert _differ().diff(local, _snap(open_orders=[])) == []


def test_local_only_untimestamped_emits_nothing():
    local = _state(orders=[_order(last_update_time=None, submitted_at=None)])
    assert _differ().diff(local, _snap(open_orders=[])) == []


def test_field_mismatch_within_grace_emits_nothing():
    # order present on BOTH sides with a real price drift, but changed within grace ->
    # the gate suppresses ALL atoms for it (not just the missing case)
    local = _state(orders=[_order(price=50_000.0, last_update_time=FRESH)])
    origin = _snap(open_orders=[_order(price=50_100.0)])
    assert _differ().diff(local, origin) == []


def test_terminal_local_absent_from_snapshot_ignored():
    local = _state(orders=[_order(status=OrderStatus.FILLED)])
    assert _differ().diff(local, _snap(open_orders=[])) == []


def test_snapshot_only_emits_original_missing():
    origin = _snap(open_orders=[_order(cid="qubx_ext", venue_id="vX")])
    diffs = _differ().diff(_state(orders=[]), origin)
    assert [type(d) for d in diffs] == [OriginalOrderMissing]
    assert diffs[0].order.venue_order_id == "vX"


def test_cid_match_with_new_venue_id_emits_venue_id_mismatch():
    # unacked framework order: local has no venue id yet, snapshot reports it under our cid
    local = _state(orders=[_order(cid="qubx_a", venue_id=None)])
    origin = _snap(open_orders=[_order(cid="qubx_a", venue_id="v9")])
    diffs = _differ().diff(local, origin)
    assert [type(d) for d in diffs] == [OrderVenueIdMismatch]
    assert diffs[0].origin.venue_order_id == "v9"


def test_subtick_price_drift_below_tolerance_emits_nothing():
    local = _state(orders=[_order(price=50_000.00)])
    origin = _snap(open_orders=[_order(price=50_000.03)])  # < tick/2 = 0.05
    assert _differ().diff(local, origin) == []


def test_sublot_filled_drift_below_tolerance_emits_nothing():
    local = _state(orders=[_order(filled_quantity=0.4000)])
    origin = _snap(open_orders=[_order(filled_quantity=0.4003)])  # < lot/2 = 0.0005
    assert _differ().diff(local, origin) == []


def _bal(currency="USDT", *, free=1000.0, locked=0.0, total=1000.0) -> Balance:
    return Balance(exchange=EXCHANGE, currency=currency, free=free, locked=locked, total=total)


def _figs(
    *, equity=50_000.0, available_margin=40_000.0, margin_ratio=0.1, withdrawable=40_000.0
) -> VenueAccountFigures:
    return VenueAccountFigures(
        as_of=AS_OF,
        equity=equity,
        available_margin=available_margin,
        margin_ratio=margin_ratio,
        withdrawable=withdrawable,
    )


# --------------------------------------------------------------------------- #
# positions
# --------------------------------------------------------------------------- #


def test_in_sync_position_yields_no_diff():
    local = _state(positions=[_pos(quantity=1.0, avg=50_000.0, maint_margin=100.0)])
    origin = _snap(positions=[_pos(quantity=1.0, avg=50_000.0, maint_margin=100.0)])
    assert _differ().diff(local, origin) == []


def test_position_size_drift_emits_size_mismatch():
    local = _state(positions=[_pos(quantity=1.5)])
    origin = _snap(positions=[_pos(quantity=1.2)])
    assert [type(d) for d in _differ().diff(local, origin)] == [PositionSizeMismatch]


def test_position_avg_price_drift_emits_avg_price_mismatch():
    local = _state(positions=[_pos(avg=50_000.0)])
    origin = _snap(positions=[_pos(avg=50_100.0)])
    assert [type(d) for d in _differ().diff(local, origin)] == [PositionAvgPriceMismatch]


def test_position_margin_drift_emits_margin_mismatch():
    local = _state(positions=[_pos(maint_margin=100.0)])
    origin = _snap(positions=[_pos(maint_margin=120.0)])
    assert [type(d) for d in _differ().diff(local, origin)] == [PositionMarginMismatch]


def test_position_multi_field_emits_one_atom_per_field():
    local = _state(positions=[_pos(quantity=1.5, maint_margin=100.0)])
    origin = _snap(positions=[_pos(quantity=1.2, maint_margin=120.0)])
    kinds = {type(d) for d in _differ().diff(local, origin)}
    assert kinds == {PositionSizeMismatch, PositionMarginMismatch}


def test_position_sublot_size_drift_below_tolerance_emits_nothing():
    local = _state(positions=[_pos(quantity=1.0000)])
    origin = _snap(positions=[_pos(quantity=1.0003)])  # < lot/2 = 0.0005
    assert _differ().diff(local, origin) == []


# --------------------------------------------------------------------------- #
# balances
# --------------------------------------------------------------------------- #


def test_in_sync_balance_yields_no_diff():
    local = _state(balances=[_bal(free=1000.0, total=1000.0)])
    origin = _snap(balances=[_bal(free=1000.0, total=1000.0)])
    assert _differ().diff(local, origin) == []


def test_balance_free_drift_emits_balance_mismatch():
    local = _state(balances=[_bal(free=1000.0, total=1200.0)])
    origin = _snap(balances=[_bal(free=980.0, total=1200.0)])
    assert [type(d) for d in _differ().diff(local, origin)] == [BalanceMismatch]


def test_balance_total_drift_emits_balance_mismatch():
    local = _state(balances=[_bal(free=1000.0, total=1200.0)])
    origin = _snap(balances=[_bal(free=1000.0, total=1180.0)])
    assert [type(d) for d in _differ().diff(local, origin)] == [BalanceMismatch]


def test_balance_below_tolerance_emits_nothing():
    local = _state(balances=[_bal(free=1000.0, total=1000.0)])
    origin = _snap(balances=[_bal(free=1000.0 + 1e-9, total=1000.0)])
    assert _differ().diff(local, origin) == []


# --------------------------------------------------------------------------- #
# venue figures
# --------------------------------------------------------------------------- #


def test_in_sync_figures_yield_no_diff():
    local = _state(figures=_figs(equity=50_000.0))
    origin = _snap(equity=50_000.0, available_margin=40_000.0, margin_ratio=0.1, withdrawable=40_000.0)
    assert _differ().diff(local, origin) == []


def test_figures_equity_drift_emits_figures_mismatch():
    local = _state(figures=_figs(equity=50_000.0))
    origin = _snap(equity=49_800.0, available_margin=40_000.0, margin_ratio=0.1, withdrawable=40_000.0)
    assert [type(d) for d in _differ().diff(local, origin)] == [VenueFiguresMismatch]


def test_figures_none_leg_in_snapshot_skipped():
    # snapshot reports no figures at all -> nothing to compare
    local = _state(figures=_figs(equity=50_000.0))
    assert _differ().diff(local, _snap()) == []


# --------------------------------------------------------------------------- #
# presence — positions / balances (None = not observed -> silent; [] / absent = flag)
# --------------------------------------------------------------------------- #


def test_local_position_absent_from_observed_snapshot_emits_local_missing():
    local = _state(positions=[_pos(quantity=10.0)])
    diffs = _differ().diff(local, _snap(positions=[]))  # venue observed, reports flat
    assert [type(d) for d in diffs] == [LocalPositionMissing]
    assert diffs[0].position.quantity == 10.0


def test_local_position_with_positions_not_observed_emits_nothing():
    local = _state(positions=[_pos(quantity=10.0)])
    assert _differ().diff(local, _snap(positions=None)) == []  # leg not observed


def test_flat_local_position_absent_from_snapshot_emits_nothing():
    local = _state(positions=[_pos(quantity=0.0)])  # immaterial
    assert _differ().diff(local, _snap(positions=[])) == []


def test_snapshot_position_absent_locally_emits_original_missing():
    origin = _snap(positions=[_pos(quantity=10.0)])
    diffs = _differ().diff(_state(positions=[]), origin)
    assert [type(d) for d in diffs] == [OriginalPositionMissing]
    assert diffs[0].position.quantity == 10.0


def test_local_balance_absent_from_observed_snapshot_emits_local_missing():
    local = _state(balances=[_bal(currency="USDT", free=1000.0, total=1000.0)])
    diffs = _differ().diff(local, _snap(balances=[]))  # venue observed, no balances
    assert [type(d) for d in diffs] == [LocalBalanceMissing]
    assert diffs[0].balance.currency == "USDT"


def test_local_balance_with_balances_not_observed_emits_nothing():
    local = _state(balances=[_bal(free=1000.0, total=1000.0)])
    assert _differ().diff(local, _snap(balances=None)) == []


def test_zero_local_balance_absent_from_snapshot_emits_nothing():
    local = _state(balances=[_bal(currency="BNB", free=0.0, locked=0.0, total=0.0)])
    assert _differ().diff(local, _snap(balances=[])) == []


def test_snapshot_balance_absent_locally_emits_original_missing():
    origin = _snap(balances=[_bal(currency="USDT", free=990.0, total=990.0)])
    diffs = _differ().diff(_state(balances=[]), origin)
    assert [type(d) for d in diffs] == [OriginalBalanceMissing]
    assert diffs[0].balance.currency == "USDT"


# ---------------------------------------------------------------------------


def test_diffs_repr():
    from qubx.utils.misc import green, red

    local = _state(
        orders=[
            _order(cid="X1", venue_id="543210", price=50_000.00),
            _order(cid="X3", venue_id="012345", price=1_000.00),
        ],
        balances=[_bal(free=1000.0, total=1000.0)],
    )
    origin = _snap(
        open_orders=[
            _order(cid="X1", venue_id="012345", price=50_000.00),
            _order(cid="X2", venue_id="123", price=50_000.00),
            _order(cid="X3", venue_id="456", status=OrderStatus.FILLED),
        ],
        positions=[_pos(quantity=10, maint_margin=100.0)],
        balances=[_bal(free=990.0, total=1000.0)],
    )

    print(red("\n- - - - Diffs - - - - "))

    for d in _differ().diff(local, origin):
        print(f" - {green(d)}")

    print(red("- - - - - - - - - - - "))
