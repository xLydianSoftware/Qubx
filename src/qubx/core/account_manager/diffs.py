"""Snapshot Differ — stage 1 of the reconciliator redesign (git-style diff/merge).

``Differ.diff(local, origin)`` compares local ``AccountState`` against a venue
``AccountSnapshot`` and returns a flat list of fine-grained ``Diff`` atoms — the
analogue of ``git diff`` producing a list of deltas. It is pure (never mutates either
side) and clock-free except for the ``as_of``-based grace gate on orders. Deciding what
to DO with the diffs — merging them into local state — is stage 2 (a separate module).

See docs/account-management/reconciliation-redesign.md.
"""

from dataclasses import dataclass
from typing import ClassVar, dataclass_transform

import numpy as np

from qubx.core.account_manager.state import AccountState, VenueAccountFigures
from qubx.core.basics import Balance, Order, Position
from qubx.core.events import AccountSnapshot
from qubx.utils.time import to_timedelta

# Relative tolerance for floats that have no instrument lot/tick scale (margins,
# venue figures): abs(a - b) <= RTOL * max(|a|, |b|).
RTOL = 1e-9

# Balances ignore sub-cent drift: margin/unrealized-PnL float noise nudges free/locked by a
# tiny amount on every snapshot, which would otherwise reconcile (and log) each poll. A change
# under 1 cent is not a reconcilable difference. Absolute, because "1 cent" is absolute.
BALANCE_ABS_TOL = 0.01


# - Diff atoms ------------------------------------------------------------------ #
@dataclass_transform(frozen_default=True, kw_only_default=True)
def diffatom[T](cls: type[T]) -> type[T]:
    # repr=False so the base Diff.__repr__ (→ describe()) is inherited by every atom
    # rather than each leaf getting a verbose dataclass-generated one.
    return dataclass(frozen=True, slots=True, kw_only=True, repr=False)(cls)


def _fmt(v: object) -> str:
    # .8g, not .2f: quantities can be far smaller than 0.01 (e.g. 0.001 BTC), and .2f
    # rendered them as "0.00" in diff logs. %g keeps significant digits across scales.
    return f"{v:.8g}" if isinstance(v, float) else repr(v)


def _delta(local: object, origin: object) -> str:
    if isinstance(local, float) and isinstance(origin, float):
        return f" (Δ {origin - local:+.8g})"
    return ""


def _qeps(instrument) -> float:
    # quantity/size dust tolerance: half a lot (cleanly separates float dust from a real lot)
    return instrument.lot_size * 0.5 if instrument is not None else 0.0


def _peps(instrument) -> float:
    # price dust tolerance: half a tick
    return instrument.tick_size * 0.5 if instrument is not None else 0.0


def _close_rel(a: float, b: float) -> bool:
    # relative tolerance for floats with no lot/tick scale (margins, figures)
    return abs(a - b) <= RTOL * max(abs(a), abs(b))


def _close_bal(a: float, b: float) -> bool:
    # balance legs: absolute 1-cent tolerance (ignore sub-cent margin/PnL float drift)
    return abs(a - b) <= BALANCE_ABS_TOL


def _material_pos(position: Position) -> bool:
    # a position smaller than half a lot is effectively flat — never a presence diff
    return abs(position.quantity) > _qeps(position.instrument)


def _material_bal(balance: Balance) -> bool:
    # a balance that is zero on every leg is nothing — never a presence diff
    return not (_close_rel(balance.free, 0.0) and _close_rel(balance.locked, 0.0) and _close_rel(balance.total, 0.0))


@diffatom
class Diff:
    """Base for every diff atom. ``__repr__`` delegates to ``describe()`` so a list of
    diffs logs as readable git-style lines."""

    def __repr__(self) -> str:
        return self.describe()

    def describe(self) -> str:
        return f"{type(self).__name__}()"


# -- orders -- #
@diffatom
class LocalOrderMissing(Diff):
    """Order present locally but absent from the snapshot (a DELETE; grace-gated)."""

    order: Order

    def describe(self) -> str:
        o = self.order
        return (
            f"LocalOrderMissing[{o.instrument} cid={o.client_order_id} "
            f"status={o.status} qty={o.quantity}]  present locally, absent from snapshot"
        )


@diffatom
class OriginalOrderMissing(Diff):
    """Order present in the snapshot but absent locally (an ADD; not grace-gated)."""

    order: Order

    def describe(self) -> str:
        o = self.order
        return (
            f"OriginalOrderMissing[{o.instrument} vid={o.venue_order_id} "
            f"status={o.status} qty={o.quantity}]  present in snapshot, absent locally"
        )


@diffatom
class OrderFieldMismatch(Diff):
    """Base for per-field order MODIFY atoms. Leaves set ``FIELD`` (the Order attribute
    to compare) and optionally ``LABEL`` (the human name to render)."""

    local: Order
    origin: Order
    FIELD: ClassVar[str] = ""
    LABEL: ClassVar[str] = ""

    def describe(self) -> str:
        lv = getattr(self.local, self.FIELD)
        ov = getattr(self.origin, self.FIELD)
        label = self.LABEL or self.FIELD
        return (
            f"{type(self).__name__}[{self.local.instrument} cid={self.local.client_order_id}] "
            f"{label}: {_fmt(lv)} → {_fmt(ov)}{_delta(lv, ov)}"
        )


@diffatom
class OrderStatusMismatch(OrderFieldMismatch):
    FIELD = "status"


@diffatom
class OrderFilledQtyMismatch(OrderFieldMismatch):
    FIELD = "filled_quantity"


@diffatom
class OrderPriceMismatch(OrderFieldMismatch):
    FIELD = "price"


@diffatom
class OrderVenueIdMismatch(OrderFieldMismatch):
    FIELD = "venue_order_id"


@diffatom
class OrderQuantityMismatch(OrderFieldMismatch):
    FIELD = "quantity"


@diffatom
class OrderAvgFillPriceMismatch(OrderFieldMismatch):
    FIELD = "avg_fill_price"


# -- positions -- #
@diffatom
class PositionFieldMismatch(Diff):
    """Base for per-field position MODIFY atoms."""

    local: Position
    origin: Position
    FIELD: ClassVar[str] = ""
    LABEL: ClassVar[str] = ""

    def describe(self) -> str:
        lv = getattr(self.local, self.FIELD)
        ov = getattr(self.origin, self.FIELD)
        label = self.LABEL or self.FIELD
        return f"{type(self).__name__}[{self.local.instrument}] {label}: {_fmt(lv)} → {_fmt(ov)}{_delta(lv, ov)}"


@diffatom
class PositionSizeMismatch(PositionFieldMismatch):
    FIELD = "quantity"
    LABEL = "size"


@diffatom
class PositionAvgPriceMismatch(PositionFieldMismatch):
    FIELD = "position_avg_price"
    LABEL = "avg_price"


@diffatom
class PositionMarginMismatch(PositionFieldMismatch):
    FIELD = "maint_margin"
    LABEL = "maint_margin"


@diffatom
class LocalPositionMissing(Diff):
    """Material position present locally but absent from an observed snapshot."""

    position: Position

    def describe(self) -> str:
        p = self.position
        return f"LocalPositionMissing[{p.instrument}] size={p.quantity}  present locally, absent from snapshot"


@diffatom
class OriginalPositionMissing(Diff):
    """Material position present in the snapshot but absent locally."""

    position: Position

    def describe(self) -> str:
        p = self.position
        return f"OriginalPositionMissing[{p.instrument}] size={p.quantity}  present in snapshot, absent locally"


# -- balances / venue figures -- #
@diffatom
class LocalBalanceMissing(Diff):
    """Material balance present locally but absent from an observed snapshot."""

    balance: Balance

    def describe(self) -> str:
        b = self.balance
        return f"LocalBalanceMissing[{b.currency}] free={b.free} total={b.total}  present locally, absent from snapshot"


@diffatom
class OriginalBalanceMissing(Diff):
    """Material balance present in the snapshot but absent locally."""

    balance: Balance

    def describe(self) -> str:
        b = self.balance
        return (
            f"OriginalBalanceMissing[{b.currency}] free={b.free} total={b.total}  present in snapshot, absent locally"
        )


@diffatom
class BalanceMismatch(Diff):
    local: Balance
    origin: Balance

    def describe(self) -> str:
        return (
            f"BalanceMismatch[{self.origin.currency}] "
            f"free: {self.local.free:.2f} → {self.origin.free:.2f}{_delta(self.local.free, self.origin.free)} | "
            f"total: {self.local.total:.2f} → {self.origin.total:.2f}"
        )


@diffatom
class VenueFiguresMismatch(Diff):
    local: VenueAccountFigures
    origin: VenueAccountFigures

    def describe(self) -> str:
        return f"VenueFiguresMismatch equity: {self.local.equity} → {self.origin.equity}"


class Differ:
    _grace: np.timedelta64

    def __init__(self, grace: str | np.timedelta64 = "5s") -> None:
        self._grace = to_timedelta(grace).asm8 if isinstance(grace, str) else grace

    def diff(self, local: AccountState, origin: AccountSnapshot) -> list[Diff]:
        """Compare local state against a venue snapshot; return the list of diff atoms.

        Read-only on both arguments.
        """
        if local.exchange != origin.exchange:
            raise ValueError(
                f"Cannot diff states from different exchanges: local={local.exchange} vs origin={origin.exchange}"
            )
        diffs: list[Diff] = []
        self._diff_orders(local, origin, diffs)
        self._diff_positions(local, origin, diffs)
        self._diff_balances(local, origin, diffs)
        self._diff_venue_figures(local, origin, diffs)
        return diffs

    # -- orders -- #
    def _diff_orders(self, local: AccountState, origin: AccountSnapshot, diffs: list[Diff]) -> None:
        # None = the order-fetch leg FAILED (not "venue has no orders") → skip entirely, never
        # engage missing-handling. [] IS the venue's answer (no open orders) and is processed.
        if origin.open_orders is None:
            return
        snap_orders = origin.open_orders
        by_vid = {o.venue_order_id: o for o in snap_orders if o.venue_order_id}
        by_cid = {o.client_order_id: o for o in snap_orders if o.client_order_id}

        local_orders = list(local.get_orders().values())
        local_active = [o for o in local_orders if not o.status.is_terminal]
        # "present locally" for the OriginalOrderMissing check spans ALL retained orders,
        # active AND terminal: a snapshot taken before a fill can still list a since-completed
        # order as open, arriving after we already terminalized it. Matching only active orders
        # would flag it missing -> recover -> re-add an existing cid (crash) / resurrect a
        # terminal order. Our terminal view wins; the stale open copy is ignored.
        present_vids = {o.venue_order_id for o in local_orders if o.venue_order_id}
        present_cids = {o.client_order_id for o in local_orders}

        # local side: missing-from-snapshot + field mismatches (grace-gated)
        for lo in local_active:
            so = (by_vid.get(lo.venue_order_id) if lo.venue_order_id else None) or by_cid.get(lo.client_order_id)
            if not self._past_grace(lo, origin.as_of):
                continue
            if so is None:
                diffs.append(LocalOrderMissing(order=lo))
            else:
                self._compare_orders(lo, so, diffs)

        # origin side: snapshot orders we do not hold locally (no grace)
        for so in snap_orders:
            matched = (so.venue_order_id in present_vids) or (so.client_order_id in present_cids)
            if not matched:
                diffs.append(OriginalOrderMissing(order=so))

    def _compare_orders(self, lo: Order, so: Order, diffs: list[Diff]) -> None:
        inst = lo.instrument
        qeps = _qeps(inst)
        peps = _peps(inst)
        if lo.status != so.status:
            diffs.append(OrderStatusMismatch(local=lo, origin=so))
        if lo.venue_order_id != so.venue_order_id:
            diffs.append(OrderVenueIdMismatch(local=lo, origin=so))
        if abs(lo.filled_quantity - so.filled_quantity) > qeps:
            diffs.append(OrderFilledQtyMismatch(local=lo, origin=so))
        if abs(lo.quantity - so.quantity) > qeps:
            diffs.append(OrderQuantityMismatch(local=lo, origin=so))
        if abs((lo.price or 0.0) - (so.price or 0.0)) > peps:
            diffs.append(OrderPriceMismatch(local=lo, origin=so))
        if abs((lo.avg_fill_price or 0.0) - (so.avg_fill_price or 0.0)) > peps:
            diffs.append(OrderAvgFillPriceMismatch(local=lo, origin=so))

    def _past_grace(self, order: Order, as_of: np.datetime64) -> bool:
        seen_at = order.last_update_time if order.last_update_time is not None else order.submitted_at
        if seen_at is None:
            return False
        return bool((as_of - seen_at) >= self._grace)

    # -- positions (matched by instrument; no grace) -- #
    # origin.positions is None => leg NOT observed => stay silent. A list (incl. empty)
    # means observed: a material position on only one side is a presence diff.
    def _diff_positions(self, local: AccountState, origin: AccountSnapshot, diffs: list[Diff]) -> None:
        if origin.positions is None:
            return
        local_pos = local.get_positions()
        snap_instruments = {sp.instrument for sp in origin.positions}

        for sp in origin.positions:
            lp = local_pos.get(sp.instrument)
            if lp is None:
                if _material_pos(sp):
                    diffs.append(OriginalPositionMissing(position=sp))
                continue

            if abs(lp.quantity - sp.quantity) > _qeps(sp.instrument):
                diffs.append(PositionSizeMismatch(local=lp, origin=sp))

            if abs(lp.position_avg_price - sp.position_avg_price) > _peps(sp.instrument):
                diffs.append(PositionAvgPriceMismatch(local=lp, origin=sp))

            if not _close_rel(lp.maint_margin, sp.maint_margin):
                diffs.append(PositionMarginMismatch(local=lp, origin=sp))

        for inst, lp in local_pos.items():
            if inst not in snap_instruments and _material_pos(lp):
                diffs.append(LocalPositionMissing(position=lp))

    # -- balances (matched by currency) -- #
    # origin.balances is None => leg NOT observed => stay silent. A list (incl. empty)
    # means observed: a material balance on only one side is a presence diff.
    def _diff_balances(self, local: AccountState, origin: AccountSnapshot, diffs: list[Diff]) -> None:
        if origin.balances is None:
            return
        snap_currencies = {sb.currency for sb in origin.balances}

        for sb in origin.balances:
            lb = local.get_balance(sb.currency)
            if lb is None:
                if _material_bal(sb):
                    diffs.append(OriginalBalanceMissing(balance=sb))
                continue

            if not (
                _close_bal(lb.free, sb.free) and _close_bal(lb.locked, sb.locked) and _close_bal(lb.total, sb.total)
            ):
                diffs.append(BalanceMismatch(local=lb, origin=sb))

        for lb in local.get_balances():
            if lb.currency not in snap_currencies and _material_bal(lb):
                diffs.append(LocalBalanceMissing(balance=lb))

    # -- venue figures (singleton; per-figure rtol, None legs skipped) -- #
    def _diff_venue_figures(self, local: AccountState, origin: AccountSnapshot, diffs: list[Diff]) -> None:
        of = self._origin_figures(origin)
        if of is None:
            return
        lf = local.get_venue_figures()
        if lf is None:
            # nothing local to compare against is deferred to stage 2
            return
        for name in ("equity", "available_margin", "margin_ratio", "withdrawable"):
            lv = getattr(lf, name)
            ov = getattr(of, name)
            if lv is None or ov is None:
                continue
            if not _close_rel(lv, ov):
                diffs.append(VenueFiguresMismatch(local=lf, origin=of))
                return  # one atom per figures mismatch

    @staticmethod
    def _origin_figures(origin: AccountSnapshot) -> VenueAccountFigures | None:
        if (
            origin.equity is None
            and origin.available_margin is None
            and origin.margin_ratio is None
            and origin.withdrawable is None
        ):
            return None
        return VenueAccountFigures(
            as_of=origin.as_of,
            equity=origin.equity,
            available_margin=origin.available_margin,
            margin_ratio=origin.margin_ratio,
            withdrawable=origin.withdrawable,
        )
