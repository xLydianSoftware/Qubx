"""Balance-backed spot: a spot position's size is its base asset's net holding.

Covers a BINANCE spot leg on a BINANCE.PM account (one state, holdings from the cross-margin
balance) and a plain spot account (holding = balance total). Accounting (avg, r_pnl, commissions)
is booked from deals; the size always follows the holding.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.account_manager.manager import AccountManager, SimulatedAccountManager
from qubx.core.account_manager.reconciler import ConfirmPositionBySnapshot
from qubx.core.basics import (
    Balance,
    Deal,
    Instrument,
    ITimeProvider,
    MarketType,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from qubx.core.events import (
    AccountSnapshot,
    AccountSnapshotEvent,
    BalanceUpdateEvent,
    OrderAcceptedEvent,
    OrderFilledEvent,
)
from qubx.core.lookups import lookup
from qubx.core.series import Quote

T0 = np.datetime64("2026-10-23T00:00:00", "ns")
T1 = np.datetime64("2026-10-23T00:01:00", "ns")
T2 = np.datetime64("2026-10-23T00:02:00", "ns")
T3 = np.datetime64("2026-10-23T00:03:00", "ns")

UM = "BINANCE.UM"
TRX = lookup.find_symbol("BINANCE", "TRXUSDT")  # lot 0.1, min notional 5
TRX_PERP = lookup.find_symbol(UM, "TRXUSDT")
assert TRX is not None and TRX_PERP is not None


def _spot(base: str, lot: float = 0.1, min_notional: float = 5.0) -> Instrument:
    return Instrument(
        f"{base}USDT", MarketType.SPOT, "BINANCE", base, "USDT", "USDT", f"{base}USDT", 0.0001, lot, lot, min_notional
    )


INIT = _spot("INIT")
AERO = _spot("AERO", lot=0.01)


class _Time(ITimeProvider):
    def __init__(self, now: np.datetime64 = T1):
        self.now = now

    def time(self) -> np.datetime64:
        return self.now


def _pm(**kw) -> tuple[AccountManager, _Time]:
    clock = _Time()
    am = AccountManager(
        connectors={UM: MagicMock()},
        base_currencies={UM: "USDT"},
        time=clock,
        account_markets={UM: (UM, "BINANCE")},
        **kw,
    )
    return am, clock


def _mark(am: AccountManager, instrument: Instrument, price: float) -> None:
    am.get_position(instrument)
    am.on_market_quote(instrument, Quote(T0, price, price, 1.0, 1.0))


_seq = iter(range(1_000_000))


def _fill(am, instrument, amount, price, *, fee=0.0, fee_ccy="USDT", time=T0) -> None:
    n = next(_seq)
    cid, vid = f"c{n}", f"v{n}"
    side = OrderSide.BUY if amount > 0 else OrderSide.SELL
    am.add_order(
        Order(
            client_order_id=cid,
            type=OrderType.MARKET,
            instrument=instrument,
            quantity=abs(amount),
            side=side,
            time_in_force="gtc",
            status=OrderStatus.SUBMITTED,
            origin=OrderOrigin.FRAMEWORK,
        )
    )
    am.apply(OrderAcceptedEvent(instrument=instrument, client_order_id=cid, venue_order_id=vid, accepted_at=time))
    deal = Deal(f"t{n}", vid, time, amount, price, True, fee_amount=fee, fee_currency=fee_ccy)
    am.apply(
        OrderFilledEvent(
            instrument=instrument, client_order_id=cid, venue_order_id=vid, last_update_time=time, fill=deal
        )
    )


def _bal(currency: str, holding: float, *, total: float | None = None, since=None) -> Balance:
    total = holding if total is None else total
    return Balance(UM, currency, free=total, total=total, holding=holding, last_update_time=since)


def _snapshot(am: AccountManager, as_of, *balances: Balance, positions=()) -> None:
    am.apply(
        AccountSnapshotEvent(
            instrument=None,
            snapshot=AccountSnapshot(exchange=UM, as_of=as_of, positions=list(positions), balances=list(balances)),
        )
    )


def _position_of(diff) -> Instrument | None:
    pos = getattr(diff, "position", None) or getattr(diff, "local", None)
    return pos.instrument if isinstance(pos, Position) else None


def _confirm_tasks(am: AccountManager) -> set:
    return {k for k, t in am._reconcilers[UM]._tasks.items() if isinstance(t, ConfirmPositionBySnapshot)}


class TestFills:
    def test_a_base_fee_leaves_the_size_equal_to_the_holding(self):
        am, _ = _pm()
        _fill(am, TRX, 36.3, 0.3315, fee=0.0363, fee_ccy="TRX")
        pos = am.get_position(TRX)
        assert pos.quantity == am.get_state(UM).holding_of("TRX") == pytest.approx(36.2637)
        assert pos.position_avg_price == pytest.approx(0.3315)
        assert not pos.instrument.is_dust(pos.quantity, 0.3315)

    def test_a_push_covered_deal_books_accounting_and_takes_the_size_from_the_push(self):
        am, _ = _pm()
        _fill(am, TRX, 60.0, 0.30)
        # - the push (venue time T1) already counts the sell executed at T0
        am.apply(BalanceUpdateEvent(instrument=None, balance=_bal("TRX", 0.0), as_of=T1))
        pos = am.get_position(TRX)
        assert pos.quantity == 0.0
        _fill(am, TRX, -60.0, 0.33)
        assert pos.quantity == 0.0
        assert pos.r_pnl == pytest.approx(60 * 0.03)
        assert am.get_state(UM).holding_of("TRX") == 0.0

    def test_a_push_covered_opening_deal_moves_the_average(self):
        am, _ = _pm()
        am.get_position(TRX)
        am.apply(BalanceUpdateEvent(instrument=None, balance=_bal("TRX", 60.0), as_of=T1))
        _fill(am, TRX, 60.0, 0.30)
        pos = am.get_position(TRX)
        assert pos.quantity == 60.0
        assert pos.position_avg_price == pytest.approx(0.30)

    def test_a_close_leaves_the_untradeable_tail_and_reads_closed(self):
        am, _ = _pm()
        _mark(am, INIT, 0.134)
        _fill(am, INIT, 333.3, 0.134, fee=0.3336, fee_ccy="INIT")  # - holds 332.9664
        pos = am.get_position(INIT)
        assert pos.tradeable_quantity == pytest.approx(332.9)
        _fill(am, INIT, -pos.tradeable_quantity, 0.135)
        assert pos.quantity == pytest.approx(0.0664)
        assert pos.is_dust() and not pos.is_open()


class TestSnapshot:
    def test_spot_gets_no_position_atoms(self):
        am, _ = _pm()
        _fill(am, TRX, 60.0, 0.30)
        _fill(am, TRX_PERP, -60.0, 0.30)
        perp = Position(TRX_PERP, quantity=-60.0, pos_average_price=0.30)
        diffs = am._reconcilers[UM]._differ.diff(
            am.get_state(UM),
            AccountSnapshot(exchange=UM, as_of=T2, positions=[perp], balances=[_bal("TRX", 59.0)]),
        )
        assert [d for d in diffs if _position_of(d) == TRX] == []
        # - an observed snapshot without the spot row is not a missed close
        diffs = am._reconcilers[UM]._differ.diff(
            am.get_state(UM), AccountSnapshot(exchange=UM, as_of=T2, positions=[], balances=[])
        )
        assert [d.position.instrument for d in diffs if type(d).__name__ == "LocalPositionMissing"] == [TRX_PERP]

    def test_a_stale_snapshot_does_not_rewind_the_holding(self):
        am, clock = _pm()
        _snapshot(am, T0, _bal("USDT", 100.0))
        clock.now = T2
        _fill(am, TRX, 60.0, 0.30)  # - booked at T2
        _snapshot(am, T1, _bal("USDT", 82.0))  # - requested before the fill: holds no TRX yet
        assert am.get_position(TRX).quantity == 60.0
        assert am.get_state(UM).holding_of("TRX") == 60.0
        _snapshot(am, T3, _bal("USDT", 82.0))  # - a fresh one does
        assert am.get_position(TRX).quantity == 0.0

    def test_a_negative_holding_is_a_short_and_interest_drift_is_a_commission(self):
        am, clock = _pm()
        _mark(am, TRX, 0.30)
        _snapshot(am, T0, _bal("TRX", -35.0, total=10.0))
        pos = am.get_position(TRX)
        assert pos.quantity == -35.0 and pos.position_avg_price == pytest.approx(0.30)
        clock.now = T2
        _snapshot(am, T2, _bal("TRX", -35.05, total=10.0))
        assert pos.quantity == pytest.approx(-35.05)
        assert pos.commissions == pytest.approx(0.05 * 0.30)
        assert _confirm_tasks(am) == set()

    def test_a_dust_drift_snaps_silently_a_lot_spawns_a_confirm(self):
        am, clock = _pm()
        _mark(am, TRX, 0.30)
        _snapshot(am, T0, _bal("TRX", 60.0))
        pos = am.get_position(TRX)
        commissions = pos.commissions
        clock.now = T2
        _snapshot(am, T2, _bal("TRX", 59.95))  # - below a lot: a conversion, not a missed deal
        assert pos.quantity == pytest.approx(59.95)
        assert pos.commissions == commissions
        assert _confirm_tasks(am) == set()
        clock.now = T3
        _snapshot(am, T3, _bal("TRX", 9.95, since=T2))  # - 50 sold, the fill never arrived
        assert pos.quantity == pytest.approx(9.95)
        assert _confirm_tasks(am) == {str(TRX)}
        assert am.get_state(UM).get_position_reconcile_as_of(TRX) == T2

    def test_a_sub_tolerance_balance_drift_is_not_a_mismatch(self):
        am, _ = _pm()
        _fill(am, TRX, 60.0, 0.30)

        def trx_diffs(holding):
            diffs = am._reconcilers[UM]._differ.diff(
                am.get_state(UM), AccountSnapshot(exchange=UM, as_of=T2, positions=[], balances=[_bal("TRX", holding)])
            )
            return [d for d in diffs if getattr(d, "origin", None) is not None and d.origin.currency == "TRX"]

        assert trx_diffs(60.00001) == []
        assert len(trx_diffs(60.001)) == 1  # - 0.01 units no longer hides a base-asset move

    def test_a_late_fill_the_snapshot_counted_books_accounting_only(self):
        am, clock = _pm()
        _mark(am, TRX, 0.30)
        _snapshot(am, T0, _bal("USDT", 100.0))
        clock.now = T2
        _snapshot(am, T2, _bal("TRX", 60.0, since=T1))  # - holds the T1 buy before its fill arrived
        _fill(am, TRX, 60.0, 0.31, time=T1)
        pos = am.get_position(TRX)
        assert pos.quantity == 60.0
        assert pos.position_avg_price == pytest.approx(0.31)
        assert am.get_state(UM).holding_of("TRX") == 60.0


class TestBinding:
    def test_one_spot_instrument_per_asset(self):
        am, _ = _pm()
        am.get_position(TRX)
        trx_usdc = Instrument(
            "TRXUSDC", MarketType.SPOT, "BINANCE", "TRX", "USDC", "USDC", "TRXUSDC", 0.0001, 0.1, 0.1, 5.0
        )
        with pytest.raises(ValueError, match="one spot instrument per asset"):
            am.get_position(trx_usdc)

    def test_an_ignored_asset_stays_a_balance(self):
        am, _ = _pm(ignore_assets={UM: ("TRX",)})
        am.get_position(TRX)
        _snapshot(am, T0, _bal("TRX", 60.0))
        assert am.get_position(TRX).quantity == 0.0
        assert am.get_state(UM).spot_instrument_of("TRX") is None

    def test_stables_are_not_bound(self):
        am, _ = _pm()
        usdc = lookup.find_symbol("BINANCE", "USDCUSDT")
        assert usdc is not None
        am.get_position(usdc)
        assert am.get_state(UM).spot_instrument_of("USDC") is None


class TestRestart:
    def test_restored_dust_the_venue_no_longer_holds_goes_to_zero_keeping_r_pnl(self):
        am, _ = _pm()
        am.seed_position(Position(AERO, quantity=0.087, pos_average_price=0.9, r_pnl=1.5))
        assert not am.is_synced()
        _snapshot(am, T1, _bal("USDT", 100.0))
        pos = am.get_position(AERO)
        assert am.is_synced()
        assert pos.quantity == 0.0
        assert pos.r_pnl == 1.5

    def test_restored_holding_off_the_lot_grid_closes_to_its_dust(self):
        am, clock = _pm()
        am.seed_position(Position(INIT, quantity=332.964, pos_average_price=0.13, r_pnl=0.7))
        _mark(am, INIT, 0.134)
        _snapshot(am, T1, _bal("INIT", 332.964))
        pos = am.get_position(INIT)
        assert pos.quantity == 332.964 and pos.r_pnl == 0.7
        assert pos.tradeable_quantity == pytest.approx(332.9) and not pos.is_dust()
        clock.now = T2
        _fill(am, INIT, -332.9, 0.134, time=T2)
        assert pos.quantity == pytest.approx(0.064)
        assert not pos.is_open()


class TestPlainSpot:
    def test_the_holding_is_the_balance_total(self):
        clock = _Time()
        am = AccountManager(connectors={"BINANCE": MagicMock()}, base_currencies={"BINANCE": "USDT"}, time=clock)
        _fill(am, TRX, 60.0, 0.30)
        am.apply(
            AccountSnapshotEvent(
                instrument=None,
                snapshot=AccountSnapshot(
                    exchange="BINANCE",
                    as_of=T1,
                    positions=[],
                    balances=[Balance("BINANCE", "TRX", free=59.9, total=59.9)],
                ),
            )
        )
        assert am.get_position(TRX).quantity == pytest.approx(59.9)


XYZ = _spot("XYZ")


class TestSimulation:
    def test_quote_fee_fills_book_the_size_bit_identical_to_the_ledger(self):
        # - a float sum of the balance drifts off the grid (0.1 + 0.2); the size must not move with it
        am = SimulatedAccountManager(
            connectors={"BINANCE": MagicMock()}, base_currencies={"BINANCE": "USDT"}, time=_Time()
        )
        ledger = Position(XYZ)
        for i, amount in enumerate((0.1, 0.2, 0.7, -0.3, 1.1, -1.8, 0.3)):
            price = 10.0 + i
            _fill(am, XYZ, amount, price, fee=0.001)
            ledger.update_position_by_deal(
                Deal(f"r{i}", "r", T0, amount, price, True, fee_amount=0.001, fee_currency="USDT")
            )
            pos = am.get_position(XYZ)
            assert pos.quantity == ledger.quantity
            assert pos.position_avg_price == ledger.position_avg_price
            assert pos.r_pnl == ledger.r_pnl
            assert am.get_state("BINANCE").holding_of("XYZ") == pos.quantity


class TestBookAccounting:
    @pytest.mark.parametrize("seed", range(20))
    def test_matches_the_size_writing_path(self, seed):
        # - a holding that already carries each deal, then the deal: same avg / r_pnl / commissions
        rng = np.random.default_rng(seed)
        ledger, held = Position(TRX), Position(TRX)
        for i in range(30):
            amount = round(float(rng.integers(-400, 400)) * 0.1, 1) or 0.1
            price = round(float(rng.uniform(0.2, 0.4)), 4)
            fee_ccy = "TRX" if amount > 0 and rng.random() < 0.5 else "USDT"
            fee = round(abs(amount) * 0.001, 4)
            deal = Deal(
                f"d{i}", "o", T0 + np.timedelta64(i, "s"), amount, price, True, fee_amount=fee, fee_currency=fee_ccy
            )
            ledger.update_position_by_deal(deal)
            held.reconcile_size(ledger.quantity, held.position_avg_price)
            held.book_accounting(deal)
            assert held.quantity == ledger.quantity
            assert held.position_avg_price == pytest.approx(ledger.position_avg_price)
            assert held.r_pnl == pytest.approx(ledger.r_pnl)
            assert held.commissions == pytest.approx(ledger.commissions)


def test_a_short_with_no_wallet_total_sizes_the_position():
    am, _ = _pm()
    _mark(am, TRX, 0.30)
    _snapshot(am, T0, _bal("TRX", -16.0, total=0.0))
    assert am.get_position(TRX).quantity == -16.0


def test_a_holding_adopted_before_its_first_mark_is_averaged_at_that_mark():
    am, _ = _pm()
    am.get_position(TRX)
    _snapshot(am, T0, _bal("TRX", 60.0))
    pos = am.get_position(TRX)
    assert pos.quantity == 60.0 and pos.position_avg_price == 0.0
    am.on_market_quote(TRX, Quote(T0, 0.30, 0.32, 1.0, 1.0))
    assert pos.position_avg_price == pytest.approx(0.31)
    assert pos.unrealized_pnl() == pytest.approx(0.0)
