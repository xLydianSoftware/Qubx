"""
Tests for Position "episodes" - the span from one flat->open transition to the next return to flat.

Episodes are a view over the lifetime accumulators (r_pnl / commissions / cumulative_funding): the
four baseline fields (episode_start_time, r_pnl_at_open, commissions_at_open, cumulative_funding_at_open)
are stamped synchronously in the deal-processing path so strategies can read P&L scoped to the currently
open position, entry costs included.

Design: docs/specs/2026-07-21-position-episodes-design.md
Impl:   docs/specs/2026-07-21-position-episodes-impl.md
"""

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from qubx.core.basics import Instrument, MarketType, Position
from qubx.core.series import time_as_nsec

TIME = lambda x: pd.Timestamp(x).asm8  # noqa: E731


@pytest.fixture
def instr() -> Instrument:
    return Instrument(
        symbol="BTCUSDT",
        market_type=MarketType.SWAP,
        exchange="binance",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol="BTCUSDT",
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


class TestPositionEpisodes:
    # 1
    def test_open_from_flat_stamps_pre_deal(self, instr):
        p = Position(instr)
        assert not p.is_open()

        t0 = TIME("2025-01-08 00:00:00")
        p.update_position(t0, position=1.0, exec_price=50000.0, fee_amount=5.0)

        # episode stamped at the opening deal, baselines taken from the PRE-deal accumulators
        assert p.episode_start_time == t0
        assert p.r_pnl_at_open == 0.0
        assert p.commissions_at_open == 0.0  # stamped BEFORE the opening fee is booked
        assert p.cumulative_funding_at_open == 0.0

        # the entry fee lands inside the episode -> episode_net_pnl ~ -fee right after open
        assert p.episode_pnl() == approx(0.0)
        assert p.episode_commissions() == approx(5.0)
        assert p.episode_net_pnl() == approx(-5.0)

    # 2
    def test_partial_trim_and_add_keep_baselines(self, instr):
        p = Position(instr)
        t0 = TIME("2025-01-08 00:00:00")
        p.update_position(t0, position=2.0, exec_price=50000.0, fee_amount=4.0)

        start, r_open, c_open, f_open = (
            p.episode_start_time,
            p.r_pnl_at_open,
            p.commissions_at_open,
            p.cumulative_funding_at_open,
        )

        # partial trim (same-side reduce) -> no re-stamp
        p.change_position_by(TIME("2025-01-08 01:00:00"), amount=-1.0, exec_price=51000.0, fee_amount=2.0)
        assert p.episode_start_time == start
        assert (p.r_pnl_at_open, p.commissions_at_open, p.cumulative_funding_at_open) == (r_open, c_open, f_open)

        # add (same-side increase) -> no re-stamp
        p.change_position_by(TIME("2025-01-08 02:00:00"), amount=2.0, exec_price=52000.0, fee_amount=3.0)
        assert p.episode_start_time == start
        assert (p.r_pnl_at_open, p.commissions_at_open, p.cumulative_funding_at_open) == (r_open, c_open, f_open)

    # 3
    def test_full_close_preserves_final_episode(self, instr):
        p = Position(instr)
        t0 = TIME("2025-01-08 00:00:00")
        p.update_position(t0, position=1.0, exec_price=50000.0, fee_amount=5.0)

        # full close (long 1 @ 50000 -> flat @ 51000): +1000 realized
        p.update_position(TIME("2025-01-08 01:00:00"), position=0.0, exec_price=51000.0, fee_amount=5.0)

        assert not p.is_open()
        # closed episode remains readable via the accessors (no re-stamp on close)
        assert p.episode_start_time == t0
        assert p.r_pnl_at_open == 0.0
        assert p.commissions_at_open == 0.0
        assert p.episode_pnl() == approx(1000.0)  # realized since open
        assert p.episode_commissions() == approx(10.0)  # entry + exit fee
        assert p.episode_net_pnl() == approx(990.0)

    # 4
    def test_reopen_restamps(self, instr):
        p = Position(instr)
        p.update_position(TIME("2025-01-08 00:00:00"), position=1.0, exec_price=50000.0, fee_amount=5.0)
        p.update_position(TIME("2025-01-08 01:00:00"), position=0.0, exec_price=51000.0, fee_amount=5.0)

        t_reopen = TIME("2025-01-08 02:00:00")
        p.update_position(t_reopen, position=1.0, exec_price=52000.0, fee_amount=5.0)

        # new episode: baselines capture everything realized in the closed episode
        assert p.episode_start_time == t_reopen
        assert p.r_pnl_at_open == approx(1000.0)  # closed episode's realized rolls into the baseline
        assert p.commissions_at_open == approx(10.0)  # prior fees excluded from the new episode
        assert p.episode_pnl() == approx(0.0)
        assert p.episode_commissions() == approx(5.0)  # only the reopen fee
        assert p.episode_net_pnl() == approx(-5.0)  # fresh entry honestly starts at -fee

    # 5
    def test_sign_flip_splits_pnl_and_fee(self, instr):
        p = Position(instr)
        p.update_position(TIME("2025-01-08 00:00:00"), position=4.0, exec_price=50000.0, fee_amount=4.0)

        # flip long 4 -> short 2 in one deal @ 51000, total fee 8
        #   closing half |4|, opening half |2| -> fee_closing = 8 * 4/6, fee_opening = 8 * 2/6
        t_flip = TIME("2025-01-08 01:00:00")
        p.update_position(t_flip, position=-2.0, exec_price=51000.0, fee_amount=8.0)

        fee_closing = 8.0 * 4.0 / 6.0
        fee_opening = 8.0 - fee_closing

        # flip always re-stamps; closing pnl (+4000) and fee_closing belong to the OLD episode (in baselines)
        assert p.episode_start_time == t_flip
        assert p.r_pnl_at_open == approx(4000.0)
        assert p.commissions_at_open == approx(4.0 + fee_closing)  # entry fee + closing share
        # new (short) episode starts at ~ -fee_opening
        assert p.episode_pnl() == approx(0.0)
        assert p.episode_commissions() == approx(fee_opening)
        assert p.episode_net_pnl() == approx(-fee_opening)

    # 6
    def test_funding_while_flat_attributed_to_old_episode(self, instr):
        p = Position(instr)
        p.update_position(TIME("2025-01-08 00:00:00"), position=1.0, exec_price=50000.0, fee_amount=0.0)
        start = p.episode_start_time

        # funding accrues DURING the episode -> attributed to it; a mid-episode settle never re-stamps
        p.apply_funding_payment(TIME("2025-01-08 04:00:00"), -5.0)  # long pays
        assert p.cumulative_funding == approx(-5.0)
        assert p.episode_funding() == approx(-5.0)
        assert p.episode_start_time == start  # book-without-size-change does not stamp

        # close, then settle again while flat: a settlement is account truth even at qty=0, so it books
        # into the lifetime accumulator and lands (baselines unchanged) in the just-closed episode.
        p.update_position(TIME("2025-01-08 08:00:00"), position=0.0, exec_price=50000.0, fee_amount=0.0)
        assert p.apply_funding_payment(TIME("2025-01-08 08:00:01"), -2.0) == -2.0
        assert p.cumulative_funding == approx(-7.0)
        assert p.episode_start_time == start  # close + flat settle never stamp

        # reopen -> the prior episode's funding (incl. the while-flat settle) is baked into the baseline;
        # the new episode's funding starts at 0.
        t_reopen = TIME("2025-01-08 09:00:00")
        p.update_position(t_reopen, position=1.0, exec_price=50000.0, fee_amount=0.0)
        assert p.episode_start_time == t_reopen
        assert p.cumulative_funding_at_open == approx(-7.0)
        assert p.episode_funding() == approx(0.0)

    # 7
    def test_realize_only_never_stamps(self, instr):
        # realize_only books the closing pnl + fee of a stale deal but leaves size/avg to a snapshot
        # reconcile (situation II). It must NEVER stamp; the deltas flow into the CURRENT episode's
        # lifetime accumulators (the correct recovery semantic).
        p = Position(instr)
        t0 = TIME("2025-01-08 00:00:00")
        p.update_position(t0, position=2.0, exec_price=50000.0, fee_amount=4.0)

        start = p.episode_start_time
        r_open, c_open, f_open = p.r_pnl_at_open, p.commissions_at_open, p.cumulative_funding_at_open
        prev_qty = p.quantity
        assert p.episode_commissions() == approx(4.0)

        # recover a stale partial-close deal (1 @ 51000, fee 2) already reflected in the venue size
        deal_pnl, comms = p.change_position_by(
            TIME("2025-01-08 01:00:00"), amount=-1.0, exec_price=51000.0, fee_amount=2.0, realize_only=True
        )

        # size untouched (snapshot owns it), and NO stamp happened
        assert p.quantity == prev_qty
        assert p.episode_start_time == start
        assert (p.r_pnl_at_open, p.commissions_at_open, p.cumulative_funding_at_open) == (r_open, c_open, f_open)

        # the recovered pnl + fee landed inside the current episode's lifetime accumulators
        assert deal_pnl == approx(1000.0)
        assert comms == approx(2.0)
        assert p.r_pnl == approx(1000.0)
        assert p.episode_commissions() == approx(6.0)  # entry 4 + recovered 2

    # 8
    def test_reconcile_size_flat_to_open_stamps_at_now(self, instr):
        # An authoritative venue snapshot that takes a flat position to open (first-connect recovery of
        # a pre-existing position whose true opening was never observed) stamps an episode "at now" from
        # the current accumulators.
        p = Position(instr)
        assert not p.is_open()

        t_recon = TIME("2025-01-08 12:00:00")
        p.reconcile_size(0.5, 60000.0, timestamp=t_recon)

        assert p.is_open()
        assert p.episode_start_time == t_recon
        assert p.r_pnl_at_open == approx(0.0)
        assert p.commissions_at_open == approx(0.0)
        assert p.cumulative_funding_at_open == approx(0.0)

        # a subsequent reconcile that does NOT cross flat->open (open -> resized open) must not re-stamp
        p.reconcile_size(0.75, 60500.0, timestamp=TIME("2025-01-08 13:00:00"))
        assert p.episode_start_time == t_recon

    # 9
    def test_reset_clears_reset_by_position_copies_flatten_keeps(self, instr):
        # reset() clears the four episode fields
        p = Position(instr)
        p.update_position(TIME("2025-01-08 00:00:00"), position=1.0, exec_price=50000.0, fee_amount=5.0)
        assert not np.isnat(p.episode_start_time)
        p.reset()
        assert np.isnat(p.episode_start_time)
        assert p.r_pnl_at_open == 0.0
        assert p.commissions_at_open == 0.0
        assert p.cumulative_funding_at_open == 0.0

        # reset_by_position() copies the four episode fields
        src = Position(instr)
        src.update_position(TIME("2025-01-08 00:00:00"), position=2.0, exec_price=50000.0, fee_amount=4.0)
        dst = Position(instr)
        dst.reset_by_position(src)
        assert dst.episode_start_time == src.episode_start_time
        assert dst.r_pnl_at_open == src.r_pnl_at_open
        assert dst.commissions_at_open == src.commissions_at_open
        assert dst.cumulative_funding_at_open == src.cumulative_funding_at_open

        # flatten() ends the episode via the flat predicate but PRESERVES the episode baselines
        q = Position(instr)
        t0 = TIME("2025-01-08 00:00:00")
        q.update_position(t0, position=1.0, exec_price=50000.0, fee_amount=5.0)
        q.update_market_price(time_as_nsec(TIME("2025-01-08 01:00:00")), 51000.0, 1.0)
        q.flatten()
        assert not q.is_open()
        assert q.episode_start_time == t0  # preserved through the flatten-to-flat
        assert q.r_pnl_at_open == 0.0
        assert q.commissions_at_open == 0.0

    # 10
    def test_legacy_zero_baselines_degrade_to_lifetime(self, instr):
        # accumulate lifetime state, then force legacy defaults (zero baselines / NaT start)
        p = Position(instr)
        p.update_position(TIME("2025-01-08 00:00:00"), position=2.0, exec_price=50000.0, fee_amount=4.0)
        p.apply_funding_payment(TIME("2025-01-08 04:00:00"), -5.0)
        p.update_market_price(time_as_nsec(TIME("2025-01-08 05:00:00")), 51000.0, 1.0)

        p.episode_start_time = np.datetime64("NaT")
        p.r_pnl_at_open = 0.0
        p.commissions_at_open = 0.0
        p.cumulative_funding_at_open = 0.0

        # with zero baselines every episode accessor equals its lifetime counterpart
        assert p.episode_pnl() == approx(p.pnl)
        assert p.episode_funding() == approx(p.cumulative_funding)
        assert p.episode_commissions() == approx(p.commissions)
        assert p.episode_price_pnl() == approx(p.get_total_price_pnl())
        assert p.episode_net_pnl() == approx(p.pnl - p.commissions)

    # 11
    def test_constructor_open_position_stamps_episode_at_init(self, instr):
        # with explicit episode kwargs -> assigned verbatim (restorer round-trip path)
        est = TIME("2025-01-01 00:00:00")
        p = Position(
            instr,
            quantity=1.0,
            pos_average_price=50000.0,
            r_pnl=100.0,
            cumulative_funding=-5.0,
            commissions=3.0,
            episode_start_time=est,
            r_pnl_at_open=90.0,
            commissions_at_open=2.0,
            cumulative_funding_at_open=-4.0,
        )
        assert p.episode_start_time == est
        assert p.r_pnl_at_open == 90.0
        assert p.commissions_at_open == 2.0
        assert p.cumulative_funding_at_open == -4.0

        # constructed open WITHOUT episode kwargs -> episode-at-init from the supplied lifetime accumulators
        p2 = Position(
            instr, quantity=1.0, pos_average_price=50000.0, r_pnl=100.0, cumulative_funding=-5.0, commissions=3.0
        )
        assert np.isnat(p2.episode_start_time)  # opening never observed
        assert p2.r_pnl_at_open == 100.0
        assert p2.commissions_at_open == 3.0
        assert p2.cumulative_funding_at_open == -5.0

        # flat construction leaves the zero defaults
        p3 = Position(instr)
        assert np.isnat(p3.episode_start_time)
        assert p3.r_pnl_at_open == 0.0
        assert p3.commissions_at_open == 0.0
        assert p3.cumulative_funding_at_open == 0.0

    # 13
    def test_simulation_open_funding_trim_close_reopen(self, instr):
        p = Position(instr)

        # --- open long 2 @ 50000, fee 4 ---
        t0 = TIME("2025-01-08 00:00:00")
        p.update_position(t0, position=2.0, exec_price=50000.0, fee_amount=4.0)
        assert p.episode_start_time == t0
        assert p.episode_pnl() == approx(0.0)
        assert p.episode_commissions() == approx(4.0)
        assert p.episode_net_pnl() == approx(-4.0)

        # --- price moves to 51000 ---
        p.update_market_price(time_as_nsec(TIME("2025-01-08 00:30:00")), 51000.0, 1.0)
        assert p.episode_price_pnl() == approx(2000.0)  # 2 * (51000 - 50000)
        assert p.episode_net_pnl() == approx(1996.0)

        # --- funding settle (long pays): -(2 * 51000 * 0.0001) = -10.2 ---
        p.apply_funding_payment(TIME("2025-01-08 01:00:00"), -10.2)
        assert p.episode_funding() == approx(-10.2)
        assert p.episode_price_pnl() == approx(2000.0)  # price component unchanged
        assert p.episode_pnl() == approx(1989.8)
        assert p.episode_start_time == t0  # funding never re-stamps

        # --- trim to long 1 @ 51000, fee 2 (partial close, realizes +1000) ---
        p.update_position(TIME("2025-01-08 02:00:00"), position=1.0, exec_price=51000.0, fee_amount=2.0)
        assert p.episode_start_time == t0
        assert p.episode_commissions() == approx(6.0)
        assert p.episode_funding() == approx(-10.2)
        assert p.episode_pnl() == approx(1989.8)  # total episode P&L unaffected by realizing part of it

        # --- full close to 0 @ 52000, fee 2 (realizes +2000 more) ---
        p.update_position(TIME("2025-01-08 03:00:00"), position=0.0, exec_price=52000.0, fee_amount=2.0)
        assert not p.is_open()
        assert p.episode_start_time == t0
        assert p.episode_commissions() == approx(8.0)
        assert p.episode_pnl() == approx(2989.8)  # 1000 + 2000 price - 10.2 funding
        assert p.episode_net_pnl() == approx(2981.8)

        # --- reopen long 1 @ 52000, fee 3 -> fresh episode ---
        t_reopen = TIME("2025-01-08 04:00:00")
        p.update_position(t_reopen, position=1.0, exec_price=52000.0, fee_amount=3.0)
        assert p.episode_start_time == t_reopen
        assert p.r_pnl_at_open == approx(2989.8)
        assert p.cumulative_funding_at_open == approx(-10.2)
        assert p.episode_pnl() == approx(0.0)
        assert p.episode_funding() == approx(0.0)  # prior funding excluded from the new episode
        assert p.episode_commissions() == approx(3.0)
        assert p.episode_net_pnl() == approx(-3.0)
