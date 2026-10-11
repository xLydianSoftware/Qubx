"""One backtest trading BINANCE spot and BINANCE.UM perps: long spot / short perp carry.

Funding is booked on the perp only. With ``unified_margin={"BINANCE": "BINANCE.UM"}`` both legs
book into one account (a portfolio-margin account): one capital pool holding the spot purchase
and the perp PnL; without it each exchange is a separate account with half the capital.
"""

import numpy as np
import pandas as pd
import pytest

from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType
from qubx.core.interfaces import IStrategy, IStrategyContext, TriggerEvent
from qubx.data.storages.handy import HandyStorage

CAPITAL = 100_000.0
QTY = 0.5


class Carry(IStrategy):
    def on_init(self, ctx) -> None:
        ctx.set_base_subscription(DataType.OHLC["1h"])
        ctx.set_event_schedule("0 * * * *")

    def _legs(self, ctx: IStrategyContext):
        spot = next(i for i in ctx.instruments if i.exchange == "BINANCE")
        perp = next(i for i in ctx.instruments if i.exchange == "BINANCE.UM")
        return spot, perp

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
        spot, perp = self._legs(ctx)
        if ctx.positions[spot].quantity < spot.min_size:
            ctx.trade(spot, QTY)
        if ctx.positions[perp].quantity > -perp.min_size:
            ctx.trade(perp, -QTY)
        return []

    def on_stop(self, ctx: IStrategyContext) -> None:
        spot, perp = self._legs(ctx)
        am = ctx.account
        self.result = {
            "spot_qty": ctx.positions[spot].quantity,
            "perp_qty": ctx.positions[perp].quantity,
            "spot_funding": ctx.positions[spot].cumulative_funding,
            "perp_funding": ctx.positions[perp].cumulative_funding,
            "shared": am.get_state("BINANCE") is am.get_state("BINANCE.UM"),
            "capital": am.get_total_capital(),
            "capital_spot": am.get_total_capital("BINANCE"),
            "capital_perp": am.get_total_capital("BINANCE.UM"),
            "pnl": ctx.positions[spot].pnl + ctx.positions[perp].pnl,
            "spot_commissions": ctx.positions[spot].commissions,
            "perp_commissions": ctx.positions[perp].commissions,
        }


def _storage() -> HandyStorage:
    idx = pd.date_range("2023-12-30", "2024-01-05", freq="1h", inclusive="left")
    rs = np.random.RandomState(7)
    close = 50_000 + np.cumsum(rs.normal(0, 20, len(idx)))

    def ohlc(px):
        return pd.DataFrame({"open": px, "high": px + 30, "low": px - 30, "close": px, "volume": 100.0}, index=idx)

    fidx = pd.date_range("2023-12-30", "2024-01-05", freq="8h", inclusive="left")
    funding = pd.DataFrame({"funding_rate": 0.0001, "funding_interval_hours": 8.0}, index=fidx)
    funding.index.name = "timestamp"
    storage = HandyStorage({"BINANCE:SPOT:BTCUSDT": ohlc(close), "BINANCE.UM:SWAP:BTCUSDT": ohlc(close + 5.0)})
    storage.get_reader("BINANCE.UM", "SWAP").add("BTCUSDT", "funding_payment", funding)
    return storage


def _run(**kwargs) -> dict:
    strategy = Carry()
    simulate(
        {"carry": strategy},
        data=_storage(),
        capital=CAPITAL,
        instruments=["BINANCE:BTCUSDT", "BINANCE.UM:BTCUSDT"],
        commissions="vip0_usdt",
        start="2024-01-01",
        stop="2024-01-04",
        enable_funding=True,
        silent=True,
        debug="ERROR",
        n_jobs=1,
        **kwargs,
    )
    return strategy.result


def test_unified_margin_books_both_legs_into_one_account():
    r = _run(unified_margin={"BINANCE": "BINANCE.UM"})
    assert r["shared"]
    assert r["spot_qty"] == pytest.approx(QTY)
    assert r["perp_qty"] == pytest.approx(-QTY)
    # - funding on the perp only: a short receives a positive rate
    assert r["perp_funding"] > 0
    assert r["spot_funding"] == 0.0
    # - one pool: whichever exchange is named, the same capital — counted once in the total
    assert r["capital"] == r["capital_spot"] == r["capital_perp"]
    # - the whole capital plus both legs' PnL (perp PnL carries the funding) less commissions on both legs
    fees = r["spot_commissions"] + r["perp_commissions"]
    assert fees > 0
    assert r["capital"] == pytest.approx(CAPITAL + r["pnl"] - fees, abs=0.01)


def test_separate_accounts_split_capital():
    r = _run()
    assert not r["shared"]
    assert r["perp_funding"] > 0 and r["spot_funding"] == 0.0
    assert r["capital"] == pytest.approx(r["capital_spot"] + r["capital_perp"])
    # - each account started with half; the spot account holds the coin, the perp account the margin
    assert r["capital_spot"] == pytest.approx(CAPITAL / 2, rel=0.01)
    assert r["capital_perp"] == pytest.approx(CAPITAL / 2, rel=0.01)
