"""Funding routing in simulation: market tuples + SimFundingBooker.

Market funding rides (instrument, d_type, data, is_historical) tuples like any market data;
the runner no longer dual-emits. Booking into the simulated account is SimFundingBooker's
job (wired into the ProcessingManager's funding_payment handler), account-scoped by
construction: events are produced only when our position is open. Warmup funding is never
booked — it rides the cache-only hist tuple path.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from qubx.backtester.funding import SimFundingBooker
from qubx.backtester.runner import SimulationRunner
from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType, FundingPayment, Position
from qubx.core.events import FundingPaymentEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, TriggerEvent
from qubx.core.lookups import lookup
from qubx.core.series import Quote
from qubx.data.storages.handy import HandyStorage


def _runner() -> SimulationRunner:
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.channel = MagicMock()
    return runner


def _payment() -> FundingPayment:
    return FundingPayment(time=0, funding_rate=0.0001, funding_interval_hours=8)


def test_live_funding_payment_rides_tuple_path_only():
    # The runner emits funding as a plain market tuple — no typed event; booking is
    # SimFundingBooker's job inside the processing manager's funding_payment handler.
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    payment = _payment()

    runner._send_market_data(instr, DataType.FUNDING_PAYMENT, payment, is_hist=False)

    assert runner.channel.send.call_count == 1
    (sent,), _ = runner.channel.send.call_args
    assert sent == (instr, DataType.FUNDING_PAYMENT, payment, False)


def test_warmup_funding_payment_stays_on_tuple_path():
    # Historical/warmup funding is NOT booked into the account; it rides the cache-only tuple path.
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    payment = _payment()

    runner._send_market_data(instr, DataType.FUNDING_PAYMENT, payment, is_hist=True)

    assert runner.channel.send.call_count == 1
    (sent,), _ = runner.channel.send.call_args
    assert sent == (instr, DataType.FUNDING_PAYMENT, payment, True)


def test_market_data_rides_tuple_path():
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)

    runner._send_market_data(instr, DataType.QUOTE, quote, is_hist=False)

    assert runner.channel.send.call_count == 1
    (sent,), _ = runner.channel.send.call_args
    assert sent == (instr, DataType.QUOTE, quote, False)


class TestSimFundingBooker:
    def test_open_position_yields_sim_event(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None
        account = MagicMock()
        account.get_position.return_value = Position(instrument=instr, quantity=0.5, pos_average_price=50_000.0)
        payment = _payment()

        event = SimFundingBooker(account).on_market_funding(instr, payment)

        assert isinstance(event, FundingPaymentEvent)
        assert event.instrument is instr
        assert event.payment is payment
        assert event.amount is None  # sim has no venue cash truth -> computed booking

    def test_flat_position_yields_nothing(self):
        # account-scoped by construction: no open position -> no account event
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None
        account = MagicMock()
        account.get_position.return_value = Position(instrument=instr, quantity=instr.min_size / 2)

        assert SimFundingBooker(account).on_market_funding(instr, _payment()) is None

    def test_missing_position_record_yields_nothing(self):
        instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
        assert instr is not None
        account = MagicMock()
        account.get_position.return_value = None

        assert SimFundingBooker(account).on_market_funding(instr, _payment()) is None


class FundingHold(IStrategy):
    """Opens 0.5 BTC at the first hourly event and holds across funding settlements."""

    def on_init(self, ctx) -> None:
        ctx.set_base_subscription(DataType.OHLC["1h"])
        ctx.set_event_schedule("0 * * * *")

    def on_start(self, ctx: IStrategyContext) -> None:
        self.observed: list[dict] = []

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
        instr = ctx.instruments[0]
        if abs(ctx.positions[instr].quantity) < instr.min_size:
            ctx.trade(instr, 0.5)
        return []

    def on_market_data(self, ctx: IStrategyContext, event) -> None:
        if str(event.type) != "funding_payment":
            return
        # book-before-react: by the time the strategy sees the funding tuple the payment is
        # already in cumulative_funding, and the mark has not moved since booking
        pos = ctx.positions[ctx.instruments[0]]
        self.observed.append(
            {
                "cf": pos.cumulative_funding,
                "qty": pos.quantity,
                "mark": pos.last_update_price,
                "rate": event.data.funding_rate,
            }
        )

    def on_stop(self, ctx: IStrategyContext) -> None:
        pos = ctx.positions[ctx.instruments[0]]
        self.final_cumulative_funding = pos.cumulative_funding
        self.final_n_payments = len(pos.funding_payments)


def test_backtest_books_funding_for_held_position():
    # End-to-end: market funding tuples from simulated data are booked into the simulated
    # account (computed: -qty * mark * rate) exactly once per settlement while the position
    # is open, and still reach the strategy's on_market_data.
    idx = pd.date_range("2023-12-30", "2024-01-05", freq="1h", inclusive="left")
    rs = np.random.RandomState(42)
    close = 50_000 + np.cumsum(rs.normal(0, 20, len(idx)))
    ohlc = pd.DataFrame(
        {"open": close, "high": close + 30, "low": close - 30, "close": close, "volume": 100.0}, index=idx
    )
    fidx = pd.date_range("2023-12-30", "2024-01-05", freq="8h", inclusive="left")
    funding = pd.DataFrame(
        {
            "funding_rate": np.where(np.arange(len(fidx)) % 3 == 0, 0.0001, -0.00005),
            "funding_interval_hours": 8.0,
        },
        index=fidx,
    )
    funding.index.name = "timestamp"
    storage = HandyStorage({"BTCUSDT": ohlc}, exchange="BINANCE.UM")
    storage.get_reader("BINANCE.UM", "SWAP").add("BTCUSDT", "funding_payment", funding)

    strategy = FundingHold()
    simulate(
        {"funding_hold": strategy},
        data=storage,
        capital=100_000,
        instruments=["BINANCE.UM:BTCUSDT"],
        commissions="vip0_usdt",
        start="2024-01-01",
        stop="2024-01-04",
        enable_funding=True,
        silent=True,
        debug="ERROR",
        n_jobs=1,
    )

    booked = [o for o in strategy.observed if abs(o["qty"]) >= 0.001]
    assert len(booked) > 0
    assert strategy.final_n_payments == len(booked)

    expected = sum(-o["qty"] * o["mark"] * o["rate"] for o in booked)
    assert strategy.final_cumulative_funding == pytest.approx(expected)
    assert strategy.final_cumulative_funding != 0.0

    # cumulative deltas between consecutive funding observations match each computed payment
    prev_cf = 0.0
    for o in booked:
        assert o["cf"] - prev_cf == pytest.approx(-o["qty"] * o["mark"] * o["rate"])
        prev_cf = o["cf"]
