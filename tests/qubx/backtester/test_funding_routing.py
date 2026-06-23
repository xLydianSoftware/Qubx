"""Regression test for the backtester funding-payment routing.

A live (non-warmup) funding payment is dual-emitted, mirroring the live CCXT funding handler: the
FundingPaymentEvent rides the typed channel FIRST (the AccountManager books it into position
PnL/balances), then the same payment rides a (instrument, d_type, data, is_historical) tuple so the
strategy still reacts in on_market_data. Event-first preserves book-before-react ordering. Warmup
funding is never booked (matches main) — it rides the cache-only tuple path alone.
"""

from unittest.mock import MagicMock

from qubx.backtester.runner import SimulationRunner
from qubx.core.basics import DataType, FundingPayment
from qubx.core.events import FundingPaymentEvent
from qubx.core.lookups import lookup
from qubx.core.series import Quote


def _runner() -> SimulationRunner:
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.channel = MagicMock()
    return runner


def test_live_funding_payment_rides_typed_account_channel():
    # Dual-emit, mirroring the live CCXT handler: the FundingPaymentEvent rides the typed channel
    # FIRST (the AccountManager books it), then the same payment rides a tuple so the strategy still
    # reacts in on_market_data. Event-first preserves book-before-react ordering.
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    payment = FundingPayment(time=0, funding_rate=0.0001, funding_interval_hours=8)

    runner._send_market_data(instr, DataType.FUNDING_PAYMENT, payment, is_hist=False)

    assert runner.channel.send.call_count == 2
    (first_arg,), _ = runner.channel.send.call_args_list[0]
    (second_arg,), _ = runner.channel.send.call_args_list[1]
    assert isinstance(first_arg, FundingPaymentEvent)
    assert first_arg.instrument is instr
    assert first_arg.payment is payment
    assert second_arg == (instr, DataType.FUNDING_PAYMENT, payment, False)


def test_warmup_funding_payment_stays_on_tuple_path():
    # Historical/warmup funding is NOT booked into the account (matches main); it rides the
    # cache-only tuple path.
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    payment = FundingPayment(time=0, funding_rate=0.0001, funding_interval_hours=8)

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
