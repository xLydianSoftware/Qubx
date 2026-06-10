"""Regression test for the backtester funding-payment routing.

Market data rides (instrument, d_type, data, is_historical) tuples through process_data, but a
funding payment is an ACCOUNT event (the AccountManager books it into position PnL/balances), so a
live (non-warmup) funding payment must ride the typed channel as a FundingPaymentEvent — mirroring
the live CCXT funding handler. If it rode a bare tuple it would dead-end in _process_custom_event and
simulated funding would silently stop affecting PnL (the regression this test guards against).
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
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    payment = FundingPayment(time=0, funding_rate=0.0001, funding_interval_hours=8)

    runner._send_market_data(instr, DataType.FUNDING_PAYMENT, payment, is_hist=False)

    (sent,), _ = runner.channel.send.call_args
    assert isinstance(sent, FundingPaymentEvent)
    assert sent.instrument is instr
    assert sent.payment is payment


def test_warmup_funding_payment_stays_on_tuple_path():
    # Historical/warmup funding is NOT booked into the account (matches main); it rides the
    # cache-only tuple path.
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    payment = FundingPayment(time=0, funding_rate=0.0001, funding_interval_hours=8)

    runner._send_market_data(instr, DataType.FUNDING_PAYMENT, payment, is_hist=True)

    (sent,), _ = runner.channel.send.call_args
    assert sent == (instr, DataType.FUNDING_PAYMENT, payment, True)


def test_market_data_rides_tuple_path():
    runner = _runner()
    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)

    runner._send_market_data(instr, DataType.QUOTE, quote, is_hist=False)

    (sent,), _ = runner.channel.send.call_args
    assert sent == (instr, DataType.QUOTE, quote, False)
