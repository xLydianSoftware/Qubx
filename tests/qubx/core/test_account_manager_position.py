from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager, reducer
from qubx.core.basics import (
    Balance,
    Deal,
    Instrument,
    MarketType,
    Position,
)
from qubx.core.events import FundingPaymentEvent


class _T:
    def __init__(self, t="2026-05-28T00:00:00"):
        self.t = np.datetime64(t)

    def time(self):
        return self.t


def _instrument(symbol="BTCUSDT", exchange="binance") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _spot_instrument(symbol="BTCUSDT", exchange="binance") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SPOT,
        exchange=exchange,
        base=symbol.replace("USDT", ""),
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
        contract_size=1.0,
    )


def _am(exchange="binance"):
    return AccountManager(
        connectors={exchange: MagicMock()},
        base_currencies={exchange: "USDT"},
        time=_T(),
        account_id="test",
    )


def _fill(trade_id="t1", amount=0.5, price=50_000.0):
    return Deal(
        trade_id=trade_id,
        order_id="V1",
        time=np.datetime64("2026-05-28T00:00:00"),
        amount=amount,
        price=price,
        aggressive=True,
    )


T_SETTLE = np.datetime64("2026-05-28T00:00:00")


def test_funding_payment_books_amount_and_cash():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))

    am.apply(FundingPaymentEvent(instrument=inst, time=T_SETTLE, amount=-5.0))
    assert abs(pos.cumulative_funding - (-5.0)) < 1e-9
    assert abs(state.get_balance("USDT").total - 995.0) < 1e-9


def test_funding_payment_duplicate_skipped():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))

    am.apply(FundingPaymentEvent(instrument=inst, time=T_SETTLE, amount=-5.0))
    funding_after_first = pos.cumulative_funding
    balance_after_first = state.get_balance("USDT").total
    # a second delivery of the same settlement (same settle hour) is a no-op
    am.apply(FundingPaymentEvent(instrument=inst, time=T_SETTLE, amount=-5.0))
    assert pos.cumulative_funding == funding_after_first
    assert state.get_balance("USDT").total == balance_after_first


def test_funding_payment_moves_free_and_total_together():
    # Regression for I4: funding affects free cash; bal.free must move by the
    # same amount as bal.total (Balance invariant free == total - locked).
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=900.0, locked=100.0))

    am.apply(FundingPaymentEvent(instrument=inst, time=T_SETTLE, amount=-5.0))
    bal = state.get_balance("USDT")
    assert abs(bal.total - 995.0) < 1e-9
    assert abs(bal.free - 895.0) < 1e-9
    assert bal.locked == 100.0


def test_funding_payment_different_bucket_applies_again():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))

    am.apply(FundingPaymentEvent(instrument=inst, time=T_SETTLE, amount=-5.0))
    first = pos.cumulative_funding
    am.apply(FundingPaymentEvent(instrument=inst, time=T_SETTLE + np.timedelta64(8, "h"), amount=-5.0))
    assert pos.cumulative_funding != first


def test_futures_realized_pnl_folds_into_total_capital():
    # A futures round-trip realizing +X must credit the settle balance by +X, and
    # get_total_capital() (seeded capital + realized) must reflect it.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=100_000.0, free=100_000.0))

    # open long 1.0 @ 50k
    reducer._book_deal(state, inst, _fill(trade_id="t1", amount=1.0, price=50_000.0))
    # opening a futures position does not touch cash
    assert state.get_balance("USDT").total == 100_000.0
    assert am.get_total_capital("binance") == 100_000.0

    # close 1.0 @ 60k -> realized +10k
    reducer._book_deal(state, inst, _fill(trade_id="t2", amount=-1.0, price=60_000.0))
    bal = state.get_balance("USDT")
    assert abs(bal.total - 110_000.0) < 1e-6
    assert abs(bal.free - 110_000.0) < 1e-6
    assert abs(am.get_total_capital("binance") - 110_000.0) < 1e-6


def test_spot_fill_credits_base_and_debits_quote():
    # A spot buy debits the quote currency by the notional and credits the base asset;
    # total capital is unchanged (cash converted to a holding of equal value).
    am = _am()
    state = am._states["binance"]
    inst = _spot_instrument()
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=100_000.0, free=100_000.0))

    deal = _fill(trade_id="t1", amount=0.5, price=100_000.0)
    reducer._book_deal(state, inst, deal)
    # mark the position so its market value contributes to total capital
    state.get_position(inst).update_market_price(am._time.time(), 100_000.0, 1.0)

    assert abs(state.get_balance("USDT").total - 50_000.0) < 1e-6
    assert abs(state.get_balance("USDT").free - 50_000.0) < 1e-6
    assert abs(state.get_balance("BTC").free - 0.5) < 1e-9
    assert abs(am.get_total_capital("binance") - 100_000.0) < 1e-6
