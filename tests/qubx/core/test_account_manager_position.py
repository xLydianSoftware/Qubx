from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager, AccountManagerConfig
from qubx.core.basics import (
    Balance,
    Deal,
    FundingPayment,
    Instrument,
    MarketType,
    Order,
    OrderOrigin,
    OrderStatus,
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
    am = AccountManager.__new__(AccountManager)
    am._init_state(
        connectors={exchange: MagicMock()}, strategy=MagicMock(), time=_T(),
        cfg=AccountManagerConfig(), account_id="test", tcc=None,
    )
    return am


def add_order(state, inst, cid="cid-1", status=OrderStatus.ACCEPTED, qty=1.0):
    state.add_order(
        Order(
            client_order_id=cid,
            venue_order_id="V1",
            origin=OrderOrigin.FRAMEWORK,
            type="LIMIT",
            instrument=inst,
            time=np.datetime64("2026-05-28T00:00:00"),
            quantity=qty,
            price=50_000.0,
            side="BUY",
            status=status,
            time_in_force="gtc",
        )
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


def test_partial_fill_updates_position_quantity_and_avg():
    from qubx.core.events import OrderPartiallyFilledEvent

    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    add_order(state, inst)
    am.apply(
        OrderPartiallyFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(amount=0.5, price=50_000.0)
        )
    )
    pos = state.get_position(inst)
    assert pos is not None
    assert pos.quantity == 0.5
    assert pos.position_avg_price == 50_000.0


def test_two_fills_average_into_position():
    from qubx.core.events import OrderFilledEvent, OrderPartiallyFilledEvent

    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    add_order(state, inst, qty=1.0)
    am.apply(
        OrderPartiallyFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t1", amount=0.5, price=50_000.0)
        )
    )
    am.apply(
        OrderFilledEvent(
            instrument=inst, client_order_id="cid-1", venue_order_id="V1", fill=_fill(trade_id="t2", amount=0.5, price=51_000.0)
        )
    )
    pos = state.get_position(inst)
    assert pos.quantity == 1.0
    assert abs(pos.position_avg_price - 50_500.0) < 1e-6


def test_funding_payment_applied_once_per_bucket():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # open a long position and mark it so funding has a mark price
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(am._time.time(), 50_000.0, 1.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))

    payment = FundingPayment(
        time=np.datetime64("2026-05-28T00:00:00").astype("datetime64[ns]").astype(np.int64),
        funding_rate=0.0001,
        funding_interval_hours=8,
    )
    am.apply(FundingPaymentEvent(instrument=inst, payment=payment))
    # long pays positive funding: cumulative_funding negative
    expected = -(1.0 * 50_000.0 * 0.0001)
    assert abs(pos.cumulative_funding - expected) < 1e-9
    assert abs(state.get_balance("USDT").total - (1000.0 + expected)) < 1e-9


def test_funding_payment_duplicate_skipped():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(am._time.time(), 50_000.0, 1.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))

    payment = FundingPayment(
        time=np.datetime64("2026-05-28T00:00:00").astype("datetime64[ns]").astype(np.int64),
        funding_rate=0.0001,
        funding_interval_hours=8,
    )
    am.apply(FundingPaymentEvent(instrument=inst, payment=payment))
    funding_after_first = pos.cumulative_funding
    balance_after_first = state.get_balance("USDT").total
    # a second payment in the same bucket (same time/interval) is a no-op
    am.apply(FundingPaymentEvent(instrument=inst, payment=payment))
    assert pos.cumulative_funding == funding_after_first
    assert state.get_balance("USDT").total == balance_after_first


def test_simulation_account_manager_constructs_without_pm():
    from qubx.core.account_manager import SimulatedAccountManager

    sam = SimulatedAccountManager(connectors={"binance": object()}, strategy=None, time=_T())
    assert sam._pm is None
    assert "binance" in sam._states
    # position math is inherited and works in the sim variant
    inst = _instrument()
    state = sam._states["binance"]
    deal = _fill(amount=1.0, price=50_000.0)
    sam._apply_deal_to_position(state, inst, deal)
    assert state.get_position(inst).quantity == 1.0


def test_funding_on_unmarked_position_skipped_without_consuming_bucket():
    # Regression for I2: a freshly created position has last_update_price = NaN.
    # Funding must NOT poison balance/cumulative_funding with NaN, and must NOT
    # consume the dedup bucket — so a re-delivered event applies once a mark exists.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    # position with no quote/deal -> last_update_price is NaN
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    assert np.isnan(pos.last_update_price)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))

    payment = FundingPayment(
        time=np.datetime64("2026-05-28T00:00:00").astype("datetime64[ns]").astype(np.int64),
        funding_rate=0.0001,
        funding_interval_hours=8,
    )
    am.apply(FundingPaymentEvent(instrument=inst, payment=payment))
    # nothing applied: no NaN anywhere, bucket not consumed
    assert not np.isnan(state.get_balance("USDT").total)
    assert state.get_balance("USDT").total == 1000.0
    assert not np.isnan(pos.cumulative_funding)
    assert pos.cumulative_funding == 0.0

    # now mark the position and re-deliver the SAME bucket -> it applies this time
    pos.update_market_price(am._time.time(), 50_000.0, 1.0)
    am.apply(FundingPaymentEvent(instrument=inst, payment=payment))
    expected = -(1.0 * 50_000.0 * 0.0001)
    assert abs(pos.cumulative_funding - expected) < 1e-9
    assert abs(state.get_balance("USDT").total - (1000.0 + expected)) < 1e-9


def test_funding_payment_moves_free_and_total_together():
    # Regression for I4: funding affects free cash; bal.free must move by the
    # same amount as bal.total (Balance invariant free == total - locked).
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(am._time.time(), 50_000.0, 1.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=900.0, locked=100.0))

    payment = FundingPayment(
        time=np.datetime64("2026-05-28T00:00:00").astype("datetime64[ns]").astype(np.int64),
        funding_rate=0.0001,
        funding_interval_hours=8,
    )
    am.apply(FundingPaymentEvent(instrument=inst, payment=payment))
    amount = -(1.0 * 50_000.0 * 0.0001)
    bal = state.get_balance("USDT")
    assert abs(bal.total - (1000.0 + amount)) < 1e-9
    assert abs(bal.free - (900.0 + amount)) < 1e-9
    assert bal.locked == 100.0


def test_funding_payment_different_bucket_applies_again():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    pos.update_market_price(am._time.time(), 50_000.0, 1.0)
    state.set_position(inst, pos)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=1000.0))

    base_ns = np.datetime64("2026-05-28T00:00:00").astype("datetime64[ns]").astype(np.int64)
    p1 = FundingPayment(time=base_ns, funding_rate=0.0001, funding_interval_hours=8)
    next_bucket_ns = base_ns + 8 * 3_600_000_000_000
    p2 = FundingPayment(time=next_bucket_ns, funding_rate=0.0001, funding_interval_hours=8)
    am.apply(FundingPaymentEvent(instrument=inst, payment=p1))
    first = pos.cumulative_funding
    am.apply(FundingPaymentEvent(instrument=inst, payment=p2))
    assert pos.cumulative_funding != first


def test_futures_realized_pnl_folds_into_total_capital():
    # A futures round-trip realizing +X must credit the settle balance by +X, and
    # get_total_capital() (seeded capital + realized) must reflect it.
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=100_000.0, free=100_000.0))

    # open long 1.0 @ 50k
    am._apply_deal_to_position(state, inst, _fill(trade_id="t1", amount=1.0, price=50_000.0))
    # opening a futures position does not touch cash
    assert state.get_balance("USDT").total == 100_000.0
    assert am.get_total_capital("binance") == 100_000.0

    # close 1.0 @ 60k -> realized +10k
    am._apply_deal_to_position(state, inst, _fill(trade_id="t2", amount=-1.0, price=60_000.0))
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
    am._apply_deal_to_position(state, inst, deal)
    # mark the position so its market value contributes to total capital
    state.get_position(inst).update_market_price(am._time.time(), 100_000.0, 1.0)

    assert abs(state.get_balance("USDT").total - 50_000.0) < 1e-6
    assert abs(state.get_balance("USDT").free - 50_000.0) < 1e-6
    assert abs(state.get_balance("BTC").free - 0.5) < 1e-9
    assert abs(am.get_total_capital("binance") - 100_000.0) < 1e-6
