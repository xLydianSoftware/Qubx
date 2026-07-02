from unittest.mock import MagicMock

import numpy as np

from qubx.core.account_manager import AccountManager
from qubx.core.basics import Balance, Instrument, MarketType, Position


class _T:
    def time(self):
        return np.datetime64("2026-05-28T00:00:00")


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


def _am(exchanges=("binance",)):
    return AccountManager(
        connectors={ex: MagicMock() for ex in exchanges},
        base_currencies={ex: "USDT" for ex in exchanges},
        time=_T(),
    )


def _marked_position(inst, quantity, avg_price, mark_price):
    pos = Position(instrument=inst, quantity=quantity, pos_average_price=avg_price)
    pos.update_market_price(np.datetime64("2026-05-28T00:00:00"), mark_price, 1.0)
    return pos


def test_total_capital_single_exchange():
    am = _am()
    state = am._states["binance"]
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=800.0))
    # flat (no positions): total capital equals base balance total
    assert am.get_total_capital("binance") == 1000.0
    assert am.get_total_capital() == 1000.0


def test_total_capital_includes_unrealized_pnl():
    am = _am()
    state = am._states["binance"]
    inst = _instrument()
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=1000.0, free=800.0))
    # long 1 BTC @ 50k, marked at 51k -> +1000 unrealized
    state.set_position(inst, _marked_position(inst, 1.0, 50_000.0, 51_000.0))
    assert abs(am.get_total_capital("binance") - 2000.0) < 1e-6


# cross-exchange total-capital aggregation is pinned by the ported PR #302 suite:
# tests/qubx/core/account_manager/manager_test.py (test_total_capital_aggregates_across_exchanges).


def test_gross_leverage_two_positions():
    am = _am()
    state = am._states["binance"]
    inst_a = _instrument("BTCUSDT")
    inst_b = _instrument("ETHUSDT")
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=10_000.0))
    # long 1 BTC @ 50k marked 50k -> notional 50k; short 10 ETH @ 3k marked 3k -> notional -30k
    state.set_position(inst_a, _marked_position(inst_a, 1.0, 50_000.0, 50_000.0))
    state.set_position(inst_b, _marked_position(inst_b, -10.0, 3_000.0, 3_000.0))
    # gross = (50k + 30k) / 10k = 8.0 (capital includes 0 unrealized at mark==avg)
    assert abs(am.get_gross_leverage("binance") - 8.0) < 1e-6


def test_net_leverage_cancels_long_short():
    am = _am()
    state = am._states["binance"]
    inst_a = _instrument("BTCUSDT")
    inst_b = _instrument("ETHUSDT")
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=10_000.0))
    # +50k and -30k notional -> net 20k -> 20k/10k = 2.0
    state.set_position(inst_a, _marked_position(inst_a, 1.0, 50_000.0, 50_000.0))
    state.set_position(inst_b, _marked_position(inst_b, -10.0, 3_000.0, 3_000.0))
    assert abs(am.get_net_leverage("binance") - 2.0) < 1e-6


def test_net_leverage_negative_for_short_book():
    # Regression for I7: leverage is signed (merge-base + PR #302 behavior) — a
    # net-short book must report NEGATIVE net leverage, not its absolute value.
    am = _am()
    state = am._states["binance"]
    inst = _instrument("BTCUSDT")
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=10_000.0))
    # short 1 BTC @ 50k marked 50k -> notional -50k -> net -5.0, gross +5.0
    state.set_position(inst, _marked_position(inst, -1.0, 50_000.0, 50_000.0))
    assert abs(am.get_net_leverage("binance") - (-5.0)) < 1e-6
    assert abs(am.get_gross_leverage("binance") - 5.0) < 1e-6


def test_instrument_leverage():
    am = _am()
    state = am._states["binance"]
    inst = _instrument("BTCUSDT")
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=10_000.0))
    state.set_position(inst, _marked_position(inst, 1.0, 50_000.0, 50_000.0))
    # 50k / 10k = 5.0
    assert abs(am.get_leverage(inst) - 5.0) < 1e-6


def test_instrument_leverage_signed_for_short():
    am = _am()
    state = am._states["binance"]
    inst = _instrument("BTCUSDT")
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=10_000.0))
    state.set_position(inst, _marked_position(inst, -1.0, 50_000.0, 50_000.0))
    # -50k / 10k = -5.0 (sign preserved per position direction)
    assert abs(am.get_leverage(inst) - (-5.0)) < 1e-6


def test_leverage_aggregation_capital_weighted():
    # Cross-exchange aggregation is capital-weighted: Σ(lev * capital) / Σcapital
    # recovers Σnotional / Σcapital, not a naive average of per-exchange leverages.
    am = _am(exchanges=("binance", "bybit"))
    inst_a = _instrument("BTCUSDT", exchange="binance")
    inst_b = _instrument("ETHUSDT", exchange="bybit")
    am._states["binance"].update_balance("USDT", Balance(exchange="binance", currency="USDT", total=10_000.0))
    am._states["bybit"].update_balance("USDT", Balance(exchange="bybit", currency="USDT", total=30_000.0))
    # binance: 50k notional / 10k capital = 5.0; bybit: 30k notional / 30k capital = 1.0
    am._states["binance"].set_position(inst_a, _marked_position(inst_a, 1.0, 50_000.0, 50_000.0))
    am._states["bybit"].set_position(inst_b, _marked_position(inst_b, 10.0, 3_000.0, 3_000.0))
    # total notional 80k / total capital 40k = 2.0
    assert abs(am.get_net_leverage() - 2.0) < 1e-6
    assert abs(am.get_gross_leverage() - 2.0) < 1e-6


def test_leverage_finite_for_unmarked_position():
    # Regression for I3: an unmarked position has notional_value == NaN, which
    # would poison every leverage aggregate. All three must return finite 0.0.
    am = _am()
    state = am._states["binance"]
    inst = _instrument("BTCUSDT")
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=10_000.0))
    # position never marked -> last_update_price is NaN
    pos = Position(instrument=inst, quantity=1.0, pos_average_price=50_000.0)
    assert np.isnan(pos.last_update_price)
    state.set_position(inst, pos)
    assert am.get_leverage(inst) == 0.0
    assert am.get_net_leverage("binance") == 0.0
    assert am.get_gross_leverage("binance") == 0.0
    assert not np.isnan(am.get_leverage(inst))
    assert not np.isnan(am.get_net_leverage("binance"))
    assert not np.isnan(am.get_gross_leverage("binance"))


def test_leverage_zero_when_no_capital():
    am = _am()
    inst = _instrument("BTCUSDT")
    am._states["binance"].set_position(inst, _marked_position(inst, 1.0, 50_000.0, 50_000.0))
    assert am.get_gross_leverage("binance") == 0.0
    assert am.get_net_leverage("binance") == 0.0
    assert am.get_leverage(inst) == 0.0


def test_total_margins_aggregate_across_positions():
    # Regression guard: get_total_initial_margin / get_total_maint_margin sum the
    # per-position margins (venue-reported, set externally) across the book. This
    # aggregation lost its dedicated coverage when the old account-processor tests were
    # removed; the per-position setter is tested elsewhere, the AM summation was not.
    am = _am()
    state = am._states["binance"]
    inst1 = _instrument("BTCUSDT")
    inst2 = _instrument("ETHUSDT")
    p1 = _marked_position(inst1, 1.0, 50_000.0, 50_000.0)
    p2 = _marked_position(inst2, 10.0, 3_000.0, 3_000.0)
    p1.set_external_initial_margin(500.0)
    p1.set_external_maint_margin(250.0)
    p2.set_external_initial_margin(300.0)
    p2.set_external_maint_margin(150.0)
    state.set_position(inst1, p1)
    state.set_position(inst2, p2)
    state.update_balance("USDT", Balance(exchange="binance", currency="USDT", total=2_000.0))

    assert am.get_total_initial_margin("binance") == 800.0
    assert am.get_total_maint_margin("binance") == 400.0
    # exchange=None aggregates across all states
    assert am.get_total_initial_margin() == 800.0
    assert am.get_total_maint_margin() == 400.0
    # available margin = total capital - total initial margin
    assert am.get_available_margin("binance") == 2_000.0 - 800.0


# ---- per-instrument venue settings: read side (backed by Position) ---------- #


def test_venue_settings_read_off_position():
    am = _am()
    state = am._states["binance"]
    inst = _instrument("BTCUSDT")
    pos = _marked_position(inst, 1.0, 50_000.0, 50_000.0)
    pos.leverage = 10.0
    pos.margin_mode = "isolated"
    pos.max_notional = 1_000_000.0
    pos.adl_level = 2
    state.set_position(inst, pos)

    assert am.get_max_instrument_leverage(inst) == 10.0
    assert am.get_max_instrument_notional(inst) == 1_000_000.0
    assert am.get_margin_mode(inst) == "isolated"
    assert am.get_adl_level(inst) == 2


def test_venue_settings_neutral_when_not_reported():
    # An instrument the venue never reported reads as None / inf (get_position materializes flat).
    am = _am()
    inst = _instrument("BTCUSDT")
    assert am.get_max_instrument_leverage(inst) is None
    assert am.get_max_instrument_notional(inst) == float("inf")
    assert am.get_margin_mode(inst) is None
    assert am.get_adl_level(inst) is None


# ---- per-instrument venue settings: write side ------------------------------ #


def test_set_max_instrument_leverage_routes_to_connector():
    am = _am()
    inst = _instrument("BTCUSDT")
    am._connectors["binance"].set_max_instrument_leverage.return_value = True
    assert am.set_max_instrument_leverage(inst, 7.0) is True
    am._connectors["binance"].set_max_instrument_leverage.assert_called_once_with(inst, 7.0)


def test_set_margin_mode_routes_to_connector():
    am = _am()
    inst = _instrument("BTCUSDT")
    am._connectors["binance"].set_margin_mode.return_value = True
    assert am.set_margin_mode(inst, "cross") is True
    am._connectors["binance"].set_margin_mode.assert_called_once_with(inst, "cross")


def test_set_leverage_unmanaged_exchange_returns_false():
    am = _am(exchanges=("binance",))
    inst = _instrument("BTCUSDT", exchange="bybit")  # no connector for bybit
    assert am.set_max_instrument_leverage(inst, 7.0) is False
