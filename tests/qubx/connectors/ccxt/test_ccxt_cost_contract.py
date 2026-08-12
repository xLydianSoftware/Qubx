"""Pins the installed ccxt's REST cost contract.

The rate limit configs and the storage's "charge ccxt's real weight" path are sized against these
numbers; a ccxt version bump that changes them silently invalidates those decisions. Verified
against ccxt 4.5.50.
"""

import ccxt
import ccxt.pro as cxp


def test_ccxt_version_is_visible_on_failure():
    # not a pin on the version — it just puts the version in the failure output of the tests below
    assert ccxt.__version__


def test_binanceusdm_rate_limit_is_50ms():
    assert cxp.binanceusdm().rateLimit == 50


def test_fapi_open_orders_without_symbol_costs_40():
    exchange = cxp.binanceusdm()
    config = exchange.api["fapiPrivate"]["get"]["openOrders"]
    assert config["noSymbol"] == 40
    assert exchange.calculate_rate_limiter_cost("fapiPrivate", "GET", "openOrders", {}, config) == 40


def test_fapi_open_orders_with_symbol_costs_1():
    # negative control: the 40 is the no-symbol surcharge, not the endpoint's base cost
    exchange = cxp.binanceusdm()
    config = exchange.api["fapiPrivate"]["get"]["openOrders"]
    assert config["cost"] == 1
    assert exchange.calculate_rate_limiter_cost("fapiPrivate", "GET", "openOrders", {"symbol": "BTCUSDT"}, config) == 1


def test_binance_spot_trades_costs_2():
    exchange = cxp.binance()
    assert exchange.api["public"]["get"]["trades"] == 2
    assert exchange.calculate_rate_limiter_cost("public", "GET", "trades", {}, {"cost": 2}) == 2
