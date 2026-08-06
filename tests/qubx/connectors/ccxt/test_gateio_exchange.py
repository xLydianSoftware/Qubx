"""Tests for Gate.io exchange registration and custom class."""

import asyncio

import ccxt.pro as cxp
import pytest

from qubx.connectors.ccxt.exchanges import EXCHANGE_ALIASES, READER_CAPABILITIES, GateioFutures


def run(coro):
    return asyncio.run(coro)


class TestGateioRegistration:
    """Test that Gate.io exchange is properly registered."""

    def test_exchange_alias_exists(self):
        assert "gateio.f" in EXCHANGE_ALIASES
        assert EXCHANGE_ALIASES["gateio.f"] == "gateio_futures"

    def test_custom_class_registered_in_ccxt(self):
        assert hasattr(cxp, "gateio_futures")
        assert cxp.gateio_futures is GateioFutures
        assert "gateio_futures" in cxp.exchanges

    def test_reader_capabilities(self):
        assert "gateio.f" in READER_CAPABILITIES
        caps = READER_CAPABILITIES["gateio.f"]
        assert caps.supports_bulk_funding is False
        assert caps.default_funding_interval_hours == 8.0

    def test_gateio_futures_inherits_gate(self):
        assert issubclass(GateioFutures, cxp.gate)

    def test_gateio_futures_has_watch_funding_rates(self):
        assert hasattr(GateioFutures, "watch_funding_rates")
        assert callable(getattr(GateioFutures, "watch_funding_rates"))

    def test_gateio_futures_has_un_watch_funding_rates(self):
        assert hasattr(GateioFutures, "un_watch_funding_rates")
        assert callable(getattr(GateioFutures, "un_watch_funding_rates"))

    def test_gateio_futures_instantiation(self):
        exchange = GateioFutures()
        assert exchange._funding_rate_adapter is None
        assert exchange.id == "gate"


def _gate_swap_market(base: str) -> dict:
    return {
        "id": f"{base}_USDT",
        "symbol": f"{base}/USDT:USDT",
        "base": base,
        "quote": "USDT",
        "settle": "USDT",
        "baseId": base,
        "quoteId": "USDT",
        "settleId": "usdt",
        "type": "swap",
        "spot": False,
        "swap": True,
        "future": False,
        "option": False,
        "contract": True,
        "linear": True,
        "inverse": False,
        "contractSize": 1.0,
        "active": True,
        "precision": {"amount": 1, "price": 0.1},
        "limits": {"amount": {"min": 1}, "price": {}, "cost": {}},
        "info": {},
    }


@pytest.fixture
def offline_gateio_futures():
    """GateioFutures with preseeded swap markets — no network calls."""
    ex = GateioFutures({"options": {"defaultType": "swap"}})
    ex.set_markets([_gate_swap_market("BTC")])
    yield ex
    run(ex.close())


class TestGateioCidOnlyEdit:
    """ccxt's base edit_order_with_client_order_id routes to edit_order_request with
    id='' and the cid in params; upstream gate has no cid substitution for amends
    (unlike its fetch_order/cancel_order builders): order_id='' lands in the URL path
    and a stray ``clientOrderId`` body key rides along. The override must produce the
    documented custom-id path form: order_id = 't-<cid>' (qubx-placed gate orders carry
    text='t-<cid>', which the venue resolves while the order is live)."""

    def test_cid_only_edit_request_substitutes_t_prefixed_order_id(self, offline_gateio_futures):
        request = offline_gateio_futures.edit_order_request(
            "", "BTC/USDT:USDT", "limit", "buy", 1.0, 100.0, {"clientOrderId": "myCid123"}
        )
        assert request["order_id"] == "t-myCid123"
        assert "clientOrderId" not in request
        assert request["settle"] == "usdt"

    def test_venue_id_edit_request_unchanged(self, offline_gateio_futures):
        request = offline_gateio_futures.edit_order_request(
            "123456789", "BTC/USDT:USDT", "limit", "buy", 1.0, 100.0, {}
        )
        assert request["order_id"] == "123456789"
