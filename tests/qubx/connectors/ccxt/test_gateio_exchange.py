"""Tests for Gate.io exchange registration and custom class."""

import ccxt.pro as cxp

from qubx.connectors.ccxt.exchanges import EXCHANGE_ALIASES, READER_CAPABILITIES, GateioFutures


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
