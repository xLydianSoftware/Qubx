"""Integration tests for Lighter instrument loader

These tests connect to live Lighter API (read-only operations).
Run with: pytest tests/qubx/connectors/lighter/test_instruments_integration.py -v -m integration
"""

import os

import pytest

from qubx.connectors.xlighter.client import LighterClient
from qubx.connectors.xlighter.instruments import LighterInstrumentLoader
from qubx.core.basics import AssetType, MarketType


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def lighter_credentials():
    """
    Get Lighter credentials from accounts file.

    Set LIGHTER_ACCOUNT_PATH to override default location.
    """
    account_path = os.getenv("LIGHTER_ACCOUNT_PATH", "/home/yuriy/accounts/xlydian1_lighter.toml")

    if not os.path.exists(account_path):
        pytest.skip(f"Lighter account file not found: {account_path}")

    import toml

    config = toml.load(account_path)

    accounts = config.get("accounts", [])
    lighter_account = None

    for account in accounts:
        if account.get("exchange", "").upper() == "LIGHTER":
            lighter_account = account
            break

    if not lighter_account:
        pytest.skip("No LIGHTER account found in config")

    return {
        "api_key": lighter_account.get("api_key"),
        "secret": lighter_account.get("secret"),
        "account_index": lighter_account.get("account_index"),
        "api_key_index": lighter_account.get("api_key_index", 0),
    }


@pytest.fixture
async def lighter_client(lighter_credentials):
    """Create Lighter client for testing (async fixture for aiohttp compatibility)"""
    client = LighterClient(
        api_key=lighter_credentials["api_key"],
        private_key=lighter_credentials["secret"],
        account_index=lighter_credentials["account_index"],
        api_key_index=lighter_credentials["api_key_index"],
        testnet=False,  # Use mainnet for integration tests
    )
    yield client
    client.close()


class TestLighterInstrumentLoaderIntegration:
    """Integration tests for Lighter instrument loader"""

    async def test_load_instruments(self, lighter_client):
        """Test loading instruments from live Lighter API"""
        loader = LighterInstrumentLoader(lighter_client)
        instruments = await loader.load_instruments()

        # Should have loaded some instruments
        assert len(instruments) > 0, "Should load at least one instrument"

        # Check instrument format
        for full_id, instrument in instruments.items():
            # Check full ID format
            assert full_id.startswith("LIGHTER:SWAP:"), f"Invalid full_id format: {full_id}"

            # Check instrument properties
            assert instrument.exchange == "LIGHTER"
            assert instrument.market_type == MarketType.SWAP
            assert instrument.asset_type == AssetType.CRYPTO
            assert instrument.tick_size > 0
            assert instrument.lot_size > 0
            assert instrument.min_size > 0

            # Check symbol format (should be "BTCUSDC" style - normalized Qubx format)
            assert "USDC" in instrument.symbol, f"Invalid symbol format: {instrument.symbol}"
            assert instrument.quote == "USDC"

    async def test_load_btc_instrument(self, lighter_client):
        """Test that BTCUSDC instrument loads correctly"""
        loader = LighterInstrumentLoader(lighter_client)
        instruments = await loader.load_instruments()

        # Look for BTC instrument (should be "BTCUSDC")
        btc_instrument = None
        for full_id, instrument in instruments.items():
            if instrument.symbol == "BTCUSDC" or (instrument.base == "BTC" and instrument.quote == "USDC"):
                btc_instrument = instrument
                break

        assert btc_instrument is not None, "Should find BTC instrument"

        # Verify BTC instrument properties
        assert btc_instrument.symbol == "BTCUSDC", "Symbol should be normalized to BTCUSDC"
        assert btc_instrument.base == "BTC"
        assert btc_instrument.quote == "USDC"
        assert btc_instrument.settle == "USDC"
        assert btc_instrument.tick_size > 0
        assert btc_instrument.lot_size > 0
        assert btc_instrument.contract_size == 1.0  # Perpetuals

    async def test_market_id_mappings(self, lighter_client):
        """Test that market ID mappings are created correctly"""
        loader = LighterInstrumentLoader(lighter_client)
        await loader.load_instruments()

        # Should have mappings
        assert len(loader.market_id_to_symbol) > 0
        assert len(loader.symbol_to_market_id) > 0

        # Mappings should be consistent
        for market_id, symbol in loader.market_id_to_symbol.items():
            # Symbol -> market_id should work
            assert loader.symbol_to_market_id.get(symbol) == market_id

            # Should be able to get instrument by market_id
            instrument = loader.get_instrument_by_market_id(market_id)
            assert instrument is not None
            assert instrument.symbol == symbol

    async def test_get_instrument_by_symbol(self, lighter_client):
        """Test getting instrument by symbol"""
        loader = LighterInstrumentLoader(lighter_client)
        await loader.load_instruments()

        # Get first symbol from mappings
        if not loader.symbol_to_market_id:
            pytest.skip("No instruments loaded")

        symbol = next(iter(loader.symbol_to_market_id.keys()))

        instrument = loader.get_instrument_by_symbol(symbol)
        assert instrument is not None
        assert instrument.symbol == symbol
        # Verify symbol is in normalized format (no dashes)
        assert "-" not in symbol, f"Symbol should be normalized: {symbol}"

    async def test_get_market_id(self, lighter_client):
        """Test getting market ID by symbol"""
        loader = LighterInstrumentLoader(lighter_client)
        await loader.load_instruments()

        # Get first symbol from mappings
        if not loader.symbol_to_market_id:
            pytest.skip("No instruments loaded")

        symbol = next(iter(loader.symbol_to_market_id.keys()))
        market_id = loader.get_market_id(symbol)

        assert market_id is not None
        assert isinstance(market_id, int)
        assert market_id >= 0


class TestLighterClientIntegration:
    """Integration tests for Lighter client"""

    async def test_get_markets(self, lighter_client):
        """Test getting markets from API"""
        markets = await lighter_client.get_markets()

        assert len(markets) > 0, "Should return at least one market"

        # Check market structure
        first_market = markets[0]
        assert "id" in first_market
        assert "symbol" in first_market

    async def test_get_market_info(self, lighter_client):
        """Test getting specific market info"""
        markets = await lighter_client.get_markets()
        if not markets:
            pytest.skip("No markets available")

        market_id = markets[0].get("id")
        market_info = await lighter_client.get_market_info(market_id)

        assert market_info is not None
        assert market_info["id"] == market_id

    async def test_get_orderbook(self, lighter_client):
        """Test getting orderbook"""
        markets = await lighter_client.get_markets()
        if not markets:
            pytest.skip("No markets available")

        market_id = markets[0].get("id")
        orderbook = await lighter_client.get_orderbook(market_id)

        assert "asks" in orderbook
        assert "bids" in orderbook
        # Orderbook might be empty, but keys should exist
        assert isinstance(orderbook["asks"], list)
        assert isinstance(orderbook["bids"], list)
