import os
import tempfile
from unittest import mock

import pytest

from qubx.core.basics import TransactionCostsCalculator
from qubx.core.lookups import FeesLookupFile


class TestFeesLookup:
    def test_find_existing_fees(self):
        """Test finding fees for an existing exchange and specification."""
        fees_lookup = FeesLookupFile()

        # Test finding fees for binance with vip0_usdt specification
        costs = fees_lookup.find_fees("binance", "vip0_usdt")

        assert isinstance(costs, TransactionCostsCalculator)
        assert costs.name == "binance_vip0_usdt"
        assert costs.maker == 0.1000 / 100.0
        assert costs.taker == 0.1000 / 100.0

        # Test finding fees for kraken with k0 specification
        costs = fees_lookup.find_fees("kraken", "k0")

        assert isinstance(costs, TransactionCostsCalculator)
        assert costs.name == "kraken_k0"
        assert costs.maker == 0.25 / 100.0
        assert costs.taker == 0.40 / 100.0

    def test_find_with_direct_maker_taker_spec(self):
        """Test finding fees using direct maker/taker specification."""
        fees_lookup = FeesLookupFile()

        # Test with direct maker/taker specification
        costs = fees_lookup.find_fees("any_exchange", "maker=0.05,taker=0.08")

        assert isinstance(costs, TransactionCostsCalculator)
        assert costs.name == "any_exchange_maker=0.05,taker=0.08"
        assert costs.maker == 0.05 / 100.0
        assert costs.taker == 0.08 / 100.0

    def test_find_with_direct_maker_taker_rebates(self):
        """Test finding fees using direct maker/taker specification."""
        fees_lookup = FeesLookupFile()

        # Test with direct maker/taker specification
        costs = fees_lookup.find_fees("any_exchange", "maker=-0.05 taker=0.02")

        assert isinstance(costs, TransactionCostsCalculator)
        assert costs.name == "any_exchange_maker=-0.05 taker=0.02"
        assert costs.maker == -0.05 / 100.0
        assert costs.taker == 0.02 / 100.0

    def test_find_nonexistent_fees(self):
        """Test that finding fees for a nonexistent specification raises ValueError."""
        fees_lookup = FeesLookupFile()

        # Test with nonexistent specification
        with pytest.raises(ValueError, match="No fees found for nonexistent_exchange_nonexistent_spec"):
            fees_lookup.find_fees("nonexistent_exchange", "nonexistent_spec")

    def test_refresh_and_load(self):
        """Test refreshing and loading fees from a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a new FeesLookup with the temporary directory
            with mock.patch("qubx.core.lookups.makedirs", return_value=temp_dir):
                fees_lookup = FeesLookupFile(path=temp_dir)

                # Verify that the default fees file was created
                assert os.path.exists(os.path.join(temp_dir, "default.ini"))

                # Verify that some fees were loaded
                assert len(fees_lookup._lookup) > 0

                # Test finding a known fee
                costs = fees_lookup.find_fees("binance", "vip0_usdt")
                assert costs.maker == 0.1000 / 100.0
                assert costs.taker == 0.1000 / 100.0
