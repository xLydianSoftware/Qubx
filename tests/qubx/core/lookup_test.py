import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from unittest import mock

import pytest

from qubx.core.basics import TransactionCostsCalculator
from qubx.core.lookups import FeesLookupFile, FileInstrumentsLookupWithCCXT


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


class TestFileInstrumentsLookupCache:
    @pytest.mark.skipif(sys.platform == "win32", reason="fork start method")
    def test_concurrent_cold_starts_all_see_the_full_cache(self, tmp_path):
        """xdist workers (or bots) sharing a fresh ~/.qubx/instruments must never load a half-built cache."""
        ctx = mp.get_context("fork")
        with ctx.Pool(8) as pool:
            for round_ in range(10):
                path = str(tmp_path / f"round{round_}" / "instruments")
                os.makedirs(path)
                # - staggered starts put readers inside another process's write, as CI's workers did
                starts = [(path, random.random() * 0.4) for _ in range(8)]
                assert all(pool.map(_binance_btc_known, starts, chunksize=1))

    def test_a_cold_start_publishes_the_cache_and_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "instruments"
        path.mkdir()
        lookup = FileInstrumentsLookupWithCCXT(str(path))
        assert lookup.find_symbol("BINANCE", "BTCUSDT") is not None
        assert list(path.glob("*.json"))
        assert [p.name for p in tmp_path.iterdir()] == ["instruments"]
        assert not [p for p in path.iterdir() if not p.name.endswith(".json")]

    def test_a_cold_start_that_loses_the_publish_race_uses_the_winner_cache(self, tmp_path, monkeypatch):
        path = tmp_path / "instruments"
        path.mkdir()
        original = FileInstrumentsLookupWithCCXT.refresh

        def refresh_then_lose(self, query_exchanges=False, path=None):
            original(self, query_exchanges, path)
            # - another process publishes its cache while this one was building
            winner = tmp_path / "instruments"
            if not any(winner.iterdir()):
                original(self, query_exchanges, str(winner))

        monkeypatch.setattr(FileInstrumentsLookupWithCCXT, "refresh", refresh_then_lose)
        lookup = FileInstrumentsLookupWithCCXT(str(path))
        assert lookup.find_symbol("BINANCE", "BTCUSDT") is not None
        assert [p.name for p in tmp_path.iterdir()] == ["instruments"]


def _binance_btc_known(start: tuple[str, float]) -> bool:
    path, delay = start
    time.sleep(delay)
    return FileInstrumentsLookupWithCCXT(path).find_symbol("BINANCE", "BTCUSDT") is not None
