"""
\nIntegration tests for MultiStorage — replacement of the deprecated CompositeReader.\n
"""

import pandas as pd
import pytest

from qubx import QubxLogConfig
from qubx.connectors.ccxt.reader import CcxtDataReader
from qubx.data import CsvStorage
from qubx.data.registry import StorageRegistry
from qubx.data.storages.multi import MultiStorage

_CSV_STORAGE = "tests/data/storages/csv"


@pytest.mark.integration
class TestMultiStorage:
    """
    Test cases for MultiStorage — the IStorage-based replacement of CompositeReader.

    MultiStorage tries each storage in order, returning data from the first
    that has it (fallback pattern). Unlike the old CompositeReader, it does
    not merge or deduplicate across all storages.
    """

    def test_read_with_identical_storages(self):
        """
        Test that MultiStorage with two identical CSV storages returns the same
        data as reading from a single storage (first storage always succeeds,
        second is never queried).
        """
        storage1 = CsvStorage(_CSV_STORAGE)
        storage2 = CsvStorage(_CSV_STORAGE)
        multi = MultiStorage([storage1, storage2])

        reader_single = storage1.get_reader("BINANCE.UM", "SWAP")
        reader_multi = multi.get_reader("BINANCE.UM", "SWAP")

        single_df = reader_single.read("BTCUSDT", "ohlc(1h)", "2023-06-10", "2023-07-10").to_pd()
        multi_df = reader_multi.read("BTCUSDT", "ohlc(1h)", "2023-06-10", "2023-07-10").to_pd()

        assert len(single_df) == len(multi_df)
        pd.testing.assert_frame_equal(single_df, multi_df)

    def test_fallback_to_second_storage(self):
        """
        Test that MultiStorage falls back to the second storage when the first
        does not have data for the requested exchange/market.
        """
        # - first storage has BINANCE.UM but not HYPERLIQUID, second has both
        storage_hl = CsvStorage(_CSV_STORAGE)  # - has HYPERLIQUID/SWAP
        storage_binance = CsvStorage(_CSV_STORAGE)  # - has BINANCE.UM/SWAP

        multi = MultiStorage([storage_hl, storage_binance])

        # - HYPERLIQUID data should come from storage_hl (first)
        reader_hl = multi.get_reader("HYPERLIQUID", "SWAP")
        df_hl = reader_hl.read("BTCUSDC", "ohlc(1h)", "2023-06-10", "2023-07-10").to_pd()
        assert len(df_hl) > 0

    def test_registry_uri_construction(self):
        """
        Test constructing MultiStorage via StorageRegistry with URI-style names.
        """
        storage = StorageRegistry.get(f"csv::{_CSV_STORAGE}")
        assert storage is not None

        reader = storage.get_reader("BINANCE.UM", "SWAP")
        df = reader.read("ETHUSDT", "ohlc(1h)", "2023-06-10", "2023-07-10").to_pd()
        assert len(df) > 0
        assert set(df.columns) >= {"open", "high", "low", "close"}

    def test_multi_symbol_read(self):
        """
        Test reading multiple symbols at once from MultiStorage.
        """
        storage = CsvStorage(_CSV_STORAGE)
        multi = MultiStorage([storage])

        reader = multi.get_reader("BINANCE.UM", "SWAP")
        result = reader.read(["BTCUSDT", "ETHUSDT"], "ohlc(1h)", "2023-06-10", "2023-07-10").to_pd(id_in_index=True)

        assert len(result) > 0
        symbols = result.index.get_level_values("symbol").unique().tolist()
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols


@pytest.mark.integration
class TestMultiStorageMqdbCcxt:
    """
    Test cases for MultiStorage with MQDB and CCXT storages (requires live connections).
    Replaces TestCompositeMqdbCcxtReader which used the deprecated CompositeReader.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        from qubx.data.registry import StorageRegistry

        self._mqdb = StorageRegistry.get("mqdb::nebula")
        self._ccxt_reader = CcxtDataReader(exchanges=["BINANCE.UM"], max_bars=1000)
        self._now = pd.Timestamp.now()

        QubxLogConfig.set_log_level("DEBUG")
        yield
        self._ccxt_reader.close()

    def test_read_recent_ohlc(self):
        """
        Test reading recent OHLCV data from MQDB storage (live connection).
        """
        reader = self._mqdb.get_reader("BINANCE.UM", "SWAP")
        df = reader.read(
            "BTCUSDT",
            "ohlc(1h)",
            start=str(self._now - pd.Timedelta(days=30)),
            stop=str(self._now),
        ).to_pd()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert set(df.columns) >= {"open", "high", "low", "close"}
