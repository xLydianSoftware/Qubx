import pandas as pd
import pytest
from pytest import approx

from qubx import QubxLogConfig, logger
from qubx.connectors.ccxt.reader import CcxtDataReader
from qubx.data.composite import CompositeReader
from qubx.data.readers import AsPandasFrame
from qubx.data.registry import ReaderRegistry


class TestCompositeReader:
    """
    Test cases for the CompositeReader class.
    """

    def test_non_chunked_read_with_identical_readers(self):
        """
        Test non-chunked read operation with two identical CSV readers.

        This test creates two identical CSV readers pointing to the same data,
        combines them with CompositeReader, and verifies that the result is
        the same as reading from a single reader (no duplicates).
        """
        # Create two identical CSV readers
        reader1 = ReaderRegistry.get("csv::tests/data/csv_1h/")
        reader2 = ReaderRegistry.get("csv::tests/data/csv_1h/")

        # Create a composite reader from the two readers
        composite_reader = CompositeReader([reader1, reader2])

        # Read data from the first reader
        data_id = "BINANCE.UM:BTCUSDT"
        reader1_data = reader1.read(data_id=data_id, chunksize=0)

        # Ensure we have a list
        if not isinstance(reader1_data, list):
            reader1_data = list(reader1_data)

        # Read data from the composite reader
        composite_data = composite_reader.read(data_id=data_id, chunksize=0)

        # Ensure we have a list
        if not isinstance(composite_data, list):
            composite_data = list(composite_data)

        # Verify that the composite reader returns the same data as a single reader
        # (i.e., duplicates are removed)
        assert len(reader1_data) == len(composite_data)

        # Check that all records are identical
        for i in range(len(reader1_data)):
            assert reader1_data[i][0] == composite_data[i][0]  # Compare timestamps

            # Compare all other fields
            for j in range(1, len(reader1_data[i])):
                assert reader1_data[i][j] == approx(composite_data[i][j], rel=1e-6)

        # Verify that the data is sorted by timestamp
        for i in range(1, len(composite_data)):
            assert composite_data[i - 1][0] < composite_data[i][0]

    def test_chunked_read_with_identical_readers(self):
        """
        Test chunked read operation with two identical CSV readers.

        This test creates two identical CSV readers pointing to the same data,
        combines them with CompositeReader, and verifies that the chunked reading
        works correctly.
        """
        # Create two identical CSV readers
        reader1 = ReaderRegistry.get("csv::tests/data/csv_1h/")
        reader2 = ReaderRegistry.get("csv::tests/data/csv_1h/")

        # Create a composite reader from the two readers
        composite_reader = CompositeReader([reader1, reader2])

        # Read all data from the first reader for comparison
        data_id = "BINANCE.UM:BTCUSDT"
        reader1_data = reader1.read(data_id=data_id, chunksize=0, transform=AsPandasFrame())

        # Read data from the composite reader in chunks
        chunk_size = 10
        composite_data_chunks = composite_reader.read(data_id=data_id, chunksize=chunk_size, transform=AsPandasFrame())

        # Collect all chunks
        all_chunks = []
        if composite_data_chunks is not None:
            for chunk in composite_data_chunks:
                all_chunks.append(chunk)

        all_chunks = pd.concat(all_chunks)

        # Verify that the combined chunks have the same length as the non-chunked data
        assert len(reader1_data) == len(all_chunks)

        # Compare the dataframes to ensure they contain the same data
        pd.testing.assert_frame_equal(reader1_data, all_chunks)


class TestCompositeMqdbCcxtReader:
    """
    Test cases for the CompositeReader class with MQDB and CCXT readers.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        # Initialize the reader with a max of 1000 bars
        self._ccxt = CcxtDataReader(exchanges=["BINANCE.UM"], max_bars=1000)
        self._mqdb = ReaderRegistry.get("mqdb::nebula")
        self._now = pd.Timestamp.now()

        QubxLogConfig.set_log_level("DEBUG")
        # Set the log level to DEBUG for more detailed output
        yield
        # Clean up after the test
        self._ccxt.close()
        if hasattr(self._mqdb, "close") and callable(self._mqdb.close):
            self._mqdb.close()

    @pytest.mark.integration
    def test_non_chunked_read_with_mqdb_and_ccxt_readers(self):
        _composite = CompositeReader([self._mqdb, self._ccxt])

        data_id = "BINANCE.UM:BTCUSDT"
        data = _composite.read(
            data_id=data_id,
            start=str(self._now - pd.Timedelta(days=30)),
            stop=str(self._now),
            timeframe="1h",
            chunksize=0,
            transform=AsPandasFrame(),
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert data.shape[1] == 5
        assert data.shape[0] > 0

    @pytest.mark.integration
    def test_chunked_read_with_mqdb_and_ccxt_readers(self):
        _composite = CompositeReader([self._mqdb, self._ccxt])

        data_id = "BINANCE.UM:BTCUSDT"
        data = _composite.read(
            data_id=data_id,
            start=str(self._now - pd.Timedelta(days=30)),
            stop=str(self._now),
            timeframe="1h",
            chunksize=200,
            transform=AsPandasFrame(),
        )

        chunks = []
        if data is not None:
            for chunk in data:
                chunks.append(chunk)

        chunks = pd.concat(chunks)

        assert len(chunks) > 0
