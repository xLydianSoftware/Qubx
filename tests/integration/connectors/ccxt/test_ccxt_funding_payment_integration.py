"""
Integration tests for CcxtStorage / CcxtReader funding payment functionality.

Verifies that:
1. funding_payment reads return correct schema and value ranges
2. Relative delta strings ("-12h", "-3d") produce correctly bounded results
3. Explicit ISO start/stop ranges work
4. Multi-symbol reads return one RawData per symbol
5. Chunked reads reassemble correctly
"""

import pandas as pd
import pytest

from qubx import QubxLogConfig, logger
from qubx.data.storages.ccxt import CcxtStorage


@pytest.mark.integration
class TestCcxtStorageFundingPayment:
    """
    Integration tests for CcxtStorage funding_payment reads via BINANCE.UM.

    All tests require network access to Binance Futures API.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.stor = CcxtStorage()
        self.reader = self.stor.get_reader("BINANCE.UM", "SWAP")
        QubxLogConfig.set_log_level("DEBUG")
        yield
        self.stor.close()

    # ------------------------------------------------------------------
    # Schema / format
    # ------------------------------------------------------------------

    def test_funding_columns_present(self):
        """RawData contains funding_rate and funding_interval_hours columns."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-3d", "now").to_pd()
        logger.info(f"Funding columns: {list(df.columns)}")
        assert "funding_rate" in df.columns
        assert "funding_interval_hours" in df.columns

    def test_funding_data_types(self):
        """funding_rate and funding_interval_hours are float64."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-3d", "now").to_pd()
        assert df["funding_rate"].dtype == "float64"
        assert df["funding_interval_hours"].dtype == "float64"

    def test_funding_sorted_index(self):
        """Result index (timestamp) is monotonically increasing."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-7d", "now").to_pd()
        assert df.index.is_monotonic_increasing

    # ------------------------------------------------------------------
    # Value ranges
    # ------------------------------------------------------------------

    def test_funding_value_ranges(self):
        """funding_rate < 1.0 (< 100%) and funding_interval_hours > 0."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-7d", "now").to_pd()
        if len(df) > 0:
            assert (df["funding_rate"].abs() < 1.0).all(), "funding_rate should be < 100%"
            assert (df["funding_interval_hours"] > 0).all(), "interval should be positive"

    def test_funding_binance_interval_is_8h(self):
        """Binance UM perpetuals use an 8-hour funding interval."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-3d", "now").to_pd()
        if len(df) > 0:
            assert (df["funding_interval_hours"] == 8.0).all()

    # ------------------------------------------------------------------
    # Row count / time bounding
    # ------------------------------------------------------------------

    def test_funding_row_count_3d(self):
        """3 days at 8h interval = ~9 funding rows; expect at least 6."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-3d", "now").to_pd()
        logger.info(f"Funding 3d (len={len(df)}):\n{df.tail()}")
        assert len(df) >= 6, f"Expected ≥6 rows for 3d, got {len(df)}"

    def test_funding_short_delta_12h(self):
        """'-12h' delta returns 1-2 funding rows, NOT 30 days worth."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-12h", "now").to_pd()
        logger.info(f"Funding -12h (len={len(df)}):\n{df}")
        assert 0 < len(df) <= 4, f"Expected 1-2 rows for -12h, got {len(df)}"

    def test_funding_explicit_time_range(self):
        """Explicit ISO start/stop strings produce correctly bounded results."""
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=2)

        df = self.reader.read("BTCUSDT", "funding_payment", start.isoformat(), end.isoformat()).to_pd()
        logger.info(f"Explicit range funding (len={len(df)}):\n{df}")

        if len(df) > 0:
            # - allow one interval of slack on each side (flooring to interval boundary)
            assert df.index.min() >= start - pd.Timedelta(hours=8)
            assert df.index.max() <= end + pd.Timedelta(hours=1)

    # ------------------------------------------------------------------
    # Multi-symbol
    # ------------------------------------------------------------------

    def test_funding_multiple_symbols(self):
        """Passing a list of symbols returns one RawData per symbol."""
        result = self.reader.read(["BTCUSDT", "ETHUSDT"], "funding_payment", "-3d", "now")
        raws = list(result)
        syms = {r.data_id for r in raws}
        logger.info(f"Multi-symbol funding symbols: {syms}")

        assert "BTCUSDT" in syms
        assert "ETHUSDT" in syms
        for r in raws:
            df = r.to_pd()
            assert "funding_rate" in df.columns
            assert len(df) >= 6, f"{r.data_id}: expected ≥6 rows for 3d"

    # ------------------------------------------------------------------
    # Chunked read
    # ------------------------------------------------------------------

    def test_funding_chunked_read(self):
        """Chunked funding read yields RawData slices that reassemble to the full result."""
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=7)

        chunks = list(  # pyright: ignore[reportArgumentType]
            self.reader.read("BTCUSDT", "funding_payment", start.isoformat(), end.isoformat(), chunksize=3) # pyright: ignore[reportArgumentType]
        )
        assert len(chunks) > 0

        df = pd.concat([c.to_pd() for c in chunks])
        logger.info(f"Chunked funding reassembled (len={len(df)}):\n{df}")
        assert "funding_rate" in df.columns
        assert df.index.is_monotonic_increasing
        assert len(df) >= 6, f"Expected ≥6 rows for 7d, got {len(df)}"
