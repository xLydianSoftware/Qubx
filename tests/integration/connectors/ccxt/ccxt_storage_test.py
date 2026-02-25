import pandas as pd
import pytest

from qubx import QubxLogConfig, logger
from qubx.data.storages.ccxt import CcxtStorage


@pytest.mark.integration
class TestCcxtStorage:
    """
    Integration tests for CcxtStorage / CcxtReader.

    All tests use BINANCE.UM (perpetual futures) as the exchange and target
    BTCUSDT as the primary instrument.  Tests are marked ``integration`` and
    are skipped in the normal unit-test run.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.stor = CcxtStorage()
        self.reader = self.stor.get_reader("BINANCE.UM", "SWAP")
        QubxLogConfig.set_log_level("DEBUG")
        yield
        self.stor.close()

    # ------------------------------------------------------------------
    # Symbol / metadata
    # ------------------------------------------------------------------

    def test_get_symbols(self):
        """get_data_id() returns a non-empty list that includes BTCUSDT."""
        symbols = self.reader.get_data_id()
        logger.info(f"Symbols (len={len(symbols)}): {symbols[:10]}")
        assert len(symbols) > 0
        assert "BTCUSDT" in symbols

    def test_get_time_range(self):
        """get_time_range() returns a ~30-day window ending near now."""
        start, end = self.reader.get_time_range("BTCUSDT", "ohlc(1h)")
        assert start is not None
        assert end is not None
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        assert (end_dt - start_dt).days >= 25, f"Expected ~30-day look-back, got {end_dt - start_dt}"

    # ------------------------------------------------------------------
    # OHLCV reads
    # ------------------------------------------------------------------

    def test_read_ohlcv(self):
        """read() returns correct OHLCV columns, sorted index, and reasonable row count."""
        df = self.reader.read("BTCUSDT", "ohlc(1h)", "-2d", "now").to_pd()
        logger.info(f"OHLCV (len={len(df)}):\n{df.tail()}")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"Missing column: {col}"
        assert df.index.is_monotonic_increasing
        # - 2 days * 24 bars = 48; allow a few missing bars
        assert len(df) >= 40, f"Expected ~48 bars for 2d/1h, got {len(df)}"

    def test_read_ohlcv_short_delta(self):
        """Negative sub-day delta '-6h' returns ~6 1h bars (not 30 days)."""
        df = self.reader.read("BTCUSDT", "ohlc(1h)", "-6h", "now").to_pd()
        logger.info(f"6h OHLCV (len={len(df)}):\n{df}")
        # - last bar may be incomplete → allow ±2 rows
        assert 4 <= len(df) <= 8, f"Expected ~6 bars for -6h/1h, got {len(df)}"

    def test_read_ohlcv_uppercase_timeframe(self):
        """Upper-case timeframe string '1H' is treated as a 1-hour delta (not year-0001)."""
        df = self.reader.read("BTCUSDT", "ohlc(1H)", "-2h", "now").to_pd()
        logger.info(f"1H OHLCV (len={len(df)}):\n{df}")
        assert 1 <= len(df) <= 4

    def test_read_ohlcv_chunked(self):
        """Chunked reads yield RawData objects that reassemble to the correct total rows."""
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(hours=24)

        chunks = list(self.reader.read("BTCUSDT", "ohlc(1h)", start.isoformat(), end.isoformat(), chunksize=5))  # pyright: ignore[reportArgumentType]
        assert len(chunks) > 0

        df = pd.concat([c.to_pd() for c in chunks])
        logger.info(f"Chunked OHLCV reassembled (len={len(df)}):\n{df.tail()}")
        assert len(df) > 0
        assert df.index.is_monotonic_increasing

    def test_read_multi_symbols(self):
        """Passing a list of symbols returns one RawData per symbol."""
        result = self.reader.read(["BTCUSDT", "ETHUSDT"], "ohlc(1h)", "-6h", "now")
        raws = list(result)
        assert len(raws) == 2
        syms = {r.data_id for r in raws}
        assert "BTCUSDT" in syms
        assert "ETHUSDT" in syms
        for r in raws:
            df = r.to_pd()
            logger.info(f"  {r.data_id}: {len(df)} bars")
            assert len(df) > 0

    # ------------------------------------------------------------------
    # Funding payment reads
    # ------------------------------------------------------------------

    def test_read_funding_payment(self):
        """funding_payment returns rows with funding_rate and funding_interval_hours."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-3d", "now").to_pd()
        logger.info(f"Funding (len={len(df)}):\n{df.tail()}")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "funding_rate" in df.columns
        assert "funding_interval_hours" in df.columns
        assert df.index.is_monotonic_increasing
        # - 3 days / 8h interval = 9 payments; allow some slack
        assert len(df) >= 6, f"Expected ≥6 funding rows for 3d, got {len(df)}"

    def test_read_funding_payment_short_delta(self):
        """Negative sub-day delta '-12h' returns only 1-2 funding rows."""
        df = self.reader.read("BTCUSDT", "funding_payment", "-12h", "now").to_pd()
        logger.info(f"12h Funding (len={len(df)}):\n{df}")
        # - 12h / 8h interval = 1-2 rows
        assert 0 < len(df) <= 4, f"Expected 1-2 funding rows for -12h, got {len(df)}"

    def test_read_funding_payment_chunked(self):
        """Chunked funding reads yield RawData chunks that reassemble correctly."""
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=3)

        chunks = list(
            self.reader.read(
                "BTCUSDT",
                "funding_payment",
                start.isoformat(),
                end.isoformat(),
                chunksize=3,
            )
        )
        assert len(chunks) > 0
        df = pd.concat([c.to_pd() for c in chunks])
        logger.info(f"Chunked funding reassembled (len={len(df)}):\n{df}")
        assert len(df) > 0
        assert "funding_rate" in df.columns
