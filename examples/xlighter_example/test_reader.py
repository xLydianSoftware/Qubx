"""
Quick validation script for XLighterDataReader.

Tests:
1. Reader instantiation
2. Fetching OHLC data for BTCUSDC
3. Verifying data format
4. Testing funding payment data (optional)
"""

import asyncio
import sys

from qubx import logger
from qubx.connectors.xlighter.reader import XLighterDataReader


async def test_reader():
    """Test XLighterDataReader functionality"""

    logger.info("=" * 60)
    logger.info("Testing XLighterDataReader")
    logger.info("=" * 60)

    # 1. Test reader instantiation
    logger.info("\n1. Creating XLighterDataReader...")
    try:
        reader = XLighterDataReader(
            # No credentials needed for read-only operations
            max_history="30d",
        )
        logger.info("✅ Reader created successfully")
        logger.info(f"   - Loaded {len(reader.instrument_loader.instruments)} instruments")
    except Exception as e:
        logger.error(f"❌ Failed to create reader: {e}")
        return False

    # 2. Test getting exchange names
    logger.info("\n2. Testing get_names()...")
    try:
        names = reader.get_names()
        assert names == ["LIGHTER"], f"Expected ['LIGHTER'], got {names}"
        logger.info(f"✅ Exchange names: {names}")
    except Exception as e:
        logger.error(f"❌ Failed to get names: {e}")
        return False

    # 3. Test getting symbols
    logger.info("\n3. Testing get_symbols()...")
    try:
        symbols = reader.get_symbols("LIGHTER", "ohlc")
        logger.info(f"✅ Found {len(symbols)} symbols")
        if symbols:
            logger.info(f"   - First 3: {symbols[:3]}")
    except Exception as e:
        logger.error(f"❌ Failed to get symbols: {e}")
        return False

    # 4. Test fetching OHLC data
    logger.info("\n4. Testing OHLC data fetch for BTCUSDC...")
    try:
        import pandas as pd

        # Fetch 1 hour of data
        stop = pd.Timestamp.now()
        start = stop - pd.Timedelta(hours=1)

        data = reader.read(
            data_id="LIGHTER:SWAP:BTCUSDC",
            start=str(start),
            stop=str(stop),
            timeframe="1m",
            data_type="ohlc",
        )

        if hasattr(data, "__len__"):
            logger.info(f"✅ Fetched {len(data)} OHLC records")
            if len(data) > 0:
                logger.info(f"   - First record: {data[0]}")
                logger.info(f"   - Last record: {data[-1]}")
        else:
            logger.warning("⚠️  Data returned as iterator (not counted)")

    except Exception as e:
        logger.error(f"❌ Failed to fetch OHLC data: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 5. Test funding payment data (optional)
    logger.info("\n5. Testing funding payment data...")
    try:
        funding_df = reader.get_funding_payment(
            exchange="LIGHTER",
            symbols=["BTCUSDC"],
            start=str(start),
            stop=str(stop),
        )

        logger.info(f"✅ Fetched {len(funding_df)} funding records")
        if len(funding_df) > 0:
            logger.info(f"   - Columns: {funding_df.columns.tolist()}")
            logger.info(f"   - First record:\n{funding_df.head(1)}")

            # Verify funding interval is 1 hour
            if "funding_interval_hours" in funding_df.columns:
                unique_intervals = funding_df["funding_interval_hours"].unique()
                logger.info(f"   - Funding intervals: {unique_intervals}")
                assert all(
                    interval == 1.0 for interval in unique_intervals
                ), f"Expected 1.0 hour intervals, got {unique_intervals}"
                logger.info("   ✅ Funding interval is 1.0 hours (correct for Lighter)")

    except Exception as e:
        logger.warning(f"⚠️  Funding data test failed (may be expected): {e}")
        # Don't fail the test if funding data is not available

    # 6. Test time ranges
    logger.info("\n6. Testing get_time_ranges()...")
    try:
        start_time, end_time = reader.get_time_ranges("LIGHTER:SWAP:BTCUSDC", "ohlc")
        logger.info(f"✅ Time range: {start_time} to {end_time}")
    except Exception as e:
        logger.error(f"❌ Failed to get time ranges: {e}")
        return False

    # Cleanup
    logger.info("\n7. Closing reader...")
    try:
        reader.close()
        logger.info("✅ Reader closed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to close reader: {e}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests passed!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_reader())
    sys.exit(0 if success else 1)
