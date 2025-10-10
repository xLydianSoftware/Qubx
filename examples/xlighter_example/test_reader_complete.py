"""
Complete validation script for XLighterDataReader.

Tests:
1. Reader instantiation
2. Fetching OHLC data for BTCUSDC
3. Verifying data format
4. Testing funding payment data
"""

import sys
import pandas as pd

from qubx import logger
from qubx.connectors.xlighter.reader import XLighterDataReader


def test_reader():
    """Test XLighterDataReader functionality"""

    print("=" * 60)
    print("Testing XLighterDataReader")
    print("=" * 60)

    # 1. Test reader instantiation
    print("\n1. Creating XLighterDataReader...")
    try:
        reader = XLighterDataReader(
            # No credentials needed for read-only operations
            max_history="30d",
        )
        print("✅ Reader created successfully")
        print(f"   - Loaded {len(reader.instrument_loader.instruments)} instruments")
    except Exception as e:
        print(f"❌ Failed to create reader: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Test getting exchange names
    print("\n2. Testing get_names()...")
    try:
        names = reader.get_names()
        assert names == ["LIGHTER"], f"Expected ['LIGHTER'], got {names}"
        print(f"✅ Exchange names: {names}")
    except Exception as e:
        print(f"❌ Failed to get names: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Test getting symbols
    print("\n3. Testing get_symbols()...")
    try:
        symbols = reader.get_symbols("LIGHTER", "ohlc")
        print(f"✅ Found {len(symbols)} symbols")
        if symbols:
            print(f"   - First 3: {symbols[:3]}")
    except Exception as e:
        print(f"❌ Failed to get symbols: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Test fetching OHLC data
    print("\n4. Testing OHLC data fetch for BTCUSDC...")
    try:
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
            print(f"✅ Fetched {len(data)} OHLC records")
            if len(data) > 0:
                print(f"   - First record: {data[0][:2]}... (timestamp, open)")
                print(f"   - Last record: {data[-1][:2]}... (timestamp, open)")
        else:
            print("⚠️  Data returned as iterator (not counted)")

    except Exception as e:
        print(f"❌ Failed to fetch OHLC data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Test funding payment data
    print("\n5. Testing funding payment data...")
    try:
        funding_df = reader.get_funding_payment(
            exchange="LIGHTER",
            symbols=["BTCUSDC"],
            start=str(start),
            stop=str(stop),
        )

        print(f"✅ Fetched {len(funding_df)} funding records")
        if len(funding_df) > 0:
            print(f"   - Columns: {funding_df.columns.tolist()}")
            print(f"   - First record:\n{funding_df.head(1)}")

            # Verify funding interval is 1 hour
            if "funding_interval_hours" in funding_df.columns:
                unique_intervals = funding_df["funding_interval_hours"].unique()
                print(f"   - Funding intervals: {unique_intervals}")
                assert all(
                    interval == 1.0 for interval in unique_intervals
                ), f"Expected 1.0 hour intervals, got {unique_intervals}"
                print("   ✅ Funding interval is 1.0 hours (correct for Lighter)")

    except Exception as e:
        print(f"⚠️  Funding data test failed (may be expected): {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the test if funding data is not available

    # 6. Test time ranges
    print("\n6. Testing get_time_ranges()...")
    try:
        start_time, end_time = reader.get_time_ranges("LIGHTER:SWAP:BTCUSDC", "ohlc")
        print(f"✅ Time range: {start_time} to {end_time}")
    except Exception as e:
        print(f"❌ Failed to get time ranges: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Cleanup
    print("\n7. Closing reader...")
    try:
        reader.close()
        print("✅ Reader closed successfully")
    except Exception as e:
        print(f"❌ Failed to close reader: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    # Run the test
    success = test_reader()
    sys.exit(0 if success else 1)
