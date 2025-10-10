"""
Test XLighterDataReader initialization to find where it hangs.
"""

import asyncio
from qubx import logger
from qubx.connectors.xlighter.reader import XLighterDataReader


def test_reader_init():
    """Test reader initialization"""

    print("=" * 60)
    print("Testing XLighterDataReader Initialization")
    print("=" * 60)

    print("\n1. Creating XLighterDataReader...")
    print("   (This loads instruments via AsyncThreadLoop)")

    try:
        # Set a timer to see how long this takes
        import time
        start = time.time()

        reader = XLighterDataReader(max_history="30d")

        elapsed = time.time() - start
        print(f"   ✅ Reader created in {elapsed:.2f}s")
        print(f"   - Loaded {len(reader.instrument_loader.instruments)} instruments")

        # Test a simple method
        print("\n2. Testing get_names()...")
        names = reader.get_names()
        print(f"   ✅ Got names: {names}")

        print("\n3. Closing reader...")
        reader.close()
        print("   ✅ Reader closed")

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        return True

    except KeyboardInterrupt:
        print("\n   ❌ Interrupted by user")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_reader_init()
    sys.exit(0 if success else 1)
