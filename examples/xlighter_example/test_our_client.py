"""
Test our LighterClient wrapper to find where it hangs.
"""

import asyncio
from qubx import logger
from qubx.connectors.xlighter.client import LighterClient


async def test_our_client():
    """Test our LighterClient wrapper"""

    print("=" * 60)
    print("Testing Our LighterClient Wrapper")
    print("=" * 60)

    print("\n1. Creating LighterClient...")
    try:
        client = LighterClient(
            api_key="0x0000000000000000000000000000000000000000",
            private_key="0" * 64,
            account_index=0,
            api_key_index=0,
            testnet=False,
        )
        print("   ✅ Client created")
    except Exception as e:
        print(f"   ❌ Failed to create client: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n2. Fetching markets with timeout...")
    try:
        markets = await asyncio.wait_for(client.get_markets(), timeout=10.0)
        print(f"   ✅ Got {len(markets)} markets")
        if markets:
            print(f"   - First market: {markets[0].get('symbol', 'N/A')}")
            print(f"   - Market ID: {markets[0].get('id', 'N/A')}")
    except asyncio.TimeoutError:
        print("   ❌ get_markets() timed out!")
    except Exception as e:
        print(f"   ❌ Error in get_markets(): {e}")
        import traceback
        traceback.print_exc()

    print("\n3. Closing client...")
    try:
        await client.close()
        print("   ✅ Client closed")
    except Exception as e:
        print(f"   ❌ Error closing: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_our_client())
