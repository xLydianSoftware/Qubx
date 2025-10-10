"""
Test the lighter-python SDK directly to see if it works.
"""

import asyncio
from lighter import ApiClient, Configuration, OrderApi


async def test_lighter_sdk():
    """Test lighter-python SDK"""

    print("=" * 60)
    print("Testing Lighter Python SDK")
    print("=" * 60)

    print("\n1. Creating SDK client...")
    config = Configuration(host="https://mainnet.zklighter.elliot.ai")
    api_client = ApiClient(configuration=config)
    order_api = OrderApi(api_client)

    print("   ✅ SDK client created")

    print("\n2. Fetching orderbooks...")
    try:
        response = await asyncio.wait_for(order_api.order_books(), timeout=10.0)

        if hasattr(response, 'order_books'):
            print(f"   ✅ Got {len(response.order_books)} orderbooks")
            if response.order_books:
                first = response.order_books[0]
                print(f"   - First market: {getattr(first, 'symbol', 'N/A')}")
                print(f"   - Market ID: {getattr(first, 'market_id', 'N/A')}")
        else:
            print(f"   ⚠️  Unexpected response format: {type(response)}")

    except asyncio.TimeoutError:
        print("   ❌ Request timed out!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n3. Closing client...")
        await api_client.close()
        print("   ✅ Client closed")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_lighter_sdk())
