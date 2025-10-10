"""
Direct API test to diagnose connection issues with Lighter API.
"""

import asyncio
import aiohttp


async def test_lighter_api():
    """Test direct connection to Lighter API"""

    print("=" * 60)
    print("Testing Direct Lighter API Connection")
    print("=" * 60)

    # Test endpoint
    url = "https://mainnet.zklighter.elliot.ai/api/v1/orderBooks"

    print(f"\n1. Testing GET {url}")
    print("   (This should return market/orderbook data)")

    try:
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print("   - Created aiohttp session")

            async with session.get(url) as response:
                print(f"   - Response status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ SUCCESS! Got {len(data.get('order_books', []))} orderbooks")

                    if data.get('order_books'):
                        first_market = data['order_books'][0]
                        print(f"   - First market: {first_market.get('symbol', 'N/A')}")
                        print(f"   - Market ID: {first_market.get('market_id', 'N/A')}")
                else:
                    text = await response.text()
                    print(f"   ❌ Error response: {text[:200]}")

    except asyncio.TimeoutError:
        print("   ❌ Request timed out!")
    except aiohttp.ClientError as e:
        print(f"   ❌ Connection error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_lighter_api())
