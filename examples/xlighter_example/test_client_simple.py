"""
Simple test to check if LighterClient can be instantiated and make basic API calls.
"""

import asyncio
import sys

from qubx import logger
from qubx.connectors.xlighter.client import LighterClient


async def test_client():
    """Test basic LighterClient functionality"""

    logger.info("Creating LighterClient with dummy credentials...")

    # Create client with dummy credentials
    client = LighterClient(
        api_key="0x0000000000000000000000000000000000000000",
        private_key="0" * 64,
        account_index=0,
        api_key_index=0,
        testnet=False,
    )

    logger.info("Client created successfully")

    # Try to fetch markets (read-only operation)
    logger.info("Fetching markets...")
    try:
        markets = await client.get_markets()
        logger.info(f"✅ Found {len(markets)} markets")
        if markets:
            logger.info(f"   - First market: {markets[0].get('symbol', 'N/A')}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to fetch markets: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    success = asyncio.run(test_client())
    sys.exit(0 if success else 1)
