from functools import lru_cache

from .client import LighterClient
from .instruments import LighterInstrumentLoader
from .websocket import LighterWebSocketManager


@lru_cache(maxsize=32)
def get_lighter_client(
    api_key: str,
    private_key: str,
    account_index: int = 0,
    api_key_index: int = 0,
    testnet: bool = False,
) -> LighterClient:
    """
    Get a LighterClient instance with caching.

    Clients are cached by (api_key, private_key, account_index, api_key_index, testnet)
    to ensure rate limit tracking is shared across all components.
    """
    return LighterClient(
        api_key=api_key,
        private_key=private_key,
        account_index=account_index,
        api_key_index=api_key_index,
        testnet=testnet,
        loop=None,
    )


@lru_cache(maxsize=32)
def get_lighter_ws_manager(
    api_key: str,
    private_key: str,
    account_index: int = 0,
    api_key_index: int = 0,
    testnet: bool = False,
) -> LighterWebSocketManager:
    """
    Get a LighterWebSocketManager instance with caching.

    WebSocket managers are cached to ensure connection sharing across components.
    """
    client = get_lighter_client(
        api_key=api_key,
        private_key=private_key,
        account_index=account_index,
        api_key_index=api_key_index,
        testnet=testnet,
    )
    return LighterWebSocketManager(client=client, testnet=testnet)


@lru_cache(maxsize=1)
def get_lighter_instrument_loader() -> LighterInstrumentLoader:
    """Get a cached LighterInstrumentLoader instance."""
    return LighterInstrumentLoader()


def clear_lighter_cache() -> None:
    """Clear all lighter caches. Useful for testing."""
    get_lighter_client.cache_clear()
    get_lighter_ws_manager.cache_clear()
    get_lighter_instrument_loader.cache_clear()
