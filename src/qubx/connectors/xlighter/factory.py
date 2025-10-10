"""
Factory functions for creating XLighter connector components.

This module provides factory functions that integrate XLighter exchange components
into the Qubx framework, following patterns similar to the CCXT connector.
"""

from qubx.core.basics import CtrlChannel, ITimeProvider
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider

from .account import LighterAccountProcessor
from .broker import LighterBroker
from .client import LighterClient
from .data import LighterDataProvider


def get_xlighter_client(
    api_key: str,
    secret: str,
    account_index: int,
    api_key_index: int | None = None,
    testnet: bool = False,
    **kwargs,
) -> LighterClient:
    """
    Create a LighterClient instance.

    Args:
        api_key: Ethereum address (e.g., "0x...")
        secret: Private key (e.g., "0x...")
        account_index: Lighter account index
        api_key_index: API key index (optional, defaults to 0)
        testnet: Use testnet (not currently supported by Lighter)
        **kwargs: Additional parameters

    Returns:
        Configured LighterClient instance

    Example:
        ```python
        client = get_xlighter_client(
            api_key="0xYourAddress",
            secret="0xYourPrivateKey",
            account_index=225671,
            api_key_index=2
        )
        ```
    """
    if api_key_index is None:
        api_key_index = 0

    # Create client first
    client = LighterClient(
        api_key=api_key,
        private_key=secret,
        account_index=account_index,
        api_key_index=api_key_index,
    )

    # Note: Instruments are loaded on-demand by components that need them
    return client


def get_xlighter_data_provider(
    client: LighterClient,
    time_provider: ITimeProvider,
    channel: CtrlChannel,
    **kwargs,
) -> IDataProvider:
    """
    Create a LighterDataProvider instance.

    Args:
        client: Configured LighterClient
        time_provider: Time provider for timestamps
        channel: Control channel for data distribution
        **kwargs: Additional parameters

    Returns:
        Configured LighterDataProvider instance

    Example:
        ```python
        data_provider = get_xlighter_data_provider(
            client=client,
            time_provider=time_provider,
            channel=channel
        )
        ```
    """
    # Create instrument loader (loads instruments from cache or API)
    from .instruments import LighterInstrumentLoader
    import asyncio
    import concurrent.futures

    instrument_loader = LighterInstrumentLoader(client=client)

    # Load instruments using the client's event loop (required by aiohttp)
    init_future = concurrent.futures.Future()

    def create_load_task():
        """Create load task in the client's event loop"""
        task = asyncio.create_task(instrument_loader.load_instruments())
        task.add_done_callback(
            lambda t: init_future.set_result(None) if not t.exception() else init_future.set_exception(t.exception())
        )

    client._loop.call_soon_threadsafe(create_load_task)
    init_future.result()  # Wait for loading to complete

    return LighterDataProvider(
        client=client,
        instrument_loader=instrument_loader,
        time_provider=time_provider,
        channel=channel,
        loop=client._loop,
    )


def get_xlighter_account(
    client: LighterClient,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    **kwargs,
) -> IAccountProcessor:
    """
    Create a LighterAccountProcessor instance.

    Args:
        client: Configured LighterClient
        channel: Control channel for account events
        time_provider: Time provider for timestamps
        **kwargs: Additional parameters (e.g., base_currency, initial_capital, account_id)

    Returns:
        Configured LighterAccountProcessor instance

    Example:
        ```python
        account = get_xlighter_account(
            client=client,
            channel=channel,
            time_provider=time_provider,
            base_currency="USDC",
            initial_capital=100000.0
        )
        ```
    """
    from .instruments import LighterInstrumentLoader
    from .websocket import LighterWebSocketManager

    # Extract parameters from kwargs
    base_currency = kwargs.get("base_currency", "USDC")
    initial_capital = kwargs.get("initial_capital", 100_000.0)
    account_id = kwargs.get("account_id", str(client.account_index))

    # Create shared components
    import asyncio
    import concurrent.futures

    instrument_loader = LighterInstrumentLoader(client=client)

    # Load instruments using the client's event loop (required by aiohttp)
    init_future = concurrent.futures.Future()

    def create_load_task():
        """Create load task in the client's event loop"""
        task = asyncio.create_task(instrument_loader.load_instruments())
        task.add_done_callback(
            lambda t: init_future.set_result(None) if not t.exception() else init_future.set_exception(t.exception())
        )

    client._loop.call_soon_threadsafe(create_load_task)
    init_future.result()  # Wait for loading to complete

    ws_manager = LighterWebSocketManager()

    return LighterAccountProcessor(
        account_id=account_id,
        client=client,
        instrument_loader=instrument_loader,
        ws_manager=ws_manager,
        channel=channel,
        time_provider=time_provider,
        base_currency=base_currency,
        initial_capital=initial_capital,
    )


def get_xlighter_broker(
    client: LighterClient,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    data_provider: IDataProvider,
    **kwargs,
) -> IBroker:
    """
    Create a LighterBroker instance.

    Args:
        client: Configured LighterClient
        channel: Control channel for order events
        time_provider: Time provider for timestamps
        account: Account processor for order tracking
        data_provider: Data provider for market data
        **kwargs: Additional parameters

    Returns:
        Configured LighterBroker instance

    Example:
        ```python
        broker = get_xlighter_broker(
            client=client,
            channel=channel,
            time_provider=time_provider,
            account=account,
            data_provider=data_provider
        )
        ```
    """
    from .data import LighterDataProvider
    from .instruments import LighterInstrumentLoader

    # Get instrument loader from data provider if available
    import asyncio
    import concurrent.futures

    if isinstance(data_provider, LighterDataProvider):
        instrument_loader = data_provider.instrument_loader
    else:
        # Fallback: create new loader
        instrument_loader = LighterInstrumentLoader(client=client)

        # Load instruments using the client's event loop (required by aiohttp)
        init_future = concurrent.futures.Future()

        def create_load_task():
            """Create load task in the client's event loop"""
            task = asyncio.create_task(instrument_loader.load_instruments())
            task.add_done_callback(
                lambda t: init_future.set_result(None) if not t.exception() else init_future.set_exception(t.exception())
            )

        client._loop.call_soon_threadsafe(create_load_task)
        init_future.result()  # Wait for loading to complete

    return LighterBroker(
        client=client,
        instrument_loader=instrument_loader,
        channel=channel,
        time_provider=time_provider,
        account=account,
        data_provider=data_provider,
    )


def create_xlighter_components(
    api_key: str,
    secret: str,
    account_index: int,
    api_key_index: int | None,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    base_currency: str = "USDC",
    initial_capital: float = 100_000.0,
    testnet: bool = False,
    **kwargs,
) -> tuple[LighterClient, IDataProvider, IAccountProcessor, IBroker]:
    """
    Create all XLighter connector components in the correct order.

    This is a convenience function that creates and wires together all
    components needed for XLighter exchange integration.

    Args:
        api_key: Ethereum address
        secret: Private key
        account_index: Lighter account index
        api_key_index: API key index (optional)
        channel: Control channel
        time_provider: Time provider
        base_currency: Base currency for account (default: USDC)
        initial_capital: Initial capital for paper trading (default: 100000)
        testnet: Use testnet (not currently supported)
        **kwargs: Additional parameters

    Returns:
        Tuple of (client, data_provider, account, broker)

    Example:
        ```python
        client, data_provider, account, broker = create_xlighter_components(
            api_key="0xYourAddress",
            secret="0xYourPrivateKey",
            account_index=225671,
            api_key_index=2,
            channel=channel,
            time_provider=time_provider,
            base_currency="USDC"
        )
        ```
    """
    # Create client
    client = get_xlighter_client(
        api_key=api_key,
        secret=secret,
        account_index=account_index,
        api_key_index=api_key_index,
        testnet=testnet,
        **kwargs,
    )

    # Create data provider
    data_provider = get_xlighter_data_provider(
        client=client, time_provider=time_provider, channel=channel, **kwargs
    )

    # Create account processor
    account = get_xlighter_account(
        client=client,
        channel=channel,
        time_provider=time_provider,
        base_currency=base_currency,
        initial_capital=initial_capital,
        **kwargs,
    )

    # Create broker
    broker = get_xlighter_broker(
        client=client,
        channel=channel,
        time_provider=time_provider,
        account=account,
        data_provider=data_provider,
        **kwargs,
    )

    return client, data_provider, account, broker
