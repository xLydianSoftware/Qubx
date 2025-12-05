"""
Factory functions for creating XLighter connector components.

This module provides factory functions that integrate XLighter exchange components
into the Qubx framework, following patterns similar to the CCXT connector.
"""

from qubx import logger
from qubx.core.basics import CtrlChannel, ITimeProvider
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider, IHealthMonitor

from .account import LighterAccountProcessor
from .broker import LighterBroker
from .client import LighterClient
from .data import LighterDataProvider
from .instruments import LighterInstrumentLoader
from .websocket import LighterWebSocketManager


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
        api_key: Lighter API key (Ethereum address)
        secret: Private key for signing
        account_index: Lighter account index
        api_key_index: API key index (default: 0)
        testnet: If True, use testnet (default: False)
        **kwargs: Additional parameters including:
            - account_type: "premium" or "standard" (default: "premium")
            - rest_rate_limit: Override REST rate limit in req/min (optional)

    Returns:
        Configured LighterClient instance
    """
    return LighterClient(
        api_key=api_key,
        private_key=secret,
        account_index=account_index,
        api_key_index=api_key_index or 0,
        testnet=testnet,
        account_type=kwargs.get("account_type", "premium"),
        rest_rate_limit=kwargs.get("rest_rate_limit"),
    )


def get_xlighter_data_provider(
    client: LighterClient,
    time_provider: ITimeProvider,
    channel: CtrlChannel,
    ws_manager: "LighterWebSocketManager | None" = None,
    instrument_loader: "LighterInstrumentLoader | None" = None,
    health_monitor: "IHealthMonitor | None" = None,
    **kwargs,
) -> IDataProvider:
    """
    Create a LighterDataProvider instance.

    Args:
        client: Configured LighterClient
        time_provider: Time provider for timestamps
        channel: Control channel for data distribution
        ws_manager: WebSocket manager (optional, created if not provided)
            WARNING: If creating multiple components (broker, account), create a shared
            instance and pass it to all components to ensure proper resource sharing!
        instrument_loader: Instrument loader (optional, created if not provided)
        **kwargs: Additional parameters

    Returns:
        Configured LighterDataProvider instance

    Example:
        ```python
        # Standalone data provider (creates own ws_manager)
        data_provider = get_xlighter_data_provider(
            client=client,
            time_provider=time_provider,
            channel=channel
        )

        # Multiple components (share ws_manager) - RECOMMENDED
        ws_manager = LighterWebSocketManager()
        data_provider = get_xlighter_data_provider(
            client=client,
            time_provider=time_provider,
            channel=channel,
            ws_manager=ws_manager
        )
        ```
    """
    if ws_manager is None:
        ws_manager = LighterWebSocketManager(
            client=client,
            testnet=kwargs.get("testnet", False),
            ws_subscription_rate_limit=kwargs.get("ws_subscription_rate_limit"),
        )
        logger.warning(
            "Creating new WebSocket manager for data provider. "
            "If you're creating multiple components (broker, account), "
            "consider creating a shared WebSocket manager and passing it to all components!"
        )

    if instrument_loader is None:
        instrument_loader = LighterInstrumentLoader()

    return LighterDataProvider(
        client=client,
        instrument_loader=instrument_loader,
        time_provider=time_provider,
        channel=channel,
        loop=client._loop,
        ws_manager=ws_manager,
        health_monitor=health_monitor,
    )


def get_xlighter_account(
    client: LighterClient,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    health_monitor: IHealthMonitor,
    ws_manager: "LighterWebSocketManager | None" = None,
    instrument_loader: "LighterInstrumentLoader | None" = None,
    **kwargs,
) -> IAccountProcessor:
    """
    Create a LighterAccountProcessor instance.

    Args:
        client: Configured LighterClient
        channel: Control channel for account events
        time_provider: Time provider for timestamps
        ws_manager: WebSocket manager (optional, created if not provided)
            WARNING: If creating multiple components (broker, data_provider), create a shared
            instance and pass it to all components to ensure proper resource sharing!
        instrument_loader: Instrument loader (optional, created if not provided)
        **kwargs: Additional parameters (e.g., base_currency, initial_capital, account_id)

    Returns:
        Configured LighterAccountProcessor instance

    Example:
        ```python
        # Multiple components (share ws_manager) - RECOMMENDED
        ws_manager = LighterWebSocketManager()
        instrument_loader = LighterInstrumentLoader(client)
        account = get_xlighter_account(
            client=client,
            channel=channel,
            time_provider=time_provider,
            ws_manager=ws_manager,
            instrument_loader=instrument_loader,
            base_currency="USDC",
            initial_capital=100000.0
        )
        ```
    """

    # Create WebSocket manager if not provided
    if ws_manager is None:
        from .websocket import LighterWebSocketManager

        testnet = kwargs.get("testnet", False)
        ws_manager = LighterWebSocketManager(
            client=client,
            testnet=testnet,
            ws_subscription_rate_limit=kwargs.get("ws_subscription_rate_limit"),
        )
        logger.warning(
            "Creating new WebSocket manager for account processor. "
            "If you're creating multiple components (broker, data_provider), "
            "consider creating a shared WebSocket manager and passing it to all components!"
        )

    if instrument_loader is None:
        instrument_loader = LighterInstrumentLoader()

    # Extract parameters from kwargs
    base_currency = kwargs.get("base_currency", "USDC")
    initial_capital = kwargs.get("initial_capital", 100_000.0)
    account_id = kwargs.get("account_id", str(client.account_index))
    restored_state = kwargs.get("restored_state")

    return LighterAccountProcessor(
        account_id=account_id,
        client=client,
        instrument_loader=instrument_loader,
        ws_manager=ws_manager,
        channel=channel,
        time_provider=time_provider,
        loop=client._loop,
        health_monitor=health_monitor,
        base_currency=base_currency,
        initial_capital=initial_capital,
        restored_state=restored_state,
    )


def get_xlighter_broker(
    client: LighterClient,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    data_provider: IDataProvider,
    ws_manager: "LighterWebSocketManager",
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
        ws_manager: WebSocket manager (optional, tries to get from data_provider or creates new)
            WARNING: Should be the same instance used by account and data_provider!
            Create a shared instance and pass it to all components to ensure proper resource sharing.
        instrument_loader: Instrument loader (optional, tries to get from data_provider or creates new)
        **kwargs: Additional parameters
    """
    return LighterBroker(
        client=client,
        ws_manager=ws_manager,
        channel=channel,
        time_provider=time_provider,
        account=account,
        data_provider=data_provider,
        loop=client._loop,
    )
