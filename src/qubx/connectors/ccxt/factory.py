import asyncio
from threading import Thread
from typing import Any

import ccxt.pro as cxp
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.core.basics import CtrlChannel
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider, ITimeProvider

from .account import CcxtAccountProcessor
from .exchanges import CUSTOM_ACCOUNTS, CUSTOM_BROKERS, EXCHANGE_ALIASES
from .exchange_manager import ExchangeManager


def get_ccxt_exchange(
    exchange: str,
    api_key: str | None = None,
    secret: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    use_testnet: bool = False,
    **kwargs,
) -> cxp.Exchange:
    """
    Get a raw CCXT exchange object.
    
    Creates and configures a CCXT exchange instance without any stability wrapper.
    Use get_ccxt_exchange_manager() if you need automatic stability management.
    
    Parameters:
        exchange (str): The exchange name.
        api_key (str, optional): The API key. Default is None.
        secret (str, optional): The API secret. Default is None.
        loop (asyncio.AbstractEventLoop, optional): Event loop. Default is None.
        use_testnet (bool): Use testnet/sandbox mode. Default is False.
        **kwargs: Additional parameters for exchange configuration.
        
    Returns:
        Raw CCXT Exchange instance
    """
    # Resolve exchange ID
    _exchange = exchange.lower()
    if kwargs.get("enable_mm", False):
        _exchange = f"{_exchange}.mm"

    _exchange = EXCHANGE_ALIASES.get(_exchange, _exchange)

    if _exchange not in cxp.exchanges:
        raise ValueError(f"Exchange {exchange} is not supported by ccxt.")

    # Build exchange options
    options: dict[str, Any] = {"name": exchange}

    if loop is not None:
        options["asyncio_loop"] = loop
    else:
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        options["thread_asyncio_loop"] = thread
        options["asyncio_loop"] = loop

    # Add API credentials
    api_key, secret = _get_api_credentials(api_key, secret, kwargs)
    if api_key and secret:
        options["apiKey"] = api_key
        options["secret"] = secret

    # Create raw CCXT exchange
    ccxt_exchange = getattr(cxp, _exchange)(options | kwargs)

    # Apply post-creation configuration
    if ccxt_exchange.name.startswith("HYPERLIQUID") and api_key and secret:
        ccxt_exchange.walletAddress = api_key
        ccxt_exchange.privateKey = secret

    if use_testnet:
        ccxt_exchange.set_sandbox_mode(True)

    return ccxt_exchange


def get_ccxt_exchange_manager(
    exchange: str,
    api_key: str | None = None,
    secret: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    use_testnet: bool = False,
    max_recreations: int = 3,
    reset_interval_hours: float = 24.0,
    check_interval_seconds: float = 30.0,
    **kwargs,
) -> ExchangeManager:
    """
    Get a CCXT exchange with automatic stability management.
    
    Returns ExchangeManager wrapper that handles exchange recreation
    during data stall scenarios via self-monitoring.
    
    Parameters:
        exchange (str): The exchange name.
        api_key (str, optional): The API key. Default is None.
        secret (str, optional): The API secret. Default is None.
        loop (asyncio.AbstractEventLoop, optional): Event loop. Default is None.
        use_testnet (bool): Use testnet/sandbox mode. Default is False.
        max_recreations (int): Maximum recreation attempts before circuit breaker. Default is 3.
        reset_interval_hours (float): Hours between recreation count resets. Default is 24.0.
        check_interval_seconds (float): How often to check for stalls. Default is 30.0.
        **kwargs: Additional parameters for exchange configuration.
        
    Returns:
        ExchangeManager wrapping the CCXT Exchange
    """
    # Prepare factory parameters for ExchangeManager recreation
    factory_params = {
        'exchange': exchange,
        'api_key': api_key,  
        'secret': secret,
        'loop': loop,
        'use_testnet': use_testnet,
        **{k: v for k, v in kwargs.items() if k not in {
            'max_recreations', 'reset_interval_hours', 'check_interval_seconds'
        }}
    }
    
    # Create raw CCXT exchange using public factory method
    ccxt_exchange = get_ccxt_exchange(
        exchange=exchange,
        api_key=api_key,
        secret=secret,
        loop=loop,
        use_testnet=use_testnet,
        **kwargs
    )

    # Wrap in ExchangeManager for stability management
    return ExchangeManager(
        exchange_name=exchange,
        factory_params=factory_params,
        initial_exchange=ccxt_exchange,
        max_recreations=max_recreations,
        reset_interval_hours=reset_interval_hours,
        check_interval_seconds=check_interval_seconds,
    )


def get_ccxt_broker(
    exchange_name: str,
    exchange_manager: ExchangeManager,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    data_provider: IDataProvider,
    **kwargs,
) -> IBroker:
    broker_cls = CUSTOM_BROKERS.get(exchange_name.lower(), CcxtBroker)
    return broker_cls(exchange_manager, channel, time_provider, account, data_provider, **kwargs)


def get_ccxt_account(
    exchange_name: str,
    **kwargs,
) -> IAccountProcessor:
    account_cls = CUSTOM_ACCOUNTS.get(exchange_name.lower(), CcxtAccountProcessor)
    return account_cls(**kwargs)


def _get_api_credentials(
    api_key: str | None, secret: str | None, kwargs: dict[str, Any]
) -> tuple[str | None, str | None]:
    if api_key is None:
        if "apiKey" in kwargs:
            api_key = kwargs.pop("apiKey")
        elif "key" in kwargs:
            api_key = kwargs.pop("key")
        elif "API_KEY" in kwargs:
            api_key = kwargs.get("API_KEY")
    if secret is None:
        if "secret" in kwargs:
            secret = kwargs.pop("secret")
        elif "apiSecret" in kwargs:
            secret = kwargs.pop("apiSecret")
        elif "API_SECRET" in kwargs:
            secret = kwargs.get("API_SECRET")
        elif "SECRET" in kwargs:
            secret = kwargs.get("SECRET")
    return api_key, secret
