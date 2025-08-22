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
    # Stability configuration
    enable_stability_manager: bool = True,
    max_recreations: int = 3,
    reset_interval_hours: float = 4.0,
    **kwargs,
) -> cxp.Exchange:  # Return type stays same for compatibility
    """
    Get a CCXT exchange object with automatic stability management.
    
    Returns ExchangeManager wrapper that transparently handles exchange recreation
    during data stall scenarios via BaseHealthMonitor integration.
    
    Parameters:
        exchange (str): The exchange name.
        api_key (str, optional): The API key. Default is None.
        secret (str, optional): The API secret. Default is None.
        loop (asyncio.AbstractEventLoop, optional): Event loop. Default is None.
        use_testnet (bool): Use testnet/sandbox mode. Default is False.
        enable_stability_manager (bool): Enable recreation management. Default is True.
        max_recreations (int): Maximum recreation attempts before circuit breaker. Default is 3.
        reset_interval_hours (float): Hours between recreation count resets. Default is 24.0.
        
    Returns:
        ExchangeManager or raw CCXT Exchange (transparent to caller)
    """
    
    # Prepare factory parameters for ExchangeManager recreation
    factory_params = {
        'exchange': exchange,
        'api_key': api_key,  
        'secret': secret,
        'loop': loop,
        'use_testnet': use_testnet,
        'enable_stability_manager': False,  # Prevent recursive wrapping
        **{k: v for k, v in kwargs.items() if k not in {
            'max_recreations', 'reset_interval_hours'
        }}
    }
    
    # Create raw CCXT exchange (existing logic unchanged)
    _exchange = exchange.lower()
    if kwargs.get("enable_mm", False):
        _exchange = f"{_exchange}.mm"

    _exchange = EXCHANGE_ALIASES.get(_exchange, _exchange)

    if _exchange not in cxp.exchanges:
        raise ValueError(f"Exchange {exchange} is not supported by ccxt.")

    options: dict[str, Any] = {"name": exchange}

    if loop is not None:
        options["asyncio_loop"] = loop
    else:
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        options["thread_asyncio_loop"] = thread
        options["asyncio_loop"] = loop

    api_key, secret = _get_api_credentials(api_key, secret, kwargs)
    if api_key and secret:
        options["apiKey"] = api_key
        options["secret"] = secret

    ccxt_exchange = getattr(cxp, _exchange)(options | kwargs)

    if ccxt_exchange.name.startswith("HYPERLIQUID") and api_key and secret:
        ccxt_exchange.walletAddress = api_key
        ccxt_exchange.privateKey = secret

    if use_testnet:
        ccxt_exchange.set_sandbox_mode(True)

    # Wrap in ExchangeManager if stability management enabled
    if enable_stability_manager:
        return ExchangeManager(  # type: ignore  # ExchangeManager is transparent proxy for Exchange
            exchange_name=exchange,
            factory_params=factory_params,
            initial_exchange=ccxt_exchange,
            max_recreations=max_recreations,
            reset_interval_hours=reset_interval_hours,
        )
    else:
        # Return raw exchange for backwards compatibility or testing
        return ccxt_exchange


def get_ccxt_broker(
    exchange_name: str,
    exchange: cxp.Exchange,
    channel: CtrlChannel,
    time_provider: ITimeProvider,
    account: IAccountProcessor,
    data_provider: IDataProvider,
    **kwargs,
) -> IBroker:
    broker_cls = CUSTOM_BROKERS.get(exchange_name.lower(), CcxtBroker)
    return broker_cls(exchange, channel, time_provider, account, data_provider, **kwargs)


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
