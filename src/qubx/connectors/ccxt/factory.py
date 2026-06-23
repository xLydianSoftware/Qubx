import asyncio
from threading import Thread
from typing import Any

import ccxt.pro as cxp

from qubx.connectors.registry import CredentialsProvider, connector
from qubx.core.basics import CtrlChannel
from qubx.core.interfaces import IDataProvider, IHealthMonitor, ITimeProvider
from qubx.core.mixins.utils import canonical_exchange

from .connector import CcxtConnector
from .exchange_manager import ExchangeManager
from .exchanges import CUSTOM_CONNECTORS, EXCHANGE_ALIASES


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

    # Binance-specific ccxt guards that otherwise raise NotSupported inside the account snapshot
    # (fetch_balance / fetch_positions / fetch_open_orders), leaving balance and positions empty:
    #   - ccxt >= 4.x gates the Binance FUTURES TESTNET behind disableFuturesSandboxWarning; without
    #     it every authenticated fapi call on testnet.binancefuture.com raises "testnet/sandbox mode
    #     is not supported for futures anymore" (see binance.py). We acknowledge the deprecation and
    #     keep using the futures testnet.
    #   - fetch_open_orders without a symbol raises unless warnOnFetchOpenOrdersWithoutSymbol is off;
    #     the snapshot fetches venue-wide, so disable it (applies to prod too).
    if "binance" in ccxt_exchange.id.lower():
        if use_testnet:
            ccxt_exchange.options["disableFuturesSandboxWarning"] = True
        ccxt_exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False

    return ccxt_exchange


_exchange_manager_cache: dict[tuple, ExchangeManager] = {}


def get_ccxt_exchange_manager(
    exchange: str,
    health_monitor: IHealthMonitor,
    time_provider: ITimeProvider,
    api_key: str | None = None,
    secret: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    use_testnet: bool = False,
    check_interval_seconds: float = 30.0,
    **kwargs,
) -> ExchangeManager:
    """
    Get a CCXT exchange with automatic stability management.

    Returns ExchangeManager wrapper that handles exchange recreation
    during data stall scenarios via self-monitoring. Instances are cached
    by (exchange, api_key, secret, use_testnet) to enable sharing across
    data provider, account processor, and broker.
    """
    cache_key = (exchange.lower(), api_key, secret, use_testnet, check_interval_seconds)

    if cache_key in _exchange_manager_cache:
        return _exchange_manager_cache[cache_key]

    factory_params = {
        "exchange": exchange,
        "api_key": api_key,
        "secret": secret,
        "loop": loop,
        "use_testnet": use_testnet,
        **kwargs,
    }

    ccxt_exchange = get_ccxt_exchange(
        exchange=exchange, api_key=api_key, secret=secret, loop=loop, use_testnet=use_testnet, **kwargs
    )

    manager = ExchangeManager(
        exchange_name=exchange,
        factory_params=factory_params,
        health_monitor=health_monitor,
        time_provider=time_provider,
        initial_exchange=ccxt_exchange,
        check_interval_seconds=check_interval_seconds,
    )

    _exchange_manager_cache[cache_key] = manager
    return manager


def clear_exchange_manager_cache() -> None:
    """Clear the exchange manager cache. Useful for testing."""
    _exchange_manager_cache.clear()


def get_ccxt_connector(
    exchange_name: str,
    **kwargs,
) -> CcxtConnector:
    """Construct the right CcxtConnector subclass for the exchange.

    Resolves the per-exchange subclass from ``CUSTOM_CONNECTORS`` keyed by the
    lowercased framework exchange name (OKX/Bitfinex get the split orders/fills
    streams), falling back to the base ``CcxtConnector`` for any unlisted exchange
    (Binance, Hyperliquid, ...). The ``CUSTOM_CONNECTORS`` map carries both the dotted
    (``okx.f``) and bare (``okx``) names, like ``EXCHANGE_ALIASES``.
    """
    connector_cls = CUSTOM_CONNECTORS.get(exchange_name.lower(), CcxtConnector)
    return connector_cls(exchange_name=exchange_name, **kwargs)


@connector("ccxt")
def create_ccxt_connector(
    exchange_name: str,
    time_provider: ITimeProvider,
    channel: CtrlChannel,
    credentials: CredentialsProvider,
    data_provider: IDataProvider,
    health_monitor: IHealthMonitor,
    loop: asyncio.AbstractEventLoop | None = None,
    **kwargs,
) -> CcxtConnector:
    """Registered ``IConnector`` factory for ccxt venues (``ConnectorRegistry.get_connector('ccxt')``).

    Builds the authenticated ccxt ExchangeManager from the venue credentials — a separate
    cached manager from the unauthenticated one ``CcxtDataProvider`` uses for market data
    (the manager cache keys on api_key/secret) — and resolves the per-exchange
    ``CcxtConnector`` subclass via ``get_ccxt_connector``.
    """
    creds = credentials.get_exchange_credentials(exchange_name)
    exchange_manager = get_ccxt_exchange_manager(
        exchange=exchange_name,
        use_testnet=creds.testnet,
        api_key=creds.api_key,
        secret=creds.secret,
        health_monitor=health_monitor,
        time_provider=time_provider,
        loop=loop,
        **(creds.model_extra or {}),
    )
    # The connector self-reports the canonical (instrument-universe) exchange so the
    # account events it stamps (balances/snapshots) route to the same AM state its
    # instruments do: a BINANCE.PM account trades BINANCE.UM instruments — the venue
    # name is plumbing only (credentials lookup + ccxt exchange class, both above).
    return get_ccxt_connector(
        canonical_exchange(exchange_name),
        channel=channel,
        time_provider=time_provider,
        exchange_manager=exchange_manager,
        data_provider=data_provider,
        loop=loop,
    )


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
