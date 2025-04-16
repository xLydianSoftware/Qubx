import asyncio
from threading import Thread
from typing import Any

import ccxt.pro as cxp
from qubx.connectors.ccxt.broker import CcxtBroker
from qubx.core.basics import CtrlChannel
from qubx.core.interfaces import IAccountProcessor, IBroker, IDataProvider, ITimeProvider

from .account import CcxtAccountProcessor
from .exchanges import CUSTOM_ACCOUNTS, CUSTOM_BROKERS, EXCHANGE_ALIASES


def get_ccxt_exchange(
    exchange: str,
    api_key: str | None = None,
    secret: str | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    use_testnet: bool = False,
    **kwargs,
) -> cxp.Exchange:
    """
    Get a ccxt exchange object with the given api_key and api_secret.
    Parameters:
        exchange (str): The exchange name.
        api_key (str, optional): The API key. Default is None.
        api_secret (str, optional): The API secret. Default is None.
    Returns:
        ccxt.Exchange: The ccxt exchange object.
    """
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

    if use_testnet:
        ccxt_exchange.set_sandbox_mode(True)

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
