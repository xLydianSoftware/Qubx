import asyncio
from threading import Thread
from typing import Any

import ccxt.pro as cxp

from .customizations import BinancePortfolioMargin, BinanceQV, BinanceQVUSDM

EXCHANGE_ALIASES = {
    "binance": "binanceqv",
    "binance.um": "binanceqv_usdm",
    "binance.cm": "binancecoinm",
    "binance.pm": "binancepm",
    "kraken.f": "krakenfutures",
}

cxp.binanceqv = BinanceQV  # type: ignore
cxp.binanceqv_usdm = BinanceQVUSDM  # type: ignore
cxp.binancepm = BinancePortfolioMargin  # type: ignore

cxp.exchanges.append("binanceqv")
cxp.exchanges.append("binanceqv_usdm")
cxp.exchanges.append("binancepm")
cxp.exchanges.append("binancepm_usdm")


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
