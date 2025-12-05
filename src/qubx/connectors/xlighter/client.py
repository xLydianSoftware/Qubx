"""Lighter API client wrapper"""

import asyncio
import logging
import threading
import time
from typing import Awaitable, Optional, TypeVar, cast

import pandas as pd
from lighter import (  # type: ignore
    AccountApi,
    ApiClient,
    CandlestickApi,
    Configuration,
    FundingApi,
    InfoApi,
    OrderApi,
    SignerClient,
    TransactionApi,
)

# Reset logging - lighter SDK sets root logger to DEBUG on import
logging.root.setLevel(logging.WARNING)

from qubx import logger
from qubx.utils.rate_limiter import rate_limited

from .constants import API_BASE_MAINNET, API_BASE_TESTNET
from .rate_limits import (
    WEIGHT_CANDLESTICKS,
    WEIGHT_DEFAULT,
    WEIGHT_FUNDING,
    create_lighter_rate_limiters,
)

T = TypeVar("T")


class LighterClient:
    """
    Wrapper for Lighter SDK client.

    Provides:
    - REST API access (market data, account info, orders)
    - Order signing and submission
    - Account management

    Usage:
        ```python
        client = LighterClient(
            api_key="0xAddress",
            private_key="0xPrivateKey",
            account_index=225671,
            api_key_index=2,
            testnet=False
        )

        # Get markets
        markets = client.get_markets()

        # Get orderbook
        ob = client.get_orderbook(market_id=0)
        ```
    """

    _config: Configuration
    _account_api: AccountApi
    _api_client: ApiClient
    _info_api: InfoApi
    _order_api: OrderApi
    _candlestick_api: CandlestickApi
    _funding_api: FundingApi
    _transaction_api: TransactionApi
    signer_client: SignerClient

    def __init__(
        self,
        api_key: str,
        private_key: str,
        account_index: int,
        api_key_index: int = 0,
        testnet: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        account_type: str = "premium",
        rest_rate_limit: int | None = None,
    ):
        """
        Initialize Lighter client.

        Args:
            api_key: Lighter API key (Ethereum address)
            private_key: Private key for signing (without 0x prefix)
            account_index: Lighter account index
            api_key_index: API key index for the account
            testnet: If True, use testnet. Otherwise mainnet.
            loop: Event loop to use (optional)
            account_type: "premium" or "standard" for rate limiting (default: "premium")
            rest_rate_limit: Override REST API rate limit in requests/minute (optional)
        """
        self.api_key = api_key
        self.private_key = private_key.replace("0x", "")  # Remove 0x prefix if present
        self.account_index = account_index
        self.api_key_index = api_key_index
        self.testnet = testnet

        # Determine URLs
        self.api_url = API_BASE_TESTNET if testnet else API_BASE_MAINNET

        # Create and start event loop in background thread
        # This is required because lighter-python SDK needs a running event loop
        if loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self._loop_thread.start()
            time.sleep(0.1)
        else:
            self._loop = loop
            self._loop_thread = None

        # Build SDK objects on the client loop
        async def _init_sdks():
            self._config = Configuration(host=self.api_url)
            self._api_client = ApiClient(configuration=self._config)
            self._account_api = AccountApi(self._api_client)
            self._info_api = InfoApi(self._api_client)
            self._order_api = OrderApi(self._api_client)
            self._candlestick_api = CandlestickApi(self._api_client)
            self._funding_api = FundingApi(self._api_client)
            self._transaction_api = TransactionApi(self._api_client)
            self.signer_client = SignerClient(
                url=self.api_url,
                private_key=self.private_key,
                api_key_index=self.api_key_index,
                account_index=self.account_index,
            )

        asyncio.run_coroutine_threadsafe(_init_sdks(), self._loop).result()

        # Initialize rate limiters
        self._rate_limiters = create_lighter_rate_limiters(
            account_type=account_type,
            rest_rate_limit=rest_rate_limit,
        )

        logger.info(
            f"Initialized LighterClient (testnet={testnet}, account_index={account_index}, "
            f"api_key_index={api_key_index}, account_type={account_type})"
        )

    def _run_event_loop(self):
        """Run the event loop in a background thread"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _run_on_client_loop(self, coro: Awaitable[T]) -> T:
        try:
            # If we're already on the client loop, just await directly.
            if asyncio.get_running_loop() is self._loop:
                return await coro
        except RuntimeError:
            # No running loop in this thread; fall through to thread-safe submit.
            pass

        # Submit to the client loop from another loop or a sync/thread context.
        cfut = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore
        # Make it awaitable from the *caller*'s loop:
        return await asyncio.wrap_future(cfut)

    @rate_limited("rest", weight=WEIGHT_DEFAULT * 2)
    async def get_markets(self) -> list[dict]:
        """
        Get list of all markets.

        Returns:
            List of market dictionaries with metadata
        """

        async def _get_markets_impl():
            response = await self._order_api.order_books()
            if hasattr(response, "order_books"):
                # Convert OrderBook objects to dicts and normalize field names
                markets = []
                for ob in response.order_books:
                    market_dict = ob.to_dict() if hasattr(ob, "to_dict") else ob.model_dump()
                    # Normalize field names: market_id -> id
                    if "market_id" in market_dict and "id" not in market_dict:
                        market_dict["id"] = market_dict["market_id"]
                    markets.append(market_dict)
                return markets
            return []

        try:
            return await self._run_on_client_loop(_get_markets_impl())
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            raise

    async def next_nonce(self) -> int:
        """
        Get next nonce for the account.
        """
        response = await self._run_on_client_loop(
            self._transaction_api.next_nonce(account_index=self.account_index, api_key_index=self.api_key_index)
        )
        return response.nonce

    async def get_market_info(self, market_id: int) -> Optional[dict]:
        """
        Get information for a specific market.

        Args:
            market_id: Market ID

        Returns:
            Market info dict or None if not found
        """
        markets = await self.get_markets()
        for market in markets:
            if market.get("id") == market_id:
                return market
        return None

    @rate_limited("rest", weight=WEIGHT_CANDLESTICKS * 2.0)
    async def get_candlesticks(
        self,
        market_id: int,
        resolution: str = "1h",
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
        count_back: int = 1000,
    ) -> list[dict]:
        """
        Get historical candlestick data for a market.

        Args:
            market_id: Market ID
            resolution: Candlestick resolution (e.g., "1m", "5m", "1h", "1d")
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            count_back: Number of candles to fetch

        Returns:
            List of candlestick dictionaries with timestamp, open, high, low, close, volume
        """
        try:
            resolution = resolution.lower()

            # Set default timestamps if not provided
            if end_timestamp is None:
                end_timestamp = int(time.time() * 1000)
            if start_timestamp is None:
                # Default to 1000 periods back
                resolution_ms = self._resolution_to_milliseconds(resolution)
                start_timestamp = end_timestamp - (count_back * resolution_ms)

            start_td = cast(pd.Timestamp, pd.Timestamp(start_timestamp, unit="ms"))
            end_td = cast(pd.Timestamp, pd.Timestamp(end_timestamp, unit="ms"))
            tf = pd.Timedelta(resolution)
            if start_td + tf > end_td:
                start_td = end_td - tf
                start_timestamp = int(start_td.timestamp() * 1000)  # type: ignore

            count_back = int((end_td - start_td) / tf)  # type: ignore

            response = await self._run_on_client_loop(
                self._candlestick_api.candlesticks(
                    market_id=market_id,
                    resolution=resolution,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    count_back=count_back,
                )
            )

            # Convert response to list of dicts
            if hasattr(response, "candlesticks") and response.candlesticks:
                candles = []
                for candle in response.candlesticks:
                    candle_dict = candle.to_dict() if hasattr(candle, "to_dict") else candle.model_dump()
                    candles.append(candle_dict)
                return candles
            return []

        except Exception as e:
            logger.error(f"Failed to get candlesticks for market {market_id}: {e}")
            raise

    @rate_limited("rest", weight=WEIGHT_FUNDING * 2.0)
    async def get_fundings(
        self,
        market_id: int,
        resolution: str = "1h",
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
        count_back: int = 1000,
    ) -> list[dict]:
        """
        Get historical funding rate data for a market.

        Args:
            market_id: Market ID
            resolution: Funding resolution (typically "1h" for Lighter)
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            count_back: Number of funding records to fetch

        Returns:
            List of funding dictionaries
        """
        try:
            resolution_ms = self._resolution_to_milliseconds(resolution)
            if end_timestamp is None:
                end_timestamp = int(time.time() * 1000)
            if start_timestamp is None:
                start_timestamp = end_timestamp - (count_back * resolution_ms)

            count_back = int((end_timestamp - start_timestamp) / resolution_ms)

            response = await self._run_on_client_loop(
                self._candlestick_api.fundings(
                    market_id=market_id,
                    resolution=resolution,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    count_back=count_back,
                )
            )

            # Convert response to list of dicts
            if hasattr(response, "fundings") and response.fundings:
                fundings = []
                for funding in response.fundings:
                    funding_dict = funding.to_dict() if hasattr(funding, "to_dict") else funding.model_dump()
                    fundings.append(funding_dict)
                return fundings
            return []

        except Exception as e:
            logger.error(f"Failed to get funding data for market {market_id}: {e}")
            raise

    @rate_limited("rest", weight=WEIGHT_DEFAULT * 1.2)
    async def get_funding_rates(self) -> dict[int, float]:
        """
        Get current funding rates for all markets.

        Returns:
            Dictionary mapping market_id to current funding rate
        """
        try:
            response = await self._run_on_client_loop(self._funding_api.funding_rates())

            # Parse response - format depends on API response structure
            if hasattr(response, "funding_rates"):
                rates = {}
                for market_id, rate in response.funding_rates.items():
                    rates[int(market_id)] = float(rate)
                return rates
            return {}

        except Exception as e:
            logger.error(f"Failed to get funding rates: {e}")
            raise

    def _resolution_to_milliseconds(self, resolution: str) -> int:
        """
        Convert resolution string to milliseconds.

        Args:
            resolution: Resolution string (e.g., "1m", "5m", "1h", "1d")

        Returns:
            Milliseconds for the resolution
        """
        resolution_lower = resolution.lower()

        # Extract number and unit
        import re

        match = re.match(r"(\d+)([smhd])", resolution_lower)
        if not match:
            # Default to 1 hour if parsing fails
            logger.warning(f"Failed to parse resolution '{resolution}', defaulting to 1h")
            return 3600000

        value = int(match.group(1))
        unit = match.group(2)

        # Convert to milliseconds
        multipliers = {
            "s": 1000,  # seconds
            "m": 60000,  # minutes
            "h": 3600000,  # hours
            "d": 86400000,  # days
        }

        return value * multipliers.get(unit, 3600000)

    async def close(self):
        """Close the client and release resources"""
        if self._api_client:
            # Close API client if it has a close method
            if hasattr(self._api_client, "close"):
                await self._api_client.close()

        # Stop the event loop
        if hasattr(self, "_loop") and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for loop thread to finish
        if hasattr(self, "_loop_thread") and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)

        logger.debug("LighterClient closed")
