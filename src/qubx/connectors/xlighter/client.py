"""Lighter API client wrapper"""

import asyncio
import logging
import threading
from typing import Any, Optional

from lighter import ApiClient, CandlestickApi, Configuration, FundingApi, InfoApi, OrderApi, SignerClient

# Reset logging - lighter SDK sets root logger to DEBUG on import
logging.root.setLevel(logging.WARNING)

from qubx import logger

from .constants import API_BASE_MAINNET, API_BASE_TESTNET


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

    def __init__(
        self,
        api_key: str,
        private_key: str,
        account_index: int,
        api_key_index: int = 0,
        testnet: bool = False,
    ):
        """
        Initialize Lighter client.

        Args:
            api_key: Lighter API key (Ethereum address)
            private_key: Private key for signing (without 0x prefix)
            account_index: Lighter account index
            api_key_index: API key index for the account
            testnet: If True, use testnet. Otherwise mainnet.
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
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be ready
        import time

        time.sleep(0.1)

        # Initialize API clients in the event loop context
        # Use asyncio.ensure_future to create a proper Task (required by aiohttp)
        import concurrent.futures

        init_future = concurrent.futures.Future()

        def create_init_task():
            """Create init task in the event loop"""
            task = asyncio.create_task(self._async_init())
            task.add_done_callback(
                lambda t: init_future.set_result(t.result())
                if not t.exception()
                else init_future.set_exception(t.exception())
            )

        self._loop.call_soon_threadsafe(create_init_task)
        init_future.result()  # Wait for initialization to complete

        # Initialize signer client for order operations
        self._signer_client: Optional[SignerClient] = None

        logger.info(
            f"Initialized LighterClient (testnet={testnet}, account_index={account_index}, api_key_index={api_key_index})"
        )

    def _run_event_loop(self):
        """Run the event loop in a background thread"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _async_init(self):
        """Initialize API clients in async context"""
        self._config = Configuration(host=self.api_url)
        self._api_client = ApiClient(configuration=self._config)
        self._info_api = InfoApi(self._api_client)
        self._order_api = OrderApi(self._api_client)
        self._candlestick_api = CandlestickApi(self._api_client)
        self._funding_api = FundingApi(self._api_client)

    @property
    def signer_client(self) -> SignerClient:
        """
        Get or create signer client for order operations.

        Returns:
            SignerClient instance
        """
        if self._signer_client is None:
            self._signer_client = SignerClient(
                url=self.api_url,
                private_key=self.private_key,
                api_key_index=self.api_key_index,
                account_index=self.account_index,
            )
        return self._signer_client

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
            # Ensure we're running as a task in our event loop
            if asyncio.get_running_loop() == self._loop:
                # Already in our loop, just run directly
                return await _get_markets_impl()
            else:
                # Not in our loop, schedule as task
                return await asyncio.create_task(_get_markets_impl())
        except RuntimeError:
            # No running loop, schedule on our loop
            future = asyncio.run_coroutine_threadsafe(_get_markets_impl(), self._loop)
            return future.result()
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            raise

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

    async def get_orderbook(self, market_id: int) -> dict:
        """
        Get current orderbook for a market.

        Args:
            market_id: Market ID

        Returns:
            Orderbook dict with bids and asks
        """
        try:
            response = await self._order_api.order_books(market_id=market_id)
            # Response is OrderBooks which may have order_books list
            if hasattr(response, "order_books") and response.order_books:
                orderbook = response.order_books[0]
                return {
                    "asks": getattr(orderbook, "asks", []),
                    "bids": getattr(orderbook, "bids", []),
                }
            return {"asks": [], "bids": []}
        except Exception as e:
            logger.error(f"Failed to get orderbook for market {market_id}: {e}")
            raise

    def get_account_positions(self) -> list[dict]:
        """
        Get account positions.

        Returns:
            List of position dictionaries
        """
        try:
            # TODO: Implement when SDK supports it
            logger.warning("get_account_positions not yet implemented")
            return []
        except Exception as e:
            logger.error(f"Failed to get account positions: {e}")
            raise

    def get_account_balances(self) -> dict:
        """
        Get account balances.

        Returns:
            Balance dictionary
        """
        try:
            # TODO: Implement when SDK supports it
            logger.warning("get_account_balances not yet implemented")
            return {}
        except Exception as e:
            logger.error(f"Failed to get account balances: {e}")
            raise

    def get_open_orders(self, market_id: Optional[int] = None) -> list[dict]:
        """
        Get open orders.

        Args:
            market_id: Optional market ID filter

        Returns:
            List of order dictionaries
        """
        try:
            # TODO: Implement using SDK
            logger.warning("get_open_orders not yet implemented")
            return []
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise

    async def create_order(
        self,
        market_id: int,
        is_buy: bool,
        size: float,
        price: Optional[float] = None,
        order_type: int = 0,  # 0=limit, 1=market
        time_in_force: int = 1,  # 1=GTT (default)
        reduce_only: bool = False,
        post_only: bool = False,
        **kwargs,
    ) -> tuple[Any, Any, Optional[str]]:
        """
        Create an order using Lighter SignerClient.

        Args:
            market_id: Market ID
            is_buy: True for buy, False for sell
            size: Order size (float)
            price: Limit price (float, required for limit orders)
            order_type: Order type (0=limit, 1=market)
            time_in_force: Time in force (0=IOC, 1=GTT, 2=POST_ONLY)
            reduce_only: If True, order will only reduce existing position
            post_only: If True, order will only be maker (post-only)
            **kwargs: Additional order parameters

        Returns:
            Tuple of (created_tx, response, error_string)
        """
        try:
            logger.info(
                f"Creating order: market={market_id}, is_buy={is_buy}, type={order_type}, "
                f"size={size}, price={price}, tif={time_in_force}"
            )

            # Use SignerClient to create order
            result = await self.signer_client.create_order(
                market_id=market_id,
                is_buy=is_buy,
                size=str(size),  # SDK expects string
                price=str(price) if price else None,  # SDK expects string
                order_type=order_type,
                time_in_force=time_in_force,
                reduce_only=reduce_only,
                post_only=post_only,
            )

            # SignerClient returns (created_tx, response, error)
            return result

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return None, None, str(e)

    async def cancel_order(self, order_id: int, market_id: int) -> tuple[Any, Any, Optional[str]]:
        """
        Cancel an order using Lighter SignerClient.

        Args:
            order_id: Order ID to cancel (integer)
            market_id: Market ID where the order exists

        Returns:
            Tuple of (created_tx, response, error_string)
        """
        try:
            logger.info(f"Cancelling order: order_id={order_id}, market_id={market_id}")

            # Use SignerClient to cancel order
            result = await self.signer_client.cancel_order(
                order_id=order_id,
                market_id=market_id,
            )

            # SignerClient returns (created_tx, response, error)
            return result

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return None, None, str(e)

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
            # Set default timestamps if not provided
            if end_timestamp is None:
                import time

                end_timestamp = int(time.time() * 1000)
            if start_timestamp is None:
                # Default to 1000 periods back
                resolution_ms = self._resolution_to_milliseconds(resolution)
                start_timestamp = end_timestamp - (count_back * resolution_ms)

            logger.debug(
                f"Fetching candlesticks for market {market_id}: {resolution}, from {start_timestamp} to {end_timestamp}"
            )

            response = await self._candlestick_api.candlesticks(
                market_id=market_id,
                resolution=resolution,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                count_back=count_back,
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
            # Set default timestamps if not provided
            if end_timestamp is None:
                import time

                end_timestamp = int(time.time() * 1000)
            if start_timestamp is None:
                # Default to 1000 periods back
                resolution_ms = self._resolution_to_milliseconds(resolution)
                start_timestamp = end_timestamp - (count_back * resolution_ms)

            logger.debug(
                f"Fetching funding data for market {market_id}: {resolution}, from {start_timestamp} to {end_timestamp}"
            )

            response = await self._candlestick_api.fundings(
                market_id=market_id,
                resolution=resolution,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                count_back=count_back,
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

    async def get_funding_rates(self) -> dict[int, float]:
        """
        Get current funding rates for all markets.

        Returns:
            Dictionary mapping market_id to current funding rate
        """
        try:
            response = await self._funding_api.funding_rates()

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
