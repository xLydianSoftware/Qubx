"""Lighter API client wrapper"""

from typing import Any, Optional

from lighter import ApiClient, Configuration, InfoApi, OrderApi, SignerClient

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

        # Initialize API clients
        self._config = Configuration(host=self.api_url)
        self._api_client = ApiClient(configuration=self._config)
        self._info_api = InfoApi(self._api_client)
        self._order_api = OrderApi(self._api_client)

        # Initialize signer client for order operations
        self._signer_client: Optional[SignerClient] = None

        logger.info(
            f"Initialized LighterClient (testnet={testnet}, account_index={account_index}, api_key_index={api_key_index})"
        )

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
        try:
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

    async def cancel_order(
        self, order_id: int, market_id: int
    ) -> tuple[Any, Any, Optional[str]]:
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

    def close(self):
        """Close the client and release resources"""
        if self._api_client:
            # Close API client if it has a close method
            if hasattr(self._api_client, "close"):
                self._api_client.close()

        logger.debug("LighterClient closed")
