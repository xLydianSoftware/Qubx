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

    def get_markets(self) -> list[dict]:
        """
        Get list of all markets.

        Returns:
            List of market dictionaries with metadata
        """
        try:
            response = self._info_api.get_order_books()
            if hasattr(response, "order_books"):
                return response.order_books
            return []
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            raise

    def get_market_info(self, market_id: int) -> Optional[dict]:
        """
        Get information for a specific market.

        Args:
            market_id: Market ID

        Returns:
            Market info dict or None if not found
        """
        markets = self.get_markets()
        for market in markets:
            if market.get("id") == market_id:
                return market
        return None

    def get_orderbook(self, market_id: int) -> dict:
        """
        Get current orderbook for a market.

        Args:
            market_id: Market ID

        Returns:
            Orderbook dict with bids and asks
        """
        try:
            response = self._order_api.get_order_book(market_id)
            return {
                "asks": response.asks if hasattr(response, "asks") else [],
                "bids": response.bids if hasattr(response, "bids") else [],
            }
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
        side: str,
        order_type: int,
        size: str,
        price: Optional[str] = None,
        client_order_id: Optional[str] = None,
        time_in_force: int = 1,
        **kwargs,
    ) -> tuple[Any, Any, Optional[str]]:
        """
        Create an order.

        Args:
            market_id: Market ID
            side: "B" for buy, "S" for sell
            order_type: Order type (0=limit, 1=market, etc.)
            size: Order size as string
            price: Limit price as string (for limit orders)
            client_order_id: Optional client order ID
            time_in_force: Time in force (0=IOC, 1=GTT, 2=POST_ONLY)
            **kwargs: Additional order parameters

        Returns:
            Tuple of (created_order, response, error_string)
        """
        try:
            # Use signer client to create and sign order
            # This is a simplified version - actual implementation depends on SDK
            logger.info(
                f"Creating order: market={market_id}, side={side}, type={order_type}, size={size}, price={price}"
            )

            # TODO: Implement actual order creation using SDK
            # For now, return placeholder
            return None, None, "Not yet implemented"

        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            return None, None, str(e)

    async def cancel_order(self, order_id: str) -> tuple[bool, Optional[str]]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Tuple of (success, error_string)
        """
        try:
            logger.info(f"Cancelling order: {order_id}")

            # TODO: Implement using SDK
            return False, "Not yet implemented"

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False, str(e)

    def close(self):
        """Close the client and release resources"""
        if self._api_client:
            # Close API client if it has a close method
            if hasattr(self._api_client, "close"):
                self._api_client.close()

        logger.debug("LighterClient closed")
