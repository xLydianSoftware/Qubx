"""
Lighter exchange-specific API extensions.

Provides access to Lighter-specific features like leverage management,
transfers, and liquidity pool creation.
"""

# TODO: implement this

from typing import TYPE_CHECKING

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.interfaces import IExchangeExtensions

if TYPE_CHECKING:
    from .broker import LighterBroker
    from .client import LighterClient


class LighterExchangeAPI(IExchangeExtensions):
    """
    Exchange-specific API for Lighter.

    Provides access to Lighter-specific operations:
    - update_leverage: Update leverage for an instrument
    - transfer: Transfer USDC between accounts
    - create_pool: Create a public liquidity pool
    - create_subaccount: Create a new sub-account
    """

    def __init__(self, client: "LighterClient", broker: "LighterBroker"):
        """
        Initialize Lighter extensions.

        Args:
            client: LighterClient instance for API access
            broker: LighterBroker instance for instrument lookups
        """
        self.client = client
        self.broker = broker

    async def update_leverage(
        self, instrument: Instrument, leverage: float, margin_mode: str = "cross"
    ) -> tuple[bool, str | None]:
        """
        Update leverage for an instrument.

        Args:
            instrument: Instrument to update leverage for
            leverage: Target leverage ratio (e.g., 10.0 for 10x)
            margin_mode: "cross" or "isolated" (default: "cross")

        Returns:
            Tuple of (success: bool, error: str | None)

        Example:
            >>> # Update leverage to 10x for BTC/USDC
            >>> broker = ctx.get_broker("LIGHTER")
            >>> btc_usdc = ctx.query_instrument("BTC/USDC", "LIGHTER")
            >>> success, error = await broker.extensions.update_leverage(btc_usdc, 10.0)
            >>> if success:
            >>>     print("Leverage updated successfully")
            >>> else:
            >>>     print(f"Failed to update leverage: {error}")
        """
        try:
            # Get market_id from instrument via broker's instrument loader
            market_id = self.broker.instrument_loader.get_market_id(instrument.symbol)
            if market_id is None:
                return False, f"Instrument {instrument.symbol} has no market_id"

            # Convert margin mode to integer (0=cross, 1=isolated)
            margin_mode_int = 0 if margin_mode.lower() == "cross" else 1

            logger.info(
                f"Updating leverage for {instrument.symbol} (market_id={market_id}): "
                f"leverage={leverage}x, margin_mode={margin_mode}"
            )

            # Use SignerClient to update leverage
            # The update_leverage method takes: market_index, margin_mode, leverage
            result = await self.client.signer_client.update_leverage(
                market_index=market_id, margin_mode=margin_mode_int, leverage=leverage
            )

            # Result format: (created_tx, response, error)
            _, _, error = result

            if error:
                logger.error(f"Failed to update leverage for {instrument.symbol}: {error}")
                return False, error

            logger.info(f"Successfully updated leverage for {instrument.symbol} to {leverage}x ({margin_mode})")
            return True, None

        except Exception as e:
            error_msg = f"Exception updating leverage: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    async def transfer(
        self, to_account_index: int, amount: float, fee: float = 0.0, memo: str = ""
    ) -> tuple[bool, str | None]:
        """
        Transfer USDC between accounts.

        Args:
            to_account_index: Destination account index
            amount: Amount of USDC to transfer
            fee: Transfer fee (default: 0.0)
            memo: Optional memo/note for the transfer

        Returns:
            Tuple of (success: bool, error: str | None)

        Example:
            >>> # Transfer 100 USDC to account 225672
            >>> broker = ctx.get_broker("LIGHTER")
            >>> success, error = await broker.extensions.transfer(
            >>>     to_account_index=225672,
            >>>     amount=100.0,
            >>>     memo="Test transfer"
            >>> )
            >>> if success:
            >>>     print("Transfer successful")
            >>> else:
            >>>     print(f"Transfer failed: {error}")
        """
        try:
            logger.info(f"Transferring {amount} USDC to account {to_account_index} (fee={fee}, memo='{memo}')")

            # Use SignerClient to transfer
            # Note: eth_private_key is already stored in signer_client
            result = await self.client.signer_client.transfer(
                eth_private_key=self.client.private_key,
                to_account_index=to_account_index,
                usdc_amount=amount,
                fee=fee,
                memo=memo,
            )

            # Result format: (created_tx, response, error)
            _, _, error = result

            if error:
                logger.error(f"Failed to transfer USDC: {error}")
                return False, error

            logger.info(f"Successfully transferred {amount} USDC to account {to_account_index}")
            return True, None

        except Exception as e:
            error_msg = f"Exception during transfer: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    async def create_pool(
        self, operator_fee: float, initial_total_shares: float, min_operator_share_rate: float
    ) -> tuple[bool, str | None]:
        """
        Create a public liquidity pool (Lighter-specific feature).

        Args:
            operator_fee: Fee charged by pool operator (as fraction, e.g., 0.01 for 1%)
            initial_total_shares: Initial number of shares to create
            min_operator_share_rate: Minimum operator share rate (as fraction)

        Returns:
            Tuple of (success: bool, error: str | None)

        Example:
            >>> # Create a pool with 1% operator fee
            >>> broker = ctx.get_broker("LIGHTER")
            >>> success, error = await broker.extensions.create_pool(
            >>>     operator_fee=0.01,
            >>>     initial_total_shares=1000000.0,
            >>>     min_operator_share_rate=0.05
            >>> )
            >>> if success:
            >>>     print("Pool created successfully")
            >>> else:
            >>>     print(f"Failed to create pool: {error}")
        """
        try:
            logger.info(
                f"Creating public pool: operator_fee={operator_fee}, "
                f"initial_shares={initial_total_shares}, min_operator_rate={min_operator_share_rate}"
            )

            # Use SignerClient to create pool
            result = await self.client.signer_client.create_public_pool(
                operator_fee=operator_fee,
                initial_total_shares=initial_total_shares,
                min_operator_share_rate=min_operator_share_rate,
            )

            # Result format: (created_tx, response, error)
            _, _, error = result

            if error:
                logger.error(f"Failed to create public pool: {error}")
                return False, error

            logger.info("Successfully created public pool")
            return True, None

        except Exception as e:
            error_msg = f"Exception creating pool: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    async def create_subaccount(self) -> tuple[bool, str | None]:
        """
        Create a new sub-account.

        Returns:
            Tuple of (success: bool, error: str | None)

        Example:
            >>> # Create a new sub-account
            >>> broker = ctx.get_broker("LIGHTER")
            >>> success, error = await broker.extensions.create_subaccount()
            >>> if success:
            >>>     print("Sub-account created successfully")
            >>> else:
            >>>     print(f"Failed to create sub-account: {error}")
        """
        try:
            logger.info("Creating new sub-account")

            # Use SignerClient to create sub-account
            result = await self.client.signer_client.create_sub_account()

            # Result format: (created_tx, response, error)
            _, _, error = result

            if error:
                logger.error(f"Failed to create sub-account: {error}")
                return False, error

            logger.info("Successfully created sub-account")
            return True, None

        except Exception as e:
            error_msg = f"Exception creating sub-account: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
