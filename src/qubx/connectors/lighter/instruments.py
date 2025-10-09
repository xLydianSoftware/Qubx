"""Lighter instrument loader and management"""

from pathlib import Path
from typing import Optional

from qubx import logger
from qubx.core.basics import AssetType, Instrument, MarketType

from .client import LighterClient
from .utils import lighter_symbol_to_qubx


class LighterInstrumentLoader:
    """
    Loads and manages Lighter market instruments.

    Fetches market metadata from Lighter API and converts to Qubx Instrument format.
    Creates bidirectional mappings between Lighter market IDs and Qubx symbols.
    """

    def __init__(self, client: LighterClient):
        """
        Initialize instrument loader.

        Args:
            client: LighterClient instance
        """
        self.client = client

        # Mappings
        self.market_id_to_symbol: dict[int, str] = {}
        self.symbol_to_market_id: dict[str, int] = {}
        self.instruments: dict[str, Instrument] = {}  # Full Qubx ID -> Instrument

    def load_instruments(self) -> dict[str, Instrument]:
        """
        Load all instruments from Lighter API.

        Returns:
            Dictionary mapping full instrument ID to Instrument object
            Format: "LIGHTER:SWAP:BTC-USDC" -> Instrument(...)
        """
        try:
            logger.info("Loading instruments from Lighter API...")

            markets = self.client.get_markets()
            logger.info(f"Found {len(markets)} markets")

            instruments = {}

            for market in markets:
                try:
                    instrument = self._convert_market_to_instrument(market)
                    if instrument:
                        # Store in multiple formats for easy lookup
                        full_id = f"LIGHTER:{instrument.market_type}:{instrument.symbol}"
                        instruments[full_id] = instrument

                        # Store mappings
                        market_id = market.get("id")
                        self.market_id_to_symbol[market_id] = instrument.symbol
                        self.symbol_to_market_id[instrument.symbol] = market_id

                        logger.debug(f"Loaded instrument: {full_id} (market_id={market_id})")

                except Exception as e:
                    logger.error(f"Failed to convert market {market.get('id')}: {e}")
                    continue

            self.instruments = instruments
            logger.info(f"Successfully loaded {len(instruments)} instruments")

            return instruments

        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            raise

    def _convert_market_to_instrument(self, market: dict) -> Optional[Instrument]:
        """
        Convert Lighter market metadata to Qubx Instrument.

        Lighter market format:
        {
            "id": 0,
            "symbol": "BTC-USDC",
            "supported_price_decimals": 2,
            "supported_size_decimals": 3,
            "min_base_amount": "0.001",
            "min_quote_amount": "5.0",
            ...
        }

        Args:
            market: Market metadata dict from Lighter API

        Returns:
            Qubx Instrument object or None if conversion fails
        """
        try:
            symbol_lighter = market.get("symbol", "")
            if not symbol_lighter or "-" not in symbol_lighter:
                logger.warning(f"Invalid symbol format: {symbol_lighter}")
                return None

            base, quote = symbol_lighter.split("-")

            # Extract precision and limits
            price_decimals = market.get("supported_price_decimals", 2)
            size_decimals = market.get("supported_size_decimals", 3)
            min_base_amount = float(market.get("min_base_amount", 0.001))
            min_quote_amount = float(market.get("min_quote_amount", 5.0))

            # Calculate tick_size and lot_size from decimals
            tick_size = 10 ** -price_decimals
            lot_size = 10 ** -size_decimals

            # All Lighter markets are perpetual swaps (SWAP)
            instrument = Instrument(
                symbol=symbol_lighter,  # Keep Lighter format: "BTC-USDC"
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="LIGHTER",
                base=base,
                quote=quote,
                settle=quote,  # Perpetuals settle in quote currency
                exchange_symbol=symbol_lighter,
                tick_size=tick_size,
                lot_size=lot_size,
                min_size=min_base_amount,
                min_notional=min_quote_amount,
                initial_margin=market.get("initial_margin_requirement", 0.0),
                maint_margin=market.get("maintenance_margin_requirement", 0.0),
                contract_size=1.0,  # Perpetuals have 1:1 contract size
            )

            return instrument

        except Exception as e:
            logger.error(f"Failed to convert market to instrument: {e}")
            return None

    def get_instrument_by_market_id(self, market_id: int) -> Optional[Instrument]:
        """
        Get instrument by Lighter market ID.

        Args:
            market_id: Lighter market ID

        Returns:
            Instrument object or None
        """
        symbol = self.market_id_to_symbol.get(market_id)
        if not symbol:
            return None

        full_id = f"LIGHTER:SWAP:{symbol}"
        return self.instruments.get(full_id)

    def get_instrument_by_symbol(self, symbol: str) -> Optional[Instrument]:
        """
        Get instrument by symbol.

        Args:
            symbol: Symbol in Lighter format (e.g., "BTC-USDC")

        Returns:
            Instrument object or None
        """
        full_id = f"LIGHTER:SWAP:{symbol}"
        return self.instruments.get(full_id)

    def get_market_id(self, symbol: str) -> Optional[int]:
        """
        Get Lighter market ID for a symbol.

        Args:
            symbol: Symbol in Lighter format (e.g., "BTC-USDC")

        Returns:
            Market ID or None
        """
        return self.symbol_to_market_id.get(symbol)

    def save_to_file(self, file_path: Path) -> None:
        """
        Save instruments to JSON file.

        Args:
            file_path: Path to save JSON file
        """
        try:
            import json

            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert instruments to serializable format
            instruments_data = []
            for instrument in self.instruments.values():
                # Convert to dict (would need proper serialization)
                # TODO: Implement proper serialization
                pass

            with open(file_path, "w") as f:
                json.dump(instruments_data, f, indent=2)

            logger.info(f"Saved {len(instruments_data)} instruments to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save instruments to {file_path}: {e}")
            raise


def load_lighter_instruments(
    api_key: str,
    private_key: str,
    account_index: int,
    api_key_index: int = 0,
    testnet: bool = False,
) -> dict[str, Instrument]:
    """
    Convenience function to load Lighter instruments.

    Args:
        api_key: Lighter API key
        private_key: Private key for signing
        account_index: Lighter account index
        api_key_index: API key index
        testnet: If True, use testnet

    Returns:
        Dictionary of instruments
    """
    client = LighterClient(
        api_key=api_key,
        private_key=private_key,
        account_index=account_index,
        api_key_index=api_key_index,
        testnet=testnet,
    )

    loader = LighterInstrumentLoader(client)
    return loader.load_instruments()
