"""Lighter instrument loader and management"""

from pathlib import Path
from typing import Optional

from qubx import logger
from qubx.core.basics import AssetType, Instrument, MarketType

from .client import LighterClient


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

    async def load_instruments(self) -> dict[str, Instrument]:
        """
        Load all instruments from Lighter API.

        Returns:
            Dictionary mapping full instrument ID to Instrument object
            Format: "LIGHTER:SWAP:BTCUSDC" -> Instrument(...)
        """
        try:
            logger.info("Loading instruments from Lighter API...")

            markets = await self.client.get_markets()
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
            "symbol": "BTC",  # Single token
            "supported_price_decimals": 2,
            "supported_size_decimals": 3,
            "min_base_amount": "0.001",
            "min_quote_amount": "5.0",
            ...
        }

        Converts to Qubx normalized format: "BTC" -> "BTCUSDC"

        Args:
            market: Market metadata dict from Lighter API

        Returns:
            Qubx Instrument object or None if conversion fails
        """
        try:
            symbol_lighter = market.get("symbol", "")
            if not symbol_lighter:
                logger.warning("Empty symbol")
                return None

            # Lighter uses single-token symbols (e.g., "BTC", "ETH")
            # All perpetuals are settled in USDC on Lighter
            base = symbol_lighter
            quote = "USDC"

            # Extract precision and limits
            price_decimals = market.get("supported_price_decimals", 2)
            size_decimals = market.get("supported_size_decimals", 3)
            min_base_amount = float(market.get("min_base_amount", 0.001))
            min_quote_amount = float(market.get("min_quote_amount", 5.0))

            # Calculate tick_size and lot_size from decimals
            tick_size = 10**-price_decimals
            lot_size = 10**-size_decimals

            # Ensure min_size is at least lot_size (some markets have 0 min_base_amount)
            min_size = max(min_base_amount, lot_size)

            # All Lighter markets are perpetual swaps (SWAP)
            # Convert single-token symbol to Qubx normalized format: "BTC" -> "BTCUSDC"
            qubx_symbol = f"{base}{quote}"  # Qubx normalized format: no separator

            instrument = Instrument(
                symbol=qubx_symbol,  # Qubx format: "BTCUSDC"
                asset_type=AssetType.CRYPTO,
                market_type=MarketType.SWAP,
                exchange="LIGHTER",  # Exchange name is LIGHTER
                base=base,
                quote=quote,
                settle=quote,  # Perpetuals settle in USDC
                exchange_symbol=symbol_lighter,  # Lighter format: "BTC"
                tick_size=tick_size,
                lot_size=lot_size,
                min_size=min_size,
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
            symbol: Symbol in Qubx normalized format (e.g., "BTCUSDC")

        Returns:
            Instrument object or None
        """
        full_id = f"LIGHTER:SWAP:{symbol}"
        return self.instruments.get(full_id)

    def get_market_id(self, symbol: str) -> Optional[int]:
        """
        Get Lighter market ID for a symbol.

        Args:
            symbol: Symbol in Qubx normalized format (e.g., "BTCUSDC")

        Returns:
            Market ID or None
        """
        return self.symbol_to_market_id.get(symbol)
