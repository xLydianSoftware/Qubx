from unittest.mock import Mock

import pytest

from qubx.backtester.simulated_data import IterableSimulationData
from qubx.core.basics import AssetType, DataType, Instrument, MarketType
from qubx.data.readers import DataReader


class TestLiquidationSubscription:
    """Test suite for liquidation subscription filtering."""

    def test_liquidation_non_swap_instruments_filtered(self):
        """Test that non-SWAP instruments are filtered out from liquidation subscriptions."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"liquidation": mock_reader}
        sim_data = IterableSimulationData(readers=mock_readers)
        
        # Create mix of SWAP and non-SWAP instruments
        swap_instrument = Instrument(
            "BTCUSDT", AssetType.CRYPTO, MarketType.SWAP, "binance", "BTC", "USDT", "USDT",
            "BTCUSDT", 0.01, 0.001, 0.001
        )
        spot_instrument = Instrument(
            "BTCUSDT", AssetType.CRYPTO, MarketType.SPOT, "binance", "BTC", "USDT", "USDT",
            "BTCUSDT", 0.01, 0.001, 0.001
        )
        future_instrument = Instrument(
            "BTCUSDT", AssetType.CRYPTO, MarketType.FUTURE, "binance", "BTC", "USDT", "USDT",
            "BTCUSDT", 0.01, 0.001, 0.001
        )
        
        # Subscribe with mixed instruments
        instruments = [swap_instrument, spot_instrument, future_instrument]
        sim_data.add_instruments_for_subscription(DataType.LIQUIDATION, instruments)
        
        # Verify only SWAP instrument was subscribed
        liquidation_instruments = sim_data.get_instruments_for_subscription(DataType.LIQUIDATION)
        assert len(liquidation_instruments) == 1
        assert liquidation_instruments[0] == swap_instrument
        assert liquidation_instruments[0].market_type == MarketType.SWAP

    def test_liquidation_mixed_instrument_types(self):
        """Test liquidation subscriptions with mixed instrument types."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"liquidation": mock_reader}
        sim_data = IterableSimulationData(readers=mock_readers)
        
        # Create instruments of different types
        swap1 = Instrument(
            "BTCUSDT", AssetType.CRYPTO, MarketType.SWAP, "binance", "BTC", "USDT", "USDT",
            "BTCUSDT", 0.01, 0.001, 0.001
        )
        swap2 = Instrument(
            "ETHUSDT", AssetType.CRYPTO, MarketType.SWAP, "binance", "ETH", "USDT", "USDT",
            "ETHUSDT", 0.01, 0.001, 0.001
        )
        spot = Instrument(
            "ADAUSDT", AssetType.CRYPTO, MarketType.SPOT, "binance", "ADA", "USDT", "USDT",
            "ADAUSDT", 0.0001, 0.1, 0.1
        )
        
        instruments = [swap1, spot, swap2]
        sim_data.add_instruments_for_subscription(DataType.LIQUIDATION, instruments)
        
        # Should only have the SWAP instruments
        liquidation_instruments = sim_data.get_instruments_for_subscription(DataType.LIQUIDATION)
        assert len(liquidation_instruments) == 2
        assert all(instr.market_type == MarketType.SWAP for instr in liquidation_instruments)
        
        # Should have both SWAP instruments
        symbols = {instr.symbol for instr in liquidation_instruments}
        assert symbols == {"BTCUSDT", "ETHUSDT"}

    def test_liquidation_only_swap_instruments(self):
        """Test liquidation subscriptions with only SWAP instruments work normally."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"liquidation": mock_reader}
        sim_data = IterableSimulationData(readers=mock_readers)
        
        swap_instruments = [
            Instrument(
                "BTCUSDT", AssetType.CRYPTO, MarketType.SWAP, "binance", "BTC", "USDT", "USDT",
                "BTCUSDT", 0.01, 0.001, 0.001
            ),
            Instrument(
                "ETHUSDT", AssetType.CRYPTO, MarketType.SWAP, "binance", "ETH", "USDT", "USDT",
                "ETHUSDT", 0.01, 0.001, 0.001
            )
        ]
        
        sim_data.add_instruments_for_subscription(DataType.LIQUIDATION, swap_instruments)
        
        # All instruments should be subscribed
        liquidation_instruments = sim_data.get_instruments_for_subscription(DataType.LIQUIDATION)
        assert len(liquidation_instruments) == 2
        assert liquidation_instruments == swap_instruments

    def test_liquidation_empty_after_filtering(self):
        """Test behavior when no SWAP instruments remain after filtering."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"liquidation": mock_reader}
        sim_data = IterableSimulationData(readers=mock_readers)
        
        # Only non-SWAP instruments
        non_swap_instruments = [
            Instrument(
                "BTCUSDT", AssetType.CRYPTO, MarketType.SPOT, "binance", "BTC", "USDT", "USDT",
                "BTCUSDT", 0.01, 0.001, 0.001
            ),
            Instrument(
                "ETHUSDT", AssetType.CRYPTO, MarketType.FUTURE, "binance", "ETH", "USDT", "USDT",
                "ETHUSDT", 0.01, 0.001, 0.001
            )
        ]
        
        sim_data.add_instruments_for_subscription(DataType.LIQUIDATION, non_swap_instruments)
        
        # Should have no liquidation instruments
        liquidation_instruments = sim_data.get_instruments_for_subscription(DataType.LIQUIDATION)
        assert len(liquidation_instruments) == 0

    def test_other_subscription_types_unaffected(self):
        """Test that other subscription types are not affected by liquidation filtering."""
        mock_reader = Mock(spec=DataReader)
        mock_readers = {"ohlc": mock_reader}
        sim_data = IterableSimulationData(readers=mock_readers)
        
        # Mix of instrument types
        instruments = [
            Instrument(
                "BTCUSDT", AssetType.CRYPTO, MarketType.SWAP, "binance", "BTC", "USDT", "USDT",
                "BTCUSDT", 0.01, 0.001, 0.001
            ),
            Instrument(
                "ETHUSDT", AssetType.CRYPTO, MarketType.SPOT, "binance", "ETH", "USDT", "USDT",
                "ETHUSDT", 0.01, 0.001, 0.001
            )
        ]
        
        # Subscribe to OHLC (should not be filtered)
        sim_data.add_instruments_for_subscription("ohlc(1h)", instruments)
        
        # All instruments should be subscribed for OHLC
        ohlc_instruments = sim_data.get_instruments_for_subscription("ohlc(1h)")
        assert len(ohlc_instruments) == 2
        assert ohlc_instruments == instruments