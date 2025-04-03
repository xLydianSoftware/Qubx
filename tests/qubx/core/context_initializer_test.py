from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qubx.core.context import StrategyContext
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategy, IStrategyInitializer
from qubx.core.lookups import lookup
from qubx.core.utils import recognize_timeframe


class MockStrategy(IStrategy):
    """Mock strategy for testing."""

    def __init__(self):
        self.on_init_called = False
        self.initializer = None

    def on_init(self, initializer: IStrategyInitializer):
        """Record that on_init was called and store the initializer."""
        self.on_init_called = True
        self.initializer = initializer

        # Configure the initializer
        initializer.set_base_subscription("OHLC[1m]")
        initializer.set_warmup("14d")
        initializer.set_fit_schedule("0 0 * * *")
        initializer.set_event_schedule("0 * * * *")


class TestContextInitializer:
    """Tests for the integration between BasicStrategyInitializer and StrategyContext."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for the StrategyContext."""
        broker = MagicMock()
        data_provider = MagicMock()
        account = MagicMock()
        scheduler = MagicMock()
        time_provider = MagicMock()
        time_provider.time.return_value = np.datetime64("2023-01-01", "ns")

        # Use lookup.find_symbol to get a real Instrument instance
        instruments = [lookup.find_symbol("BINANCE.UM", "BTCUSDT")]
        logging = MagicMock()

        return {
            "broker": broker,
            "data_provider": data_provider,
            "account": account,
            "scheduler": scheduler,
            "time_provider": time_provider,
            "instruments": instruments,
            "logging": logging,
        }

    def test_initializer_passed_to_strategy(self, mock_components):
        """Test that the initializer is passed to the strategy's on_init method."""
        # Create a strategy
        strategy = MockStrategy()

        # Create an initializer
        initializer = BasicStrategyInitializer(simulation=True)

        # Create a context with the initializer
        with patch("qubx.core.context.CachedMarketDataHolder"):
            with patch("qubx.core.context.MarketManager"):
                with patch("qubx.core.context.UniverseManager"):
                    with patch("qubx.core.context.SubscriptionManager"):
                        with patch("qubx.core.context.TradingManager"):
                            with patch("qubx.core.context.ProcessingManager"):
                                ctx = StrategyContext(
                                    strategy=strategy,
                                    brokers=[mock_components["broker"]],
                                    data_providers=[mock_components["data_provider"]],
                                    account=mock_components["account"],
                                    scheduler=mock_components["scheduler"],
                                    time_provider=mock_components["time_provider"],
                                    instruments=mock_components["instruments"],
                                    logging=mock_components["logging"],
                                    initializer=initializer,
                                )

        # Check that on_init was called
        assert strategy.on_init_called

        # Check that the initializer was passed to on_init
        assert strategy.initializer is initializer

        # Check that the initializer is accessible through the context
        assert ctx.initializer is initializer

    def test_initializer_created_if_not_provided(self, mock_components):
        """Test that an initializer is created if one is not provided."""
        # Create a strategy
        strategy = MockStrategy()

        # Create a context without an initializer
        with patch("qubx.core.context.CachedMarketDataHolder"):
            with patch("qubx.core.context.MarketManager"):
                with patch("qubx.core.context.UniverseManager"):
                    with patch("qubx.core.context.SubscriptionManager"):
                        with patch("qubx.core.context.TradingManager"):
                            with patch("qubx.core.context.ProcessingManager"):
                                ctx = StrategyContext(
                                    strategy=strategy,
                                    brokers=[mock_components["broker"]],
                                    data_providers=[mock_components["data_provider"]],
                                    account=mock_components["account"],
                                    scheduler=mock_components["scheduler"],
                                    time_provider=mock_components["time_provider"],
                                    instruments=mock_components["instruments"],
                                    logging=mock_components["logging"],
                                )

        # Check that on_init was called
        assert strategy.on_init_called

        # Check that an initializer was created and passed to on_init
        assert strategy.initializer is not None
        assert isinstance(strategy.initializer, BasicStrategyInitializer)

        # Check that the initializer is accessible through the context
        assert ctx.initializer is strategy.initializer

    def test_initializer_settings_applied_to_context(self, mock_components):
        """Test that the initializer settings are applied to the context."""
        # Create a strategy
        strategy = MockStrategy()

        # Create a context
        with patch("qubx.core.context.CachedMarketDataHolder"):
            with patch("qubx.core.context.MarketManager"):
                with patch("qubx.core.context.UniverseManager") as mock_universe_manager:
                    with patch("qubx.core.context.SubscriptionManager") as mock_subscription_manager:
                        with patch("qubx.core.context.TradingManager"):
                            with patch("qubx.core.context.ProcessingManager") as mock_processing_manager:
                                ctx = StrategyContext(
                                    strategy=strategy,
                                    brokers=[mock_components["broker"]],
                                    data_providers=[mock_components["data_provider"]],
                                    account=mock_components["account"],
                                    scheduler=mock_components["scheduler"],
                                    time_provider=mock_components["time_provider"],
                                    instruments=mock_components["instruments"],
                                    logging=mock_components["logging"],
                                )

        # Check that the base subscription was set
        mock_subscription_manager.return_value.set_base_subscription.assert_called_with("OHLC[1m]")

        # Check that the warmup period was set
        # This is more complex because it depends on the subscriptions, which are mocked
        # We'll just check that the initializer has the correct warmup period
        assert ctx.initializer.get_warmup() == recognize_timeframe("14d")

        # Check that the fit schedule was set
        mock_processing_manager.return_value.set_fit_schedule.assert_called_with("0 0 * * *")

        # Check that the event schedule was set
        mock_processing_manager.return_value.set_event_schedule.assert_called_with("0 * * * *")
