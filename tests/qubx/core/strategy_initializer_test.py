import inspect
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qubx.core.basics import RestoredState, dt_64
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategy, IStrategyInitializer
from qubx.core.lookups import lookup
from qubx.restarts.state_resolvers import StateResolver
from qubx.restarts.time_finders import TimeFinder


class TestStrategy(IStrategy):
    """Test strategy that configures warmup using the initializer."""

    def __init__(self, warmup_period="14d", use_custom_time_finder=False, use_custom_resolver=False):
        self.warmup_period = warmup_period
        self.use_custom_time_finder = use_custom_time_finder
        self.use_custom_resolver = use_custom_resolver
        self.on_init_called = False

    def on_init(self, initializer: IStrategyInitializer):
        """Configure the strategy using the initializer."""
        self.on_init_called = True

        # Set base subscription
        initializer.set_base_subscription("OHLC[1m]")

        # Set warmup period with optional custom time finder
        if self.use_custom_time_finder:
            initializer.set_warmup(self.warmup_period, self.custom_time_finder)
        else:
            initializer.set_warmup(self.warmup_period)

        # Set position mismatch resolver if requested
        if self.use_custom_resolver:
            initializer.set_mismatch_resolver(self.custom_resolver)

    def custom_time_finder(self, state: RestoredState) -> dt_64:
        """Custom time finder that returns a fixed date."""
        return np.datetime64("2023-01-01", "ns")

    def custom_resolver(self, ctx, sim_positions, sim_orders):
        """Custom position mismatch resolver."""
        pass


class TestStrategyInitializer:
    """Tests for strategies that use the initializer to configure warmup."""

    def test_strategy_configures_warmup(self):
        """Test that a strategy can configure warmup using the initializer."""
        # Create a strategy
        strategy = TestStrategy(warmup_period="14d")

        # Create an initializer
        initializer = BasicStrategyInitializer()

        # Call on_init
        strategy.on_init(initializer)

        # Check that on_init was called
        assert strategy.on_init_called

        # Check that the base subscription was set
        assert initializer.base_subscription == "OHLC[1m]"

        # Check that the warmup period was set
        assert initializer.warmup_period == "14d"

        # Check that no custom time finder was set
        assert initializer.start_time_finder is None

        # Check that no custom resolver was set
        assert initializer.mismatch_resolver is None

    def test_strategy_configures_custom_time_finder(self):
        """Test that a strategy can configure a custom time finder."""
        # Create a strategy with a custom time finder
        strategy = TestStrategy(warmup_period="14d", use_custom_time_finder=True)

        # Create an initializer
        initializer = BasicStrategyInitializer()

        # Call on_init
        strategy.on_init(initializer)

        # Check that the warmup period was set
        assert initializer.warmup_period == "14d"

        # Check that the custom time finder was set
        assert initializer.start_time_finder is not None
        assert callable(initializer.start_time_finder)

        # Create a test state and check that the function returns the expected value
        test_state = MagicMock(spec=RestoredState)
        assert initializer.start_time_finder(test_state) == np.datetime64("2023-01-01", "ns")

    def test_strategy_configures_custom_resolver(self):
        """Test that a strategy can configure a custom position mismatch resolver."""
        # Create a strategy with a custom resolver
        strategy = TestStrategy(warmup_period="14d", use_custom_resolver=True)

        # Create an initializer
        initializer = BasicStrategyInitializer()

        # Call on_init
        strategy.on_init(initializer)

        # Check that the custom resolver was set
        assert initializer.mismatch_resolver is not None
        assert callable(initializer.mismatch_resolver)

    def test_strategy_configures_both_custom_components(self):
        """Test that a strategy can configure both custom time finder and resolver."""
        # Create a strategy with both custom components
        strategy = TestStrategy(warmup_period="14d", use_custom_time_finder=True, use_custom_resolver=True)

        # Create an initializer
        initializer = BasicStrategyInitializer()

        # Call on_init
        strategy.on_init(initializer)

        # Check that the custom time finder was set
        assert initializer.start_time_finder is not None
        assert callable(initializer.start_time_finder)

        # Create a test state and check that the function returns the expected value
        test_state = MagicMock(spec=RestoredState)
        assert initializer.start_time_finder(test_state) == np.datetime64("2023-01-01", "ns")

        # Check that the custom resolver was set
        assert initializer.mismatch_resolver is not None
        assert callable(initializer.mismatch_resolver)
