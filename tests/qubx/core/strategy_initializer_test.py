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
            initializer.set_state_resolver(self.custom_resolver)

    def custom_time_finder(self, time: dt_64, state: RestoredState) -> dt_64:
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
        assert initializer.start_time_finder(dt_64(), test_state) == np.datetime64("2023-01-01", "ns")

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

    def test_strategy_schedules_custom_methods(self):
        """Test that a strategy can schedule custom methods using the initializer."""
        # Create a test strategy class that schedules custom methods
        class SchedulingTestStrategy(IStrategy):
            def __init__(self):
                self.custom_method_called = False
                self.another_method_called = False

            def on_init(self, initializer: IStrategyInitializer):
                # Schedule custom methods
                initializer.schedule("0 9 * * *", self.daily_rebalance)
                initializer.schedule("0 */4 * * *", self.risk_check)

            def daily_rebalance(self, ctx):
                self.custom_method_called = True

            def risk_check(self, ctx):
                self.another_method_called = True

        # Create strategy and initializer
        strategy = SchedulingTestStrategy()
        initializer = BasicStrategyInitializer()

        # Call on_init
        strategy.on_init(initializer)

        # Check that custom schedules were stored
        custom_schedules = initializer.get_custom_schedules()
        assert len(custom_schedules) == 2

        # Check that the schedules contain the expected cron strings and methods
        schedules_data = list(custom_schedules.values())
        cron_schedules = [schedule[0] for schedule in schedules_data]
        methods = [schedule[1] for schedule in schedules_data]

        assert "0 9 * * *" in cron_schedules
        assert "0 */4 * * *" in cron_schedules
        assert strategy.daily_rebalance in methods
        assert strategy.risk_check in methods

    def test_schedule_method_validation(self):
        """Test that the schedule method properly validates arguments."""
        initializer = BasicStrategyInitializer()

        def test_method(ctx):
            pass

        # Valid cron schedule should work
        initializer.schedule("0 0 * * *", test_method)
        assert len(initializer.get_custom_schedules()) == 1

        # Check that method is callable
        custom_schedules = initializer.get_custom_schedules()
        schedule_id, (cron_schedule, method) = next(iter(custom_schedules.items()))
        assert cron_schedule == "0 0 * * *"
        assert callable(method)
        assert method == test_method

    def test_multiple_custom_schedules(self):
        """Test that multiple custom schedules can be registered."""
        initializer = BasicStrategyInitializer()

        def method1(ctx):
            pass

        def method2(ctx):
            pass

        def method3(ctx):
            pass

        # Register multiple schedules
        initializer.schedule("0 9 * * *", method1)
        initializer.schedule("0 12 * * *", method2)
        initializer.schedule("0 18 * * *", method3)

        # Check that all schedules were stored
        custom_schedules = initializer.get_custom_schedules()
        assert len(custom_schedules) == 3

        # Check that all methods are present
        methods = [schedule[1] for schedule in custom_schedules.values()]
        assert method1 in methods
        assert method2 in methods
        assert method3 in methods

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
        assert initializer.start_time_finder(dt_64(), test_state) == np.datetime64("2023-01-01", "ns")

        # Check that the custom resolver was set
        assert initializer.mismatch_resolver is not None
        assert callable(initializer.mismatch_resolver)
