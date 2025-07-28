import numpy as np
import pandas as pd

from qubx.core.basics import Instrument, Order, Position, RestoredState, dt_64
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategyContext


class TestBasicStrategyInitializer:
    """Tests for the BasicStrategyInitializer class."""

    def test_initialization(self):
        """Test that the initializer can be created with default values."""
        initializer = BasicStrategyInitializer()

        assert initializer.base_subscription is None
        assert initializer.fit_schedule is None
        assert initializer.event_schedule is None
        assert initializer.warmup_period is None
        assert initializer.start_time_finder is None
        assert initializer.mismatch_resolver is None
        assert initializer.config == {}
        assert initializer.auto_subscribe is None

    def test_set_base_subscription(self):
        """Test setting the base subscription."""
        initializer = BasicStrategyInitializer()

        initializer.set_base_subscription("OHLC[1m]")
        assert initializer.base_subscription == "OHLC[1m]"

    def test_auto_subscribe_property(self):
        """Test the auto_subscribe property."""
        initializer = BasicStrategyInitializer()

        assert initializer.auto_subscribe is None

        initializer.auto_subscribe = True
        assert initializer.auto_subscribe is True

        initializer.auto_subscribe = False
        assert initializer.auto_subscribe is False

    def test_set_fit_schedule(self):
        """Test setting the fit schedule."""
        initializer = BasicStrategyInitializer()

        initializer.set_fit_schedule("0 0 * * *")
        assert initializer.fit_schedule == "0 0 * * *"

    def test_set_event_schedule(self):
        """Test setting the event schedule."""
        initializer = BasicStrategyInitializer()

        initializer.set_event_schedule("0 * * * *")
        assert initializer.event_schedule == "0 * * * *"

    def test_schedule_custom_methods(self):
        """Test scheduling custom methods."""
        initializer = BasicStrategyInitializer()

        def custom_method1(ctx):
            pass

        def custom_method2(ctx):
            pass

        # Schedule methods
        initializer.schedule("0 9 * * *", custom_method1)
        initializer.schedule("0 17 * * *", custom_method2)

        # Check that schedules were stored
        custom_schedules = initializer.get_custom_schedules()
        assert len(custom_schedules) == 2

        # Verify the schedules
        schedules_data = list(custom_schedules.values())
        cron_schedules = [schedule[0] for schedule in schedules_data]
        methods = [schedule[1] for schedule in schedules_data]

        assert "0 9 * * *" in cron_schedules
        assert "0 17 * * *" in cron_schedules
        assert custom_method1 in methods
        assert custom_method2 in methods

        # Test get_custom_schedules returns a copy
        schedules_copy = initializer.get_custom_schedules()
        schedules_copy.clear()
        assert len(initializer.get_custom_schedules()) == 2

    def test_set_warmup(self):
        """Test setting the warmup period."""
        initializer = BasicStrategyInitializer()

        # Test with just a period
        initializer.set_warmup("14d")
        assert initializer.warmup_period == "14d"
        assert initializer.start_time_finder is None

        # Test with a period and a start time finder
        def dummy_start_time_finder(time: dt_64, state: RestoredState) -> dt_64:
            return np.datetime64("2023-01-01", "ns")

        initializer.set_warmup("7d", dummy_start_time_finder)
        assert initializer.warmup_period == "7d"
        assert initializer.start_time_finder is dummy_start_time_finder

    def test_get_warmup(self):
        """Test getting the warmup period as a timedelta."""
        initializer = BasicStrategyInitializer()

        # Set a warmup period
        initializer.set_warmup("14d")

        # Get the warmup period as a timedelta
        warmup = initializer.get_warmup()

        # Check that it's a timedelta of 14 days
        expected = pd.Timedelta("14d").to_numpy()
        assert warmup == expected

    def test_get_start_time_finder(self):
        """Test getting the start time finder."""
        initializer = BasicStrategyInitializer()

        # Initially None
        assert initializer.get_start_time_finder() is None

        # Set a start time finder
        def dummy_start_time_finder(time: dt_64, state: RestoredState) -> dt_64:
            return np.datetime64("2023-01-01", "ns")

        initializer.set_warmup("14d", dummy_start_time_finder)
        assert initializer.get_start_time_finder() is dummy_start_time_finder

    def test_set_state_resolver(self):
        """Test setting the mismatch resolver."""
        initializer = BasicStrategyInitializer()

        # Initially None
        assert initializer.get_state_resolver() is None

        # Set a state resolver
        def dummy_state_resolver(
            ctx: IStrategyContext, sim_positions: dict[Instrument, Position], sim_orders: dict[Instrument, list[Order]]
        ) -> None:
            pass

        initializer.set_state_resolver(dummy_state_resolver)
        assert initializer.get_state_resolver() is dummy_state_resolver

    def test_config_methods(self):
        """Test the config methods."""
        initializer = BasicStrategyInitializer()

        # Initially empty
        assert initializer.config == {}

        # Set a config value
        initializer.set_config("test_key", "test_value")
        assert initializer.config == {"test_key": "test_value"}

        # Get a config value
        assert initializer.get_config("test_key") == "test_value"

        # Get a non-existent config value with default
        assert initializer.get_config("non_existent", "default") == "default"

        # Get a non-existent config value without default
        assert initializer.get_config("non_existent") is None
