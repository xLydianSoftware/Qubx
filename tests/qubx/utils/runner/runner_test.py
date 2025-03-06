from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qubx.core.basics import RestoredState
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategyContext
from qubx.core.lookups import lookup
from qubx.restarts.state_resolvers import StateResolver
from qubx.restarts.time_finders import TimeFinder
from qubx.utils.runner.runner import _run_warmup


@pytest.mark.skip(reason="This logic is not implemented yet in the runner.")
class TestRunWarmup:
    """Tests for the _run_warmup function."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock context for testing."""
        context = MagicMock(spec=IStrategyContext)

        # Create a mock initializer
        initializer = BasicStrategyInitializer()
        initializer.set_warmup("14d")

        # Set up the context to return the initializer
        context.initializer = initializer

        return context

    @pytest.fixture
    def mock_restored_state(self):
        """Create a mock restored state for testing."""
        return MagicMock(spec=RestoredState)

    @patch("qubx.utils.runner.runner.TimeFinder")
    @patch("qubx.utils.runner.runner.StateResolver")
    def test_run_warmup_with_defaults(self, mock_state_resolver, mock_time_finder, mock_context, mock_restored_state):
        """Test running warmup with default time finder and state resolver."""
        # Set up the mock time finder and state resolver
        mock_time_finder.LAST_SIGNAL = MagicMock()
        mock_state_resolver.REDUCE_ONLY = MagicMock()

        # Run warmup
        _run_warmup(mock_context, mock_restored_state)

        # Check that the default time finder and state resolver were used
        assert mock_context.initializer.get_start_time_finder() is mock_time_finder.LAST_SIGNAL
        assert mock_context.initializer.get_mismatch_resolver() is mock_state_resolver.REDUCE_ONLY

    @patch("qubx.utils.runner.runner.TimeFinder")
    @patch("qubx.utils.runner.runner.StateResolver")
    def test_run_warmup_with_custom_time_finder(
        self, mock_state_resolver, mock_time_finder, mock_context, mock_restored_state
    ):
        """Test running warmup with a custom time finder."""

        # Create a custom time finder
        def custom_time_finder(state: RestoredState) -> np.datetime64:
            return np.datetime64("2023-01-01", "ns")

        # Set the custom time finder in the initializer
        mock_context.initializer.set_warmup("14d", custom_time_finder)

        # Run warmup
        _run_warmup(mock_context, mock_restored_state)

        # Check that the custom time finder was used
        assert mock_context.initializer.get_start_time_finder() is custom_time_finder

        # Check that the default state resolver was used
        assert mock_context.initializer.get_mismatch_resolver() is mock_state_resolver.REDUCE_ONLY

    @patch("qubx.utils.runner.runner.TimeFinder")
    @patch("qubx.utils.runner.runner.StateResolver")
    def test_run_warmup_with_custom_state_resolver(
        self, mock_state_resolver, mock_time_finder, mock_context, mock_restored_state
    ):
        """Test running warmup with a custom state resolver."""

        # Create a custom state resolver
        def custom_state_resolver(ctx, sim_positions, sim_orders):
            pass

        # Set the custom state resolver in the initializer
        mock_context.initializer.set_mismatch_resolver(custom_state_resolver)

        # Run warmup
        _run_warmup(mock_context, mock_restored_state)

        # Check that the default time finder was used
        assert mock_context.initializer.get_start_time_finder() is mock_time_finder.LAST_SIGNAL

        # Check that the custom state resolver was used
        assert mock_context.initializer.get_mismatch_resolver() is custom_state_resolver

    @patch("qubx.utils.runner.runner.TimeFinder")
    @patch("qubx.utils.runner.runner.StateResolver")
    def test_run_warmup_without_warmup_period(
        self, mock_state_resolver, mock_time_finder, mock_context, mock_restored_state
    ):
        """Test running warmup without a warmup period."""
        # Set the warmup period to None
        mock_context.initializer.warmup_period = None

        # Run warmup
        _run_warmup(mock_context, mock_restored_state)

        # Check that no time finder or state resolver was set
        assert mock_context.initializer.get_start_time_finder() is None
        assert mock_context.initializer.get_mismatch_resolver() is None
