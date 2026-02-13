"""Tests for the pending signal retry mechanism in ProcessingManager."""

from unittest.mock import MagicMock, Mock, PropertyMock
import numpy as np
import pytest

from qubx.core.basics import Instrument, Signal
from qubx.core.mixins.processing import ProcessingManager
from qubx.core.series import Quote


class TestPendingSignalRetry:
    """Tests for pending signal retry mechanism when quote is unavailable."""

    @pytest.fixture
    def mock_instrument(self):
        """Create a mock instrument."""
        instrument = MagicMock(spec=Instrument)
        instrument.symbol = "BTCUSDT"
        instrument.min_size = 0.001
        return instrument

    @pytest.fixture
    def mock_signal(self, mock_instrument):
        """Create a mock signal."""
        signal = MagicMock(spec=Signal)
        signal.instrument = mock_instrument
        signal.is_service = False
        signal.group = None
        signal.reference_price = None
        return signal

    @pytest.fixture
    def mock_quote(self):
        """Create a mock quote."""
        quote = MagicMock(spec=Quote)
        quote.mid_price.return_value = 50000.0
        quote.time = np.datetime64("2023-01-01T00:00:00", "ns")
        return quote

    @pytest.fixture
    def processing_manager(self, mock_instrument):
        """Create a ProcessingManager with mocked dependencies."""
        # Mock all dependencies
        context = MagicMock()
        context.is_simulation = True
        context.instruments = [mock_instrument]
        context._strategy_state = MagicMock()
        context._strategy_state.is_on_fit_called = True
        context._strategy_state.is_on_start_called = True
        context._strategy_state.is_on_warmup_finished_called = True

        strategy = MagicMock()
        strategy.__class__.__name__ = "TestStrategy"

        logging = MagicMock()

        cache = MagicMock()
        cache.default_timeframe = "1h"

        market_data = MagicMock()
        market_data.quote.return_value = None  # No quote by default
        market_data.get_market_data_cache.return_value = cache

        subscription_manager = MagicMock()
        subscription_manager.get_base_subscription.return_value = "quote"

        time_provider = MagicMock()
        time_provider.time.return_value = np.datetime64("2023-01-01T00:00:00", "ns")

        account = MagicMock()
        position_tracker = MagicMock()
        position_tracker.process_signals.return_value = []

        position_gathering = MagicMock()
        universe_manager = MagicMock()
        universe_manager.is_trading_allowed.return_value = True

        scheduler = MagicMock()

        health_monitor = MagicMock()
        health_monitor.return_value.__enter__ = MagicMock(return_value=None)
        health_monitor.return_value.__exit__ = MagicMock(return_value=False)

        delisting_detector = MagicMock()

        pm = ProcessingManager(
            context=context,
            strategy=strategy,
            logging=logging,
            market_data=market_data,
            subscription_manager=subscription_manager,
            time_provider=time_provider,
            account=account,
            position_tracker=position_tracker,
            position_gathering=position_gathering,
            universe_manager=universe_manager,
            scheduler=scheduler,
            is_simulation=True,
            health_monitor=health_monitor,
            delisting_detector=delisting_detector,
        )

        # Store references for testing
        pm._test_context = context
        pm._test_market_data = market_data
        pm._test_position_tracker = position_tracker

        return pm

    def test_signal_without_quote_stored_as_pending(self, processing_manager, mock_signal, mock_instrument):
        """Test that a signal without quote is stored in pending signals."""
        # Ensure no quote is available
        processing_manager._market_data.quote.return_value = None

        # Process the signal
        processing_manager._ProcessingManager__process_signals([mock_signal])

        # Verify signal was stored as pending
        assert mock_instrument in processing_manager._pending_no_quote_signals
        assert processing_manager._pending_no_quote_signals[mock_instrument] == mock_signal

        # Verify signal was NOT processed by tracker
        processing_manager._test_position_tracker.process_signals.assert_not_called()

    def test_signal_with_quote_processed_normally(self, processing_manager, mock_signal, mock_quote, mock_instrument):
        """Test that a signal with quote is processed normally."""
        # Ensure quote is available
        processing_manager._market_data.quote.return_value = mock_quote

        # Process the signal
        processing_manager._ProcessingManager__process_signals([mock_signal])

        # Verify signal was NOT stored as pending
        assert mock_instrument not in processing_manager._pending_no_quote_signals

        # Verify signal was processed by tracker
        processing_manager._test_position_tracker.process_signals.assert_called_once()

    def test_latest_signal_replaces_older_pending(self, processing_manager, mock_instrument):
        """Test that latest signal replaces older pending signal for same instrument."""
        # Create two signals for the same instrument
        signal1 = MagicMock(spec=Signal)
        signal1.instrument = mock_instrument
        signal1.is_service = False
        signal1.group = None
        signal1.reference_price = None

        signal2 = MagicMock(spec=Signal)
        signal2.instrument = mock_instrument
        signal2.is_service = False
        signal2.group = None
        signal2.reference_price = None

        # Ensure no quote is available
        processing_manager._market_data.quote.return_value = None

        # Process first signal
        processing_manager._ProcessingManager__process_signals([signal1])
        assert processing_manager._pending_no_quote_signals[mock_instrument] == signal1

        # Process second signal
        processing_manager._ProcessingManager__process_signals([signal2])
        assert processing_manager._pending_no_quote_signals[mock_instrument] == signal2

    def test_pending_signal_retried_when_quote_arrives(self, processing_manager, mock_signal, mock_quote, mock_instrument):
        """Test that pending signal is retried when quote arrives."""
        # First, store signal as pending (no quote)
        processing_manager._market_data.quote.return_value = None
        processing_manager._ProcessingManager__process_signals([mock_signal])
        assert mock_instrument in processing_manager._pending_no_quote_signals

        # Now simulate quote arrival by calling __update_base_data
        # Set up the cache to recognize this as base data
        processing_manager._cache.update = MagicMock()

        # Make _is_base_data return True
        processing_manager._is_base_data = MagicMock(return_value=(True, mock_quote))

        # Mock _get_tracker_for to return a mock tracker
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = []
        processing_manager._get_tracker_for = MagicMock(return_value=mock_tracker)

        # Call __update_base_data
        processing_manager._ProcessingManager__update_base_data(
            mock_instrument, "quote", mock_quote, is_historical=False
        )

        # Verify pending signal was removed and added to emitted signals
        assert mock_instrument not in processing_manager._pending_no_quote_signals
        assert mock_signal in processing_manager._emitted_signals

    def test_service_signals_bypass_quote_check(self, processing_manager, mock_signal, mock_instrument):
        """Test that service signals bypass the quote check."""
        # Mark signal as service signal
        mock_signal.is_service = True

        # Ensure no quote is available
        processing_manager._market_data.quote.return_value = None

        # Process the signal
        processing_manager._ProcessingManager__process_signals([mock_signal])

        # Verify signal was NOT stored as pending (service signals bypass check)
        assert mock_instrument not in processing_manager._pending_no_quote_signals

    def test_multiple_instruments_pending_signals(self, processing_manager):
        """Test pending signals for multiple instruments."""
        # Create two instruments
        instrument1 = MagicMock(spec=Instrument)
        instrument1.symbol = "BTCUSDT"
        instrument1.min_size = 0.001

        instrument2 = MagicMock(spec=Instrument)
        instrument2.symbol = "ETHUSDT"
        instrument2.min_size = 0.01

        # Create signals for both instruments
        signal1 = MagicMock(spec=Signal)
        signal1.instrument = instrument1
        signal1.is_service = False
        signal1.group = None
        signal1.reference_price = None

        signal2 = MagicMock(spec=Signal)
        signal2.instrument = instrument2
        signal2.is_service = False
        signal2.group = None
        signal2.reference_price = None

        # Ensure no quote is available
        processing_manager._market_data.quote.return_value = None

        # Process both signals
        processing_manager._ProcessingManager__process_signals([signal1, signal2])

        # Verify both are stored as pending
        assert instrument1 in processing_manager._pending_no_quote_signals
        assert instrument2 in processing_manager._pending_no_quote_signals

    def test_pending_signals_initialized_empty(self, processing_manager):
        """Test that pending signals dict is initialized empty."""
        assert processing_manager._pending_no_quote_signals == {}
