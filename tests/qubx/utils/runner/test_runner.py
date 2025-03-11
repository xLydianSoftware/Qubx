import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from qubx import logger
from qubx.data.helpers import loader

# Add tests/strategies to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../strategies")))

from macd_crossover.models.macd_crossover import MacdCrossoverStrategy  # type: ignore

from qubx import QubxLogConfig
from qubx.core.basics import CtrlChannel, DataType, Instrument, LiveTimeProvider, RestoredState
from qubx.core.context import IStrategyContext
from qubx.core.interfaces import IDataProvider
from qubx.core.lookups import lookup
from qubx.data.readers import AsBars
from qubx.utils.runner.accounts import AccountConfigurationManager
from qubx.utils.runner.runner import run_strategy_yaml


class TestRunStrategyYaml:
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="wb") as temp_file:
            config = {
                "strategy": "macd_crossover.models.macd_crossover.MacdCrossoverStrategy",
                "parameters": {
                    "signal_period": 9,
                    "slow_period": 26,
                    "fast_period": 12,
                    "leverage": 1.0,
                    "timeframe": "1h",
                },
                "exchanges": {
                    "BINANCE.UM": {
                        "connector": "ccxt",
                        "universe": ["BTCUSDT", "ETHUSDT"],
                    }
                },
                "logging": {
                    "logger": "CsvFileLogsWriter",
                    "position_interval": "10Sec",
                    "portfolio_interval": "5Min",
                    "heartbeat_interval": "1m",
                },
                "warmup": {
                    "readers": [
                        {
                            "data_type": "ohlc(1h)",
                            "readers": [
                                {
                                    "reader": "csv::tests/data/csv_1h/",
                                    "args": {},
                                }
                            ],
                        }
                    ]
                },
                "aux": None,
            }
            yaml_content = yaml.dump(config, encoding="utf-8")
            temp_file.write(yaml_content)
            temp_file_path = temp_file.name

        yield Path(temp_file_path)

        # Clean up the temporary file
        import os

        os.unlink(temp_file_path)

    @pytest.fixture
    def start_time(self):
        """Return the start time for data loading."""
        return "2023-07-01"

    @pytest.fixture
    def stop_time(self):
        """Return the stop time for data loading."""
        return "2023-08-01"

    @pytest.fixture
    def mock_time_provider(self, start_time):
        """Create a mock time provider that returns a configurable time."""
        mock = MagicMock(spec=LiveTimeProvider)
        # Default time - use the start_time fixture
        mock.current_time = np.datetime64(f"{start_time}T00:00:00")

        # Allow setting the time
        def set_time(new_time):
            mock.current_time = new_time

        # Return the current time
        def get_time():
            return mock.current_time

        mock.time.side_effect = get_time
        mock.set_time = set_time
        return mock

    def _find_instrument(self, exchange: str, symbol: str) -> Instrument:
        instr = lookup.find_symbol(exchange, symbol)
        assert instr is not None, f"Instrument {symbol} not found"
        return instr

    @pytest.fixture
    def mock_data_provider(self, mock_time_provider, start_time, stop_time):
        """Create a mock data provider with appropriate implementation of IDataProvider interface."""
        mock = MagicMock(spec=IDataProvider)

        # Set up basic properties
        mock.is_simulation = False

        # Set up time provider
        mock.time_provider = mock_time_provider

        # Set up a real channel
        mock.channel = CtrlChannel("databus")

        # Set up method returns
        mock.subscribe.return_value = None
        mock.unsubscribe.return_value = None
        mock.has_subscription.return_value = True
        mock.get_subscriptions.return_value = ["ohlc(1h)"]
        mock.get_subscribed_instruments.return_value = []
        mock.exchange.return_value = "BINANCE.UM"

        ldr = loader("BINANCE.UM", "1h", symbols=["BTCUSDT", "ETHUSDT"], source="csv::tests/data/csv_1h/", n_jobs=1)
        btc_ohlc = ldr.read("BTCUSDT", start=start_time, stop=stop_time, transform=AsBars())
        eth_ohlc = ldr.read("ETHUSDT", start=start_time, stop=stop_time, transform=AsBars())
        btc, eth = self._find_instrument("BINANCE.UM", "BTCUSDT"), self._find_instrument("BINANCE.UM", "ETHUSDT")
        btc_ohlc = [(btc, DataType.OHLC["1h"], bar, False) for bar in btc_ohlc]
        eth_ohlc = [(eth, DataType.OHLC["1h"], bar, False) for bar in eth_ohlc]
        ohlc = sorted([*btc_ohlc, *eth_ohlc], key=lambda x: x[2].time)

        # Store the OHLC data for pushing
        mock.ohlc_data = ohlc
        mock.current_index = 0

        # Implement push method
        def push():
            if mock.current_index >= len(mock.ohlc_data):
                return False  # No more bars to push

            # Get the next bar
            instrument, data_type, bar, is_historical = mock.ohlc_data[mock.current_index]

            # Update the time provider to match the bar's time
            bar_time = np.datetime64(bar.time, "ns")
            mock.time_provider.set_time(bar_time)

            # Send the event through the channel
            mock.channel.send((instrument, data_type, bar, is_historical))
            logger.info(
                f"Pushed bar {mock.current_index + 1}/{len(mock.ohlc_data)} for {instrument.symbol} at {bar_time}"
            )

            # Increment the index
            mock.current_index += 1
            return True

        mock.push = push

        def stream_bars(max_bars: int = 10):
            """
            Helper method to stream test bars through the channel.

            Args:
                context: The strategy context
                mock_data_provider: The mock data provider with a channel and push method
                num_bars: Number of bars to stream (default: 3)
            """
            # Push bars using the data provider's push method
            bars_pushed = 0
            for _ in range(max_bars):
                if not mock.push():
                    logger.info(f"No more bars to push after {bars_pushed} bars")
                    break
                bars_pushed += 1

            logger.info(f"Successfully pushed {bars_pushed} bars to context")

        mock.stream_bars = stream_bars

        return mock

    @pytest.fixture
    def mock_account_manager(self):
        """Create a mock account manager."""
        mock = MagicMock(spec=AccountConfigurationManager)
        mock.get_exchange_settings.return_value = MagicMock(testnet=False, commissions={})
        return mock

    @patch("qubx.utils.runner.runner.LiveTimeProvider")
    @patch("qubx.utils.runner.runner._create_data_provider")
    def test_run_strategy_yaml_with_warmup(
        self,
        mock_create_data_provider,
        mock_live_time_provider_class,
        temp_config_file,
        mock_data_provider,
        mock_time_provider,
    ):
        """Test running a strategy from a YAML file with warmup."""
        # Set up mocks
        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider

        QubxLogConfig.set_log_level("INFO")

        # Run the function under test
        context = run_strategy_yaml(temp_config_file, paper=True)

        # Verify the result is a StrategyContext
        assert isinstance(context, IStrategyContext)

        # Verify that the data provider was mocked
        mock_create_data_provider.assert_called()

        # Verify that the time provider was mocked
        mock_live_time_provider_class.assert_called_once()

        # Verify that the strategy in the context is a real MacdCrossoverStrategy
        assert isinstance(context.strategy, MacdCrossoverStrategy)

        # Stream test bars and verify they were received
        mock_data_provider.stream_bars(max_bars=10)

        time.sleep(1)

        if context.is_running():
            context.stop()

    @patch("qubx.utils.runner.runner.LiveTimeProvider")
    @patch("qubx.utils.runner.runner._create_data_provider")
    def test_run_strategy_yaml_with_state_restoration(
        self,
        mock_create_data_provider,
        mock_live_time_provider_class,
        mock_data_provider,
        temp_config_file,
        mock_time_provider,
    ):
        """Test running a strategy from a YAML file with state restoration."""
        # Set up mocks
        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider

        QubxLogConfig.set_log_level("INFO")

        # Create a mock restored state
        restored_state = MagicMock(spec=RestoredState)
        restored_state.positions = {}
        restored_state.orders = {}
        restored_state.balances = {}
        restored_state.instrument_to_target_positions = {}
        restored_state.time = np.datetime64("2023-01-01T00:00:00")

        # Patch the _restore_state function to return our mock
        with patch("qubx.utils.runner.runner._restore_state", return_value=restored_state):
            # Run the function under test with restore=True
            context = run_strategy_yaml(temp_config_file, paper=True, restore=True)

            # Verify the result is a StrategyContext
            assert isinstance(context, IStrategyContext)

            # Verify that the data provider was mocked
            mock_create_data_provider.assert_called()

            # Verify that the time provider was mocked
            mock_live_time_provider_class.assert_called_once()

            # Verify that the strategy in the context is a real MacdCrossoverStrategy
            assert isinstance(context.strategy, MacdCrossoverStrategy)

            # Stream test bars and verify they were received
            mock_data_provider.stream_bars(max_bars=10)

            time.sleep(1)

            if context.is_running():
                context.stop()
