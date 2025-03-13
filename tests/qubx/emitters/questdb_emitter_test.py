"""
Integration tests for QuestDBMetricEmitter.

This module tests the QuestDBMetricEmitter with a running strategy to ensure
metrics are properly emitted to QuestDB.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from qubx import QubxLogConfig, logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, LiveTimeProvider
from qubx.core.interfaces import IDataProvider, IStrategy, IStrategyContext, IStrategyInitializer
from qubx.core.lookups import lookup
from qubx.data.helpers import loader
from qubx.data.readers import AsBars
from qubx.emitters.questdb import QuestDBMetricEmitter
from qubx.restarts.state_resolvers import StateResolver
from qubx.utils.misc import class_import
from qubx.utils.runner.runner import run_strategy_yaml


class TestQuestDBEmitterIntegration:
    """Integration tests for QuestDBMetricEmitter."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing with QuestDB emitter configuration."""
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="wb") as temp_file:
            config = {
                "strategy": "strategy",
                "parameters": {},
                "exchanges": {
                    "BINANCE.UM": {
                        "connector": "ccxt",
                        "universe": ["BTCUSDT", "ETHUSDT"],
                    }
                },
                "logging": {
                    "logger": "InMemoryLogsWriter",
                    "position_interval": "10Sec",
                    "portfolio_interval": "5Min",
                    "heartbeat_interval": "1m",
                },
                "emission": {
                    "stats_interval": "10s",  # Emit stats frequently for testing
                    "stats_to_emit": [
                        "total_capital",
                        "net_leverage",
                        "gross_leverage",
                        "universe_size",
                        "position_count",
                    ],
                    "emitters": [
                        {
                            "emitter": "QuestDBMetricEmitter",
                            "parameters": {
                                "host": "nebula",
                                "port": 9000,
                                "table_name": "qubx_test_metrics",
                            },
                            "tags": {"environment": "test", "test_run": "integration"},
                        }
                    ],
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
            logger.info(
                f"Pushing bar {mock.current_index + 1}/{len(mock.ohlc_data)} for {instrument.symbol} at {bar_time}"
            )
            mock.channel.send((instrument, data_type, bar, is_historical))

            # Increment the index
            mock.current_index += 1
            return True

        mock.push = push

        def stream_bars(max_bars: int = 10):
            """
            Helper method to stream test bars through the channel.

            Args:
                max_bars: Number of bars to stream (default: 10)
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

    @pytest.mark.integration
    @patch("qubx.utils.runner.runner.LiveTimeProvider")
    @patch("qubx.utils.runner.runner._create_data_provider")
    @patch("qubx.utils.runner.runner.CtrlChannel")
    def test_questdb_emitter_integration(
        self,
        mock_ctrl_channel_class,
        mock_create_data_provider,
        mock_live_time_provider_class,
        temp_config_file,
        mock_data_provider,
        mock_time_provider,
    ):
        """Test that QuestDBMetricEmitter properly emits metrics when used with a strategy."""
        # Set up mocks
        channel = mock_data_provider.channel
        mock_ctrl_channel_class.return_value = channel
        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider

        QubxLogConfig.set_log_level("INFO")

        class MockStrategy(IStrategy):
            def on_init(self, initializer: IStrategyInitializer) -> None:
                initializer.set_base_subscription(DataType.OHLC["1h"])
                initializer.set_warmup("10d")
                initializer.set_state_resolver(StateResolver.SYNC_STATE)

            def on_start(self, ctx: IStrategyContext) -> None:
                instr = ctx.instruments[0]
                logger.info(f"on_start ::: <cyan>Buying {instr.symbol} qty 1</cyan>")
                ctx.trade(instr, 1)

                # Emit a custom metric directly using the emitter
                if isinstance(ctx.emitter, QuestDBMetricEmitter):
                    ctx.emitter.emit("custom_metric", 42.0, {"custom_tag": "test_value"}, ctx.time())

        with patch(
            "qubx.utils.runner.runner.class_import",
            side_effect=lambda arg: MockStrategy if arg == "strategy" else class_import(arg),
        ):
            ctx = run_strategy_yaml(temp_config_file, paper=True)

        # Stream live bars to trigger metric emissions
        mock_data_provider.stream_bars(max_bars=20)

        # Allow some time for metrics to be processed
        time.sleep(0.5)

        # Stop the context
        if ctx.is_running():
            ctx.stop()
