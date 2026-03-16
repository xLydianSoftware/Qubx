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
import pytest
import yaml

from qubx import QubxLogConfig, logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, LiveTimeProvider, TriggerEvent
from qubx.core.interfaces import IDataProvider, IStrategy, IStrategyContext, IStrategyInitializer
from qubx.core.lookups import lookup
from qubx.data import CsvStorage
from qubx.restarts.state_resolvers import StateResolver
from qubx.utils.misc import class_import
from qubx.utils.runner.runner import run_strategy_yaml


@pytest.mark.integration
class TestQuestDBEmitterIntegration:
    """Integration tests for QuestDBMetricEmitter."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing with QuestDB emitter configuration."""
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="wb") as temp_file:
            config = {
                "strategy": "strategy",
                "parameters": {},
                "live": {
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
                        "stats_interval": "10s",  # - emit stats frequently for testing
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
                        "data": {
                            "storage": "csv::tests/data/storages/csv",
                        }
                    },
                },
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
        # - default time — use the start_time fixture
        mock.current_time = np.datetime64(f"{start_time}T00:00:00")

        # - allow setting the time
        def set_time(new_time):
            mock.current_time = new_time

        # - return current time as datetime64[ns] with proper item() method
        def get_time():
            return mock.current_time.astype("datetime64[ns]")

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

        # - set up basic properties
        mock.is_simulation = False
        mock.time_provider = mock_time_provider
        mock.channel = CtrlChannel("databus")

        # - set up method returns
        mock.subscribe.return_value = None
        mock.unsubscribe.return_value = None
        mock.has_subscription.return_value = True
        mock.get_subscriptions.return_value = ["ohlc(1h)"]
        mock.get_subscribed_instruments.return_value = []
        mock.exchange.return_value = "BINANCE.UM"

        ldr = CsvStorage("tests/data/storages/csv/").get_reader("BINANCE.UM", "SWAP")
        btc_ohlc = ldr.read("BTCUSDT", "ohlc(1h)", start=start_time, stop=stop_time).to_records()
        eth_ohlc = ldr.read("ETHUSDT", "ohlc(1h)", start=start_time, stop=stop_time).to_records()

        btc, eth = self._find_instrument("BINANCE.UM", "BTCUSDT"), self._find_instrument("BINANCE.UM", "ETHUSDT")
        btc_ohlc = [(btc, DataType.OHLC["1h"], bar, False) for bar in btc_ohlc]
        eth_ohlc = [(eth, DataType.OHLC["1h"], bar, False) for bar in eth_ohlc]
        ohlc = sorted([*btc_ohlc, *eth_ohlc], key=lambda x: x[2].time)

        # - store OHLC data for sequential pushing
        mock.ohlc_data = ohlc
        mock.current_index = 0

        def push():
            if mock.current_index >= len(mock.ohlc_data):
                return False  # - no more bars

            instrument, data_type, bar, is_historical = mock.ohlc_data[mock.current_index]

            # - advance time provider to match the bar
            bar_time = np.datetime64(bar.time, "ns")
            mock.time_provider.set_time(bar_time)

            logger.info(
                f"Pushing bar {mock.current_index + 1}/{len(mock.ohlc_data)} for {instrument.symbol} at {bar_time}"
            )
            mock.channel.send((instrument, data_type, bar, is_historical))
            mock.current_index += 1
            return True

        mock.push = push

        def stream_bars(max_bars: int = 10):
            """
            Helper method to stream test bars through the channel.

            Args:
                max_bars: Number of bars to stream (default: 10)
            """
            bars_pushed = 0
            for _ in range(max_bars):
                if not mock.push():
                    logger.info(f"No more bars to push after {bars_pushed} bars")
                    break
                bars_pushed += 1
            logger.info(f"Successfully pushed {bars_pushed} bars to context")

        mock.stream_bars = stream_bars
        return mock

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
        channel = mock_data_provider.channel
        mock_ctrl_channel_class.return_value = channel
        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider

        QubxLogConfig.set_log_level("INFO")

        from qubx.ta.indicators import atr

        class MockStrategy(IStrategy):
            def on_init(self, initializer: IStrategyInitializer) -> None:
                initializer.set_event_schedule("1h")
                initializer.set_base_subscription(DataType.OHLC["1h"])
                initializer.set_warmup("10d")
                initializer.set_state_resolver(StateResolver.SYNC_STATE)

            def on_start(self, ctx: IStrategyContext) -> None:
                self._instr_to_indicator = {
                    instr: atr(ctx.ohlc(instr), period=14, smoother="sma", percentage=True) for instr in ctx.instruments
                }

            def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> None:
                for instr, ind in self._instr_to_indicator.items():
                    if len(ind) > 0 and not np.isnan(ind[0]):
                        ctx.emitter.emit(
                            "atr(14)",
                            ind[0],
                            {"symbol": instr.symbol, "exchange": instr.exchange},
                            ctx.time(),
                        )

        with patch(
            "qubx.utils.runner.runner.class_import",
            side_effect=lambda arg: MockStrategy if arg == "strategy" else class_import(arg),
        ):
            ctx = run_strategy_yaml(temp_config_file, paper=True)

        # - stream live bars to trigger metric emissions
        mock_data_provider.stream_bars(max_bars=20)

        # - allow some time for metrics to be processed
        time.sleep(0.5)

        if ctx.is_running():
            ctx.stop()
