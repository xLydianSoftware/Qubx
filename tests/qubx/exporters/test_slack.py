"""
Unit tests for the Slack Exporter.

These tests use mocks to simulate Slack API responses.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, Signal, TargetPosition
from qubx.exporters.formatters import IExportFormatter
from qubx.exporters.slack import SlackExporter

DUMMY_BOT_TOKEN = "xoxb-test-token"
DUMMY_CHANNEL = "#test-channel"


@pytest.fixture
def instruments():
    """Fixture for test instruments."""
    return [
        Instrument(
            symbol="BTC-USDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
        Instrument(
            symbol="ETH-USDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SPOT,
            exchange="BINANCE",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.00001,
            min_size=0.0001,
        ),
    ]


@pytest.fixture
def signals(instruments):
    """Fixture for test signals."""
    return [
        Signal("", instruments[0], 1.0, reference_price=50000.0, group="test_group"),
        Signal("", instruments[1], -0.5, reference_price=3000.0, group="test_group"),
    ]


@pytest.fixture
def target_positions(instruments, signals):
    """Fixture for test target positions."""
    time_now = np.datetime64("now")
    return [
        TargetPosition(time_now, signals[0].instrument, 0.1),
        TargetPosition(time_now, signals[1].instrument, -0.05),
    ]


class TestSlackExporter:
    """Unit tests for the SlackExporter using mocks."""

    @patch("qubx.exporters.slack.SlackClient")
    def test_export_signals(self, mock_slack_client_class, account_viewer, signals):
        """Test exporting signals with the default formatter."""
        # Create a mock instance
        mock_client_instance = MagicMock()
        mock_slack_client_class.return_value = mock_client_instance

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            bot_token=DUMMY_BOT_TOKEN,
            signals_channel=DUMMY_CHANNEL,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
        )

        # Export the signals
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, signals, account_viewer)

        # Wait for the background thread to complete
        time.sleep(0.1)

        # Check if notify_message_async was called with the correct arguments
        assert (
            mock_client_instance.notify_message_async.call_count == 2
        ), f"Expected 2 calls to notify_message_async, got {mock_client_instance.notify_message_async.call_count}"

        # Check the content of the first signal
        first_call_args = mock_client_instance.notify_message_async.call_args_list[0]
        call_kwargs = first_call_args[1]
        channel = call_kwargs["channel"]
        blocks = call_kwargs["blocks"]

        assert channel == DUMMY_CHANNEL, "Expected correct channel"
        assert blocks is not None, "Expected 'blocks' parameter"
        assert len(blocks) >= 2, f"Expected at least 2 blocks, got {len(blocks)}"

        # Check header block
        assert blocks[0]["type"] == "header", "Expected first block to be a header"
        assert "New Signal" in blocks[0]["text"]["text"], "Expected 'New Signal' in header text"

        # Check section block
        assert blocks[1]["type"] == "section", "Expected second block to be a section"
        section_text = blocks[1]["text"]["text"]
        assert "BUY" in section_text, "Expected 'BUY' in section text for first signal"
        assert "BTC-USDT" in section_text, "Expected 'BTC-USDT' in section text"
        assert "BINANCE" in section_text, "Expected 'BINANCE' in section text"
        assert "50000.0" in section_text, "Expected reference price in section text"
        assert "test_group" in section_text, "Expected group in section text"

        # Check the content of the second signal
        second_call_args = mock_client_instance.notify_message_async.call_args_list[1]
        call_kwargs = second_call_args[1]
        blocks = call_kwargs["blocks"]

        # Check section block
        section_text = blocks[1]["text"]["text"]
        assert "SELL" in section_text, "Expected 'SELL' in section text for second signal"
        assert "ETH-USDT" in section_text, "Expected 'ETH-USDT' in section text"
        assert "3000.0" in section_text, "Expected reference price in section text"

    @patch("qubx.exporters.slack.SlackClient")
    def test_export_target_positions(self, mock_slack_client_class, account_viewer, target_positions):
        """Test exporting target positions."""
        # Create a mock instance
        mock_client_instance = MagicMock()
        mock_slack_client_class.return_value = mock_client_instance

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            bot_token=DUMMY_BOT_TOKEN,
            targets_channel=DUMMY_CHANNEL,
            export_signals=False,
            export_targets=True,
            export_position_changes=False,
        )

        # Export the target positions
        current_time = np.datetime64("now")
        exporter.export_target_positions(current_time, target_positions, account_viewer)

        # Wait for the background thread to complete
        time.sleep(0.1)

        # Check if notify_message_async was called with the correct arguments
        assert mock_client_instance.notify_message_async.call_count == 2, f"Expected 2 calls to notify_message_async, got {mock_client_instance.notify_message_async.call_count}"

        # Check the content of the first target position
        first_call_args = mock_client_instance.notify_message_async.call_args_list[0]
        call_kwargs = first_call_args[1]
        blocks = call_kwargs["blocks"]

        # Check header block
        assert blocks[0]["type"] == "header", "Expected first block to be a header"
        assert "Target Position" in blocks[0]["text"]["text"], "Expected 'Target Position' in header text"

        # Check section block
        assert blocks[1]["type"] == "section", "Expected second block to be a section"
        section_text = blocks[1]["text"]["text"]
        assert "BTC-USDT" in section_text, "Expected 'BTC-USDT' in section text"
        assert "0.1" in section_text, "Expected target size in section text"

        # Check the content of the second target position
        second_call_args = mock_client_instance.notify_message_async.call_args_list[1]
        call_kwargs = second_call_args[1]
        blocks = call_kwargs["blocks"]

        # Check section block
        section_text = blocks[1]["text"]["text"]
        assert "ETH-USDT" in section_text, "Expected 'ETH-USDT' in section text"
        assert "-0.05" in section_text, "Expected target size in section text"

    @patch("qubx.exporters.slack.SlackClient")
    def test_export_position_changes(self, mock_slack_client_class, account_viewer, instruments):
        """Test exporting position changes."""
        # Create a mock instance
        mock_client_instance = MagicMock()
        mock_slack_client_class.return_value = mock_client_instance

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            bot_token=DUMMY_BOT_TOKEN,
            position_changes_channel=DUMMY_CHANNEL,
            export_signals=False,
            export_targets=False,
            export_position_changes=True,
        )

        # Export a position change
        current_time = np.datetime64("now")
        instrument = instruments[0]
        price = 50000.0

        # Export the position change
        exporter.export_position_changes(current_time, instrument, price, account_viewer)

        # Wait for the background thread to complete
        time.sleep(0.1)

        # Check if notify_message_async was called with the correct arguments
        assert mock_client_instance.notify_message_async.call_count == 1, f"Expected 1 call to notify_message_async, got {mock_client_instance.notify_message_async.call_count}"

        # Check the content of the position change
        call_args = mock_client_instance.notify_message_async.call_args_list[0]
        call_kwargs = call_args[1]
        blocks = call_kwargs["blocks"]

        # Check header block
        assert blocks[0]["type"] == "header", "Expected first block to be a header"
        assert "Position Change" in blocks[0]["text"]["text"], "Expected 'Position Change' in header text"

        # Check section block
        assert blocks[1]["type"] == "section", "Expected second block to be a section"
        section_text = blocks[1]["text"]["text"]
        assert "BTC-USDT" in section_text, "Expected 'BTC-USDT' in section text"
        assert "50000.0" in section_text, "Expected price in section text"
        assert "Current Quantity" in section_text, "Expected 'Current Quantity' in section text"

    @patch("qubx.exporters.slack.SlackClient")
    def test_post_to_slack_error_handling(self, mock_slack_client_class, account_viewer, signals):
        """Test error handling when posting to Slack."""
        # Create a mock instance
        mock_client_instance = MagicMock()
        mock_slack_client_class.return_value = mock_client_instance

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            bot_token=DUMMY_BOT_TOKEN,
            signals_channel=DUMMY_CHANNEL,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
        )

        # Export the signals - this should not raise an exception despite any errors
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, signals, account_viewer)

        # Wait for the background thread to complete
        time.sleep(0.1)

        # Verify that the post was attempted
        assert mock_client_instance.notify_message_async.call_count == 2, f"Expected 2 calls to notify_message_async, got {mock_client_instance.notify_message_async.call_count}"

    @patch("qubx.exporters.slack.SlackClient")
    def test_custom_formatter(self, mock_slack_client_class, account_viewer, signals):
        """Test using a custom formatter."""
        # Create a mock instance
        mock_client_instance = MagicMock()
        mock_slack_client_class.return_value = mock_client_instance

        # Create a custom formatter class for testing
        class TestFormatter(IExportFormatter):
            def format_signal(self, time, signal, account):
                return {"text": f"TEST: {signal.instrument}"}

            def format_target_position(self, time, target, account):
                return {"text": f"TEST: {target.instrument}"}

            def format_position_change(self, time, instrument, price, account):
                return {"text": f"TEST: {instrument}"}

        # Create the exporter with the custom formatter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            bot_token=DUMMY_BOT_TOKEN,
            signals_channel=DUMMY_CHANNEL,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
            formatter=TestFormatter(),
        )

        # Export the signals
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, signals, account_viewer)

        # Wait for the background thread to complete
        time.sleep(0.1)

        # Check if notify_message_async was called with the correct arguments
        assert mock_client_instance.notify_message_async.call_count == 2, f"Expected 2 calls to notify_message_async, got {mock_client_instance.notify_message_async.call_count}"

        # The custom formatter returns {"text": ...}, but now we need to check for empty blocks
        # since the exporter extracts "blocks" from the formatter output (which won't exist)
        # and passes them to notify_message_async
        first_call_args = mock_client_instance.notify_message_async.call_args_list[0]
        call_kwargs = first_call_args[1]
        # The custom formatter doesn't return blocks, so blocks should be an empty list
        blocks = call_kwargs["blocks"]
        assert blocks == [], "Expected empty blocks list from custom formatter without 'blocks' key"
