"""
Integration tests for the Slack Exporter.

These tests use a mock server to simulate Slack webhook responses.
For real integration tests, set the SLACK_WEBHOOK_URL environment variable
or add it to .env.integration file in the project root.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
import requests

from qubx.core.basics import AssetType, Instrument, MarketType, Signal, TargetPosition
from qubx.exporters.slack import SlackExporter

DUMMY_WEBHOOK_URL = "https://hooks.slack.com/services/TXXXXXXXX/BXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX"


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
        Signal(instruments[0], 1.0, reference_price=50000.0, group="test_group"),
        Signal(instruments[1], -0.5, reference_price=3000.0, group="test_group"),
    ]


@pytest.fixture
def target_positions(instruments, signals):
    """Fixture for test target positions."""
    time_now = np.datetime64("now")
    return [
        TargetPosition(time_now, signals[0], 0.1),
        TargetPosition(time_now, signals[1], -0.05),
    ]


# Mock response for requests.post
class MockResponse:
    """Mock response for requests.post."""

    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self.json_data = json_data or {}
        self.text = json.dumps(self.json_data)

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"HTTP Error: {self.status_code}")


class TestSlackExporter:
    """Unit tests for the SlackExporter using mocks."""

    @patch("requests.post")
    def test_export_signals(self, mock_post, account_viewer, signals):
        """Test exporting signals with the default formatter."""
        # Configure mock
        mock_post.return_value = MockResponse(200, {"ok": True})

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            signals_webhook_url=DUMMY_WEBHOOK_URL,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
        )

        # Export the signals
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, signals, account_viewer)

        # Check if requests.post was called with the correct arguments
        assert mock_post.call_count == 2, f"Expected 2 calls to requests.post, got {mock_post.call_count}"

        # Check the content of the first signal
        first_call_args = mock_post.call_args_list[0][1]
        assert "json" in first_call_args, "Expected 'json' in call arguments"

        json_data = first_call_args["json"]
        assert "blocks" in json_data, "Expected 'blocks' in JSON data"

        blocks = json_data["blocks"]
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
        second_call_args = mock_post.call_args_list[1][1]
        json_data = second_call_args["json"]
        blocks = json_data["blocks"]

        # Check section block
        section_text = blocks[1]["text"]["text"]
        assert "SELL" in section_text, "Expected 'SELL' in section text for second signal"
        assert "ETH-USDT" in section_text, "Expected 'ETH-USDT' in section text"
        assert "3000.0" in section_text, "Expected reference price in section text"

    @patch("requests.post")
    def test_export_target_positions(self, mock_post, account_viewer, target_positions):
        """Test exporting target positions."""
        # Configure mock
        mock_post.return_value = MockResponse(200, {"ok": True})

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            targets_webhook_url=DUMMY_WEBHOOK_URL,
            export_signals=False,
            export_targets=True,
            export_position_changes=False,
        )

        # Export the target positions
        current_time = np.datetime64("now")
        exporter.export_target_positions(current_time, target_positions, account_viewer)

        # Check if requests.post was called with the correct arguments
        assert mock_post.call_count == 2, f"Expected 2 calls to requests.post, got {mock_post.call_count}"

        # Check the content of the first target position
        first_call_args = mock_post.call_args_list[0][1]
        json_data = first_call_args["json"]
        blocks = json_data["blocks"]

        # Check header block
        assert blocks[0]["type"] == "header", "Expected first block to be a header"
        assert "Target Position" in blocks[0]["text"]["text"], "Expected 'Target Position' in header text"

        # Check section block
        assert blocks[1]["type"] == "section", "Expected second block to be a section"
        section_text = blocks[1]["text"]["text"]
        assert "BTC-USDT" in section_text, "Expected 'BTC-USDT' in section text"
        assert "0.1" in section_text, "Expected target size in section text"

        # Check the content of the second target position
        second_call_args = mock_post.call_args_list[1][1]
        json_data = second_call_args["json"]
        blocks = json_data["blocks"]

        # Check section block
        section_text = blocks[1]["text"]["text"]
        assert "ETH-USDT" in section_text, "Expected 'ETH-USDT' in section text"
        assert "-0.05" in section_text, "Expected target size in section text"

    @patch("requests.post")
    def test_export_position_changes(self, mock_post, account_viewer, instruments):
        """Test exporting position changes."""
        # Configure mock
        mock_post.return_value = MockResponse(200, {"ok": True})

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            position_changes_webhook_url=DUMMY_WEBHOOK_URL,
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

        # Check if requests.post was called with the correct arguments
        assert mock_post.call_count == 1, f"Expected 1 call to requests.post, got {mock_post.call_count}"

        # Check the content of the position change
        call_args = mock_post.call_args[1]
        json_data = call_args["json"]
        blocks = json_data["blocks"]

        # Check header block
        assert blocks[0]["type"] == "header", "Expected first block to be a header"
        assert "Position Change" in blocks[0]["text"]["text"], "Expected 'Position Change' in header text"

        # Check section block
        assert blocks[1]["type"] == "section", "Expected second block to be a section"
        section_text = blocks[1]["text"]["text"]
        assert "BTC-USDT" in section_text, "Expected 'BTC-USDT' in section text"
        assert "50000.0" in section_text, "Expected price in section text"
        assert "Current Quantity" in section_text, "Expected 'Current Quantity' in section text"

    @patch("requests.post")
    def test_post_to_slack_error_handling(self, mock_post, account_viewer, signals):
        """Test error handling when posting to Slack."""
        # Configure mock to return an error
        mock_post.return_value = MockResponse(500, {"ok": False, "error": "server_error"})

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            signals_webhook_url=DUMMY_WEBHOOK_URL,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
        )

        # Export the signals
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, signals, account_viewer)

        # Check if requests.post was called with the correct arguments
        assert mock_post.call_count == 2, f"Expected 2 calls to requests.post, got {mock_post.call_count}"

        # The test should complete without raising exceptions, as the exporter should handle errors gracefully

    @patch("requests.post")
    def test_custom_formatter(self, mock_post, account_viewer, signals):
        """Test using a custom formatter."""
        # Configure mock
        mock_post.return_value = MockResponse(200, {"ok": True})

        # Create a custom formatter with different emoji
        from qubx.exporters.formatters import SlackMessageFormatter

        custom_formatter = SlackMessageFormatter(strategy_emoji=":rocket:", include_account_info=False)

        # Create the exporter with the custom formatter
        exporter = SlackExporter(
            strategy_name="test_strategy",
            signals_webhook_url=DUMMY_WEBHOOK_URL,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
            formatter=custom_formatter,
        )

        # Export the signals
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, signals, account_viewer)

        # Check if requests.post was called with the correct arguments
        assert mock_post.call_count == 2, f"Expected 2 calls to requests.post, got {mock_post.call_count}"

        # Check the content of the first signal
        first_call_args = mock_post.call_args_list[0][1]
        json_data = first_call_args["json"]
        blocks = json_data["blocks"]

        # Check header block - should contain the custom emoji
        header_text = blocks[0]["text"]["text"]
        assert ":rocket:" in header_text, "Expected custom emoji ':rocket:' in header text"

        # Check that there's no account info block (we disabled it)
        assert len(blocks) == 2, f"Expected exactly 2 blocks (no account info), got {len(blocks)}"


@pytest.mark.integration
class TestSlackExporterIntegration:
    """Integration tests for the SlackExporter using real webhook URLs."""

    def test_real_export_signal(self, slack_webhook_url, account_viewer, instruments):
        """Test exporting a signal to a real Slack webhook."""
        # Skip the test if no webhook URL is available
        if not slack_webhook_url:
            pytest.skip("No Slack webhook URL available for integration test")

        # Create a signal
        signal = Signal(instruments[0], 1.0, reference_price=50000.0, group="test_integration")

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_integration",
            signals_webhook_url=slack_webhook_url,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
        )

        # Export the signal
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, [signal], account_viewer)

        # No assertion needed - if the request fails, an exception will be raised
        # This test is mainly to verify that the exporter can successfully send messages to Slack
