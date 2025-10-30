"""
Integration tests for the Slack Exporter.

These tests use the Slack API with bot tokens.
For real integration tests, set the SLACK_BOT_TOKEN and SLACK_TEST_CHANNEL environment variables
or add them to .env.integration file in the project root.
"""

import os

import numpy as np
import pytest

from qubx.core.basics import AssetType, Instrument, MarketType, Signal
from qubx.exporters.slack import SlackExporter


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
    ]


@pytest.mark.integration
class TestSlackExporterIntegration:
    """Integration tests for the SlackExporter using real Slack API."""

    def test_real_export_signal(self, account_viewer, instruments):
        """Test exporting a signal to a real Slack channel."""
        # Get credentials from environment
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        test_channel = os.getenv("SLACK_TEST_CHANNEL", "#test")

        # Skip the test if no bot token is available
        if not bot_token:
            pytest.skip("No Slack bot token available for integration test (set SLACK_BOT_TOKEN env variable)")

        # Create a signal
        signal = Signal("", instruments[0], 1.0, reference_price=50000.0, group="test_integration")

        # Create the exporter
        exporter = SlackExporter(
            strategy_name="test_integration",
            bot_token=bot_token,
            signals_channel=test_channel,
            export_signals=True,
            export_targets=False,
            export_position_changes=False,
        )

        # Export the signal
        current_time = np.datetime64("now")
        exporter.export_signals(current_time, [signal], account_viewer)

        # No assertion needed - if the request fails, an exception will be raised
        # This test is mainly to verify that the exporter can successfully send messages to Slack
