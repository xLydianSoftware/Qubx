"""
Integration tests for the Slack Exporter.

These tests use a mock server to simulate Slack webhook responses.
For real integration tests, set the SLACK_WEBHOOK_URL environment variable
or add it to .env.integration file in the project root.
"""

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
