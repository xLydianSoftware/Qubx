"""
Slack Exporter for trading data.

This module provides an implementation of ITradeDataExport that exports trading data to Slack channels.
"""

from typing import Dict, List, Optional

import requests

from qubx import logger
from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer, ITradeDataExport
from qubx.exporters.formatters import IExportFormatter, SlackMessageFormatter


class SlackExporter(ITradeDataExport):
    """
    Exports trading data to Slack channels.

    This exporter can be configured to export signals, target positions, and position changes
    to different Slack channels using webhook URLs.
    """

    def __init__(
        self,
        strategy_name: str,
        signals_webhook_url: Optional[str] = None,
        targets_webhook_url: Optional[str] = None,
        position_changes_webhook_url: Optional[str] = None,
        export_signals: bool = True,
        export_targets: bool = True,
        export_position_changes: bool = True,
        formatter: Optional[IExportFormatter] = None,
        strategy_emoji: str = ":chart_with_upwards_trend:",
        include_account_info: bool = True,
    ):
        """
        Initialize the Slack Exporter.

        Args:
            strategy_name: Name of the strategy (used in message headers)
            signals_webhook_url: Webhook URL for signals channel
            targets_webhook_url: Webhook URL for target positions channel
            position_changes_webhook_url: Webhook URL for position changes channel
            export_signals: Whether to export signals
            export_targets: Whether to export target positions
            export_position_changes: Whether to export position changes
            formatter: Formatter to use for formatting data (default: SlackMessageFormatter)
            strategy_emoji: Emoji to use for the strategy in messages
            include_account_info: Whether to include account information in messages
        """
        self._strategy_name = strategy_name

        self._signals_webhook_url = signals_webhook_url
        self._targets_webhook_url = targets_webhook_url
        self._position_changes_webhook_url = position_changes_webhook_url

        self._export_signals = export_signals and signals_webhook_url is not None
        self._export_targets = export_targets and targets_webhook_url is not None
        self._export_position_changes = export_position_changes and position_changes_webhook_url is not None

        # Use provided formatter or create a new SlackMessageFormatter
        self._formatter = formatter or SlackMessageFormatter(
            strategy_emoji=strategy_emoji, include_account_info=include_account_info
        )

        logger.info(
            f"[SlackExporter] Initialized for strategy '{strategy_name}' with "
            f"signals: {self._export_signals}, targets: {self._export_targets}, "
            f"position_changes: {self._export_position_changes}"
        )

    def _post_to_slack(self, webhook_url: str, data: Dict) -> bool:
        """
        Post data to a Slack webhook URL.

        Args:
            webhook_url: The Slack webhook URL to post to
            data: The data to post

        Returns:
            bool: True if the post was successful, False otherwise
        """
        try:
            response = requests.post(webhook_url, json=data)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"[SlackExporter] Failed to post to Slack: {e}")
            return False

    def export_signals(self, time: dt_64, signals: List[Signal], account: IAccountViewer) -> None:
        """
        Export signals to Slack.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to export
            account: Account viewer to get account information
        """
        if not self._export_signals or not signals or not self._signals_webhook_url:
            return

        try:
            for signal in signals:
                # Format the signal using the formatter
                data = self._formatter.format_signal(time, signal, account)

                # Post to Slack
                success = self._post_to_slack(self._signals_webhook_url, data)

                if success:
                    logger.debug(f"[SlackExporter] Exported signal for {signal.instrument} to Slack")
                else:
                    logger.warning(f"[SlackExporter] Failed to export signal for {signal.instrument} to Slack")
        except Exception as e:
            logger.error(f"[SlackExporter] Failed to export signals: {e}")

    def export_target_positions(self, time: dt_64, targets: List[TargetPosition], account: IAccountViewer) -> None:
        """
        Export target positions to Slack.

        Args:
            time: Timestamp when the target positions were generated
            targets: List of target positions to export
            account: Account viewer to get account information
        """
        if not self._export_targets or not targets or not self._targets_webhook_url:
            return

        try:
            for target in targets:
                # Format the target position using the formatter
                data = self._formatter.format_target_position(time, target, account)

                # Post to Slack
                success = self._post_to_slack(self._targets_webhook_url, data)

                if success:
                    logger.debug(f"[SlackExporter] Exported target position for {target.instrument} to Slack")
                else:
                    logger.warning(f"[SlackExporter] Failed to export target position for {target.instrument} to Slack")
        except Exception as e:
            logger.error(f"[SlackExporter] Failed to export target positions: {e}")

    def export_position_changes(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> None:
        """
        Export position changes to Slack.

        Args:
            time: Timestamp when the position change occurred
            instrument: The instrument for which the position changed
            price: Price at which the position changed
            account: Account viewer to get account information
        """
        if not self._export_position_changes or not self._position_changes_webhook_url:
            return

        try:
            # Format the position change using the formatter
            data = self._formatter.format_position_change(time, instrument, price, account)

            # Post to Slack
            success = self._post_to_slack(self._position_changes_webhook_url, data)

            if success:
                logger.debug(f"[SlackExporter] Exported position change for {instrument} to Slack")
            else:
                logger.warning(f"[SlackExporter] Failed to export position change for {instrument} to Slack")
        except Exception as e:
            logger.error(f"[SlackExporter] Failed to export position change: {e}")
