"""
Slack Exporter for trading data.

This module provides an implementation of ITradeDataExport that exports trading data to Slack channels.
"""

from typing import List, Optional

from qubx import logger
from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer, ITradeDataExport
from qubx.exporters.formatters import IExportFormatter, SlackMessageFormatter
from qubx.utils.slack import SlackClient


class SlackExporter(ITradeDataExport):
    """
    Exports trading data to Slack channels.

    This exporter can be configured to export signals, target positions, and position changes
    to different Slack channels using the Slack bot token API.
    """

    def __init__(
        self,
        strategy_name: str,
        bot_token: str,
        signals_channel: Optional[str] = None,
        targets_channel: Optional[str] = None,
        position_changes_channel: Optional[str] = None,
        export_signals: bool = True,
        export_targets: bool = True,
        export_position_changes: bool = True,
        formatter: Optional[IExportFormatter] = None,
        strategy_emoji: Optional[str] = None,
        include_account_info: bool = True,
        max_workers: int = 1,
    ):
        """
        Initialize the Slack Exporter.

        Args:
            strategy_name: Name of the strategy (used in message headers)
            bot_token: Slack bot token for authentication
            signals_channel: Channel for signals (e.g., "#signals")
            targets_channel: Channel for target positions (e.g., "#targets")
            position_changes_channel: Channel for position changes (e.g., "#positions")
            export_signals: Whether to export signals
            export_targets: Whether to export target positions
            export_position_changes: Whether to export position changes
            formatter: Formatter to use for formatting data (default: SlackMessageFormatter)
            strategy_emoji: Optional emoji to use for the strategy in messages
            include_account_info: Whether to include account information in messages
            max_workers: Maximum number of worker threads for Slack HTTP requests
        """
        self._strategy_name = strategy_name

        self._signals_channel = signals_channel
        self._targets_channel = targets_channel
        self._position_changes_channel = position_changes_channel

        self._export_signals = export_signals and signals_channel is not None
        self._export_targets = export_targets and targets_channel is not None
        self._export_position_changes = export_position_changes and position_changes_channel is not None

        # Use provided formatter or create a new SlackMessageFormatter
        self._formatter = formatter or SlackMessageFormatter(
            strategy_emoji=strategy_emoji, include_account_info=include_account_info
        )

        # Create Slack client
        self._slack_client = SlackClient(bot_token=bot_token, max_workers=max_workers)

        logger.info(
            f"Initialized for strategy '{strategy_name}' with "
            f"signals: {self._export_signals}, targets: {self._export_targets}, "
            f"position_changes: {self._export_position_changes}"
        )

    def export_signals(self, time: dt_64, signals: List[Signal], account: IAccountViewer) -> None:
        """
        Export signals to Slack.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to export
            account: Account viewer to get account information
        """
        if not self._export_signals or not signals or not self._signals_channel:
            return

        try:
            for signal in signals:
                # Format the signal using the formatter
                data = self._formatter.format_signal(time, signal, account)

                # Extract blocks and create fallback message
                blocks = data.get("blocks", [])
                message = f"Signal: {signal.instrument.symbol}"

                # Post to Slack in background thread
                self._slack_client.notify_message_async(
                    message=message,
                    channel=self._signals_channel,
                    blocks=blocks,
                )
                logger.debug(f"Queued signal for {signal.instrument} to be exported to Slack")
        except Exception as e:
            logger.error(f"Failed to export signals: {e}")

    def export_target_positions(self, time: dt_64, targets: List[TargetPosition], account: IAccountViewer) -> None:
        """
        Export target positions to Slack.

        Args:
            time: Timestamp when the target positions were generated
            targets: List of target positions to export
            account: Account viewer to get account information
        """
        if not self._export_targets or not targets or not self._targets_channel:
            return

        try:
            for target in targets:
                # Format the target position using the formatter
                data = self._formatter.format_target_position(time, target, account)

                # Extract blocks and create fallback message
                blocks = data.get("blocks", [])
                message = f"Target Position: {target.instrument.symbol}"

                # Post to Slack in background thread
                self._slack_client.notify_message_async(
                    message=message,
                    channel=self._targets_channel,
                    blocks=blocks,
                )
                logger.debug(f"Queued target position for {target.instrument} to be exported to Slack")
        except Exception as e:
            logger.error(f"Failed to export target positions: {e}")

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
        if not self._export_position_changes or not self._position_changes_channel:
            return

        try:
            # Format the position change using the formatter
            data = self._formatter.format_position_change(time, instrument, price, account)

            # Extract blocks and create fallback message
            blocks = data.get("blocks", [])
            message = f"Position Change: {instrument.symbol}"

            # Post to Slack in background thread
            self._slack_client.notify_message_async(
                message=message,
                channel=self._position_changes_channel,
                blocks=blocks,
            )
            logger.debug(f"Queued position change for {instrument} to be exported to Slack")
        except Exception as e:
            logger.error(f"Failed to export position change: {e}")
