"""
Slack message formatter for trading data.

This module provides a formatter for converting trading data into Slack message blocks
suitable for posting to Slack channels.
"""

import json
from typing import Any, Dict, List

from qubx.core.basics import Instrument, Signal, TargetPosition, dt_64
from qubx.core.interfaces import IAccountViewer
from qubx.exporters.formatters.base import DefaultFormatter, IExportFormatter


class SlackMessageFormatter(IExportFormatter):
    """
    Formatter for converting trading data into Slack message blocks.

    This formatter creates structured Slack message blocks for signals, target positions,
    and position changes, suitable for posting to Slack channels.
    """

    def __init__(self, strategy_emoji: str = ":chart_with_upwards_trend:", include_account_info: bool = True):
        """
        Initialize the Slack message formatter.

        Args:
            strategy_emoji: Emoji to use for the strategy in messages
            include_account_info: Whether to include account information in messages
        """
        self._strategy_emoji = strategy_emoji
        self._include_account_info = include_account_info
        self._default_formatter = DefaultFormatter()  # For basic data formatting

    def _format_timestamp(self, timestamp: Any) -> str:
        """Format timestamp for display in Slack messages."""
        return self._default_formatter._format_timestamp(timestamp)

    def _create_header_block(self, title: str, timestamp: str) -> Dict[str, Any]:
        """Create a header block for Slack messages."""
        return {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{self._strategy_emoji} {title} - {timestamp}", "emoji": True},
        }

    def _create_section_block(self, text: str) -> Dict[str, Any]:
        """Create a section block for Slack messages."""
        return {"type": "section", "text": {"type": "mrkdwn", "text": text}}

    def _create_divider_block(self) -> Dict[str, Any]:
        """Create a divider block for Slack messages."""
        return {"type": "divider"}

    def _format_account_info(self, account: IAccountViewer) -> str:
        """Format account information for inclusion in Slack messages."""
        if not self._include_account_info:
            return ""

        total_capital = account.get_total_capital()
        return f"*Account Info:*\nTotal Capital: {total_capital:.2f}"

    def format_signal(self, time: dt_64, signal: Signal, account: IAccountViewer) -> dict[str, Any]:
        """
        Format a signal as a Slack message.

        Args:
            time: Timestamp when the signal was generated
            signal: The signal to format
            account: Account viewer to get account information

        Returns:
            A dictionary containing the formatted Slack message blocks
        """
        timestamp = self._format_timestamp(time)
        direction = "🔴 SELL" if signal.signal < 0 else "🟢 BUY" if signal.signal > 0 else "⚪ NEUTRAL"
        strength = abs(signal.signal)

        # Basic signal data
        signal_text = (
            f"*Signal:* {direction} (Strength: {strength:.2f})\n"
            f"*Instrument:* {signal.instrument.symbol} ({signal.instrument.exchange})\n"
        )

        # Add reference price if available
        if signal.reference_price is not None:
            signal_text += f"*Reference Price:* {signal.reference_price}\n"

        # Add group if available
        if signal.group:
            signal_text += f"*Group:* {signal.group}\n"

        # Add metadata if available
        if signal.options:
            signal_text += f"*Metadata:* {json.dumps(signal.options)}\n"

        if signal.comment:
            signal_text += f"*Comment:* {signal.comment}\n"

        # Create blocks
        blocks = [self._create_header_block("New Signal", timestamp), self._create_section_block(signal_text)]

        # Add account info if enabled
        if self._include_account_info:
            account_text = self._format_account_info(account)
            if account_text:
                blocks.append(self._create_divider_block())
                blocks.append(self._create_section_block(account_text))

        return {"blocks": blocks}

    def format_target_position(self, time: dt_64, target: TargetPosition, account: IAccountViewer) -> dict[str, Any]:
        """
        Format a target position as a Slack message.

        Args:
            time: Timestamp when the target position was generated
            target: The target position to format
            account: Account viewer to get account information

        Returns:
            A dictionary containing the formatted Slack message blocks
        """
        timestamp = self._format_timestamp(time)

        # Basic target position data
        target_text = (
            f"*Instrument:* {target.instrument.symbol} ({target.instrument.exchange})\n"
            f"*Target Size:* {target.target_position_size}\n"
        )

        # Add price if available
        if target.price is not None:
            target_text += f"*Price:* {target.price}\n"

        # Create blocks
        blocks = [self._create_header_block("Target Position", timestamp), self._create_section_block(target_text)]

        # Add account info if enabled
        if self._include_account_info:
            account_text = self._format_account_info(account)
            if account_text:
                blocks.append(self._create_divider_block())
                blocks.append(self._create_section_block(account_text))

        return {"blocks": blocks}

    def format_position_change(
        self, time: dt_64, instrument: Instrument, price: float, account: IAccountViewer
    ) -> dict[str, Any]:
        """
        Format a position change as a Slack message.

        Args:
            time: Timestamp when the position change occurred
            instrument: The instrument for which the position changed
            price: Price at which the position changed
            account: Account viewer to get account information

        Returns:
            A dictionary containing the formatted Slack message blocks
        """
        timestamp = self._format_timestamp(time)
        position = account.get_position(instrument)

        # Basic position change data
        position_text = (
            f"*Instrument:* {instrument.symbol} ({instrument.exchange})\n"
            f"*Price:* {price}\n"
            f"*Current Quantity:* {position.quantity}\n"
        )

        # Create blocks
        blocks = [self._create_header_block("Position Change", timestamp), self._create_section_block(position_text)]

        # Add account info if enabled
        if self._include_account_info:
            account_text = self._format_account_info(account)
            if account_text:
                blocks.append(self._create_divider_block())
                blocks.append(self._create_section_block(account_text))

        return {"blocks": blocks}
