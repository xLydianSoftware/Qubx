"""
Slack notifications for strategy lifecycle events.

This module provides a Slack implementation of IStrategyLifecycleNotifier.
"""

import threading
from typing import Any

from qubx import logger
from qubx.core.interfaces import IStrategyNotifier
from qubx.notifications.throttler import IMessageThrottler, NoThrottling
from qubx.utils.slack import SlackClient


class SlackNotifier(IStrategyNotifier):
    """
    Notifies about strategy events via Slack.

    This notifier sends messages to a Slack channel when a strategy starts,
    stops, or encounters an error.
    """

    def __init__(
        self,
        strategy_name: str,
        bot_token: str,
        default_channel: str = "#qubx-bots",
        error_channel: str = "#qubx-bots-errors",
        message_channel: str = "#qubx-bots-audit",
        environment: str = "research",
        emoji_start: str | None = ":rocket:",
        emoji_stop: str | None = ":checkered_flag:",
        emoji_error: str | None = ":rotating_light:",
        max_workers: int = 1,
        throttler: IMessageThrottler | None = None,
    ):
        """
        Initialize the Slack Notifier.

        Args:
            strategy_name: Name of the strategy
            bot_token: Slack bot token
            default_channel: Default channel to send notifications to
            error_channel: Channel to send error notifications to
            message_channel: Channel to send message notifications to
            environment: Environment name (e.g., production, staging)
            emoji_start: Optional emoji to use for start events
            emoji_stop: Optional emoji to use for stop events
            emoji_error: Optional emoji to use for error events
            emoji_message: Optional emoji to use for message events
            max_workers: Number of worker threads for posting messages
            throttler: Optional message throttler to prevent flooding
        """
        self._strategy_name = strategy_name
        self._default_channel = default_channel
        self._error_channel = error_channel
        self._message_channel = message_channel
        self._environment = environment
        self._emoji_start = emoji_start
        self._emoji_stop = emoji_stop
        self._emoji_error = emoji_error
        self._throttler = throttler if throttler is not None else NoThrottling()

        # Add a lock for thread-safe throttling operations
        self._throttler_lock = threading.Lock()

        # Create Slack client
        self._slack_client = SlackClient(bot_token=bot_token, max_workers=max_workers, environment=environment)

        logger.info(f"Initialized for environment '{environment}'")

    def notify_start(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has started.

        Args:
            metadata: Optional dictionary with additional information
        """
        try:
            message = f"{self._emoji_start} [{self._strategy_name}] Strategy has started in {self._environment}"
            self._post_to_slack(message, metadata=metadata, channel=self._default_channel)
            logger.debug(f"Queued start notification for {self._strategy_name}")
        except Exception as e:
            logger.error(f"Failed to notify start: {e}")

    def notify_stop(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has stopped.

        Args:
            metadata: Optional dictionary with additional information
        """
        try:
            message = f"{self._emoji_stop} [{self._strategy_name}] Strategy has stopped in {self._environment}"
            self._post_to_slack(message, metadata=metadata, channel=self._default_channel)
            logger.debug(f"Queued stop notification for {self._strategy_name}")
        except Exception as e:
            logger.error(f"Failed to notify stop: {e}")

    def notify_error(self, error: Exception, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has encountered an error.

        Args:
            error: The exception that was raised
            metadata: Optional dictionary with additional information
        """
        try:
            if metadata is None:
                metadata = {}

            # Add error details to metadata
            metadata["Error Type"] = type(error).__name__
            metadata["Error Message"] = str(error)

            message = f"{self._emoji_error} [{self._strategy_name}] ALERT: Strategy error in {self._environment}"

            # Create a throttle key for this strategy/error type combination
            throttle_key = f"error:{self._strategy_name}:{type(error).__name__}"

            self._post_to_slack(message, metadata=metadata, throttle_key=throttle_key, channel=self._error_channel)
            logger.debug(f"Queued error notification for {self._strategy_name}")
        except Exception as e:
            logger.error(f"Failed to notify error: {e}")

    def notify_message(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
        channel: str | None = None,
        blocks: list[dict] | None = None,
        key: str | None = None,
    ) -> None:
        """
        Notify that a strategy has encountered an error.

        Args:
            message: The message to notify
            metadata: Optional dictionary with additional information
        """
        try:
            self._post_to_slack(
                message, metadata=metadata, channel=channel or self._message_channel, blocks=blocks, key=key
            )
        except Exception as e:
            logger.error(f"Failed to notify message: {e}")

    def _post_to_slack(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
        throttle_key: str | None = None,
        channel: str | None = None,
        blocks: list[dict] | None = None,
        key: str | None = None,
    ) -> None:
        """
        Submit a notification to be posted to Slack by the worker thread.

        Args:
            message: Main message text
            metadata: Optional dictionary with additional fields to include
            throttle_key: Optional key for throttling (if None, no throttling is applied)
            channel: Optional channel to send the message to (if None, the default channel is used)
            blocks: Optional list of blocks to send
        """
        try:
            # Thread-safe throttling check and registration
            if throttle_key is not None:
                with self._throttler_lock:
                    if not self._throttler.should_send(throttle_key):
                        logger.debug(f"Throttled message with key '{throttle_key}': {message}")
                        return
                    # Immediately register that we're about to send this message
                    # This prevents race conditions where multiple threads check should_send
                    # before any of them call register_sent
                    self._throttler.register_sent(throttle_key)

            # Post to Slack using the client
            self._slack_client.notify_message_async(
                message=message,
                channel=channel or self._default_channel,
                blocks=blocks,
                metadata=metadata,
                key=key,
            )
        except Exception as e:
            logger.error(f"Failed to queue Slack message: {e}")
