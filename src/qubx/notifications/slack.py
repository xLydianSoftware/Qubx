"""
Slack notifications for strategy lifecycle events.

This module provides a Slack implementation of IStrategyLifecycleNotifier.
"""

import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from qubx import logger
from qubx.core.interfaces import IStrategyNotifier
from qubx.notifications.throttler import IMessageThrottler, NoThrottling


class SlackNotifier(IStrategyNotifier):
    """
    Notifies about strategy events via Slack.

    This notifier sends messages to a Slack channel when a strategy starts,
    stops, or encounters an error.
    """

    SLACK_API_URL = "https://slack.com/api/chat.postMessage"

    def __init__(
        self,
        strategy_name: str,
        bot_token: str,
        default_channel: str = "#qubx-bots",
        error_channel: str = "#qubx-bots-errors",
        message_channel: str = "#qubx-bots-audit",
        environment: str = "production",
        emoji_start: str = ":rocket:",
        emoji_stop: str = ":checkered_flag:",
        emoji_error: str = ":rotating_light:",
        emoji_message: str = ":information_source:",
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
            emoji_start: Emoji to use for start events
            emoji_stop: Emoji to use for stop events
            emoji_error: Emoji to use for error events
            max_workers: Number of worker threads for posting messages
            throttler: Optional message throttler to prevent flooding
        """
        self._strategy_name = strategy_name
        self._bot_token = bot_token
        self._default_channel = default_channel
        self._error_channel = error_channel
        self._message_channel = message_channel
        self._environment = environment
        self._emoji_start = emoji_start
        self._emoji_stop = emoji_stop
        self._emoji_error = emoji_error
        self._emoji_message = emoji_message
        self._throttler = throttler if throttler is not None else NoThrottling()

        # Add a lock for thread-safe throttling operations
        self._throttler_lock = threading.Lock()

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="slack_notifier")

        logger.info(f"Initialized for environment '{environment}'")

    def notify_start(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has started.

        Args:
            metadata: Optional dictionary with additional information
        """
        try:
            message = f"[{self._strategy_name}] Strategy has started in {self._environment}"
            self._post_to_slack(message, self._emoji_start, "#36a64f", metadata, channel=self._default_channel)
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
            message = f"[{self._strategy_name}] Strategy has stopped in {self._environment}"
            self._post_to_slack(message, self._emoji_stop, "#439FE0", metadata, channel=self._default_channel)
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

            message = f"[{self._strategy_name}] ALERT: Strategy error in {self._environment}"

            # Create a throttle key for this strategy/error type combination
            throttle_key = f"error:{self._strategy_name}:{type(error).__name__}"

            self._post_to_slack(
                message, self._emoji_error, "#FF0000", metadata, throttle_key=throttle_key, channel=self._error_channel
            )
            logger.debug(f"Queued error notification for {self._strategy_name}")
        except Exception as e:
            logger.error(f"Failed to notify error: {e}")

    def notify_message(self, message: str, metadata: dict[str, Any] | None = None, channel: str | None = None) -> None:
        """
        Notify that a strategy has encountered an error.

        Args:
            message: The message to notify
            metadata: Optional dictionary with additional information
        """
        try:
            self._post_to_slack(
                message, self._emoji_message, "#439FE0", metadata, channel=channel or self._message_channel
            )
            logger.debug(f"Queued message notification for {self._strategy_name}")
        except Exception as e:
            logger.error(f"Failed to notify message: {e}")

    def _post_to_slack(
        self,
        message: str,
        emoji: str,
        color: str,
        metadata: dict[str, Any] | None = None,
        throttle_key: str | None = None,
        channel: str | None = None,
    ) -> None:
        """
        Submit a notification to be posted to Slack by the worker thread.

        Args:
            message: Main message text
            emoji: Emoji to use in the message
            color: Color for the message attachment
            metadata: Optional dictionary with additional fields to include
            throttle_key: Optional key for throttling (if None, no throttling is applied)
            channel: Optional channel to send the message to (if None, the default channel is used)
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

            # Submit the task to the executor
            self._executor.submit(self._post_to_slack_impl, message, emoji, color, metadata, throttle_key, channel)
        except Exception as e:
            logger.error(f"Failed to queue Slack message: {e}")

    def _post_to_slack_impl(
        self,
        message: str,
        emoji: str,
        color: str,
        metadata: dict[str, Any] | None = None,
        throttle_key: str | None = None,
        channel: str | None = None,
    ) -> bool:
        """
        Implementation that actually posts to Slack (called from worker thread).

        Args:
            message: Main message text
            emoji: Emoji to use in the message
            color: Color for the message attachment
            metadata: Optional dictionary with additional fields to include
            throttle_key: Optional key used for throttling
            channel: Optional channel to send the message to (if None, the default channel is used)

        Returns:
            bool: True if the post was successful, False otherwise
        """
        try:
            fields = []
            if metadata:
                for key, value in metadata.items():
                    fields.append({"title": key, "value": str(value), "short": len(str(value)) < 50})

            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Main message text that will appear in notifications
            text_message = f"{emoji} {message}"

            data = {
                "text": text_message,
                "channel": channel if channel is not None else self._default_channel,
                "attachments": [
                    {
                        "color": color,
                        "fields": fields,
                        "footer": f"Environment: {self._environment} | Time: {timestamp}",
                    }
                ],
            }

            response = requests.post(
                SlackNotifier.SLACK_API_URL,
                json=data,
                headers={
                    "Authorization": f"Bearer {self._bot_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
            )
            response.raise_for_status()

            logger.debug(f"Successfully posted message: {message}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to post to Slack: {e}")
            return False

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            self._executor.shutdown(wait=False)
        except:
            pass
