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
from qubx.core.interfaces import IStrategyLifecycleNotifier
from qubx.notifications.throttler import IMessageThrottler, NoThrottling


class SlackLifecycleNotifier(IStrategyLifecycleNotifier):
    """
    Notifies about strategy lifecycle events via Slack.

    This notifier sends messages to a Slack channel when a strategy starts,
    stops, or encounters an error.
    """

    def __init__(
        self,
        webhook_url: str,
        environment: str = "production",
        emoji_start: str = ":rocket:",
        emoji_stop: str = ":checkered_flag:",
        emoji_error: str = ":rotating_light:",
        max_workers: int = 1,
        throttler: IMessageThrottler | None = None,
    ):
        """
        Initialize the Slack Lifecycle Notifier.

        Args:
            webhook_url: Slack webhook URL to send notifications to
            environment: Environment name (e.g., production, staging)
            emoji_start: Emoji to use for start events
            emoji_stop: Emoji to use for stop events
            emoji_error: Emoji to use for error events
            max_workers: Number of worker threads for posting messages
            throttler: Optional message throttler to prevent flooding
        """
        self._webhook_url = webhook_url
        self._environment = environment
        self._emoji_start = emoji_start
        self._emoji_stop = emoji_stop
        self._emoji_error = emoji_error
        self._throttler = throttler if throttler is not None else NoThrottling()

        # Add a lock for thread-safe throttling operations
        self._throttler_lock = threading.Lock()

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="slack_notifier")

        logger.info(f"[SlackLifecycleNotifier] Initialized for environment '{environment}'")

    def _post_to_slack(
        self,
        message: str,
        emoji: str,
        color: str,
        metadata: dict[str, Any] | None = None,
        throttle_key: str | None = None,
    ) -> None:
        """
        Submit a notification to be posted to Slack by the worker thread.

        Args:
            message: Main message text
            emoji: Emoji to use in the message
            color: Color for the message attachment
            metadata: Optional dictionary with additional fields to include
            throttle_key: Optional key for throttling (if None, no throttling is applied)
        """
        try:
            # Thread-safe throttling check and registration
            if throttle_key is not None:
                with self._throttler_lock:
                    if not self._throttler.should_send(throttle_key):
                        logger.debug(f"[SlackLifecycleNotifier] Throttled message with key '{throttle_key}': {message}")
                        return
                    # Immediately register that we're about to send this message
                    # This prevents race conditions where multiple threads check should_send
                    # before any of them call register_sent
                    self._throttler.register_sent(throttle_key)

            # Submit the task to the executor
            self._executor.submit(self._post_to_slack_impl, message, emoji, color, metadata, throttle_key)
        except Exception as e:
            logger.error(f"[SlackLifecycleNotifier] Failed to queue Slack message: {e}")

    def _post_to_slack_impl(
        self,
        message: str,
        emoji: str,
        color: str,
        metadata: dict[str, Any] | None = None,
        throttle_key: str | None = None,
    ) -> bool:
        """
        Implementation that actually posts to Slack (called from worker thread).

        Args:
            message: Main message text
            emoji: Emoji to use in the message
            color: Color for the message attachment
            metadata: Optional dictionary with additional fields to include
            throttle_key: Optional key used for throttling

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
                "attachments": [
                    {
                        "color": color,
                        "fields": fields,
                        "footer": f"Environment: {self._environment} | Time: {timestamp}",
                    }
                ],
            }

            response = requests.post(self._webhook_url, json=data)
            response.raise_for_status()

            logger.debug(f"[SlackLifecycleNotifier] Successfully posted message: {message}")
            return True
        except requests.RequestException as e:
            logger.error(f"[SlackLifecycleNotifier] Failed to post to Slack: {e}")
            return False

    def notify_start(self, strategy_name: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has started.

        Args:
            strategy_name: Name of the strategy that started
            metadata: Optional dictionary with additional information
        """
        try:
            message = f"[{strategy_name}] Strategy has started in {self._environment}"
            self._post_to_slack(message, self._emoji_start, "#36a64f", metadata)
            logger.debug(f"[SlackLifecycleNotifier] Queued start notification for {strategy_name}")
        except Exception as e:
            logger.error(f"[SlackLifecycleNotifier] Failed to notify start: {e}")

    def notify_stop(self, strategy_name: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has stopped.

        Args:
            strategy_name: Name of the strategy that stopped
            metadata: Optional dictionary with additional information
        """
        try:
            message = f"[{strategy_name}] Strategy has stopped in {self._environment}"
            self._post_to_slack(message, self._emoji_stop, "#439FE0", metadata)
            logger.debug(f"[SlackLifecycleNotifier] Queued stop notification for {strategy_name}")
        except Exception as e:
            logger.error(f"[SlackLifecycleNotifier] Failed to notify stop: {e}")

    def notify_error(self, strategy_name: str, error: Exception, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has encountered an error.

        Args:
            strategy_name: Name of the strategy that encountered an error
            error: The exception that was raised
            metadata: Optional dictionary with additional information
        """
        try:
            if metadata is None:
                metadata = {}

            # Add error details to metadata
            metadata["Error Type"] = type(error).__name__
            metadata["Error Message"] = str(error)

            message = f"[{strategy_name}] ALERT: Strategy error in {self._environment}"

            # Create a throttle key for this strategy/error type combination
            throttle_key = f"error:{strategy_name}:{type(error).__name__}"

            self._post_to_slack(message, self._emoji_error, "#FF0000", metadata, throttle_key=throttle_key)
            logger.debug(f"[SlackLifecycleNotifier] Queued error notification for {strategy_name}")
        except Exception as e:
            logger.error(f"[SlackLifecycleNotifier] Failed to notify error: {e}")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            self._executor.shutdown(wait=False)
        except:
            pass
