"""
Message Throttling for Notifications.

This module defines interfaces and implementations for throttling
notification messages to prevent flooding notification channels.

Usage Examples:
    1. Basic TimeWindowThrottler with default settings (allows 1 message per key per 10 seconds):
       ```python
       from qubx.notifications.throttler import TimeWindowThrottler

       throttler = TimeWindowThrottler()
       if throttler.should_send("error:mystrategy:ValueError"):
           # Send the message
           send_message()
           # Update the throttler
           throttler.register_sent("error:mystrategy:ValueError")
       ```

    2. CountBasedThrottler (allows up to N messages per key within a time window):
       ```python
       from qubx.notifications.throttler import CountBasedThrottler

       # Allow up to 5 messages per minute for each key
       throttler = CountBasedThrottler(max_count=5, window_seconds=60.0)
       ```

    3. In a configuration file for SlackLifecycleNotifier:
       ```yaml
       notifiers:
         - notifier: SlackLifecycleNotifier
           parameters:
             webhook_url: ${SLACK_WEBHOOK_URL}
             environment: production
             throttle:
               type: TimeWindow
               window_seconds: 30.0
       ```
"""

import time
from abc import ABC, abstractmethod


class IMessageThrottler(ABC):
    """Interface for message throttlers that can limit the frequency of notifications."""

    @abstractmethod
    def should_send(self, key: str) -> bool:
        """
        Determine if a message with the given key should be sent based on throttling rules.

        Args:
            key: A unique identifier for the type of message being sent
                (e.g., "error:{strategy_name}")

        Returns:
            bool: True if the message should be sent, False if it should be throttled
        """
        pass

    @abstractmethod
    def register_sent(self, key: str) -> None:
        """
        Register that a message with the given key was sent.
        This updates the internal state of the throttler.

        Args:
            key: A unique identifier for the type of message that was sent
        """
        pass


class TimeWindowThrottler(IMessageThrottler):
    """
    Throttles messages based on a time window.

    Only allows one message per key within a specified time window.
    """

    def __init__(self, window_seconds: float = 10.0):
        """
        Initialize the time window throttler.

        Args:
            window_seconds: Minimum time between messages with the same key, in seconds
        """
        self._window_seconds = window_seconds
        self._last_sent_times: dict[str, float] = {}

    def should_send(self, key: str) -> bool:
        """
        Check if a message with the given key should be sent based on the time window.

        Args:
            key: Message key to check

        Returns:
            bool: True if enough time has passed since the last message with this key
        """
        current_time = time.time()
        last_sent = self._last_sent_times.get(key, 0)
        return (current_time - last_sent) >= self._window_seconds

    def register_sent(self, key: str) -> None:
        """
        Register that a message with the given key was sent.

        Args:
            key: Key of the message that was sent
        """
        self._last_sent_times[key] = time.time()


class CountBasedThrottler(IMessageThrottler):
    """
    Throttles messages based on a count within a time window.

    Allows a specified number of messages per key within a time window.
    """

    def __init__(self, max_count: int = 3, window_seconds: float = 60.0):
        """
        Initialize the count-based throttler.

        Args:
            max_count: Maximum number of messages allowed in the time window
            window_seconds: Time window in seconds
        """
        self._max_count = max_count
        self._window_seconds = window_seconds
        self._message_history: dict[str, list[float]] = {}

    def should_send(self, key: str) -> bool:
        """
        Check if a message with the given key should be sent based on the count limit.

        Args:
            key: Message key to check

        Returns:
            bool: True if the message count is below the limit
        """
        current_time = time.time()

        # Initialize history for this key if it doesn't exist
        if key not in self._message_history:
            self._message_history[key] = []

        # Remove timestamps older than the window
        self._message_history[key] = [
            ts for ts in self._message_history[key] if (current_time - ts) < self._window_seconds
        ]

        # Check if we're under the message count limit
        return len(self._message_history[key]) < self._max_count

    def register_sent(self, key: str) -> None:
        """
        Register that a message with the given key was sent.

        Args:
            key: Key of the message that was sent
        """
        current_time = time.time()

        if key not in self._message_history:
            self._message_history[key] = []

        self._message_history[key].append(current_time)


class NoThrottling(IMessageThrottler):
    """A throttler implementation that doesn't actually throttle - allows all messages."""

    def should_send(self, key: str) -> bool:
        """Always returns True, allowing all messages to be sent."""
        return True

    def register_sent(self, key: str) -> None:
        """No-op implementation."""
        pass
