"""Base handler interface for Lighter WebSocket messages"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from qubx import logger

# Generic type for handler output
T = TypeVar("T")


class BaseHandler(ABC, Generic[T]):
    """
    Base class for Lighter message handlers.

    Handlers transform raw Lighter WebSocket messages into Qubx data types.
    Each handler is responsible for one message type (orderbook, trades, etc.)
    """

    def __init__(self):
        """Initialize handler"""
        self._message_count = 0
        self._error_count = 0

    def handle(self, message: dict[str, Any]) -> T | list[T] | None:
        """
        Process a Lighter WebSocket message and convert to Qubx format.

        This method must be overridden by subclasses. The base implementation
        tracks statistics.

        Args:
            message: Raw message dict from Lighter WebSocket

        Returns:
            Converted Qubx object(s), or None if message should be skipped

        Raises:
            ValueError: If message format is invalid
        """
        self._message_count += 1
        return self._handle_impl(message)

    @abstractmethod
    def _handle_impl(self, message: dict[str, Any]) -> T | list[T] | None:
        """
        Implementation of message handling logic.

        Must be overridden by subclasses.

        Args:
            message: Raw message dict from Lighter WebSocket

        Returns:
            Converted Qubx object(s), or None if message should be skipped

        Raises:
            ValueError: If message format is invalid
        """
        ...

    @abstractmethod
    def can_handle(self, message: dict[str, Any]) -> bool:
        """
        Check if this handler can process the given message.

        Args:
            message: Raw message dict from Lighter WebSocket

        Returns:
            True if this handler should process the message
        """
        ...

    def handle_safe(self, message: dict[str, Any]) -> T | list[T] | None:
        """
        Safe wrapper around handle() that catches and logs exceptions.

        Args:
            message: Raw message dict from Lighter WebSocket

        Returns:
            Converted Qubx object(s), or None if error occurred
        """
        try:
            return self.handle(message)
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in {self.__class__.__name__}.handle(): {e}")
            logger.debug(f"Problematic message: {message}")
            return None

    @property
    def stats(self) -> dict[str, int]:
        """Get handler statistics"""
        return {
            "messages_processed": self._message_count,
            "errors": self._error_count,
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self._message_count = 0
        self._error_count = 0

    def reset(self):
        """
        Reset handler internal state.

        This method is called when the WebSocket connection is reestablished
        to ensure handlers start with clean state. Handlers with stateful
        components should override this method to reset their state.

        The default implementation does nothing (stateless handlers).
        """
        pass
