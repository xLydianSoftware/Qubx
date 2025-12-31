"""
Composite Strategy Notifier.

This module provides a composite implementation of IStrategyNotifier that delegates to multiple notifiers.
"""

from typing import Any, Dict, List, Optional

from qubx import logger
from qubx.core.interfaces import IStrategyNotifier


class CompositeNotifier(IStrategyNotifier):
    """
    Composite notifier that delegates to multiple notifiers.

    This notifier can be used to send notifications to multiple destinations
    by combining multiple notifiers into one.
    """

    def __init__(self, notifiers: List[IStrategyNotifier]):
        """
        Initialize the Composite Notifier.

        Args:
            notifiers: List of notifiers to delegate to
        """
        self._notifiers = notifiers

    def notify_start(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Notify that a strategy has started to all configured notifiers.

        Args:
            metadata: Optional dictionary with additional information
        """
        for notifier in self._notifiers:
            try:
                notifier.notify_start(metadata)
            except Exception as e:
                logger.error(f"Error notifying start to {notifier.__class__.__name__}: {e}")

    def notify_stop(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Notify that a strategy has stopped to all configured notifiers.

        Args:
            metadata: Optional dictionary with additional information
        """
        for notifier in self._notifiers:
            try:
                notifier.notify_stop(metadata)
            except Exception as e:
                logger.error(f"Error notifying stop to {notifier.__class__.__name__}: {e}")

    def notify_error(self, error: Exception, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Notify that a strategy has encountered an error to all configured notifiers.

        Args:
            error: The exception that was raised
            metadata: Optional dictionary with additional information
        """
        for notifier in self._notifiers:
            try:
                notifier.notify_error(error, metadata)
            except Exception as e:
                logger.error(f"Error notifying error to {notifier.__class__.__name__}: {e}")

    def notify_message(self, message: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Notify that a strategy has a message to all configured notifiers.

        Args:
            message: The message to notify
            metadata: Optional dictionary with additional information
            **kwargs: Additional keyword arguments
        """
        for notifier in self._notifiers:
            try:
                notifier.notify_message(message, metadata, **kwargs)
            except Exception as e:
                logger.error(f"Error notifying message to {notifier.__class__.__name__}: {e}")
