"""
Composite Strategy Lifecycle Notifier.

This module provides a composite implementation of IStrategyLifecycleNotifier that delegates to multiple notifiers.
"""

from typing import Dict, List, Optional

from qubx import logger
from qubx.core.interfaces import IStrategyLifecycleNotifier


class CompositeLifecycleNotifier(IStrategyLifecycleNotifier):
    """
    Composite lifecycle notifier that delegates to multiple notifiers.

    This notifier can be used to send notifications to multiple destinations
    by combining multiple notifiers into one.
    """

    def __init__(self, notifiers: List[IStrategyLifecycleNotifier]):
        """
        Initialize the Composite Lifecycle Notifier.

        Args:
            notifiers: List of notifiers to delegate to
        """
        self._notifiers = notifiers

    def notify_start(self, strategy_name: str, metadata: Optional[Dict[str, any]] = None) -> None:
        """
        Notify that a strategy has started to all configured notifiers.

        Args:
            strategy_name: Name of the strategy that started
            metadata: Optional dictionary with additional information
        """
        for notifier in self._notifiers:
            try:
                notifier.notify_start(strategy_name, metadata)
            except Exception as e:
                logger.error(f"Error notifying start to {notifier.__class__.__name__}: {e}")

    def notify_stop(self, strategy_name: str, metadata: Optional[Dict[str, any]] = None) -> None:
        """
        Notify that a strategy has stopped to all configured notifiers.

        Args:
            strategy_name: Name of the strategy that stopped
            metadata: Optional dictionary with additional information
        """
        for notifier in self._notifiers:
            try:
                notifier.notify_stop(strategy_name, metadata)
            except Exception as e:
                logger.error(f"Error notifying stop to {notifier.__class__.__name__}: {e}")

    def notify_error(self, strategy_name: str, error: Exception, metadata: Optional[Dict[str, any]] = None) -> None:
        """
        Notify that a strategy has encountered an error to all configured notifiers.

        Args:
            strategy_name: Name of the strategy that encountered an error
            error: The exception that was raised
            metadata: Optional dictionary with additional information
        """
        for notifier in self._notifiers:
            try:
                notifier.notify_error(strategy_name, error, metadata)
            except Exception as e:
                logger.error(f"Error notifying error to {notifier.__class__.__name__}: {e}")
