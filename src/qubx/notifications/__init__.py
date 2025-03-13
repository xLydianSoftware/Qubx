"""
Notifications package for strategy lifecycle events.

This package provides implementations of the IStrategyLifecycleNotifier interface
for various notification channels.
"""

from .composite import CompositeLifecycleNotifier
from .slack import SlackLifecycleNotifier

__all__ = ["CompositeLifecycleNotifier", "SlackLifecycleNotifier"]
