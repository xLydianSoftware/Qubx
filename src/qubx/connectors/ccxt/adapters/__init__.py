"""
CCXT polling adapters for converting fetch_* methods to watch_* behavior.
"""

from .polling_adapter import PollingToWebSocketAdapter

__all__ = ["PollingToWebSocketAdapter"]