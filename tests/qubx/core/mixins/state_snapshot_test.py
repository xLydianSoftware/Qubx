"""The state snapshot carries per-exchange data ages.

A fresh snapshot only proves the thread writing it is running. A bot whose thread is alive while its
market-data feed is dead writes fresh snapshots the whole time — the ages are what make that visible.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.mixins.processing import ProcessingManager


class TestSnapshotDataAges:
    def _manager(self, now, last_event_times):
        m = ProcessingManager.__new__(ProcessingManager)
        m._time_provider = MagicMock()
        m._time_provider.time.return_value = now
        m._health_monitor = MagicMock()
        m._health_monitor.get_last_event_times_by_exchange.return_value = last_event_times
        return m

    def test_ages_are_seconds_since_the_last_event(self):
        now = np.datetime64("2026-04-05T10:00:00", "ns")
        m = self._manager(now, {"orderbook": now - np.timedelta64(1500, "ms"), "trade": now - np.timedelta64(42, "s")})

        assert m._last_event_ages("BINANCE.UM") == {"orderbook": 1.5, "trade": 42.0}

    def test_a_never_seen_event_type_is_none_not_a_huge_age(self):
        """A warming-up bot has no events yet; reporting that as infinitely stale would page on
        every start."""
        now = np.datetime64("2026-04-05T10:00:00", "ns")
        m = self._manager(now, {"orderbook": None})

        assert m._last_event_ages("BINANCE.UM") == {"orderbook": None}

    def test_a_wedged_feed_shows_a_growing_age(self):
        """The 2026-08-18 shape: the writing thread is alive, so the snapshot is fresh, but no data
        has arrived for minutes."""
        now = np.datetime64("2026-04-05T10:00:00", "ns")
        m = self._manager(now, {"orderbook": now - np.timedelta64(11, "m")})

        assert m._last_event_ages("BINANCE.UM") == {"orderbook": 660.0}

    def test_ages_are_json_serialisable(self):
        """The snapshot is json.dumps'd by the persistence layer; a numpy float would raise."""
        import json

        now = np.datetime64("2026-04-05T10:00:00", "ns")
        m = self._manager(now, {"orderbook": now - np.timedelta64(3, "s"), "trade": None})

        assert json.loads(json.dumps(m._last_event_ages("BINANCE.UM"))) == {"orderbook": 3.0, "trade": None}
