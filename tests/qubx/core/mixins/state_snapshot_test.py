"""The state snapshot records when each data type last arrived.

A fresh snapshot timestamp only proves the thread writing it is running. A bot whose thread is alive
while its market-data feed is dead writes fresh snapshots the whole time — these are what make that
visible. Timestamps rather than ages, so the reader measures freshness when it reads.
"""

from unittest.mock import MagicMock

import numpy as np

from qubx.core.mixins.processing import ProcessingManager


class TestSnapshotLastEventTimes:
    def _manager(self, last_event_times):
        m = ProcessingManager.__new__(ProcessingManager)
        m._health_monitor = MagicMock()
        m._health_monitor.get_last_event_times_by_exchange.return_value = last_event_times
        return m

    def test_records_when_each_type_last_arrived(self):
        ob = np.datetime64("2026-04-05T09:59:58.500", "ns")
        tr = np.datetime64("2026-04-05T09:59:18", "ns")
        m = self._manager({"orderbook": ob, "trade": tr})

        assert m._last_event_times("BINANCE.UM") == {"orderbook": str(ob), "trade": str(tr)}

    def test_a_never_seen_type_is_none(self):
        """A warming-up bot has no events yet; a sentinel age would page on every start."""
        m = self._manager({"orderbook": None})

        assert m._last_event_times("BINANCE.UM") == {"orderbook": None}

    def test_timestamps_match_the_snapshot_timestamp_format(self):
        """The reader parses these with the same parser it uses for the snapshot's own timestamp."""
        t = np.datetime64("2026-04-05T10:00:00.123456789", "ns")
        m = self._manager({"orderbook": t})

        got = m._last_event_times("BINANCE.UM")["orderbook"]
        assert got == "2026-04-05T10:00:00.123456789"

    def test_is_json_serialisable(self):
        import json

        t = np.datetime64("2026-04-05T10:00:00", "ns")
        m = self._manager({"orderbook": t, "trade": None})

        assert json.loads(json.dumps(m._last_event_times("BINANCE.UM"))) == {"orderbook": str(t), "trade": None}
