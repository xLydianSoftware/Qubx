from types import SimpleNamespace

import pandas as pd
import pytest

from qubx.backtester.simulated_data import DataPump
from qubx.core.basics import DataType
from qubx.data.containers import RawData, RawMultiData
from qubx.data.storage import IReader

T0 = pd.Timestamp("2025-03-30")
SYMBOLS = ["AAAUSDT", "BBBUSDT"]
HOURS = 12
CHUNK_HOURS = 3


def _rows(symbol: str, hours: range) -> RawData:
    df = pd.DataFrame(
        {
            "timestamp": [T0 + pd.Timedelta(hours=h) for h in hours],
            "funding_rate": 0.0001,
            "funding_interval_hours": 8.0,
        }
    ).set_index("timestamp")
    return RawData.from_pandas(symbol, DataType.FUNDING_PAYMENT, df)


class FlakyReader(IReader):
    """
    One funding row per symbol per hour, served in chunks of CHUNK_HOURS. The first `fail_opens` reads
    fail: with `on_open` when read() is called, otherwise after `fail_after` chunks.
    """

    def __init__(self, fail_opens: int = 1, fail_after: int = 1, on_open: bool = False):
        self.fail_opens, self.fail_after, self.on_open = fail_opens, fail_after, on_open
        self.opens = 0

    def read(self, data_id, dtype, start=None, stop=None, chunksize=0, **kwargs):
        self.opens += 1
        fail = self.opens <= self.fail_opens
        if fail and self.on_open:
            raise OSError("Could not resolve host")
        return self._chunks(list(data_id), pd.Timestamp(start), pd.Timestamp(stop), fail)

    def _chunks(self, symbols, start, stop, fail):
        first = max(0, int((start - T0) / pd.Timedelta("1h")))
        last = min(HOURS, int((stop - T0) / pd.Timedelta("1h")))
        for i, h0 in enumerate(range(first, last, CHUNK_HOURS)):
            if fail and i == self.fail_after:
                raise OSError("Could not resolve host")
            yield RawMultiData([_rows(s, range(h0, min(h0 + CHUNK_HOURS, last))) for s in symbols])


def _pump(reader: IReader, attempts: int = 4) -> DataPump:
    pump = DataPump(reader, "funding_payment", "BINANCE.UM", "SWAP", read_attempts=attempts, retry_wait=0.0)
    for s in SYMBOLS:
        pump.attach_instrument(SimpleNamespace(symbol=s))
    return pump


def _drain(readers: dict) -> dict[str, list[int]]:
    """
    Timestamps each stream delivers until it ends.
    """
    got = {}
    for key, reader in readers.items():
        times = []
        try:
            while True:
                times += [r.time for r in next(reader)]
        except StopIteration:
            pass
        got[key.split(":")[-1]] = times
    return got


def _hours(times: list[int]) -> list[int]:
    return [int((pd.Timestamp(t) - T0) / pd.Timedelta("1h")) for t in times]


def test_read_error_mid_stream_is_reopened_and_every_row_arrives_once():
    reader = FlakyReader(fail_after=1)
    got = _drain(_pump(reader).start_read(T0, T0 + pd.Timedelta(hours=HOURS)))
    assert {s: _hours(t) for s, t in got.items()} == {s: list(range(HOURS)) for s in SYMBOLS}
    assert reader.opens == 2


def test_read_error_when_opening_is_retried_on_the_next_pull():
    reader = FlakyReader(on_open=True)
    got = _drain(_pump(reader).start_read(T0, T0 + pd.Timedelta(hours=HOURS)))
    assert {s: len(t) for s, t in got.items()} == {s: HOURS for s in SYMBOLS}


def test_persistent_read_error_is_raised_not_turned_into_an_ended_stream():
    readers = _pump(FlakyReader(fail_opens=10), attempts=3).start_read(T0, T0 + pd.Timedelta(hours=HOURS))
    with pytest.raises(OSError):
        _drain(readers)


def test_restart_gives_a_new_reader_to_a_subscribed_symbol_whose_stream_ended():
    pump = _pump(FlakyReader(fail_opens=0))
    first = _drain(pump.start_read(T0, T0 + pd.Timedelta(hours=6)))
    assert {s: len(t) for s, t in first.items()} == {s: 6 for s in SYMBOLS}
    # - both streams have ended; extending the read must bring them back
    second = _drain(pump.restart_read(T0 + pd.Timedelta(hours=6), T0 + pd.Timedelta(hours=HOURS)))
    assert {s: _hours(t) for s, t in second.items()} == {s: list(range(6, HOURS)) for s in SYMBOLS}
