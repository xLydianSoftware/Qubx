from dataclasses import dataclass

from qubx.core.basics import Instrument, dt_64


@dataclass(frozen=True, slots=True)
class ExchangeDataStatus:
    """Per-exchange data-flow facts, counted per instrument.

    `subscribed` and `stale_keys` cover only instruments past their grace window; a
    freshly subscribed instrument has no last-event time and would otherwise read as
    stale, which after a full universe swap would make the exchange look entirely dark.
    """

    exchange: str
    connected: bool | None
    subscribed: int
    stale_keys: frozenset[tuple[Instrument, str]]
    in_grace: int
    last_event_time: dt_64 | None

    @property
    def stale(self) -> int:
        return len(self.stale_keys)
