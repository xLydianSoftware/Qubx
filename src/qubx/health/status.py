from dataclasses import dataclass

from qubx.core.basics import dt_64


@dataclass(frozen=True, slots=True)
class ExchangeDataStatus:
    """Per-exchange data-flow facts, counted per instrument.

    `subscribed` and `stale` cover only instruments past their grace window; a freshly
    subscribed instrument has no last-event time and would otherwise read as stale,
    which after a full universe swap would make the exchange look entirely dark.
    """

    exchange: str
    connected: bool | None
    subscribed: int
    stale: int
    in_grace: int
    last_event_time: dt_64 | None
