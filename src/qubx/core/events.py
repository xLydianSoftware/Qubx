from dataclasses import dataclass
from typing import dataclass_transform

from qubx.core.basics import Instrument, dt_64


@dataclass_transform(frozen_default=True, kw_only_default=True)
def msg[T](cls: type[T]) -> type[T]:
    return dataclass(frozen=True, slots=True, kw_only=True)(cls)


@msg
class ChannelMessage:
    """Base class for all channel messages."""

    timestamp: dt_64
    instrument: Instrument | None = None
    is_historical: bool = False
