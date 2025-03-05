"""
Interfaces for the restorers module.

This module defines the core interfaces and protocols for restoring strategy state,
including positions and signals.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from qubx.core.basics import Instrument, Position, Signal


@dataclass
class RestartState:
    """
    Container for state information needed to restart a strategy.

    This includes the current time, signals by instrument, and positions.
    """

    time: np.datetime64
    instrument_to_signals: dict[Instrument, list[Signal]]
    positions: dict[Instrument, Position]


@runtime_checkable
class IPositionRestorer(Protocol):
    """
    Protocol for position restorers.

    Position restorers are responsible for retrieving position information
    when restarting a strategy or running presimulation.
    """

    def restore_positions(self, strategy_id: str) -> dict[Instrument, Position]:
        """
        Restore positions for a strategy.

        Args:
            strategy_id: The ID of the strategy to restore positions for.

        Returns:
            A dictionary mapping instruments to positions.
        """
        ...


@runtime_checkable
class ISignalRestorer(Protocol):
    """
    Protocol for signal restorers.

    Signal restorers are responsible for retrieving signal history
    when restarting a strategy or running presimulation.
    """

    def restore_signals(self, strategy_id: str) -> dict[Instrument, list[Signal]]:
        """
        Restore signals for a strategy.

        Args:
            strategy_id: The ID of the strategy to restore signals for.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """
        ...
