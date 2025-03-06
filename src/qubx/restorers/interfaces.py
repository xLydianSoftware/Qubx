"""
Interfaces for the restorers module.

This module defines the core interfaces and protocols for restoring strategy state,
including positions and signals.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from qubx.core.basics import AssetBalance, Instrument, Position, Signal


@dataclass
class RestoredState:
    """
    Container for state information needed to restart a strategy.

    This includes the current time, signals by instrument, and positions.
    """

    time: np.datetime64
    balances: dict[str, AssetBalance]
    instrument_to_signals: dict[Instrument, list[Signal]]
    positions: dict[Instrument, Position]


@runtime_checkable
class IPositionRestorer(Protocol):
    """
    Protocol for position restorers.

    Position restorers are responsible for retrieving position information
    when restarting a strategy or running presimulation.
    """

    def restore_positions(self) -> dict[Instrument, Position]:
        """
        Restore positions.

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

    def restore_signals(self) -> dict[Instrument, list[Signal]]:
        """
        Restore signals.

        Returns:
            A dictionary mapping instruments to lists of signals.
        """
        ...


@runtime_checkable
class IBalanceRestorer(Protocol):
    """
    Protocol for balance restorers.
    """

    def restore_balances(self) -> dict[str, AssetBalance]:
        """
        Restore balances.

        Returns:
            A dictionary mapping asset names to asset balances.
        """
        ...


@runtime_checkable
class IStateRestorer(Protocol):
    """
    Protocol for state restorers.
    """

    def restore_state(self) -> RestoredState:
        """
        Restore the state.

        Returns:
            The restored state.
        """
        ...
