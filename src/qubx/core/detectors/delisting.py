from typing import cast

import pandas as pd

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.interfaces import ITimeProvider


class DelistingDetector:
    def __init__(self, time_provider: ITimeProvider, delisting_check_days: int = 1, silent: bool = False):
        self._time_provider = time_provider
        self._delisting_check_days = delisting_check_days
        self._silent = silent

    @property
    def delisting_check_days(self) -> int:
        return self._delisting_check_days

    @delisting_check_days.setter
    def delisting_check_days(self, days: int) -> None:
        self._delisting_check_days = days

    def detect_delistings(self, instruments: list[Instrument]) -> list[Instrument]:
        """
        Detect instruments that are delisting within delisting_check_days.

        Args:
            instruments: List of instruments to check

        Returns:
            List of instruments that will be delisted within the configured time period
        """
        if self._delisting_check_days <= 0:
            return []

        current_time = pd.Timestamp(self._time_provider.time())
        check_ahead = cast(pd.Timestamp, current_time + pd.Timedelta(days=self._delisting_check_days))

        delisting = []
        for instrument in instruments:
            if instrument.delist_date is None:
                continue

            try:
                delist_timestamp = pd.Timestamp(instrument.delist_date).replace(tzinfo=None)
                if bool(pd.isna(delist_timestamp)):
                    continue

                if delist_timestamp <= check_ahead:
                    delisting.append(instrument)
            except Exception as e:
                self._warning(f"Error checking delist date for {instrument}: {e}")
                # On error, exclude from delisting list to be safe
                continue

        return delisting

    def filter_delistings(self, instruments: list[Instrument]) -> list[Instrument]:
        """
        Filter out instruments that are delisting within delisting_check_days.

        This returns instruments that are safe to trade (not delisting soon).

        Args:
            instruments: List of instruments to filter

        Returns:
            Filtered list of instruments without soon-to-delist instruments
        """
        # Reuse detect_delistings to find instruments that should be filtered out
        delisting_instruments = self.detect_delistings(instruments)
        delisting_set = set(delisting_instruments)

        # Log instruments being filtered out
        for instrument in delisting_instruments:
            self._info(f"Filtering out {instrument.symbol} - delisting on {instrument.delist_date}")

        # Return instruments that are NOT in the delisting set
        return [instrument for instrument in instruments if instrument not in delisting_set]

    def _info(self, msg: str) -> None:
        if not self._silent:
            logger.info(f"[DelistingDetector] {msg}")

    def _warning(self, msg: str) -> None:
        if not self._silent:
            logger.warning(f"[DelistingDetector] {msg}")

    def _error(self, msg: str) -> None:
        if not self._silent:
            logger.error(f"[DelistingDetector] {msg}")
