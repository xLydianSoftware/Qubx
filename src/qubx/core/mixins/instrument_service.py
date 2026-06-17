from __future__ import annotations

from typing import TYPE_CHECKING

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.instrument_service import IInstrumentService, NullInstrumentService
from qubx.core.interfaces import IInstrumentServiceManager

if TYPE_CHECKING:
    from qubx.core.interfaces import IStrategyContext


class InstrumentServiceManager(IInstrumentServiceManager):
    """Owns the instrument blacklist service: read helpers, the refresh→callbacks→force-close
    cycle, and the framework-automatic startup + TTL-poll scheduling. Composed by StrategyContext
    the same way UniverseManager/ProcessingManager are."""

    def __init__(self, context: "IStrategyContext", instrument_service: IInstrumentService):
        self._context = context
        self._service = instrument_service
        self._callbacks: list = []

    def set_callbacks(self, callbacks: list) -> None:
        self._callbacks = list(callbacks)

    def is_blacklisted(self, instrument: Instrument) -> bool:
        return self._service.is_blacklisted(instrument)

    def filter_blacklisted(self, instruments: list[Instrument]) -> list[Instrument]:
        return [i for i in instruments if not self._service.is_blacklisted(i)]

    def get_blacklisted_instruments(self) -> list[Instrument]:
        return self._service.matching_instruments(self._context.instruments)

    def run_cycle(self, _ctx: "IStrategyContext | None" = None) -> dict:
        """Refresh the blacklist, fire change callbacks, then force-close any still-held
        newly-blacklisted instruments. Single shared implementation used by the control action
        AND the startup/TTL-poll schedules. Runs on the strategy thread. `_ctx` is the
        scheduler-passed context (unused; the bound `self._context` is the context)."""
        diff = self._service.refresh(self._context.instruments)
        if diff.blacklisted_added or diff.blacklisted_removed:
            for cb in self._callbacks:
                try:
                    cb(self._context, diff.blacklisted_added, diff.blacklisted_removed)
                except Exception as e:
                    logger.error(f"[InstrumentService] :: change callback error: {e}")
        positions = self._context.get_positions()
        still_held = [i for i in diff.blacklisted_added if i in positions and positions[i].quantity != 0]
        if still_held:
            self._context.remove_instruments(still_held, if_has_position_then="close")
        return {
            "blacklisted_added": len(diff.blacklisted_added),
            "blacklisted_removed": len(diff.blacklisted_removed),
            "force_closed": len(still_held),
            "force_closed_instruments": [str(i) for i in still_held],
        }

    def start(self) -> None:
        """Framework-automatic refresh wiring (non-Null only): a one-shot startup refresh
        dispatched on the strategy thread via the context scheduler. The blacklist is kept
        current thereafter by the fit-time cache refresh (see `refresh_only`) and by the
        operator-triggered `refresh_instrument_service` action; there is no periodic poll."""
        if isinstance(self._service, NullInstrumentService):
            return
        self._context.delay("1s", self.run_cycle)
