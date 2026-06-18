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
    cycle, fit-time enforcement, and the framework-automatic startup refresh. Composed by StrategyContext
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
        """Refresh the blacklist, fire change callbacks, then force-close ALL held
        blacklisted instruments (full set, not just the change delta). Shared by the
        control action and the startup one-shot. Runs on the strategy thread."""
        diff = self._service.refresh(self._context.instruments)
        # Fire on ANY blacklist edit (entries_changed), not just universe-scoped add/remove:
        # un-blacklisting an instrument that was already evicted from the universe produces
        # no add/remove but must still re-fit so the strategy can re-select it.
        if diff.entries_changed or diff.blacklisted_added or diff.blacklisted_removed:
            for cb in self._callbacks:
                try:
                    cb(self._context, diff.blacklisted_added, diff.blacklisted_removed)
                except Exception as e:
                    logger.error(f"[InstrumentService] :: change callback error: {e}")
        closed = self._force_close_held_blacklisted()
        return {
            "blacklisted_added": len(diff.blacklisted_added),
            "blacklisted_removed": len(diff.blacklisted_removed),
            "force_closed": len(closed),
            "force_closed_instruments": [str(i) for i in closed],
        }

    def _force_close_held_blacklisted(self) -> list[Instrument]:
        """Force-close every currently-held position whose instrument is blacklisted.
        Idempotent and full-set (not the change delta), so already-blacklisted holdings
        are closed too. Reduce-only by construction (closes to 0), so it is allowed by the
        trade-layer blacklist gate. No-op for the Null service (is_blacklisted is False)."""
        positions = self._context.get_positions()
        held = [i for i, p in positions.items() if p.quantity != 0 and self._service.is_blacklisted(i)]
        if held:
            self._context.remove_instruments(held, if_has_position_then="close")
        return held

    def enforce_at_fit(self) -> None:
        """Fit-time enforcement: refresh the cached blacklist AND force-close any held
        blacklisted positions, WITHOUT firing change callbacks (the fit is already
        running; firing re-fit callbacks here would loop). Called immediately before
        `on_fit` so the rebalance selects over current data and never holds a blacklisted
        instrument. No-op for the Null service."""
        self._service.refresh(self._context.instruments)
        self._force_close_held_blacklisted()

    def start(self) -> None:
        """Framework-automatic refresh wiring (non-Null only): a one-shot startup refresh
        dispatched on the strategy thread via the context scheduler. The blacklist is kept
        current thereafter by the fit-time enforcement (see `enforce_at_fit`) and by the
        operator-triggered `refresh_instrument_service` action; there is no periodic poll."""
        if isinstance(self._service, NullInstrumentService):
            return
        self._context.delay("1s", self.run_cycle)
