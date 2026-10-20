from collections.abc import Callable

from qubx import logger
from qubx.core.basics import Instrument, InstrumentChanges, InstrumentsLookup, InstrumentUpdate
from qubx.core.interfaces import IDataProvider, IStrategy, IStrategyContext

INSTRUMENT_CHANGES_EVENT = "instrument_lookup_changes"


class InstrumentUpdateManager:
    """Applies instrument lookup refreshes to a live context.

    The lookup builds new objects on every refresh, while the context, positions, orders and
    connectors hold the ones resolved at startup. Changed venue facts are copied onto every held
    object in place (identity, hence every dict key, stays), then connectors re-read their markets,
    a newly set delisting date is acted on at once and the strategy gets `on_instrument_update`.
    """

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        data_providers: list[IDataProvider],
        check_delistings: Callable[[], None],
    ):
        self._context = context
        self._strategy = strategy
        self._data_providers = data_providers
        self._check_delistings = check_delistings
        self._lookup: InstrumentsLookup | None = None

    def start(self, lookup: InstrumentsLookup) -> None:
        """Live only: in simulation post_event runs inline on the posting (lookup) thread."""
        self._context.register_handler(INSTRUMENT_CHANGES_EVENT, self._on_changes_event)
        self._lookup = lookup
        lookup.add_changes_listener(self._post)

    def stop(self) -> None:
        if self._lookup is not None:
            self._lookup.remove_changes_listener(self._post)
            self._lookup = None

    def _post(self, changes: InstrumentChanges) -> None:
        # - the lookup's refresh thread: hand over to the strategy thread
        self._context.post_event(INSTRUMENT_CHANGES_EVENT, changes)

    def _on_changes_event(self, _ctx: IStrategyContext, changes: InstrumentChanges) -> None:
        self.apply(changes)

    def apply(self, changes: InstrumentChanges) -> list[InstrumentUpdate]:
        """Update held instruments from `changes`; runs on the strategy thread."""
        held = self._held_instruments()
        updates = []
        for fresh in changes.updated:
            diff = {}
            for instrument in held.get(str(fresh), ()):
                diff |= instrument.update_from(fresh)
            if diff:
                updates.append(InstrumentUpdate(held[str(fresh)][0], diff))

        if any("contract_size" in u.changes for u in updates):
            for position in self._context.get_positions().values():
                position.sync_instrument_specs()

        for update in updates:
            logger.warning(f"[InstrumentUpdates] {update.instrument} updated: {_describe(update.changes)}")
        self._notify(updates)

        for data_provider in self._data_providers:
            try:
                data_provider.on_instruments_changed(changes)
            except Exception as e:
                logger.error(f"[InstrumentUpdates] {type(data_provider).__name__} failed to apply changes: {e!r}")

        if any("delisted_at" in u.changes for u in updates):
            self._check_delistings()

        if updates:
            try:
                self._strategy.on_instrument_update(self._context, updates)
            except Exception as e:
                logger.error(f"[InstrumentUpdates] on_instrument_update failed: {e!r}")
        return updates

    def _held_instruments(self) -> dict[str, list[Instrument]]:
        # - positions and orders may hold their own (connector-built) objects for the same symbol
        held: dict[str, dict[int, Instrument]] = {}

        def add(i: Instrument) -> None:
            held.setdefault(str(i), {}).setdefault(id(i), i)

        for i in self._context.instruments:
            add(i)
        for i, position in self._context.get_positions().items():
            add(i)
            add(position.instrument)
        for order in self._context.get_orders().values():
            add(order.instrument)
        return {k: list(v.values()) for k, v in held.items()}

    def _notify(self, updates: list[InstrumentUpdate]) -> None:
        if not updates or self._context.is_simulation:
            return
        try:
            self._context.notifier.notify_message(
                f"[{self._context.strategy_name}] Instruments updated: "
                + "; ".join(f"{u.instrument.symbol} {_describe(u.changes)}" for u in updates),
                metadata={"event": "instrument_update", "instruments": ", ".join(str(u.instrument) for u in updates)},
            )
        except Exception as e:
            logger.error(f"[InstrumentUpdates] failed to notify: {e!r}")


def _describe(changes: dict) -> str:
    return ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in changes.items())
