"""Central account-state owner: per-exchange AccountStates + the event apply path.

Routes a typed AccountMessage to the right exchange's state, applies it via the reducer,
and exposes the cross-exchange read facade + aggregated metrics. Periodic ticks (reconcile,
sweep, liveness), market-data mark-to-market, and PM/connector wiring are added later.
"""

from typing import Callable

from qubx import logger
from qubx.core.account_manager import reducer
from qubx.core.account_manager.events import AccountMessage, AccountSnapshotEvent, OrderEvent
from qubx.core.account_manager.reducer import ApplyResult
from qubx.core.account_manager.state import AccountState
from qubx.core.basics import AssetBalance, Instrument, ITimeProvider, Order, Position
from qubx.core.series import Quote


class AccountManager:
    def __init__(self, base_currencies: dict[str, str], time: ITimeProvider):
        self._states: dict[str, AccountState] = {ex: AccountState(ex, bc) for ex, bc in base_currencies.items()}
        self._time = time

    def get_state(self, exchange: str) -> AccountState:
        return self._states[exchange]

    # ---- event path ---------------------------------------------------- #

    def apply(self, event: AccountMessage) -> ApplyResult:
        state = self._state_for_event(event)
        if state is None:
            return ApplyResult()
        return reducer.apply(state, event, self._time.time())

    def _state_for_event(self, event: AccountMessage) -> AccountState | None:
        if isinstance(event, AccountSnapshotEvent):
            return self._states.get(event.exchange)
        if event.instrument is not None:
            return self._states.get(event.instrument.exchange)
        if isinstance(event, OrderEvent):
            for state in self._states.values():
                if state.get_order(event.client_order_id) is not None:
                    return state
                if event.venue_order_id is not None and state.get_order_by_venue_id(event.venue_order_id) is not None:
                    return state
        if len(self._states) == 1:
            return next(iter(self._states.values()))
        logger.debug(f"cannot route {type(event).__name__} (no instrument/identifiers); dropped")
        return None

    # ---- market data --------------------------------------------------- #

    def on_market_quote(self, instrument: Instrument, quote: Quote) -> None:
        state = self._states.get(instrument.exchange)
        if state is None:
            return
        pos = state.get_position(instrument)
        if pos is None:  # only mark positions we hold; never create one per quote
            return
        pos.update_market_price(self._time.time(), quote.mid_price(), state.conversion_rate(instrument))

    # ---- reads (cross-exchange) ---------------------------------------- #

    def get_orders(self, instrument: Instrument | None = None, exchange: str | None = None) -> dict[str, Order]:
        if exchange is not None:
            orders = self._states[exchange].get_orders()
        else:
            orders = {cid: o for s in self._states.values() for cid, o in s.get_orders().items()}
        return {
            cid: o
            for cid, o in orders.items()
            if not o.status.is_terminal and (instrument is None or o.instrument == instrument)
        }

    def get_order(self, client_order_id: str, exchange: str | None = None) -> Order | None:
        if exchange is not None:
            return self._states[exchange].get_order(client_order_id)
        if len(self._states) == 1:
            return next(iter(self._states.values())).get_order(client_order_id)
        # multi-exchange, caller didn't say which: framework cids are globally unique
        for state in self._states.values():
            if (order := state.get_order(client_order_id)) is not None:
                return order
        return None

    def get_position(self, instrument: Instrument) -> Position | None:
        state = self._states.get(instrument.exchange)
        return state.get_position(instrument) if state is not None else None

    def get_positions(self, exchange: str | None = None) -> dict[Instrument, Position]:
        if exchange is not None:
            return dict(self._states[exchange].get_positions())
        return {ins: pos for s in self._states.values() for ins, pos in s.get_positions().items()}

    def get_balance(self, currency: str, exchange: str | None = None) -> AssetBalance | None:
        if exchange is not None:
            return self._states[exchange].get_balance(currency)
        if len(self._states) == 1:
            return next(iter(self._states.values())).get_balance(currency)
        for state in self._states.values():
            if (bal := state.get_balance(currency)) is not None:
                return bal
        return None

    def get_balances(self, exchange: str | None = None) -> list[AssetBalance]:
        if exchange is not None:
            return list(self._states[exchange].get_balances().values())
        return [b for s in self._states.values() for b in s.get_balances().values()]

    # ---- metrics (aggregated) ------------------------------------------ #

    def _sum(self, metric: Callable[[AccountState], float], exchange: str | None) -> float:
        if exchange is not None:
            return metric(self._states[exchange])
        return sum(metric(s) for s in self._states.values())

    def _states_for(self, exchange: str | None) -> list[AccountState]:
        return [self._states[exchange]] if exchange is not None else list(self._states.values())

    def get_total_capital(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.total_capital, exchange)

    def get_available_margin(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.available_margin, exchange)

    def get_total_initial_margin(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.total_initial_margin, exchange)

    def get_total_maint_margin(self, exchange: str | None = None) -> float:
        return self._sum(AccountState.total_maint_margin, exchange)

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        states = self._states_for(exchange)
        maint = sum(s.total_maint_margin() for s in states)
        if maint == 0:
            return 100.0
        return min(100.0, sum(s.total_capital() for s in states) / maint)

    def get_leverage(self, instrument: Instrument) -> float:
        state = self._states.get(instrument.exchange)
        return state.leverage(instrument) if state is not None else 0.0

    def get_net_leverage(self, exchange: str | None = None) -> float:
        return self._aggregate_leverage(exchange, AccountState.net_leverage)

    def get_gross_leverage(self, exchange: str | None = None) -> float:
        return self._aggregate_leverage(exchange, AccountState.gross_leverage)

    def _aggregate_leverage(self, exchange: str | None, per_state: Callable[[AccountState], float]) -> float:
        # capital-weighted: leverage is notional/capital, so recover Σnotional = Σ(lev * capital)
        states = self._states_for(exchange)
        total_capital = sum(s.total_capital() for s in states)
        if total_capital <= 0:
            return 0.0
        return sum(per_state(s) * s.total_capital() for s in states) / total_capital
