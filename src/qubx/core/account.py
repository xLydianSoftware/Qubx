from collections import defaultdict

import numpy as np

from qubx import logger
from qubx.core.basics import (
    ZERO_COSTS,
    AssetBalance,
    Deal,
    Instrument,
    ITimeProvider,
    Order,
    Position,
    Timestamped,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.helpers import extract_price
from qubx.core.interfaces import IAccountProcessor


class BasicAccountProcessor(IAccountProcessor):
    account_id: str
    time_provider: ITimeProvider
    base_currency: str
    commissions: str

    _tcc: TransactionCostsCalculator
    _balances: dict[str, AssetBalance]
    _canceled_orders: set[str]
    _active_orders: dict[str, Order]
    _processed_trades: dict[str, list[str | int]]
    _positions: dict[Instrument, Position]
    _locked_capital_by_order: dict[str, float]

    def __init__(
        self,
        account_id: str,
        time_provider: ITimeProvider,
        base_currency: str,
        tcc: TransactionCostsCalculator = ZERO_COSTS,
        initial_capital: float = 100_000,
    ) -> None:
        self.account_id = account_id
        self.time_provider = time_provider
        self.base_currency = base_currency.upper()
        self._tcc = tcc
        self._processed_trades = defaultdict(list)
        self._canceled_orders = set()
        self._active_orders = dict()
        self._positions = {}
        self._locked_capital_by_order = dict()
        self._balances = defaultdict(lambda: AssetBalance())
        self._balances[self.base_currency] += initial_capital

    def get_base_currency(self) -> str:
        return self.base_currency

    ########################################################
    # Balance and position information
    ########################################################
    def get_capital(self) -> float:
        return self.get_available_margin()

    def get_total_capital(self) -> float:
        # sum of cash + market value of all positions
        _cash_amount = self._balances[self.base_currency].total
        _positions_value = sum([p.market_value_funds for p in self._positions.values()])
        return _cash_amount + _positions_value

    def get_balances(self) -> dict[str, AssetBalance]:
        return self._balances

    def get_positions(self) -> dict[Instrument, Position]:
        return self._positions

    def get_position(self, instrument: Instrument) -> Position:
        _pos = self._positions.get(instrument)
        if _pos is None:
            _pos = Position(instrument)
            self._positions[instrument] = _pos
        return _pos

    def get_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        orders = self._active_orders.copy()
        if instrument is not None:
            orders = dict(filter(lambda x: x[1].instrument == instrument, orders.items()))
        return orders

    def position_report(self) -> dict:
        rep = {}
        for p in self._positions.values():
            rep[p.instrument.symbol] = {
                "Qty": p.quantity,
                "Price": p.position_avg_price_funds,
                "PnL": p.pnl,
                "MktValue": p.market_value_funds,
                "Leverage": self.get_leverage(p.instrument),
            }
        return rep

    ########################################################
    # Leverage information
    ########################################################
    def get_leverage(self, instrument: Instrument) -> float:
        pos = self._positions.get(instrument)
        if pos is not None:
            return pos.notional_value / self.get_total_capital()
        return 0.0

    def get_leverages(self) -> dict[Instrument, float]:
        return {s: self.get_leverage(s) for s in self._positions.keys()}

    def get_net_leverage(self) -> float:
        return sum(self.get_leverages().values())

    def get_gross_leverage(self) -> float:
        return sum(map(abs, self.get_leverages().values()))

    ########################################################
    # Margin information
    # Used for margin, swap, futures, options trading
    ########################################################
    def get_total_required_margin(self) -> float:
        # sum of margin required for all positions
        return sum([p.maint_margin for p in self._positions.values()])

    def get_available_margin(self) -> float:
        # total capital - total required margin
        return self.get_total_capital() - self.get_total_required_margin()

    def get_margin_ratio(self) -> float:
        # total capital / total required margin
        return self.get_total_capital() / self.get_total_required_margin()

    ########################################################
    # Order and trade processing
    ########################################################
    def update_balance(self, currency: str, total: float, locked: float):
        # create new asset balance if doesn't exist, otherwise update existing
        if currency not in self._balances:
            self._balances[currency] = AssetBalance(free=total - locked, locked=locked, total=total)
        else:
            self._balances[currency].free = total - locked
            self._balances[currency].locked = locked
            self._balances[currency].total = total

    def attach_positions(self, *position: Position) -> IAccountProcessor:
        for p in position:
            if p.instrument not in self._positions:
                self._positions[p.instrument] = p
            else:
                self._positions[p.instrument].reset_by_position(p)
        return self

    def add_active_orders(self, orders: dict[str, Order]):
        for oid, od in orders.items():
            if oid not in self._active_orders and oid not in self._canceled_orders:
                self._active_orders[oid] = od

    def remove_order(self, order_id: str) -> None:
        if order_id in self._active_orders:
            self._active_orders.pop(order_id)
        self._canceled_orders.add(order_id)

    def update_position_price(self, time: dt_64, instrument: Instrument, update: float | Timestamped) -> None:
        if instrument in self._positions:
            p = self._positions[instrument]
            p.update_market_price(time, extract_price(update), 1)

    def process_market_data(self, time: dt_64, instrument: Instrument, update: Timestamped) -> None: ...

    def process_order(self, order: Order, update_locked_value: bool = True) -> None:
        _new = order.status == "NEW"
        _open = order.status == "OPEN"
        _closed = order.status == "CLOSED"
        _cancel = order.status == "CANCELED"

        if _open or _new:
            if order.id not in self._canceled_orders:
                self._active_orders[order.id] = order

            # - calculate amount locked by this order
            if update_locked_value and order.type == "LIMIT":
                self._lock_limit_order_value(order)

        if _closed or _cancel:
            # TODO: (LIVE) WE NEED TO THINK HOW TO CLEANUP THIS COLLECTION !!!! -> @DM
            # if order.id in self._processed_trades:
            # self._processed_trades.pop(order.id)

            if order.id in self._active_orders:
                self._active_orders.pop(order.id)

        # - calculate amount to unlock after canceling
        if _cancel and update_locked_value and order.type == "LIMIT":
            self._unlock_limit_order_value(order)

        logger.debug(
            f"  [<y>{self.__class__.__name__}</y>(<g>{order.instrument}</g>)] :: New status for order <r>{order.id}</r> -> <y>{order.status}</y> ({order.type} {order.side} {order.quantity}"
            f"{(' @ ' + str(order.price)) if order.price else ''})"
        )

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        self._fill_missing_fee_info(instrument, deals)
        pos = self._positions.get(instrument)

        if pos is not None:
            conversion_rate = 1
            traded_amnt, realized_pnl, deal_cost = 0, 0, 0

            # - process deals
            for d in deals:
                _o_deals = self._processed_trades[d.order_id]

                if d.id not in _o_deals:
                    _o_deals.append(d.id)

                    r_pnl, fee_in_base = pos.update_position_by_deal(d, conversion_rate)
                    realized_pnl += r_pnl
                    deal_cost += d.amount * d.price / conversion_rate
                    traded_amnt += d.amount
                    total_cost = deal_cost + fee_in_base
                    logger.debug(
                        f"  [<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: traded {d.amount} @ {d.price} -> {realized_pnl:.2f} {self.base_currency} realized profit"
                    )
                    if not instrument.is_futures():
                        self._balances[self.base_currency] -= total_cost
                        self._balances[instrument.base] += d.amount
                    else:
                        self._balances[self.base_currency] -= fee_in_base
                        self._balances[instrument.settle] += realized_pnl

    def _fill_missing_fee_info(self, instrument: Instrument, deals: list[Deal]) -> None:
        for d in deals:
            if d.fee_amount is None:
                d.fee_amount = self._tcc.get_execution_fees(
                    instrument=instrument, exec_price=d.price, amount=d.amount, crossed_market=d.aggressive
                )
                # this is only true for linear contracts
                d.fee_currency = instrument.quote

    def _lock_limit_order_value(self, order: Order) -> float:
        pos = self._positions.get(order.instrument)
        excess = 0.0
        # - we handle only instruments it;s subscribed to
        if pos:
            sgn = -1 if order.side == "SELL" else +1
            pos_change = sgn * order.quantity
            direction = np.sign(pos_change)
            prev_direction = np.sign(pos.quantity)
            # how many shares are closed/open
            qty_closing = min(abs(pos.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
            qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing
            excess = abs(qty_opening) * order.price

            # TODO: locking likely doesn't work correctly for spot accounts (Account)
            # Example: if we have 1 BTC at price 100k and set a limit order for 0.1 BTC at 110k
            # it will not lock 0.1 BTC
            if excess > 0:
                self._balances[self.base_currency].lock(excess)
                self._locked_capital_by_order[order.id] = excess

        return excess

    def _unlock_limit_order_value(self, order: Order):
        if order.id in self._locked_capital_by_order:
            excess = self._locked_capital_by_order.pop(order.id)
            self._balances[self.base_currency].lock(-excess)
