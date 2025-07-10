from collections import defaultdict

import numpy as np

from qubx import logger
from qubx.core.basics import (
    ZERO_COSTS,
    AssetBalance,
    Deal,
    FundingPayment,
    Instrument,
    ITimeProvider,
    Order,
    Position,
    Timestamped,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.helpers import extract_price
from qubx.core.interfaces import IAccountProcessor, ISubscriptionManager


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

    def get_base_currency(self, exchange: str | None = None) -> str:
        return self.base_currency

    ########################################################
    # Balance and position information
    ########################################################
    def get_capital(self, exchange: str | None = None) -> float:
        return self.get_available_margin(exchange)

    def get_total_capital(self, exchange: str | None = None) -> float:
        # sum of cash + market value of all positions
        _cash_amount = self._balances[self.base_currency].total
        _positions_value = sum([p.market_value_funds for p in self._positions.values()])
        return _cash_amount + _positions_value

    def get_balances(self, exchange: str | None = None) -> dict[str, AssetBalance]:
        return self._balances

    def get_positions(self, exchange: str | None = None) -> dict[Instrument, Position]:
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

    def position_report(self, exchange: str | None = None) -> dict:
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
        capital = self.get_total_capital()
        if np.isclose(capital, 0):
            return 0.0
        if pos is not None:
            return pos.notional_value / capital
        return 0.0

    def get_leverages(self, exchange: str | None = None) -> dict[Instrument, float]:
        return {s: self.get_leverage(s) for s in self._positions.keys()}

    def get_net_leverage(self, exchange: str | None = None) -> float:
        leverages = self.get_leverages(exchange).values()
        return sum(lev for lev in leverages if lev is not None and not np.isnan(lev))

    def get_gross_leverage(self, exchange: str | None = None) -> float:
        leverages = self.get_leverages(exchange).values()
        return sum(abs(lev) for lev in leverages if lev is not None and not np.isnan(lev))

    ########################################################
    # Margin information
    # Used for margin, swap, futures, options trading
    ########################################################
    def get_total_required_margin(self, exchange: str | None = None) -> float:
        # sum of margin required for all positions
        return sum([p.maint_margin for p in self._positions.values()])

    def get_available_margin(self, exchange: str | None = None) -> float:
        # total capital - total required margin
        return self.get_total_capital(exchange) - self.get_total_required_margin(exchange)

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        # total capital / total required margin
        return self.get_total_capital(exchange) / self.get_total_required_margin(exchange)

    ########################################################
    # Order and trade processing
    ########################################################
    # TODO: remove this interface
    def update_balance(self, currency: str, total: float, locked: float, exchange: str | None = None):
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

    def remove_order(self, order_id: str, exchange: str | None = None) -> None:
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

    def process_funding_payment(self, instrument: Instrument, funding_payment: FundingPayment) -> None:
        """Process funding payment for an instrument.

        Args:
            instrument: Instrument the funding payment applies to
            funding_payment: Funding payment event to process
        """
        pos = self._positions.get(instrument)

        if pos is None or not instrument.is_futures():
            return

        # Get current market price for funding calculation
        # We need to get the mark price from the market data, but since we don't have access
        # to market data here, we'll use the current position price as a reasonable fallback
        mark_price = pos.position_avg_price_funds if pos.position_avg_price_funds > 0 else 0.0

        # Apply funding payment to position
        funding_amount = pos.apply_funding_payment(funding_payment, mark_price)

        # Update account balance with funding payment
        # For futures contracts, funding affects the settlement currency balance
        self._balances[instrument.settle] += funding_amount

        logger.debug(
            f"  [<y>{self.__class__.__name__}</y>(<g>{instrument}</g>)] :: "
            f"funding payment {funding_amount:.6f} {instrument.settle} "
            f"(rate: {funding_payment.funding_rate:.6f})"
        )

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


class CompositeAccountProcessor(IAccountProcessor):
    """
    A composite account processor that manages multiple account processors for different exchanges.
    Each exchange has its own account processor instance.
    """

    account_id: str
    time_provider: ITimeProvider
    base_currency: str

    _account_processors: dict[str, IAccountProcessor]
    _exchange_list: list[str]

    def __init__(
        self,
        time_provider: ITimeProvider,
        account_processors: dict[str, IAccountProcessor],
    ) -> None:
        """
        Initialize a composite account processor with a mapping of exchange names to account processors.

        Args:
            time_provider: Time provider instance
            account_processors: Dictionary mapping exchange names to account processor instances
        """
        self.account_id = "Composite"
        self.time_provider = time_provider
        self._account_processors = account_processors
        self._exchange_list = list(account_processors.keys())

        if not self._exchange_list:
            raise ValueError("At least one account processor must be provided")

    def get_account_processor(self, exchange: str) -> IAccountProcessor:
        return self._account_processors[exchange]

    def _get_exchange(self, exchange: str | None = None, instrument: Instrument | None = None) -> str:
        """
        Determine which exchange to use based on provided arguments.

        Priority:
        1. Explicitly provided exchange
        2. Exchange from instrument
        3. First exchange in the list
        """
        if exchange:
            if exchange not in self._account_processors:
                raise ValueError(f"Unknown exchange: {exchange}")
            return exchange

        if instrument:
            if instrument.exchange not in self._account_processors:
                raise ValueError(f"Unknown exchange: {instrument.exchange}")
            return instrument.exchange

        # Default to first exchange
        return self._exchange_list[0]

    def start(self):
        for processor in self._account_processors.values():
            processor.start()

    def stop(self):
        for processor in self._account_processors.values():
            processor.stop()

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        for processor in self._account_processors.values():
            processor.set_subscription_manager(manager)

    def get_base_currency(self, exchange: str | None = None) -> str:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_base_currency()

    ########################################################
    # Balance and position information
    ########################################################
    def get_capital(self, exchange: str | None = None) -> float:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_capital()

    def get_total_capital(self, exchange: str | None = None) -> float:
        if exchange is not None:
            # Return total capital from specific exchange
            exch = self._get_exchange(exchange)
            return self._account_processors[exch].get_total_capital()

        # Return aggregated total capital from all exchanges when no exchange is specified
        total_capital = 0.0
        for exch_name, processor in self._account_processors.items():
            total_capital += processor.get_total_capital()
        return total_capital

    def get_balances(self, exchange: str | None = None) -> dict[str, AssetBalance]:
        if exchange is not None:
            # Return balances from specific exchange
            exch = self._get_exchange(exchange)
            return self._account_processors[exch].get_balances()

        # Return aggregated balances from all exchanges when no exchange is specified
        all_balances: dict[str, AssetBalance] = defaultdict(lambda: AssetBalance())
        for exch_name, processor in self._account_processors.items():
            exch_balances = processor.get_balances()
            for currency, balance in exch_balances.items():
                if currency not in all_balances:
                    all_balances[currency] = AssetBalance(balance.free, balance.locked, balance.total)
                else:
                    all_balances[currency].free += balance.free
                    all_balances[currency].locked += balance.locked
                    all_balances[currency].total += balance.total
        return dict(all_balances)

    def get_positions(self, exchange: str | None = None) -> dict[Instrument, Position]:
        if exchange is not None:
            # Return positions from specific exchange
            exch = self._get_exchange(exchange)
            return self._account_processors[exch].get_positions()

        # Return positions from all exchanges when no exchange is specified
        all_positions: dict[Instrument, Position] = {}
        for exch_name, processor in self._account_processors.items():
            exch_positions = processor.get_positions()
            all_positions.update(exch_positions)
        return all_positions

    def get_position(self, instrument: Instrument) -> Position:
        exch = self._get_exchange(instrument=instrument)
        return self._account_processors[exch].get_position(instrument)

    def get_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        exch = self._get_exchange(instrument=instrument)
        return self._account_processors[exch].get_orders(instrument)

    def position_report(self, exchange: str | None = None) -> dict:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].position_report()

    ########################################################
    # Leverage information
    ########################################################
    def get_leverage(self, instrument: Instrument) -> float:
        exch = self._get_exchange(instrument=instrument)
        return self._account_processors[exch].get_leverage(instrument)

    def get_leverages(self, exchange: str | None = None) -> dict[Instrument, float]:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_leverages()

    def get_net_leverage(self, exchange: str | None = None) -> float:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_net_leverage()

    def get_gross_leverage(self, exchange: str | None = None) -> float:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_gross_leverage()

    ########################################################
    # Margin information
    # Used for margin, swap, futures, options trading
    ########################################################
    def get_total_required_margin(self, exchange: str | None = None) -> float:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_total_required_margin()

    def get_available_margin(self, exchange: str | None = None) -> float:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_available_margin()

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        exch = self._get_exchange(exchange)
        return self._account_processors[exch].get_margin_ratio()

    ########################################################
    # Order and trade processing
    ########################################################
    def update_balance(self, currency: str, total: float, locked: float, exchange: str | None = None):
        exch = self._get_exchange(exchange)
        self._account_processors[exch].update_balance(currency, total, locked)

    def attach_positions(self, *position: Position) -> IAccountProcessor:
        for pos in position:
            exch = self._get_exchange(instrument=pos.instrument)
            self._account_processors[exch].attach_positions(pos)
        return self

    def add_active_orders(self, orders: dict[str, Order]):
        # Group orders by exchange
        exchange_orders: dict[str, dict[str, Order]] = defaultdict(dict)
        for oid, order in orders.items():
            exch = self._get_exchange(instrument=order.instrument)
            exchange_orders[exch][oid] = order

        # Process orders for each exchange
        for exch, exch_orders in exchange_orders.items():
            self._account_processors[exch].add_active_orders(exch_orders)

    def remove_order(self, order_id: str, exchange: str | None = None):
        exch = self._get_exchange(exchange)
        self._account_processors[exch].remove_order(order_id)

    def update_position_price(self, time: dt_64, instrument: Instrument, update: float | Timestamped) -> None:
        exch = self._get_exchange(instrument=instrument)
        self._account_processors[exch].update_position_price(time, instrument, update)

    def process_market_data(self, time: dt_64, instrument: Instrument, update: Timestamped) -> None:
        exch = self._get_exchange(instrument=instrument)
        self._account_processors[exch].process_market_data(time, instrument, update)

    def process_order(self, order: Order) -> None:
        exch = self._get_exchange(instrument=order.instrument)
        self._account_processors[exch].process_order(order)

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        exch = self._get_exchange(instrument=instrument)
        self._account_processors[exch].process_deals(instrument, deals)

    def process_funding_payment(self, instrument: Instrument, funding_payment: FundingPayment) -> None:
        exch = self._get_exchange(instrument=instrument)
        self._account_processors[exch].process_funding_payment(instrument, funding_payment)
