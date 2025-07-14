from dataclasses import dataclass
from operator import neg

import numpy as np
from sortedcontainers import SortedDict

from qubx import logger
from qubx.core.basics import (
    OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION,
    OPTION_FILL_AT_SIGNAL_PRICE,
    OPTION_SIGNAL_PRICE,
    OPTION_SKIP_PRICE_CROSS_CONTROL,
    Deal,
    Instrument,
    ITimeProvider,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.exceptions import (
    ExchangeError,
    InvalidOrder,
    SimulationError,
)
from qubx.core.series import OrderBook, Quote, Trade, TradeArray


@dataclass
class SimulatedExecutionReport:
    instrument: Instrument
    timestamp: dt_64
    order: Order
    exec: Deal | None


class OrdersManagementEngine:
    """
    Orders Management Engine (OME) is a simple implementation of a management of orders for simulation of a limit order book.

    2025-06-02: Added support for deferred execution reports (mainly for stop orders). This handles following cases:
        - It's possible to send stop loss order (STOP_MARKET) in on_execution_report() from custom PositionsTracker class
        - This order may be executed immediately that can lead to calling of on_execution_report() again (when we are still in on_execution_report())
        - To avoid this, it emulate stop orders execution (when condition is met) and add deferred execution report to the list
        - Deferred executions then would be sent on next process_market_data() call
    """

    instrument: Instrument
    time_service: ITimeProvider
    active_orders: dict[str, Order]
    stop_orders: dict[str, Order]
    asks: SortedDict  # [float, list[str]]
    bids: SortedDict  # [float, list[str]]
    bbo: Quote | None  # - current best bid/ask order book
    __prev_bbo: Quote | None  # - previous best bid/ask order book
    __order_id: int
    __trade_id: int
    _fill_stops_at_price: bool
    _tick_size: float
    _last_update_time: dt_64
    _last_data_update_time_ns: int
    _deferred_exec_reports: list[SimulatedExecutionReport]

    def __init__(
        self,
        instrument: Instrument,
        time_provider: ITimeProvider,
        tcc: TransactionCostsCalculator,
        fill_stop_order_at_price: bool = False,  # emulate stop orders execution at order's exact limit price
        debug: bool = True,
    ) -> None:
        self.instrument = instrument
        self.time_service = time_provider
        self.tcc = tcc
        self.asks = SortedDict()
        self.bids = SortedDict(neg)
        self.active_orders = dict()
        self.stop_orders = dict()
        self.bbo = None
        self.__order_id = 100000
        self.__trade_id = 100000
        self._fill_stops_at_price = fill_stop_order_at_price
        self._tick_size = instrument.tick_size
        self._last_update_time = np.datetime64(0, "ns")
        self._last_data_update_time_ns = 0
        self._deferred_exec_reports = []

        if not debug:
            self._dbg = lambda message, **kwargs: None

    def _generate_order_id(self) -> str:
        self.__order_id += 1
        return "SIM-ORDER-" + self.instrument.symbol + "-" + str(self.__order_id)

    def _generate_trade_id(self) -> str:
        self.__trade_id += 1
        return "SIM-EXEC-" + self.instrument.symbol + "-" + str(self.__trade_id)

    def get_quote(self) -> Quote:
        return self.bbo  # type: ignore

    def get_open_orders(self) -> list[Order]:
        return list(self.active_orders.values()) + list(self.stop_orders.values())

    def __remove_pending_status(self, exec: SimulatedExecutionReport) -> SimulatedExecutionReport:
        if exec.order.status == "PENDING":
            exec.order.status = "CLOSED"
        return exec

    def process_market_data(self, mdata: Quote | OrderBook | Trade | TradeArray) -> list[SimulatedExecutionReport]:
        """
        Processes the new market data (quote, trade or trades array) and simulates the execution of pending orders.
        """
        timestamp = self.time_service.time()
        _exec_report = []

        # - process deferred exec reports: spit out deferred exec reports in first place
        if self._deferred_exec_reports:
            _exec_report = [self.__remove_pending_status(i) for i in self._deferred_exec_reports]
            self._deferred_exec_reports.clear()

        # - pass through data if it's older than previous update
        if mdata.time < self._last_data_update_time_ns:
            return _exec_report
        self._last_data_update_time_ns = mdata.time
        _mkt_state = "UNKNOWN"

        # - new quote
        if isinstance(mdata, Quote):
            _b, _a = mdata.bid, mdata.ask
            _bs, _as = _b, _a
            _mkt_state = "Q: " + str(mdata)

            # - update BBO by new quote
            self.__prev_bbo = self.bbo
            self.bbo = mdata

        # - bunch of trades
        elif isinstance(mdata, TradeArray):
            # - to prevent triggering of orders on past trades in array
            _, max_buy_price, min_sell_price, _ = mdata.traded_range_from(self._last_update_time)
            _b = max_buy_price - self._tick_size
            _a = min_sell_price + self._tick_size
            _bs, _as = _a, _b
            _mkt_state = "TA: " + str(mdata)

        # - single trade
        elif isinstance(mdata, Trade):
            _b, _a = mdata.price - self._tick_size, mdata.price + self._tick_size
            _bs, _as = _b, _a
            _mkt_state = "T: " + str(mdata)

        # - order book
        elif isinstance(mdata, OrderBook):
            _b, _a = mdata.top_bid, mdata.top_ask
            _bs, _as = _b, _a
            _mkt_state = "OB: " + str(mdata)

        else:
            raise SimulationError(f"Invalid market data type: {type(mdata)} for update OME({self.instrument.symbol})")

        # - when new quote bid is higher than the lowest ask order execute all affected orders
        if self.asks and _b >= self.asks.keys()[0]:
            _asks_to_execute = list(self.asks.irange(0, _b))
            for level in _asks_to_execute:
                for order_id in self.asks[level]:
                    order = self.active_orders.pop(order_id)
                    _exec_report.append(self._execute_order(timestamp, order.price, order, False, _mkt_state))
                self.asks.pop(level)

        # - when new quote ask is lower than the highest bid order execute all affected orders
        if self.bids and _a <= self.bids.keys()[0]:
            _bids_to_execute = list(self.bids.irange(np.inf, _a))
            for level in _bids_to_execute:
                for order_id in self.bids[level]:
                    order = self.active_orders.pop(order_id)
                    _exec_report.append(self._execute_order(timestamp, order.price, order, False, _mkt_state))
                self.bids.pop(level)

        # - processing stop orders
        for soid in list(self.stop_orders.keys()):
            so = self.stop_orders[soid]
            _emulate_price_exec = self._fill_stops_at_price or so.options.get(OPTION_FILL_AT_SIGNAL_PRICE, False)

            if so.side == "BUY" and _as >= so.price:
                _exec_price = _as if not _emulate_price_exec else so.price
                self.stop_orders.pop(soid)
                _exec_report.append(self._execute_order(timestamp, _exec_price, so, True, _mkt_state))

            elif so.side == "SELL" and _bs <= so.price:
                _exec_price = _bs if not _emulate_price_exec else so.price
                self.stop_orders.pop(soid)
                _exec_report.append(self._execute_order(timestamp, _exec_price, so, True, _mkt_state))

        self._last_update_time = timestamp
        return _exec_report

    def place_order(
        self,
        order_side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> SimulatedExecutionReport:
        if self.bbo is None:
            raise SimulationError(
                f"Simulator is not ready for order management - no quote for {self.instrument.symbol}"
            )

        # - validate order parameters
        self._validate_order(order_side, order_type, amount, price, time_in_force, options)

        timestamp = self.time_service.time()
        order = Order(
            self._generate_order_id(),
            order_type,
            self.instrument,
            timestamp,
            amount,
            price if price is not None else 0,
            order_side,
            "NEW",
            time_in_force,
            client_id,
            options=options,
        )

        return self._process_order(timestamp, order)

    def _dbg(self, message, **kwargs) -> None:
        logger.debug(f"    [<y>OME</y>(<g>{self.instrument}</g>)] :: {message}", **kwargs)

    def _process_order(self, timestamp: dt_64, order: Order) -> SimulatedExecutionReport:
        if order.status in ["CLOSED", "CANCELED"]:
            raise InvalidOrder(f"Order {order.id} is already closed or canceled.")

        _buy_side = order.side == "BUY"
        _c_ask = self.bbo.ask  # type: ignore
        _c_bid = self.bbo.bid  # type: ignore

        # - check if order can be "executed" immediately
        _exec_price = None
        _need_update_book = False

        match order.type:
            case "MARKET":
                if _exec_price is None:
                    _exec_price = _c_ask if _buy_side else _c_bid

                # - special case only for simulation: exact fill at signal price for market orders
                _fill_at_signal_price = order.options.get(OPTION_FILL_AT_SIGNAL_PRICE, False)
                _signal_price = order.options.get(OPTION_SIGNAL_PRICE, None)

                # - some cases require to skip price cross control
                _skip_price_cross_control = order.options.get(OPTION_SKIP_PRICE_CROSS_CONTROL, False)

                # - it's passed only if signal price is valid: market crossed this desired price on last update
                if _fill_at_signal_price and _signal_price and self.__prev_bbo:
                    _desired_fill_price = _signal_price
                    _prev_mp = self.__prev_bbo.mid_price()
                    _c_mid_price = self.bbo.mid_price()  # type: ignore

                    if (
                        _skip_price_cross_control
                        or (_prev_mp < _desired_fill_price <= _c_mid_price)
                        or (_prev_mp > _desired_fill_price >= _c_mid_price)
                    ):
                        _exec_price = _desired_fill_price
                    else:
                        raise SimulationError(
                            f"Special execution price at {_desired_fill_price} for market order {order.id} cannot be filled because market didn't cross this price on last update !"
                        )

            case "LIMIT":
                _need_update_book = True
                if (_buy_side and order.price >= _c_ask) or (not _buy_side and order.price <= _c_bid):
                    _exec_price = _c_ask if _buy_side else _c_bid

            case "STOP_MARKET":
                # - it processes stop orders separately without adding to orderbook (as on real exchanges)
                order.status = "OPEN"
                _stp_order = order
                _emulate_price_exec = self._fill_stops_at_price or _stp_order.options.get(
                    OPTION_FILL_AT_SIGNAL_PRICE, False
                )

                if _stp_order.side == "BUY" and _c_ask >= _stp_order.price:
                    # _exec_price = _c_ask if not _emulate_price_exec else so.price
                    self._deferred_exec_reports.append(
                        self._execute_order(
                            timestamp,
                            _c_ask if not _emulate_price_exec else _stp_order.price,
                            order,
                            True,
                            "BBO: " + str(self.bbo),
                            "PENDING",
                        )
                    )

                elif _stp_order.side == "SELL" and _c_bid <= _stp_order.price:
                    # _exec_price = _c_bid if not _emulate_price_exec else so.price
                    self._deferred_exec_reports.append(
                        self._execute_order(
                            timestamp,
                            _c_bid if not _emulate_price_exec else _stp_order.price,
                            order,
                            True,
                            "BBO: " + str(self.bbo),
                            "PENDING",
                        )
                    )

                else:
                    self.stop_orders[order.id] = order

            case "STOP_LIMIT":
                # TODO: (OME) check trigger conditions in options etc
                raise NotImplementedError("'STOP_LIMIT' order is not supported in Qubx simulator yet !")

            case _:
                raise SimulationError(f"Invalid order type: {order.type} for {self.instrument.symbol}")

        # - if order must be "executed" immediately
        if _exec_price is not None:
            return self._execute_order(timestamp, _exec_price, order, True, "BBO: " + str(self.bbo))

        # - processing limit orders
        if _need_update_book:
            if _buy_side:
                self.bids.setdefault(order.price, list()).append(order.id)
            else:
                self.asks.setdefault(order.price, list()).append(order.id)

            order.status = "OPEN"
            self.active_orders[order.id] = order

        self._dbg(f"registered {order.id} {order.type} {order.side} {order.quantity} {order.price}")
        return SimulatedExecutionReport(self.instrument, timestamp, order, None)

    def _execute_order(
        self,
        timestamp: dt_64,
        exec_price: float,
        order: Order,
        taker: bool,
        market_state: str,
        status: OrderStatus = "CLOSED",
    ) -> SimulatedExecutionReport:
        order.status = status
        self._dbg(
            f"<red>{order.id}</red> {order.type} {order.side} {order.quantity} executed at {exec_price} ::: {market_state} [{status}]"
        )
        return SimulatedExecutionReport(
            self.instrument,
            timestamp,
            order,
            Deal(
                id=self._generate_trade_id(),
                order_id=order.id,
                time=timestamp,
                amount=order.quantity if order.side == "BUY" else -order.quantity,
                price=exec_price,
                aggressive=taker,
                fee_amount=self.tcc.get_execution_fees(
                    instrument=self.instrument, exec_price=exec_price, amount=order.quantity, crossed_market=taker
                ),
                fee_currency=self.instrument.quote,
            ),
        )

    def _validate_order(
        self, order_side: str, order_type: str, amount: float, price: float | None, time_in_force: str, options: dict
    ) -> None:
        if order_side.upper() not in ["BUY", "SELL"]:
            raise InvalidOrder("Invalid order side. Only BUY or SELL is allowed.")

        _ot = order_type.upper()
        if _ot not in ["LIMIT", "MARKET", "STOP_MARKET", "STOP_LIMIT"]:
            raise InvalidOrder("Invalid order type. Only LIMIT, MARKET, STOP_MARKET, STOP_LIMIT are supported.")

        if amount <= 0:
            raise InvalidOrder("Invalid order amount. Amount must be positive.")

        if (_ot == "LIMIT" or _ot.startswith("STOP")) and (price is None or price <= 0):
            raise InvalidOrder("Invalid order price. Price must be positively defined for LIMIT or STOP orders.")

        if time_in_force.upper() not in ["GTC", "IOC", "GTX"]:
            raise InvalidOrder("Invalid time in force. Only GTC, IOC, GTX are supported for now.")

        if _ot.startswith("STOP"):
            # - if the option is set, we don't check the current market price against the stop price
            if options.get(OPTION_AVOID_STOP_ORDER_PRICE_VALIDATION, False):
                return

            assert self.bbo
            assert price
            c_ask, c_bid = self.bbo.ask, self.bbo.bid
            if (order_side == "BUY" and c_ask >= price) or (order_side == "SELL" and c_bid <= price):
                raise ExchangeError(
                    f"Stop price would trigger immediately: STOP_MARKET {order_side} {amount} of {self.instrument.symbol} at {price} | market: {c_ask} / {c_bid}"
                )

    def cancel_order(self, order_id: str) -> SimulatedExecutionReport | None:
        # - check limit orders
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            if order.side == "BUY":
                oids = self.bids[order.price]
                oids.remove(order_id)
                if not oids:
                    self.bids.pop(order.price)
            else:
                oids = self.asks[order.price]
                oids.remove(order_id)
                if not oids:
                    self.asks.pop(order.price)
        # - check stop orders
        elif order_id in self.stop_orders:
            order = self.stop_orders.pop(order_id)
        # - wrong order_id
        else:
            logger.error(f"Can't cancel order {order_id} for {self.instrument.symbol} because it's not found in OME !")
            return None

        order.status = "CANCELED"
        self._dbg(f"{order.id} {order.type} {order.side} {order.quantity} canceled")
        return SimulatedExecutionReport(self.instrument, self.time_service.time(), order, None)

    def __str__(self) -> str:
        _a, _b = True, True

        timestamp = self.time_service.time()
        _s = f"= = ({np.datetime64(timestamp, 'ns')}) = =\n"
        for k, v in reversed(self.asks.items()):
            _sizes = ",".join([f"{self.active_orders[o].quantity}" for o in v])
            _s += f"  {k} : [{_sizes}]\n"
            if k == self.bbo.ask:
                _a = False

        if _a:
            _s += f"  {self.bbo.ask} : \n"
        _s += "- - - - - - - - - - - - - - - - - - - -\n"

        _s1 = ""
        for k, v in self.bids.items():
            _sizes = ",".join([f"{self.active_orders[o].quantity}" for o in v])
            _s1 += f"  {k} : [{_sizes}]\n"
            if k == self.bbo.bid:
                _b = False
        _s1 += "= = = = = = = = = = = = = = = = = = = =\n"

        _s1 = f"  {self.bbo.bid} : \n" + _s1 if _b else _s1

        return _s + _s1
