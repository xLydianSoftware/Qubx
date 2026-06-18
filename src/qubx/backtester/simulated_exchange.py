from collections.abc import Generator

from qubx import logger
from qubx.backtester.ome import OrdersManagementEngine, SimulatedExecutionReport
from qubx.core.basics import (
    ZERO_COSTS,
    Instrument,
    ITimeProvider,
    Order,
    OrderStatus,
    Timestamped,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.exceptions import OrderNotFound
from qubx.core.series import Bar, OrderBook, Quote, Trade, TradeArray


class ISimulatedExchange:
    """
    Generic interface for simulated exchange.
    """

    exchange_id: str

    def __init__(self, exchange_id: str):
        self.exchange_id = exchange_id.upper()

    def get_time_provider(self) -> ITimeProvider: ...

    def get_transaction_costs_calculator(self) -> TransactionCostsCalculator: ...

    def place_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> SimulatedExecutionReport: ...

    # Returns the cancel report, or raises OrderNotFound — never returns None (the OME's
    # own None is converted to a raise here), so callers don't need a None branch.
    def cancel_order(self, order_id: str) -> SimulatedExecutionReport: ...

    def get_open_orders(self, instrument: Instrument | None = None) -> dict[str, Order]: ...

    def on_unsubscribe(self, instrument: Instrument) -> None: ...

    def on_subscribe(self, instrument: Instrument) -> None: ...

    def process_market_data(
        self, instrument: Instrument, data: Quote | OrderBook | Trade | TradeArray
    ) -> Generator[SimulatedExecutionReport]: ...


class BasicSimulatedExchange(ISimulatedExchange):
    """
    Basic implementation of generic crypto exchange.
    """

    _ome: dict[Instrument, OrdersManagementEngine]
    _order_to_instrument: dict[str, Instrument]
    _fill_stop_order_at_price: bool
    _time_provider: ITimeProvider
    _tcc: TransactionCostsCalculator

    def __init__(
        self,
        exchange_id: str,
        time_provider: ITimeProvider,
        tcc: TransactionCostsCalculator = ZERO_COSTS,
        accurate_stop_orders_execution: bool = False,
    ):
        super().__init__(exchange_id)
        self._ome = {}
        self._order_to_instrument = {}
        self._fill_stop_order_at_price = accurate_stop_orders_execution
        self._time_provider = time_provider
        self._tcc = tcc

        if self._fill_stop_order_at_price:
            logger.info(
                f"[<y>{self.__class__.__name__}</y>] :: emulation of stop orders executions at exact price is ON"
            )

    def get_time_provider(self) -> ITimeProvider:
        return self._time_provider

    def get_transaction_costs_calculator(self) -> TransactionCostsCalculator:
        return self._tcc

    def place_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **options,
    ) -> SimulatedExecutionReport:
        # - try to place order in OME
        return self._get_ome(instrument).place_order(
            order_side.upper(),  # type: ignore
            order_type.upper(),  # type: ignore
            amount,
            price,
            client_id,
            time_in_force,
            **options,
        )

    def cancel_order(self, order_id: str) -> SimulatedExecutionReport:
        # - first check in active orders
        instrument = self._order_to_instrument.get(order_id)

        if instrument is None:
            # - if not found in active orders, check in each OME
            for o in self._ome.values():
                for order in o.get_open_orders():
                    if order.venue_order_id == order_id:
                        if (result := self._process_ome_response(o.cancel_order(order_id))) is not None:
                            return result

            raise OrderNotFound(f"Order '{order_id}' not found")

        ome = self._ome.get(instrument)
        if ome is None:
            raise ValueError(
                f"{self.__class__.__name__}</y>] :: cancel_order :: No OME created for '{instrument}' - fatal error!"
            )

        # - cancel order in OME and remove from the map to free memory
        result = self._process_ome_response(ome.cancel_order(order_id))
        if result is None:
            raise OrderNotFound(f"Order '{order_id}' not found")
        return result

    def _process_ome_response(self, report: SimulatedExecutionReport | None) -> SimulatedExecutionReport | None:
        if report is not None:
            _order = report.order
            _new = _order.status == OrderStatus.SUBMITTED
            _open = _order.status == OrderStatus.ACCEPTED
            _cancel = _order.status == OrderStatus.CANCELED
            _closed = _order.status == OrderStatus.FILLED

            if _new or _open:
                self._order_to_instrument[_order.venue_order_id] = _order.instrument

            if (_cancel or _closed) and _order.venue_order_id in self._order_to_instrument:
                self._order_to_instrument.pop(_order.venue_order_id)

        return report

    def get_open_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        if instrument is not None:
            ome = self._get_ome(instrument)
            return {o.venue_order_id: o for o in ome.get_open_orders()}

        return {o.venue_order_id: o for ome in self._ome.values() for o in ome.get_open_orders()}

    def on_unsubscribe(self, instrument: Instrument) -> None:
        """
        Called when an instrument is unsubscribed.
        """
        # - clears the OME to remove stale BBO data.
        self._ome.pop(instrument, None)

    def on_subscribe(self, instrument: Instrument) -> None:
        """
        Called when new instrument is subscribed.
        """
        # - just for sanity: remove OME for this instrument if it wasn't removed in on_unsubscribe call
        if instrument in self._ome:
            self._ome.pop(instrument, None)

    def _get_ome(self, instrument: Instrument) -> OrdersManagementEngine:
        if (ome := self._ome.get(instrument)) is None:
            # - create order management engine for instrument
            self._ome[instrument] = (
                ome := OrdersManagementEngine(
                    instrument=instrument,
                    time_provider=self._time_provider,
                    tcc=self._tcc,  # type: ignore
                    fill_stop_order_at_price=self._fill_stop_order_at_price,
                )
            )
        return ome

    def process_market_data(
        self, instrument: Instrument, data: Quote | OrderBook | Trade | TradeArray
    ) -> Generator[SimulatedExecutionReport]:
        ome = self._get_ome(instrument)

        for r in ome.process_market_data(data):
            if r.exec is not None:
                if r.order.venue_order_id in self._order_to_instrument:
                    self._order_to_instrument.pop(r.order.venue_order_id)
                yield r


def get_simulated_exchange(
    exchange_name: str,
    time_provider: ITimeProvider,
    tcc: TransactionCostsCalculator,
    accurate_stop_orders_execution=False,
) -> ISimulatedExchange:
    """
    Factory function to create different types of simulated exchanges based on it's name etc
    Now it supports only basic exchange that fits for most cases of crypto trading.
    """
    return BasicSimulatedExchange(
        exchange_name, time_provider, tcc, accurate_stop_orders_execution=accurate_stop_orders_execution
    )


def emulate_quote_from_data(instrument: Instrument, timestamp: dt_64, data: float | Timestamped) -> Quote | None:
    """Emulate a tradeable quote from arbitrary market data (quote/trade/bar/orderbook/price).

    Pure helper shared by SimulatedConnector (to feed non-matchable data to the OME) and
    SimulatedDataProvider (to maintain get_quote) — neither needs the OME to shape a quote.

    TODO: we need to get rid of this in the future.
    """
    if isinstance(data, Quote):
        return data

    half_tick = instrument.tick_size / 2  # type: ignore
    if isinstance(data, Trade):
        if data.side == 1:  # type: ignore
            return Quote(timestamp, data.price - half_tick * 2, data.price, 0, 0)  # type: ignore
        return Quote(timestamp, data.price, data.price + half_tick * 2, 0, 0)  # type: ignore

    if isinstance(data, Bar):
        return Quote(timestamp, data.close - half_tick, data.close + half_tick, 0, 0)  # type: ignore

    if isinstance(data, OrderBook):
        return data.to_quote()

    if isinstance(data, float):
        return Quote(timestamp, data - half_tick, data + half_tick, 0, 0)

    return None
