from collections.abc import Generator

from qubx import logger
from qubx.backtester.ome import OrdersManagementEngine, SimulatedExecutionReport
from qubx.core.basics import (
    ZERO_COSTS,
    Instrument,
    ITimeProvider,
    Order,
    Timestamped,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.series import Bar, OrderBook, Quote, Trade, TradeArray


class ISimulatedExchange:
    """
    Generic interface for simulated exchange.
    """

    exchange_id: str
    _half_tick_size: dict[Instrument, float]

    def __init__(self, exchange_id: str):
        self.exchange_id = exchange_id.upper()
        self._half_tick_size = {}

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

    def cancel_order(self, order_id: str) -> SimulatedExecutionReport | None: ...

    def get_open_orders(self, instrument: Instrument | None = None) -> dict[str, Order]: ...

    def process_market_data(
        self, instrument: Instrument, data: Quote | OrderBook | Trade | TradeArray
    ) -> Generator[SimulatedExecutionReport]: ...

    def emulate_quote_from_data(
        self, instrument: Instrument, timestamp: dt_64, data: float | Timestamped
    ) -> Quote | None:
        """
        Emulate quote from data.

        TODO: we need to get rid of this method in the future
        """
        if instrument not in self._half_tick_size:
            self._half_tick_size[instrument] = instrument.tick_size / 2  # type: ignore

        if isinstance(data, Quote):
            return data

        elif isinstance(data, Trade):
            _ts2 = self._half_tick_size[instrument]
            if data.side == 1:  # type: ignore
                return Quote(timestamp, data.price - _ts2 * 2, data.price, 0, 0)  # type: ignore
            else:
                return Quote(timestamp, data.price, data.price + _ts2 * 2, 0, 0)  # type: ignore

        elif isinstance(data, Bar):
            _ts2 = self._half_tick_size[instrument]
            return Quote(timestamp, data.close - _ts2, data.close + _ts2, 0, 0)  # type: ignore

        elif isinstance(data, OrderBook):
            return data.to_quote()

        elif isinstance(data, float):
            _ts2 = self._half_tick_size[instrument]
            return Quote(timestamp, data - _ts2, data + _ts2, 0, 0)

        else:
            return None


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
        self._half_tick_size = {}
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

    def cancel_order(self, order_id: str) -> SimulatedExecutionReport | None:
        # - first check in active orders
        instrument = self._order_to_instrument.get(order_id)

        if instrument is None:
            # - if not found in active orders, check in each OME
            for o in self._ome.values():
                for order in o.get_open_orders():
                    if order.id == order_id:
                        return self._process_ome_response(o.cancel_order(order_id))

            logger.warning(f"[<y>{self.__class__.__name__}</y>] :: cancel_order :: can't find order '{order_id}'!")
            return None

        ome = self._ome.get(instrument)
        if ome is None:
            raise ValueError(
                f"{self.__class__.__name__}</y>] :: cancel_order :: No OME created for '{instrument}' - fatal error!"
            )

        # - cancel order in OME and remove from the map to free memory
        return self._process_ome_response(ome.cancel_order(order_id))

    def _process_ome_response(self, report: SimulatedExecutionReport | None) -> SimulatedExecutionReport | None:
        if report is not None:
            _order = report.order
            _new = _order.status == "NEW"
            _open = _order.status == "OPEN"
            _cancel = _order.status == "CANCELED"
            _closed = _order.status == "CLOSED"

            if _new or _open:
                self._order_to_instrument[_order.id] = _order.instrument

            if (_cancel or _closed) and _order.id in self._order_to_instrument:
                self._order_to_instrument.pop(_order.id)

        return report

    def get_open_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        if instrument is not None:
            ome = self._get_ome(instrument)
            return {o.id: o for o in ome.get_open_orders()}

        return {o.id: o for ome in self._ome.values() for o in ome.get_open_orders()}

    def _get_ome(self, instrument: Instrument) -> OrdersManagementEngine:
        if (ome := self._ome.get(instrument)) is None:
            self._half_tick_size[instrument] = instrument.tick_size / 2  # type: ignore
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
                if r.order.id in self._order_to_instrument:
                    self._order_to_instrument.pop(r.order.id)
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
