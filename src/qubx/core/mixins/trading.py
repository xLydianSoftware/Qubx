from typing import Any

from qubx import logger
from qubx.core.basics import Instrument, MarketType, Order, OrderRequest, OrderSide
from qubx.core.interfaces import IAccountProcessor, IBroker, ITimeProvider, ITradingManager


class ClientIdStore:
    """Manages generation of unique client order IDs."""

    def __init__(self):
        """Initialize a client ID store."""
        self._order_id: int | None = None

    def generate_id(self, time_provider: ITimeProvider, symbol: str) -> str:
        """Generate a unique client order ID.

        Args:
            time_provider: Time provider to get current timestamp
            symbol: Trading symbol for the order

        Returns:
            A unique client order ID
        """
        # Initialize order ID from timestamp if not yet set
        if self._order_id is None:
            self._order_id = self._initialize_id_from_timestamp(time_provider)

        # Increment order ID to ensure uniqueness across calls
        self._order_id += 1

        # Create and return the unique ID
        return self._create_id(symbol, self._order_id)

    def _initialize_id_from_timestamp(self, time_provider: ITimeProvider) -> int:
        """Initialize the order ID from the current timestamp.

        Args:
            time_provider: Time provider to get current timestamp

        Returns:
            Initial order ID value
        """
        return time_provider.time().astype("int64") // 100_000_000

    def _create_id(self, symbol: str, order_id: int) -> str:
        """Create the ID from symbol and order ID.

        Args:
            symbol: Trading symbol
            order_id: Current order ID counter

        Returns:
            Client ID string
        """
        return "_".join(["qubx", symbol, str(order_id)])


class TradingManager(ITradingManager):
    _time_provider: ITimeProvider
    _brokers: list[IBroker]
    _account: IAccountProcessor
    _strategy_name: str

    _client_id_store: ClientIdStore
    _exchange_to_broker: dict[str, IBroker]

    def __init__(
        self, time_provider: ITimeProvider, brokers: list[IBroker], account: IAccountProcessor, strategy_name: str
    ) -> None:
        self._time_provider = time_provider
        self._brokers = brokers
        self._account = account
        self._strategy_name = strategy_name
        self._client_id_store = ClientIdStore()
        self._exchange_to_broker = {broker.exchange(): broker for broker in brokers}

    def trade(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        client_id: str | None = None,
        **options,
    ) -> Order:
        size_adj = self._adjust_size(instrument, amount)
        side = self._get_side(amount)
        type = self._get_order_type(instrument, price, options)
        price = self._adjust_price(instrument, price, amount)
        client_id = client_id or self._generate_order_client_id(instrument.symbol)

        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Sending (blocking) {type} {side} {size_adj} {' @ ' + str(price) if price else ''} -> (client_id: <r>{client_id})</r> ..."
        )

        order = self._exchange_to_broker[instrument.exchange].send_order(
            instrument=instrument,
            order_side=side,
            order_type=type,
            amount=size_adj,
            price=price,
            time_in_force=time_in_force,
            client_id=client_id,
            **options,
        )

        if order is not None:
            self._account.add_active_orders({order.id: order})

        return order

    def trade_async(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        client_id: str | None = None,
        **options,
    ) -> None:
        size_adj = self._adjust_size(instrument, amount)
        side = self._get_side(amount)
        type = self._get_order_type(instrument, price, options)
        price = self._adjust_price(instrument, price, amount)
        client_id = client_id or self._generate_order_client_id(instrument.symbol)

        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Sending (async) {type} {side} {size_adj} {' @ ' + str(price) if price else ''} -> (client_id: <r>{client_id})</r> ..."
        )

        self._exchange_to_broker[instrument.exchange].send_order_async(
            instrument=instrument,
            order_side=side,
            order_type=type,
            amount=size_adj,
            price=price,
            time_in_force=time_in_force,
            client_id=client_id,
            **options,
        )

    def submit_orders(self, order_requests: list[OrderRequest]) -> list[Order]:
        raise NotImplementedError("Not implemented yet")

    def set_target_position(
        self, instrument: Instrument, target: float, price: float | None = None, time_in_force="gtc", **options
    ) -> Order:
        raise NotImplementedError("Not implemented yet")

    def set_target_leverage(
        self, instrument: Instrument, leverage: float, price: float | None = None, **options
    ) -> Order:
        raise NotImplementedError("Not implemented yet")

    def close_position(self, instrument: Instrument) -> None:
        raise NotImplementedError("Not implemented yet")

    def close_positions(self, market_type: MarketType | None = None) -> None:
        raise NotImplementedError("Not implemented yet")

    def cancel_order(self, order_id: str, exchange: str | None = None) -> None:
        if not order_id:
            return
        if exchange is None:
            exchange = self._brokers[0].exchange()
        self._exchange_to_broker[exchange].cancel_order(order_id)
        self._account.remove_order(order_id, exchange)

    def cancel_orders(self, instrument: Instrument) -> None:
        for o in self._account.get_orders(instrument).values():
            self.cancel_order(o.id, instrument.exchange)

    def _generate_order_client_id(self, symbol: str) -> str:
        return self._client_id_store.generate_id(self._time_provider, symbol)

    def exchanges(self) -> list[str]:
        return list(self._exchange_to_broker.keys())

    def _adjust_size(self, instrument: Instrument, amount: float) -> float:
        size_adj = instrument.round_size_down(abs(amount))
        if size_adj < instrument.min_size:
            raise ValueError(
                f"[{instrument.symbol}] Attempt to trade size {abs(amount)} less than minimal allowed {instrument.min_size} !"
            )
        return size_adj

    def _adjust_price(self, instrument: Instrument, price: float | None, amount: float) -> float | None:
        if price is None:
            return price
        return instrument.round_price_down(price) if amount > 0 else instrument.round_price_up(price)

    def _get_side(self, amount: float) -> OrderSide:
        return "BUY" if amount > 0 else "SELL"

    def _get_order_type(self, instrument: Instrument, price: float | None, options: dict[str, Any]) -> str:
        if price is None:
            return "market"
        if (stp_type := options.get("stop_type")) is not None:
            return f"stop_{stp_type}"
        return "limit"
