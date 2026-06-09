from typing import Any, cast

from qubx import logger
from qubx.core.account_manager import AccountManager
from qubx.core.basics import (
    Instrument,
    MarketType,
    Order,
    OrderOrigin,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from qubx.core.connector import IConnector
from qubx.core.exceptions import InvalidOrderSize, OrderAlreadyTerminal, OrderNotFound
from qubx.core.interfaces import (
    IHealthMonitor,
    IStrategyContext,
    ITimeProvider,
    ITradingManager,
)

from .utils import EXCHANGE_MAPPINGS


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
    _context: IStrategyContext
    _connectors: dict[str, IConnector]
    _account_manager: AccountManager
    _health_monitor: IHealthMonitor
    _strategy_name: str

    _client_id_store: ClientIdStore
    _exchange_to_connector: dict[str, IConnector]

    def __init__(
        self,
        context: IStrategyContext,
        connectors: dict[str, IConnector],
        account_manager: AccountManager,
        health_monitor: IHealthMonitor,
        strategy_name: str,
    ) -> None:
        self._context = context
        self._connectors = connectors
        self._account_manager = account_manager
        self._health_monitor = health_monitor
        self._strategy_name = strategy_name
        self._client_id_store = ClientIdStore()
        self._exchange_to_connector = dict(connectors)

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
        order_type = self._get_order_type(instrument, price, options)
        price = self._adjust_price(instrument, price, amount)
        base_cid = client_id or self._generate_order_client_id(instrument.symbol)

        connector = self._get_connector(instrument.exchange)
        cid = connector.make_client_id(base_cid)

        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Sending {order_type} {side} {size_adj} "
            f"{' @ ' + str(price) if price else ''} -> (client_id: <r>{cid})</r> ..."
        )

        order = Order(
            client_order_id=cid,
            venue_order_id=None,
            origin=OrderOrigin.FRAMEWORK,
            type=cast(OrderType, order_type.upper()),
            instrument=instrument,
            submitted_at=self._context.time(),
            quantity=size_adj,
            price=price,  # None for market orders
            side=side,
            status=OrderStatus.SUBMITTED,
            time_in_force=time_in_force,
            reduce_only=bool(options.get("reduce_only", options.get("reduceOnly", False))),
            post_only=bool(options.get("post_only", False)),
            options=options,
        )

        self._health_monitor.record_order_submit_request(
            exchange=instrument.exchange,
            client_id=cid,
            event_time=self._context.time(),
        )

        request = OrderRequest(
            client_id=cid,
            instrument=instrument,
            quantity=size_adj,
            price=price,
            order_type=cast(OrderType, order_type.upper()),
            side=side,
            time_in_force=time_in_force,
            options=options,
        )

        # Register the order BEFORE submitting. This is required for synchronous
        # connectors (the simulator): submit_order emits Accepted/Filled events inline,
        # so the account state machine must already hold the order to resolve them by cid
        # rather than materialize a phantom EXTERNAL twin. For async (live) connectors the
        # ordering is harmless — the venue events arrive later and resolve the same order.
        self._account_manager.add_order(order)
        try:
            connector.submit_order(request)
        except Exception:
            # A synchronous raise means the order never reached the venue (a framework-side
            # rejection — bad params, pre-submit error). Drop it so the cache keeps no
            # phantom in-flight order; the caller is informed by the re-raised exception.
            self._account_manager.remove_order(connector.exchange_name, cid)
            raise
        return order

    def submit_orders(self, order_requests: list[OrderRequest]) -> list[Order]:
        raise NotImplementedError("Not implemented yet")

    def set_target_position(
        self, instrument: Instrument, target: float, price: float | None = None, time_in_force="gtc", **options
    ) -> Order | None:
        """Set target position for an instrument.

        Calculates the difference between current and target position,
        then places an order to reach the target.

        Args:
            instrument: The instrument to set target position for
            target: Target position size (positive for long, negative for short)
            price: Optional limit price. If provided, uses limit order; otherwise market order
            time_in_force: Time in force for the order
            **options: Additional order options

        Returns:
            Order | None: The created order, or None if no order needed
        """
        # Get current position
        current_position = self._account_manager.get_position(instrument)
        current_quantity = current_position.quantity if current_position is not None else 0.0

        # Calculate amount to trade
        amount_to_trade = target - current_quantity

        # Check if we need to trade
        if self._is_below_min_size(instrument, amount_to_trade):
            logger.debug(
                f"[<g>{instrument.symbol}</g>] :: Target position {target} is close to current position "
                f"{current_quantity}, no trade needed"
            )
            return None

        # Place order
        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Setting target position to {target}, current: {current_quantity}, "
            f"trading: {amount_to_trade}"
        )

        return self.trade(
            instrument=instrument, amount=amount_to_trade, price=price, time_in_force=time_in_force, **options
        )

    def set_target_leverage(
        self, instrument: Instrument, leverage: float, price: float | None = None, **options
    ) -> Order | None:
        """Set target leverage for an instrument.

        Calculates target position size based on leverage percentage of total capital,
        then calls set_target_position to place the order.

        Args:
            instrument: The instrument to set target leverage for
            leverage: Target leverage as fraction (e.g., 0.03 for 3% of capital,
                      negative for short positions)
            price: Optional limit price. If provided, uses limit order and the provided price
                   for position calculation. Otherwise, uses current market price and market order.
            **options: Additional order options

        Returns:
            Order | None: The created order, or None if no order needed
        """
        # Get total capital for the exchange
        total_capital = self._account_manager.get_total_capital(instrument.exchange)
        if total_capital == 0:
            logger.warning(f"[<g>{instrument.symbol}</g>] :: Total capital is 0, cannot set target position")
            return None

        # Calculate capital to use (leverage is a fraction, e.g., 0.03 = 3%)
        capital_to_use = total_capital * abs(leverage)

        # Get price for calculation
        if price is None:
            # Get current market price from quote
            quote = self._context.quote(instrument)
            if quote is None:
                logger.error(f"[<g>{instrument.symbol}</g>] :: Cannot get current price for leverage calculation")
                return None
            # Use mid price for calculation
            calc_price = quote.mid_price()
        else:
            calc_price = price

        # Calculate target position size
        # For positive leverage: long position (positive quantity)
        # For negative leverage: short position (negative quantity)
        target_position = (capital_to_use / (calc_price * instrument.quantity_multiplier)) * (1 if leverage > 0 else -1)

        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Setting target leverage {leverage * 100:.2f}% "
            f"(capital: {total_capital:.2f}, to use: {capital_to_use:.2f}, "
            f"price: {calc_price:.2f}, target: {target_position:.6f})"
        )

        # Call set_target_position to execute the trade
        return self.set_target_position(instrument=instrument, target=target_position, price=price, **options)

    def close_position(self, instrument: Instrument, without_signals: bool = False) -> None:
        position = self._account_manager.get_position(instrument)
        quantity = position.quantity if position is not None else 0.0

        if quantity == 0:
            logger.debug(f"[<g>{instrument.symbol}</g>] :: Position already closed or zero size")
            return

        if without_signals:
            closing_amount = -quantity
            logger.debug(
                f"[<g>{instrument.symbol}</g>] :: Closing position {quantity} with market order for {closing_amount}"
            )
            self.trade(instrument, closing_amount, reduce_only=True)
        else:
            logger.debug(
                f"[<g>{instrument.symbol}</g>] :: Closing position {quantity} by emitting signal with 0 target"
            )
            signal = instrument.signal(self._context, 0, comment="Close position trade")
            self._context.emit_signal(signal)

    def close_positions(self, market_type: MarketType | None = None, without_signals: bool = False) -> None:
        positions = self._account_manager.get_positions()

        positions_to_close = []
        for instrument, position in positions.items():
            if market_type is None or instrument.market_type == market_type:
                if position.is_open():
                    positions_to_close.append(instrument)

        if not positions_to_close:
            logger.debug(f"No open positions to close{f' for market type {market_type}' if market_type else ''}")
            return

        logger.debug(
            f"Closing {len(positions_to_close)} positions{f' for market type {market_type}' if market_type else ''}"
        )

        for instrument in positions_to_close:
            self.close_position(instrument, without_signals)

    def _normalize_order_ids(self, order_id: str | None, client_order_id: str | None) -> tuple[str | None, str | None]:
        # Treat empty strings as not provided
        if order_id is not None and order_id == "":
            order_id = None
        if client_order_id is not None and client_order_id == "":
            client_order_id = None

        if (order_id is None and client_order_id is None) or (order_id is not None and client_order_id is not None):
            raise ValueError("Exactly one of order_id or client_order_id must be provided")
        return order_id, client_order_id

    def _resolve_order(self, order_id: str | None, client_order_id: str | None) -> Order | None:
        if order_id is not None:
            return self._account_manager.find_order_by_id(order_id)
        if client_order_id is not None:
            return self._account_manager.find_order_by_client_id(client_order_id)
        return None

    def cancel_order(
        self, order_id: str | None = None, client_order_id: str | None = None, exchange: str | None = None
    ) -> bool:
        """Cancel a specific order.

        Idempotent: cancelling a terminal or already-PENDING_CANCEL order is a no-op
        that reports success (the venue is already in the desired state).
        """
        order_id, client_order_id = self._normalize_order_ids(order_id, client_order_id)
        order = self._resolve_order(order_id, client_order_id)
        if order is None:
            raise OrderNotFound(client_order_id or order_id or "")

        if order.status.is_terminal or order.status is OrderStatus.PENDING_CANCEL:
            return True

        cid = order.client_order_id
        target_exchange = exchange or order.instrument.exchange

        self._health_monitor.record_order_cancel_request(
            exchange=target_exchange,
            client_id=cid,
            event_time=self._context.time(),
        )
        self._account_manager.transition_order(order.instrument.exchange, cid, OrderStatus.PENDING_CANCEL)
        self._get_connector(order.instrument.exchange).cancel_order(
            client_order_id=cid,
            venue_order_id=order.venue_order_id,
        )
        return True

    def cancel_orders(self, instrument: Instrument | None = None) -> None:
        for o in self._account_manager.get_orders(instrument).values():
            if o.status.is_terminal or o.status is OrderStatus.PENDING_CANCEL:
                continue
            self.cancel_order(client_order_id=o.client_order_id, exchange=o.instrument.exchange)

    def update_order(
        self,
        price: float,
        amount: float,
        order_id: str | None = None,
        client_order_id: str | None = None,
        exchange: str | None = None,
    ) -> None:
        """Update an existing limit order with new price and amount.

        Raises OrderAlreadyTerminal on a settled order (updating a settled order is
        meaningful misuse); a no-op while a previous update is still in flight.
        """
        order_id, client_order_id = self._normalize_order_ids(order_id, client_order_id)
        order = self._resolve_order(order_id, client_order_id)
        if order is None:
            raise OrderNotFound(client_order_id or order_id or "")

        if order.status.is_terminal:
            raise OrderAlreadyTerminal(order.client_order_id, order.status)
        if order.status is OrderStatus.PENDING_UPDATE:
            return

        instrument = order.instrument
        amount = self._adjust_size(instrument, amount)
        adjusted_price = self._adjust_price(instrument, price, amount)
        if adjusted_price is None:
            raise ValueError(f"Price adjustment failed for {instrument.symbol}")

        cid = order.client_order_id
        self._account_manager.transition_order(instrument.exchange, cid, OrderStatus.PENDING_UPDATE)
        self._get_connector(instrument.exchange).update_order(
            client_order_id=cid,
            venue_order_id=order.venue_order_id,
            price=adjusted_price,
            quantity=abs(amount),
        )

    def get_min_size(self, instrument: Instrument, amount: float | None = None) -> float:
        return self._get_min_size(instrument, amount)

    def _generate_order_client_id(self, symbol: str) -> str:
        return self._client_id_store.generate_id(self._context, symbol)

    def exchanges(self) -> list[str]:
        return list(self._exchange_to_connector.keys())

    def _is_position_reducing(self, instrument: Instrument, amount: float) -> bool:
        current_position = self._account_manager.get_position(instrument)
        current_quantity = current_position.quantity if current_position is not None else 0.0
        return (current_quantity > 0 and amount < 0 and abs(amount) <= abs(current_quantity)) or (
            current_quantity < 0 and amount > 0 and abs(amount) <= abs(current_quantity)
        )

    def _is_below_min_size(self, instrument: Instrument, amount: float) -> bool:
        return abs(amount) < self._get_min_size(instrument, amount)

    def _get_min_size(self, instrument: Instrument, amount: float | None = None) -> float:
        min_size_based_on_notional = instrument.min_size
        if instrument.min_notional > 0 and (quote := self._context.quote(instrument)) is not None:
            min_size_based_on_notional = instrument.min_notional / (quote.mid_price() * instrument.quantity_multiplier)

        return (
            instrument.lot_size
            if amount is not None and self._is_position_reducing(instrument, amount)
            else max(min_size_based_on_notional, instrument.min_size)
        )

    def _adjust_size(self, instrument: Instrument, amount: float) -> float:
        abs_amount = abs(amount)
        size_adj = instrument.round_size_down(abs_amount)
        min_size = self._get_min_size(instrument, amount)

        if size_adj >= min_size:
            return size_adj

        size_adj = instrument.round_size_up(abs_amount)
        if size_adj >= min_size:
            return size_adj

        # When amount is already at precision, round_size_up returns the same value.
        # Round up min_size instead, but only if the gap is within one lot step.
        if abs_amount + instrument.lot_size >= min_size:
            return instrument.round_size_up(min_size)

        raise InvalidOrderSize(
            f"[{instrument.symbol}] Attempt to trade size {abs_amount} less than minimal allowed {min_size} !"
        )

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

    def _get_connector(self, exchange: str) -> IConnector:
        if exchange in self._exchange_to_connector:
            return self._exchange_to_connector[exchange]
        # TODO(account-mgmt): drop the EXCHANGE_MAPPINGS fallback when every exchange
        # is keyed by its canonical name across connectors and AccountManager states.
        if exchange in EXCHANGE_MAPPINGS and EXCHANGE_MAPPINGS[exchange] in self._exchange_to_connector:
            return self._exchange_to_connector[EXCHANGE_MAPPINGS[exchange]]
        raise ValueError(f"Connector for exchange {exchange} not found")
