from typing import Any, cast

from qubx import logger
from qubx.core.basics import Instrument, MarketType, Order, OrderRequest, OrderSide, OrderType
from qubx.core.exceptions import InvalidOrderSize, OrderNotFound
from qubx.core.interfaces import (
    IAccountProcessor,
    IBroker,
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
    _brokers: list[IBroker]
    _account: IAccountProcessor
    _health_monitor: IHealthMonitor
    _strategy_name: str

    _client_id_store: ClientIdStore
    _exchange_to_broker: dict[str, IBroker]

    def __init__(
        self,
        context: IStrategyContext,
        brokers: list[IBroker],
        account: IAccountProcessor,
        health_monitor: IHealthMonitor,
        strategy_name: str,
    ) -> None:
        self._context = context
        self._brokers = brokers
        self._account = account
        self._health_monitor = health_monitor
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
    ) -> Order | None:
        size_adj = self._adjust_size(instrument, amount)
        side = self._get_side(amount)
        type = self._get_order_type(instrument, price, options)
        price = self._adjust_price(instrument, price, amount)
        client_id = client_id or self._generate_order_client_id(instrument.symbol)
        client_id = self._get_broker(instrument.exchange).make_client_id(client_id)

        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Sending (blocking) {type} {side} {size_adj} {' @ ' + str(price) if price else ''} -> (client_id: <r>{client_id})</r> ..."
        )

        request = OrderRequest(
            client_id=client_id,
            instrument=instrument,
            quantity=size_adj,
            price=price,
            order_type=cast(OrderType, type.upper()),
            side=side,
            time_in_force=time_in_force,
            options=options,
        )

        if request.client_id is not None:
            self._health_monitor.record_order_submit_request(
                exchange=instrument.exchange,
                client_id=request.client_id,
                event_time=self._context.time(),
            )

        order = self._get_broker(instrument.exchange).send_order(request)

        if order is not None:
            self._account.add_active_orders({order.id: order})
        elif request.client_id is not None:
            self._account.process_order_request(request)

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
        client_id = self._get_broker(instrument.exchange).make_client_id(client_id)

        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Sending (async) {type} {side} {size_adj} {' @ ' + str(price) if price else ''} -> (client_id: <r>{client_id})</r> ..."
        )

        request = OrderRequest(
            client_id=client_id,
            instrument=instrument,
            quantity=size_adj,
            price=price,
            order_type=cast(OrderType, type.upper()),
            side=side,
            time_in_force=time_in_force,
            options=options,
        )

        if request.client_id is not None:
            self._health_monitor.record_order_submit_request(
                exchange=instrument.exchange,
                client_id=request.client_id,
                event_time=self._context.time(),
            )

        self._get_broker(instrument.exchange).send_order_async(request)

        self._account.process_order_request(request)

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
        current_position = self._account.get_position(instrument)

        # Calculate amount to trade
        amount_to_trade = target - current_position.quantity

        # Check if we need to trade
        if self._is_below_min_size(instrument, amount_to_trade):
            logger.debug(
                f"[<g>{instrument.symbol}</g>] :: Target position {target} is close to current position {current_position.quantity}, no trade needed"
            )
            return None

        # Place order
        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Setting target position to {target}, current: {current_position.quantity}, trading: {amount_to_trade}"
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
        total_capital = self._account.get_total_capital(instrument.exchange)
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
        target_position = (capital_to_use / calc_price) * (1 if leverage > 0 else -1)

        logger.debug(
            f"[<g>{instrument.symbol}</g>] :: Setting target leverage {leverage * 100:.2f}% "
            f"(capital: {total_capital:.2f}, to use: {capital_to_use:.2f}, "
            f"price: {calc_price:.2f}, target: {target_position:.6f})"
        )

        # Call set_target_position to execute the trade
        return self.set_target_position(instrument=instrument, target=target_position, price=price, **options)

    def close_position(self, instrument: Instrument, without_signals: bool = False) -> None:
        position = self._account.get_position(instrument)

        if position.quantity == 0:
            logger.debug(f"[<g>{instrument.symbol}</g>] :: Position already closed or zero size")
            return

        if without_signals:
            closing_amount = -position.quantity
            logger.debug(
                f"[<g>{instrument.symbol}</g>] :: Closing position {position.quantity} with market order for {closing_amount}"
            )
            self.trade_async(instrument, closing_amount, reduceOnly=True)
        else:
            logger.debug(
                f"[<g>{instrument.symbol}</g>] :: Closing position {position.quantity} by emitting signal with 0 target"
            )
            signal = instrument.signal(self._context, 0, comment="Close position trade")
            self._context.emit_signal(signal)

    def close_positions(self, market_type: MarketType | None = None, without_signals: bool = False) -> None:
        positions = self._account.get_positions()

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

    def cancel_order(self, order_id: str, exchange: str | None = None) -> bool:
        """Cancel a specific order synchronously."""
        if not order_id:
            return False
        if exchange is None:
            exchange = self._brokers[0].exchange()
        try:
            order = self._account.find_order_by_id(order_id)
            if order is not None and order.client_id:
                self._health_monitor.record_order_cancel_request(
                    exchange=exchange,
                    client_id=order.client_id,
                    event_time=self._context.time(),
                )
            success = self._get_broker(exchange).cancel_order(order_id)
            if success:
                self._account.remove_order(order_id, exchange)
            return success
        except OrderNotFound:
            # Order was already cancelled or doesn't exist
            # Still try to remove it from account to keep state consistent
            self._account.remove_order(order_id, exchange)
            return False  # Return False since order wasn't found
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False  # Return False for any other errors

    def cancel_order_async(self, order_id: str, exchange: str | None = None) -> None:
        """Cancel a specific order asynchronously (non blocking)."""
        if not order_id:
            return
        if exchange is None:
            exchange = self._brokers[0].exchange()
        try:
            order = self._account.find_order_by_id(order_id)
            if order is not None and order.client_id:
                self._health_monitor.record_order_cancel_request(
                    exchange=exchange,
                    client_id=order.client_id,
                    event_time=self._context.time(),
                )
            self._get_broker(exchange).cancel_order_async(order_id)
            # Note: For async, we remove the order optimistically
            # The actual removal will be confirmed via order status updates
            self._account.remove_order(order_id, exchange)
        except OrderNotFound:
            # Order was already cancelled or doesn't exist
            # Still try to remove it from account to keep state consistent
            self._account.remove_order(order_id, exchange)

    def cancel_orders(self, instrument: Instrument) -> None:
        for o in self._account.get_orders(instrument).values():
            self.cancel_order(o.id, instrument.exchange)

    def update_order(self, order_id: str, price: float, amount: float, exchange: str | None = None) -> Order:
        """Update an existing limit order with new price and amount."""
        if not order_id:
            raise ValueError("Order ID is required")
        if exchange is None:
            exchange = self._brokers[0].exchange()

        # Get the existing order to determine instrument for adjustments
        active_orders = self._account.get_orders(exchange=exchange)
        existing_order = active_orders.get(order_id)
        if not existing_order:
            # Let broker handle the OrderNotFound - just pass through
            logger.debug(f"Updating order {order_id}: {amount} @ {price} on {exchange}")
        else:
            # Apply TradingManager-level adjustments before sending to broker
            instrument = existing_order.instrument
            adjusted_amount = self._adjust_size(instrument, amount)
            adjusted_price = self._adjust_price(instrument, price, amount)
            if adjusted_price is None:
                raise ValueError(f"Price adjustment failed for {instrument.symbol}")
            # Update the values to use adjusted ones
            amount = adjusted_amount
            price = adjusted_price

        try:
            updated_order = self._get_broker(exchange).update_order(order_id, price, abs(amount))

            if updated_order is not None:
                # Update account tracking with new order info
                self._account.process_order(updated_order)
                logger.info(f"[<g>{updated_order.instrument.symbol}</g>] :: Successfully updated order {order_id}")

            return updated_order
        except Exception as e:
            logger.error(f"Error updating order {order_id}: {e}")
            raise e

    def get_min_size(self, instrument: Instrument, amount: float | None = None) -> float:
        # TODO: maybe it's possible some exchanges have a different logic, then enable overrides via brokers
        return self._get_min_size(instrument, amount)

    def _generate_order_client_id(self, symbol: str) -> str:
        return self._client_id_store.generate_id(self._context, symbol)

    def exchanges(self) -> list[str]:
        return list(self._exchange_to_broker.keys())

    def get_broker(self, exchange: str | None = None) -> IBroker:
        """
        Get broker for a specific exchange (public access to brokers).

        Args:
            exchange: Exchange name (optional, defaults to first broker if None)

        Returns:
            IBroker: The broker instance for the exchange

        Raises:
            ValueError: If the exchange is not found
        """
        if exchange is None:
            return self._brokers[0]
        return self._get_broker(exchange)

    def list_exchange_capabilities(self, exchange: str | None = None) -> dict[str, str]:
        """
        List available extension methods for an exchange.

        Args:
            exchange: Exchange name (optional, defaults to first broker if None)

        Returns:
            dict[str, str]: Dictionary mapping method names to their descriptions

        Example:
            >>> capabilities = ctx.list_exchange_capabilities("LIGHTER")
            >>> print(capabilities)
            {'update_leverage': 'Update leverage for an instrument',
             'transfer': 'Transfer USDC between accounts',
             'create_pool': 'Create a public liquidity pool'}
        """
        broker = self.get_broker(exchange)
        return broker.extensions.list_methods()

    def get_extension_help(self, exchange: str | None = None, method: str | None = None) -> str:
        """
        Get help text for exchange extension methods.

        Args:
            exchange: Exchange name (optional, defaults to first broker if None)
            method: Specific method name (optional, lists all if None)

        Returns:
            str: Formatted help text

        Example:
            >>> # List all methods
            >>> print(ctx.get_extension_help("LIGHTER"))
            Available extension methods:
              create_pool: Create a public liquidity pool
              transfer: Transfer USDC between accounts
              update_leverage: Update leverage for an instrument

            >>> # Get detailed help for specific method
            >>> print(ctx.get_extension_help("LIGHTER", "update_leverage"))
            Method: update_leverage(instrument: Instrument, leverage: float, margin_mode: str = 'cross')

            Update leverage for an instrument.

            Args:
                instrument: Instrument to update leverage for
                leverage: Target leverage ratio (e.g., 10.0 for 10x)
                margin_mode: "cross" or "isolated" (default: "cross")

            Returns:
                True if successful
        """
        broker = self.get_broker(exchange)
        return broker.extensions.help(method)

    def _is_position_reducing(self, instrument: Instrument, amount: float) -> bool:
        current_position = self._account.get_position(instrument)
        return (current_position.quantity > 0 and amount < 0 and abs(amount) <= abs(current_position.quantity)) or (
            current_position.quantity < 0 and amount > 0 and abs(amount) <= abs(current_position.quantity)
        )

    def _is_below_min_size(self, instrument: Instrument, amount: float) -> bool:
        return abs(amount) < self._get_min_size(instrument, amount)

    def _get_min_size(self, instrument: Instrument, amount: float | None = None) -> float:
        min_size_based_on_notional = instrument.min_size
        if instrument.min_notional > 0 and (quote := self._context.quote(instrument)) is not None:
            min_size_based_on_notional = instrument.min_notional / quote.mid_price()

        return (
            instrument.lot_size
            if amount is not None and self._is_position_reducing(instrument, amount)
            else max(min_size_based_on_notional, instrument.min_size)
        )

    def _adjust_size(self, instrument: Instrument, amount: float) -> float:
        size_adj = instrument.round_size_down(abs(amount))
        min_size = self._get_min_size(instrument, amount)
        if abs(size_adj) < min_size:
            # Try just in case to round up to avoid too small orders
            size_adj = instrument.round_size_up(abs(amount))
            if abs(size_adj) < min_size:
                raise InvalidOrderSize(
                    f"[{instrument.symbol}] Attempt to trade size {abs(amount)} less than minimal allowed {min_size} !"
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

    def _get_broker(self, exchange: str) -> IBroker:
        if exchange in self._exchange_to_broker:
            return self._exchange_to_broker[exchange]
        if exchange in EXCHANGE_MAPPINGS and EXCHANGE_MAPPINGS[exchange] in self._exchange_to_broker:
            return self._exchange_to_broker[EXCHANGE_MAPPINGS[exchange]]
        raise ValueError(f"Broker for exchange {exchange} not found")
