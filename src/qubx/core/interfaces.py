"""
This module defines interfaces and classes related to trading strategies.

This module includes:
    - Trading service providers
    - Broker service providers
    - Market data providers
    - Strategy contexts
    - Position tracking and management
    - Data exporters
    - Metric emitters
    - Strategy lifecycle notifiers
"""

import inspect
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import (
    AssetBalance,
    CtrlChannel,
    Deal,
    FundingPayment,
    Instrument,
    ITimeProvider,
    MarketEvent,
    MarketType,
    Order,
    OrderRequest,
    Position,
    RestoredState,
    Signal,
    TargetPosition,
    Timestamped,
    TransactionCostsCalculator,
    TriggerEvent,
    dt_64,
    td_64,
)
from qubx.core.errors import BaseErrorEvent
from qubx.core.helpers import set_parameters_to_object
from qubx.core.series import OHLCV, Bar, Quote

if TYPE_CHECKING:
    from qubx.data.readers import DataReader

RemovalPolicy = Literal["close", "wait_for_close", "wait_for_change"]


class ITradeDataExport:
    """Interface for exporting trading data to external systems."""

    def export_signals(self, time: dt_64, signals: list[Signal], account: "IAccountViewer") -> None:
        """
        Export signals to an external system.

        Args:
            time: Timestamp when the signals were generated
            signals: list of signals to export
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        pass

    def export_target_positions(self, time: dt_64, targets: list[TargetPosition], account: "IAccountViewer") -> None:
        """
        Export target positions to an external system.

        Args:
            time: Timestamp when the target positions were generated
            targets: list of target positions to export
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        pass

    def export_position_changes(
        self, time: dt_64, instrument: Instrument, price: float, account: "IAccountViewer"
    ) -> None:
        """
        Export position changes to an external system.

        Args:
            time: Timestamp when the leverage change occurred
            instrument: The instrument for which the leverage changed
            price: Price at which the leverage changed
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        pass


class IAccountViewer:
    """Interface for viewing account information.

    If a method accepts an exchange parameter, it means that the account information is specific to the exchange.
    If not exchange is provided, it means the account information will be provided from the first exchange in the list.
    """

    account_id: str

    def get_base_currency(self, exchange: str | None = None) -> str:
        """Get the base currency for the account.

        Returns:
            str: The base currency.
        """
        ...

    ########################################################
    # Capital information
    ########################################################
    def get_capital(self, exchange: str | None = None) -> float:
        """Get the available free capital in the account.

        Returns:
            float: The amount of free capital available for trading
        """
        ...

    def get_total_capital(self, exchange: str | None = None) -> float:
        """Get the total capital in the account including positions value.

        Returns:
            float: Total account capital
        """
        ...

    ########################################################
    # Balance and position information
    ########################################################
    def get_balances(self, exchange: str | None = None) -> list[AssetBalance]:
        """Get all currency balances.

        Returns:
            list[AssetBalance]: List of AssetBalance objects (each knows its exchange and currency)
        """
        ...

    def get_balance(self, currency: str, exchange: str | None = None) -> AssetBalance:
        """Get a specific currency balance.

        Args:
            currency: The currency to get the balance for
            exchange: The exchange to get the balance for

        Returns:
            AssetBalance: The AssetBalance object
        """
        ...

    def get_positions(self, exchange: str | None = None) -> dict[Instrument, Position]:
        """Get all current positions.

        Returns:
            dict[Instrument, Position]: Dictionary mapping instruments to their positions
        """
        ...

    def get_position(self, instrument: Instrument) -> Position:
        """Get the current position for a specific instrument.

        Args:
            instrument: The instrument to get the position for

        Returns:
            Position: The position object
        """
        ...

    def get_fees_calculator(self, exchange: str | None = None) -> TransactionCostsCalculator:
        """Get the fees calculator.

        Args:
            exchange: The exchange to get the fees calculator for

        Returns:
            TransactionCostsCalculator: The transaction costs calculator
        """
        ...

    @property
    def positions(self) -> dict[Instrument, Position]:
        """[Deprecated: Use get_positions()] Get all current positions.

        Returns:
            dict[Instrument, Position]: Dictionary mapping instruments to their positions
        """
        return self.get_positions()

    def get_orders(self, instrument: Instrument | None = None, exchange: str | None = None) -> dict[str, Order]:
        """Get active orders, optionally filtered by instrument and/or exchange.

        Args:
            instrument: Optional instrument to filter orders by
            exchange: Optional exchange to filter orders by

        Returns:
            dict[str, Order]: Dictionary mapping order IDs to Order objects
        """
        ...

    def find_order_by_id(self, order_id: str) -> Order | None:
        """Find an order by its ID.

        Args:
            order_id: The ID of the order to find

        Returns:
            Order | None: The order object if found, None otherwise
        """
        ...

    def position_report(self, exchange: str | None = None) -> dict:
        """Get detailed report of all positions.

        Returns:
            dict: Dictionary containing position details including quantities, prices, PnL etc.
        """
        ...

    ########################################################
    # Leverage information
    ########################################################
    def get_leverage(self, instrument: Instrument) -> float:
        """Get the leverage used for a specific instrument.

        Args:
            instrument: The instrument to check

        Returns:
            float: Current leverage ratio for the instrument
        """
        ...

    def get_leverages(self, exchange: str | None = None) -> dict[Instrument, float]:
        """Get leverages for all instruments.

        Returns:
            dict[Instrument, float]: Dictionary mapping instruments to their leverage ratios
        """
        ...

    def get_net_leverage(self, exchange: str | None = None) -> float:
        """Get the net leverage across all positions.

        Returns:
            float: Net leverage ratio
        """
        ...

    def get_gross_leverage(self, exchange: str | None = None) -> float:
        """Get the gross leverage across all positions.

        Returns:
            float: Gross leverage ratio
        """
        ...

    ########################################################
    # Margin information
    # Used for margin, swap, futures, options trading
    ########################################################
    def get_total_required_margin(self, exchange: str | None = None) -> float:
        """Get total margin required for all positions.

        Returns:
            float: Total required margin
        """
        ...

    def get_available_margin(self, exchange: str | None = None) -> float:
        """Get available margin for new positions.

        Returns:
            float: Available margin
        """
        ...

    def get_margin_ratio(self, exchange: str | None = None) -> float:
        """Get current margin ratio.

        Formula: (total capital + positions value) / total required margin

        Example:
            If total capital is 1000, positions value is 2000, and total required margin is 3000,
            the margin ratio would be (1000 + 2000) / 3000 = 1.0

        Returns:
            float: Current margin ratio
        """
        ...

    def get_reserved(self, instrument: Instrument) -> float:
        """[Deprecated] Get reserved margin for a specific instrument.

        Args:
            instrument: The instrument to check

        Returns:
            float: Reserved margin for the instrument
        """
        return 0.0


class IExchangeExtensions:
    """
    Base class for exchange-specific API extensions.

    Provides automatic method discovery using Python's reflection capabilities.
    All public methods (not starting with '_') are automatically discoverable
    with their signatures and docstrings.
    """

    def list_methods(self) -> dict[str, str]:
        """
        List all available extension methods with brief descriptions.

        Returns:
            dict[str, str]: Dictionary mapping method names to their first-line descriptions
        """
        methods = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            # Skip private methods and the reflection methods themselves
            if name.startswith("_") or name in ["list_methods", "call_method", "get_method_signature", "help"]:
                continue

            # Get first line of docstring as description
            doc = inspect.getdoc(method)
            description = doc.split("\n")[0] if doc else "No description"
            methods[name] = description

        return methods

    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Dynamically call an extension method by name.

        Args:
            method_name: Name of the method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method

        Returns:
            Any: The result of the method call

        Raises:
            AttributeError: If the method doesn't exist
            TypeError: If invalid arguments are provided
        """
        if not hasattr(self, method_name):
            available = ", ".join(self.list_methods().keys())
            raise AttributeError(
                f"Extension method '{method_name}' not found. Available methods: {available if available else 'none'}"
            )

        method = getattr(self, method_name)
        if not callable(method) or method_name.startswith("_"):
            raise AttributeError(f"'{method_name}' is not a callable extension method")

        return method(*args, **kwargs)

    def get_method_signature(self, method_name: str) -> dict[str, Any]:
        """
        Get detailed information about a method's signature and documentation.

        Args:
            method_name: Name of the method to inspect

        Returns:
            dict: Dictionary containing:
                - 'signature': String representation of the method signature
                - 'description': Full docstring of the method
                - 'parameters': List of parameter names and their annotations
                - 'return_type': Return type annotation (if available)

        Raises:
            AttributeError: If the method doesn't exist
        """
        if not hasattr(self, method_name):
            raise AttributeError(f"Extension method '{method_name}' not found")

        method = getattr(self, method_name)
        sig = inspect.signature(method)
        doc = inspect.getdoc(method)

        # Extract parameter information
        params = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_info = {
                "name": param_name,
                "annotation": str(param.annotation) if param.annotation != param.empty else None,
                "default": str(param.default) if param.default != param.empty else None,
            }

            params.append(param_info)

        return {
            "signature": str(sig),
            "description": doc if doc else "No documentation available",
            "parameters": params,
            "return_type": str(sig.return_annotation) if sig.return_annotation != sig.empty else None,
        }

    def help(self, method_name: str | None = None) -> str:
        """
        Get formatted help text for extension methods.

        Args:
            method_name: Optional name of specific method to get help for.
                        If None, lists all available methods.

        Returns:
            str: Formatted help text
        """
        if method_name is None:
            # List all methods
            methods = self.list_methods()
            if not methods:
                return "No extension methods available"

            lines = ["Available extension methods:"]
            for name, desc in sorted(methods.items()):
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        else:
            # Detailed help for specific method
            try:
                info = self.get_method_signature(method_name)
                lines = [f"Method: {method_name}{info['signature']}", "", info["description"]]
                return "\n".join(lines)
            except AttributeError as e:
                return str(e)


class EmptyExchangeExtensions(IExchangeExtensions):
    """
    Default empty implementation of exchange extensions.

    Used for exchanges that don't have specific extension methods.
    Ensures all brokers always have an extensions property without needing hasattr checks.
    """

    def __init__(self, exchange_name: str):
        """
        Initialize empty extensions.

        Args:
            exchange_name: Name of the exchange (for error messages)
        """
        self.exchange_name = exchange_name

    def list_methods(self) -> dict[str, str]:
        """Return empty dictionary - no methods available."""
        return {}

    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Raise NotImplementedError for any method call.

        Raises:
            NotImplementedError: Always, with message about the exchange not having extensions
        """
        raise NotImplementedError(f"Exchange '{self.exchange_name}' does not have extension methods")


class IBroker:
    """Broker provider interface for managing trading operations.

    Handles account operations, order placement, and position tracking.
    """

    channel: CtrlChannel

    @property
    def extensions(self) -> IExchangeExtensions:
        """
        Access exchange-specific API extensions.

        Returns EmptyExchangeExtensions by default. Specific broker implementations
        can override this property to provide their own extensions.

        Returns:
            IExchangeExtensions: Exchange-specific extensions (always present, never None)
        """
        return EmptyExchangeExtensions(self.exchange())

    @property
    def is_simulated_trading(self) -> bool:
        """
        Check if the broker is in simulation mode.
        """
        ...

    def send_order(self, request: OrderRequest) -> Order | None:
        """Submit order synchronously and wait for result.

        The broker MAY enrich request with exchange-specific metadata
        (e.g., lighter_client_order_index) and MAY mutate request.client_id
        for exchange-specific needs.

        Args:
            request: Order request to submit

        Returns:
            Order: The created order object

        Raises:
            Various exceptions based on order creation errors
        """
        raise NotImplementedError("send_order is not implemented")

    def send_order_async(self, request: OrderRequest) -> None:
        """Submit order asynchronously.

        The broker MAY enrich request.options with exchange-specific metadata
        (e.g., lighter_client_order_index) before submitting. The broker MUST NOT
        mutate request.client_id.

        Order confirmation arrives via channel with status "NEW" or "OPEN".
        request.client_id is used for matching incoming orders.

        Args:
            request: Order request to submit. The broker can add exchange-specific
                    metadata to request.options but must preserve client_id.

        Returns:
            None: Order updates arrive asynchronously via the channel.
        """
        raise NotImplementedError("send_order_async is not implemented")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order synchronously.

        Args:
            order_id: The ID of the order to cancel.

        Returns:
            bool: True if cancellation was successful, False otherwise.
        """
        raise NotImplementedError("cancel_order is not implemented")

    def cancel_order_async(self, order_id: str) -> None:
        """Cancel an existing order asynchronously (non blocking).

        Args:
            order_id: The ID of the order to cancel.
        """
        raise NotImplementedError("cancel_order_async is not implemented")

    def cancel_orders(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument.

        Args:
            instrument: The instrument to cancel orders for.
        """
        raise NotImplementedError("cancel_orders is not implemented")

    def update_order(self, order_id: str, price: float, amount: float) -> Order:
        """Update an existing order with new price and amount.

        Args:
            order_id: The ID of the order to update.
            price: New price for the order.
            amount: New amount for the order.

        Returns:
            Order: The updated Order object if successful

        Raises:
            NotImplementedError: If the method is not implemented
            OrderNotFound: If the order is not found
            BadRequest: If the request is invalid (e.g., not a limit order)
            InvalidOrderParameters: If the order cannot be updated
        """
        raise NotImplementedError("update_order is not implemented")

    def make_client_id(self, client_id: str) -> str:
        """
        Generate a valid client_id for the broker if the provided client_id is not valid.

        Args:
            client_id: The client_id to make valid

        Returns:
            str: The valid client_id
        """
        return client_id

    def exchange(self) -> str:
        """
        Return the name of the exchange this broker is connected to.
        """
        raise NotImplementedError("exchange() is not implemented")


class IDataProvider:
    time_provider: ITimeProvider
    channel: CtrlChannel

    def subscribe(
        self,
        subscription_type: str,
        instruments: set[Instrument],
        reset: bool = False,
    ) -> None:
        """
        Subscribe to market data for a list of instruments.

        Args:
            subscription_type: Type of subscription
            instruments: Set of instruments to subscribe to
            reset: Reset existing instruments for the subscription type. Default is False.
        """
        ...

    def unsubscribe(self, subscription_type: str | None, instruments: set[Instrument]) -> None:
        """
        Unsubscribe from market data for a list of instruments.

        Args:
            subscription_type: Type of subscription to unsubscribe from (optional)
            instruments: Set of instruments to unsubscribe from
        """
        ...

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """
        Check if an instrument has a subscription.

        Args:
            instrument: Instrument to check
            subscription_type: Type of subscription to check

        Returns:
            bool: True if instrument has the subscription
        """
        ...

    def get_subscriptions(self, instrument: Instrument | None = None) -> list[str]:
        """
        Get all subscriptions for an instrument.

        Args:
            instrument (optional): Instrument to get subscriptions for. If None, all subscriptions are returned.

        Returns:
            list[str]: list of subscriptions
        """
        ...

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        """
        Get a list of instruments that are subscribed to a specific subscription type.

        Args:
            subscription_type: Type of subscription to filter by (optional)

        Returns:
            list[Instrument]: list of subscribed instruments
        """
        ...

    def warmup(self, configs: dict[tuple[str, Instrument], str]) -> None:
        """
        Run warmup for subscriptions.

        Args:
            configs: Dictionary of (subscription type, instrument) pairs and warmup periods.

        Example:
            warmup({
                (DataType.OHLC["1h"], instr1): "30d",
                (DataType.OHLC["1Min"], instr1): "6h",
                (DataType.OHLC["1Sec"], instr2): "5Min",
                (DataType.TRADE, instr2): "1h",
            })
        """
        ...

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> list[Bar]:
        """
        Get historical OHLC data for an instrument.
        """
        ...

    def get_quote(self, instrument: Instrument) -> Quote | None:
        """
        Get the latest quote for an instrument.
        """
        ...

    @property
    def is_simulation(self) -> bool:
        """
        Check if data provider is in simulation mode.
        """
        ...

    def close(self):
        """
        Close the data provider.
        """
        ...

    def exchange(self) -> str:
        """
        Return the name of the exchange this provider reads data
        """
        raise NotImplementedError("exchange() is not implemented")

    def is_connected(self) -> bool:
        """
        Check if the data provider is currently connected to the exchange.

        Returns:
            bool: True if connected, False otherwise
        """
        ...


class IMarketManager(ITimeProvider):
    """Interface for market data providing class"""

    def ohlc(self, instrument: Instrument, timeframe: str | td_64 | None = None, length: int | None = None) -> OHLCV:
        """Get OHLCV data for an instrument. If length is larger then available cached data, it will be requested from the broker.

        Args:
            instrument: The instrument to get data for
            timeframe (optional): The timeframe of the data. If None, the default timeframe is used.
            length (optional): Number of bars to retrieve. If None, full cached data is returned.

        Returns:
            OHLCV: The OHLCV data series
        """
        ...

    def ohlc_pd(
        self,
        instrument: Instrument,
        timeframe: str | td_64 | None = None,
        length: int | None = None,
        consolidated: bool = True,
    ) -> pd.DataFrame:
        """Get OHLCV data for an instrument as pandas DataFrame.

        Args:
            instrument: The instrument to get data for
            timeframe (optional): The timeframe of the data. If None, the default timeframe is used.
            length (optional): Number of bars to retrieve. If None, full cached data is returned.
            consolidated (optional): If True, only finished bars are returned.

        Returns:
            pd.DataFrame: The OHLCV data as pandas DataFrame
        """
        ...

    def quote(self, instrument: Instrument) -> Quote | None:
        """Get latest quote for an instrument.

        Args:
            instrument: The instrument to get quote for

        Returns:
            Quote | None: The latest quote or None if not available
        """
        ...

    def get_data(self, instrument: Instrument, sub_type: str) -> list[Any]:
        """Get data for an instrument. This method is used for getting data for custom subscription types.
        Could be used for orderbook, trades, liquidations, funding rates, etc.

        Args:
            instrument: The instrument to get data for
            sub_type: The subscription type of data to get

        Returns:
            list[Any]: The data
        """
        ...

    def get_aux_data(self, data_id: str, **parametes) -> pd.DataFrame | None:
        """Get auxiliary data by ID.

        Args:
            data_id: Identifier for the auxiliary data
            **parametes: Additional parameters for the data request

        Returns:
            pd.DataFrame | None: The auxiliary data or None if not found
        """
        ...

    def get_instruments(self) -> list[Instrument]:
        """Get list of subscribed instruments.

        Returns:
            list[Instrument]: list of subscribed instruments
        """
        ...

    def query_instrument(self, symbol: str, exchange: str | None = None) -> Instrument:
        """Query instrument in lookup by symbol and exchange.

        Args:
            symbol: The symbol to look up
            exchange: The exchange to look up or None (current exchange is used)

        Returns:
            Instrument: The instrument

        Raises:
            SymbolNotFound: If the instrument cannot be found
        """
        ...

    def exchanges(self) -> list[str]: ...


class ITradingManager:
    """Manages order operations."""

    def trade(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        client_id: str | None = None,
        **options,
    ) -> Order | None:
        """Place a trade order.

        Args:
            instrument: The instrument to trade
            amount: Amount to trade (positive for buy, negative for sell)
            price: Optional limit price
            time_in_force: Time in force for the order
            client_id: Client ID for the order
            **options: Additional order options

        Returns:
            Order: The created order
        """
        ...

    def trade_async(
        self,
        instrument: Instrument,
        amount: float,
        price: float | None = None,
        time_in_force="gtc",
        client_id: str | None = None,
        **options,
    ) -> None:
        """Place a trade order asynchronously.

        Args:
            instrument: The instrument to trade
            amount: Amount to trade (positive for buy, negative for sell)
            price: Optional limit price
            time_in_force: Time in force for the order
            client_id: Client ID for the order
            **options: Additional order options
        """
        ...

    def submit_orders(self, order_requests: list[OrderRequest]) -> list[Order]:
        """Submit multiple orders to the exchange."""
        ...

    def set_target_position(
        self, instrument: Instrument, target: float, price: float | None = None, **options
    ) -> Order:
        """Set target position for an instrument.

        Args:
            instrument: The instrument to set target position for
            target: Target position size
            price: Optional limit price
            time_in_force: Time in force for the order
            **options: Additional order options

        Returns:
            Order: The created order
        """
        ...

    def set_target_leverage(
        self, instrument: Instrument, leverage: float, price: float | None = None, **options
    ) -> None:
        """Set target leverage for an instrument.

        Args:
            instrument: The instrument to set target leverage for
            leverage: The target leverage
            price: Optional limit price
            **options: Additional order options
        """
        ...

    def close_position(self, instrument: Instrument, without_signals: bool = False) -> None:
        """Close position for an instrument.

        Args:
            instrument: The instrument to close position for
            without_signals: If True, trade submitted instead of emitting signal
        """
        ...

    def close_positions(self, market_type: MarketType | None = None, without_signals: bool = False) -> None:
        """Close all positions."""
        ...

    def cancel_order(self, order_id: str, exchange: str | None = None) -> bool:
        """Cancel a specific order synchronously.

        Args:
            order_id: ID of the order to cancel
            exchange: Exchange to cancel on (optional)

        Returns:
            bool: True if cancellation was successful, False otherwise.
        """
        ...

    def cancel_order_async(self, order_id: str, exchange: str | None = None) -> None:
        """Cancel a specific order asynchronously (non blocking).

        Args:
            order_id: ID of the order to cancel
            exchange: Exchange to cancel on (optional)
        """
        ...

    def cancel_orders(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument.

        Args:
            instrument: The instrument to cancel orders for
        """
        ...

    def update_order(self, order_id: str, price: float, amount: float, exchange: str | None = None) -> Order:
        """Update an existing limit order with new price and amount.

        Args:
            order_id: ID of the order to update
            price: New price for the order
            amount: New amount for the order
            exchange: Exchange to update on (optional, defaults to first exchange)

        Returns:
            Order: The updated order object

        Raises:
            OrderNotFound: If the order is not found
            BadRequest: If the order is not a limit order or other validation errors
            InvalidOrderParameters: If the update parameters are invalid
        """
        ...

    def get_min_size(self, instrument: Instrument, amount: float | None = None) -> float:
        """Get the minimum size for an instrument.

        Args:
            instrument: The instrument to get the minimum size for
            amount: The amount to be traded to determine if it's position reducing or not
        """
        ...

    def exchanges(self) -> list[str]: ...


class IUniverseManager:
    """Manages universe updates."""

    def set_universe(
        self, instruments: list[Instrument], skip_callback: bool = False, if_has_position_then: RemovalPolicy = "close"
    ):
        """Set the trading universe.

        Args:
            instruments: list of instruments in the universe
            skip_callback: skip callback to the strategy
            if_has_position_then: what to do if the instrument has a position
                - "close" (default) - close position immediatelly and remove (unsubscribe) instrument from strategy
                - "wait_for_close" - keep instrument and it's position until it's closed from strategy (or risk management), then remove instrument from strategy
                - "wait_for_change" - keep instrument and position until strategy would try to change it - then close position and remove instrument
        """
        ...

    def add_instruments(self, instruments: list[Instrument]):
        """Add instruments to the trading universe.

        Args:
            instruments: List of instruments to add
        """
        ...

    def remove_instruments(self, instruments: list[Instrument], if_has_position_then: RemovalPolicy = "close"):
        """Remove instruments from the trading universe.

        Args:
            instruments: List of instruments to remove
            if_has_position_then: What to do if the instrument has a position
                - "close" (default) - close position immediatelly and remove (unsubscribe) instrument from strategy
                - "wait_for_close" - keep instrument and it's position until it's closed from strategy (or risk management), then remove instrument from strategy
                - "wait_for_change" - keep instrument and position until strategy would try to change it - then close position and remove instrument
        """
        ...

    @property
    def instruments(self) -> list[Instrument]:
        """
        Get the list of instruments in the universe.
        """
        ...

    def on_alter_position(self, instrument: Instrument) -> None:
        """
        Called when the position of an instrument changes.
        It can be used for postponed unsubscribed events
        """
        ...

    def is_trading_allowed(self, instrument: Instrument) -> bool:
        """
        Check if trading is allowed for an instrument because of the instrument's trading policy.
        """
        ...


class ISubscriptionManager:
    """Manages subscriptions."""

    def subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        """Subscribe to market data for an instrument.

        Args:
            subscription_type: Type of subscription. If None, the base subscription type is used.
            instruments: A list of instrument of instrument to subscribe to
        """
        ...

    def unsubscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        """Unsubscribe from market data for an instrument.

        Args:
            subscription_type: Type of subscription to unsubscribe from (e.g. DataType.OHLC)
            instruments (optional): A list of instruments or instrument to unsubscribe from.
        """
        ...

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """Check if subscription exists.

        Args:
            subscription_type: Type of subscription
            instrument: Instrument to check

        Returns:
            bool: True if subscription exists
        """
        ...

    def get_base_subscription(self) -> str:
        """
        Get the main subscription which should be used for the simulation.
        This data is used for updating the internal OHLCV data series.
        By default, simulation uses 1h OHLCV bars and live trading uses orderbook data.
        """
        ...

    def set_base_subscription(self, subscription_type: str) -> None:
        """
        Set the main subscription which should be used for the simulation.

        Args:
            subscription_type: Type of subscription (e.g. DataType.OHLC, DataType.OHLC["1h"])
        """
        ...

    def get_subscriptions(self, instrument: Instrument | None = None) -> list[str]:
        """
        Get all subscriptions for an instrument.

        Args:
            instrument: Instrument to get subscriptions for (optional)

        Returns:
            list[str]: list of subscriptions
        """
        ...

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        """
        Get a list of instruments that are subscribed to a specific subscription type.

        Args:
            subscription_type: Type of subscription to filter by (optional)

        Returns:
            list[Instrument]: list of subscribed instruments
        """
        ...

    def get_warmup(self, subscription_type: str) -> str | None:
        """
        Get the warmup period for a subscription type.

        Args:
            subscription_type: Type of subscription (e.g. DataType.OHLC["1h"], etc.)

        Returns:
            str: Warmup period or None if no warmup period is set
        """
        ...

    def set_warmup(self, configs: dict[Any, str]) -> None:
        """
        Set the warmup period for different subscriptions.

        If there are multiple ohlc configs specified, they will be warmed up in parallel.

        Args:
            configs: Dictionary of subscription types and warmup periods.
                     Keys can be subscription types of dictionaries with subscription parameters.

        Example:
            set_warmup({
                DataType.OHLC["1h"]: "30d",
                DataType.OHLC["1Min"]: "6h",
                DataType.OHLC["1Sec"]: "5Min",
                DataType.TRADE: "1h",
            })
        """
        ...

    def commit(self) -> None:
        """
        Apply all pending changes.
        """
        ...

    @property
    def auto_subscribe(self) -> bool:
        """
        Get whether new instruments are automatically subscribed to existing subscriptions.

        Returns:
            bool: True if auto-subscription is enabled
        """
        ...

    @auto_subscribe.setter
    def auto_subscribe(self, value: bool) -> None:
        """
        Enable or disable automatic subscription of new instruments.

        Args:
            value: True to enable auto-subscription, False to disable
        """
        ...


class IAccountProcessor(IAccountViewer):
    time_provider: ITimeProvider

    def start(self):
        """
        Start the account processor.
        """
        ...

    def stop(self):
        """
        Stop the account processor.
        """
        ...

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        """Set the subscription manager for the account processor.

        Args:
            manager: ISubscriptionManager instance to set
        """
        ...

    def update_balance(self, currency: str, total: float, locked: float, exchange: str | None = None):
        """Update balance for a specific currency.

        Args:
            currency: Currency code
            total: Total amount of currency
            locked: Amount of locked currency
        """
        ...

    def update_position_price(self, time: dt_64, instrument: Instrument, update: float | Timestamped) -> None:
        """Update position price for an instrument.

        Args:
            time: Timestamp of the update
            instrument: Instrument being updated
            price: New price
        """
        ...

    def process_market_data(self, time: dt_64, instrument: Instrument, update: Timestamped) -> None:
        """Process market data for an instrument.

        Args:
            time: Timestamp of the update
            instrument: Instrument the data is for
            update: The data to process
        """
        ...

    def process_deals(self, instrument: Instrument, deals: list[Deal]) -> None:
        """Process executed deals for an instrument.

        Args:
            instrument: Instrument the deals belong to
            deals: List of deals to process
        """
        ...

    def process_funding_payment(self, instrument: Instrument, funding_payment: FundingPayment) -> None:
        """Process funding payment for an instrument.

        Args:
            instrument: Instrument the funding payment applies to
            funding_payment: Funding payment event to process
        """
        ...

    def process_order(self, order: Order, *args) -> None:
        """Process order updates.

        Args:
            order: Order to process
            *args: Additional arguments that may be needed by specific implementations
        """
        ...

    def process_order_request(self, request: OrderRequest) -> None:
        """Process an order request (async submission).

        Tracks pending order requests until exchange confirms with Order update.
        The broker enriches the request with exchange-specific metadata before
        this method is called.

        Args:
            request: Order request with client_id for tracking and optional
                    exchange-specific metadata in request.options
        """
        ...

    def attach_positions(self, *position: Position) -> "IAccountProcessor":
        """Attach positions to the account.

        Args:
            *position: Position objects to attach

        Returns:
            I"IAccountProcessor": Self for chaining
        """
        ...

    def add_active_orders(self, orders: dict[str, Order]) -> None:
        """Add active orders to the account.

        Warning only use in the beginning for state restoration because it does not update locked balances.

        Updates:
            - 2025-03-20: It is used now to track internally active orders, so that we can cancel the rest.

        Args:
            orders: Dictionary mapping order IDs to Order objects
        """
        ...

    def remove_order(self, order_id: str, exchange: str | None = None) -> None:
        """
        Remove an order from the account.

        Args:
            order_id: ID of the order to remove
        """
        ...


class ITransferManager:
    """Manages transfer operations between exchanges."""

    def transfer_funds(self, from_exchange: str, to_exchange: str, currency: str, amount: float) -> str:
        """
        Transfer funds between exchanges.

        Args:
            from_exchange: Source exchange identifier
            to_exchange: Destination exchange identifier
            currency: Currency to transfer
            amount: Amount to transfer

        Returns:
            str: Transaction ID
        """
        ...

    def get_transfer_status(self, transaction_id: str) -> dict[str, Any]:
        """
        Get the status of a transfer.

        Args:
            transaction_id: Transaction ID

        Returns:
            dict[str, Any]: Transfer status
        """
        ...

    def get_transfers(self) -> dict[str, dict[str, Any]]:
        """
        Get all transfers.
        """
        ...


class IProcessingManager:
    """Manages event processing."""

    def process_data(self, instrument: Instrument, d_type: str, data: Any, is_historical: bool) -> bool:
        """
        Process incoming data.

        Args:
            instrument: Instrument the data is for
            d_type: Type of the data
            data: The data to process

        Returns:
            bool: True if processing should be halted
        """
        ...

    def set_fit_schedule(self, schedule: str) -> None:
        """
        Set the schedule for fitting the strategy model (default is to trigger fit only at start).
        """
        ...

    def set_event_schedule(self, schedule: str) -> None:
        """
        Set the schedule for triggering events (default is to only trigger on data events).
        Replaces any existing event schedule.
        """
        ...

    def get_event_schedule(self, event_id: str) -> str | None:
        """
        Get defined schedule for event id.
        """
        ...

    def is_fitted(self) -> bool:
        """
        Check if the strategy is fitted.
        """
        ...

    def get_active_targets(self) -> dict[Instrument, TargetPosition]:
        """
        Get active target positions for each instrument in the universe.
        Target position (TP) is considered active if
            1. signal (S) is sent, converted to a TP, and position is open
            2. S is sent, converted to a TP, and limit order is sent for opening

        So when position is closed TP (because of opposite signal or stop loss/take profit) becomes inactive.

        Returns:
            dict[Instrument, TargetPosition]: Dictionary mapping instruments to their active targets.
        """
        ...

    def emit_signal(self, signal: Signal | list[Signal]) -> None:
        """
        Emit a signal for processing
        """
        ...

    def schedule(self, cron_schedule: str, method: Callable[["IStrategyContext"], None]) -> str:
        """
        Register a custom method to be called at specified times.

        Args:
            cron_schedule: Cron-like schedule string (e.g., "0 0 * * *" for daily at midnight)
            method: Method to call when schedule triggers
        """
        ...

    def unschedule(self, event_id: str) -> bool:
        """
        Unschedule a scheduled event.

        Args:
            event_id: ID of the event to unschedule

        Returns:
            bool: True if event was found and unscheduled, False otherwise
        """
        ...

    def configure_stale_data_detection(
        self, enabled: bool, detection_period: str | None = None, check_interval: str | None = None
    ) -> None:
        """
        Configure stale data detection settings.

        Args:
            enabled: Whether to enable stale data detection
            detection_period: Period to consider data as stale (e.g., "5Min", "1h"). If None, uses detector default.
            check_interval: Interval between stale data checks (e.g., "30s", "1Min"). If None, uses detector default.
        """
        ...


class IWarmupStateSaver:
    """
    Interface for saving warmup state. This is used for state restoration after warmup.
    """

    def set_warmup_positions(self, positions: dict[Instrument, Position]) -> None:
        """Set warmup positions."""
        ...

    def set_warmup_orders(self, orders: dict[Instrument, list[Order]]) -> None:
        """Set warmup orders."""
        ...

    def get_warmup_positions(self) -> dict[Instrument, Position]:
        """Get warmup positions."""
        ...

    def get_warmup_orders(self) -> dict[Instrument, list[Order]]:
        """Get warmup orders."""
        ...

    def set_warmup_active_targets(self, active_targets: dict[Instrument, TargetPosition]) -> None:
        """Set warmup active targets."""
        ...

    def get_warmup_active_targets(self) -> dict[Instrument, TargetPosition]:
        """Get warmup active targets."""
        ...


@dataclass
class StrategyState:
    is_on_init_called: bool = False
    is_on_start_called: bool = False
    is_on_warmup_finished_called: bool = False
    is_on_fit_called: bool = False
    is_warmup_in_progress: bool = False

    def reset_from_state(self, state: "StrategyState"):
        self.is_on_init_called = state.is_on_init_called
        self.is_on_start_called = state.is_on_start_called
        self.is_on_warmup_finished_called = state.is_on_warmup_finished_called
        self.is_on_fit_called = state.is_on_fit_called
        self.is_warmup_in_progress = state.is_warmup_in_progress


class IStrategyContext(
    IMarketManager,
    ITradingManager,
    IUniverseManager,
    ISubscriptionManager,
    IProcessingManager,
    IAccountViewer,
    IWarmupStateSaver,
    ITransferManager,
):
    strategy: "IStrategy"
    initializer: "IStrategyInitializer"
    account: IAccountProcessor
    emitter: "IMetricEmitter"
    health: "IHealthReader"
    notifier: "IStrategyNotifier"
    strategy_name: str
    aux: "DataReader | None"

    _strategy_state: StrategyState

    def start(self, blocking: bool = False):
        """Start the strategy context."""
        pass

    def stop(self):
        """Stop the strategy context."""
        pass

    @property
    def state(self) -> StrategyState:
        """Get the strategy state."""
        return StrategyState(**self._strategy_state.__dict__)

    def is_running(self) -> bool:
        """Check if the strategy context is running."""
        return False

    @property
    def is_warmup_in_progress(self) -> bool:
        """Check if the warmup is in progress."""
        return self._strategy_state.is_warmup_in_progress

    @property
    def is_simulation(self) -> bool:
        """Check if the strategy context is running in simulation mode."""
        return False

    @property
    def is_live_or_warmup(self) -> bool:
        """Check if the strategy context is running in live or warmup mode."""
        return not self.is_simulation or self.is_warmup_in_progress

    @property
    def is_paper_trading(self) -> bool:
        """Check if the strategy context is running in simulated trading mode."""
        return False

    @property
    def exchanges(self) -> list[str]:
        """Get the list of exchanges."""
        return []

    def get_restored_state(self) -> "RestoredState | None":
        """Get the restored state."""
        return None


class IPositionGathering:
    """
    Common interface for position gathering
    """

    def alter_position_size(self, ctx: IStrategyContext, target: TargetPosition) -> float: ...

    def alter_positions(
        self, ctx: IStrategyContext, targets: list[TargetPosition] | TargetPosition
    ) -> dict[Instrument, float]:
        if not isinstance(targets, list):
            targets = [targets]

        res = {}
        if targets:
            for t in targets:
                try:
                    res[t.instrument] = self.alter_position_size(ctx, t)
                except Exception as ex:
                    logger.error(f"[{ctx.time()}]: Failed processing target position {t} : {ex}")
                    logger.opt(colors=False).error(traceback.format_exc())
        return res

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal): ...

    def update(self, ctx: IStrategyContext, instrument: Instrument, update: Timestamped) -> None:
        """
        Position gatherer is being updated by new market data.

        Args:
            ctx: Strategy context object
            instrument: The instrument for which market data was updated
            update: The market data update (Quote, Trade, Bar, etc.)
        """
        pass

    def restore_from_target_positions(self, ctx: IStrategyContext, target_positions: list[TargetPosition]) -> None:
        """
        Restore gatherer state from target positions.

        Args:
            ctx: Strategy context object
            target_positions: List of target positions to restore gatherer state from
        """
        # Default implementation - subclasses can override if needed
        pass


class IPositionSizer:
    """Interface for calculating target positions from signals."""

    def calculate_target_positions(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        """Calculates target position sizes.

        Args:
            ctx: Strategy context object.
            signals: List of signals to process.

        Returns:
            List of target positions.
        """
        raise NotImplementedError("calculate_target_positions is not implemented")

    def get_signal_entry_price(
        self, ctx: IStrategyContext, signal: Signal, use_mid_price: bool = False
    ) -> float | None:
        """
        Get the entry price for a signal.
        """
        _entry = None
        if signal.price is not None and signal.price > 0:
            _entry = signal.price
        else:
            if (_q := ctx.quote(signal.instrument)) is not None:
                _entry = _q.mid_price() if use_mid_price else (_q.ask if np.sign(signal.signal) > 0 else _q.bid)
            else:
                logger.error(
                    f"{self.__class__.__name__}: Can't get actual market quote for {signal.instrument} and signal price is not set ({str(signal)}) !"
                )

        return _entry


class PositionsTracker:
    """
    Process signals from strategy and track position. It can contains logic for risk management for example.
    """

    _sizer: IPositionSizer

    def __init__(self, sizer: IPositionSizer) -> None:
        self._sizer = sizer

    def get_position_sizer(self) -> IPositionSizer:
        return self._sizer

    def is_active(self, instrument: Instrument) -> bool:
        return True

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition] | TargetPosition:
        """
        Default implementation just returns calculated target positions
        """
        return self.get_position_sizer().calculate_target_positions(ctx, signals)

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Timestamped
    ) -> list[TargetPosition] | TargetPosition:
        """
        Tracker is being updated by new market data.
        It may require to change position size or create new position because of interior tracker's logic (risk management for example).
        """
        ...

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal):
        """
        Tracker is notified when execution report is received
        """
        ...

    def cancel_tracking(self, ctx: IStrategyContext, instrument: Instrument):
        """
        Cancel tracking for instrument from outside.

        There may be cases when we need to prematurely cancel tracking for instrument from the strategy.
        """
        ...

    def restore_position_from_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> None:
        """
        Restore tracker state from signals.

        Args:
            ctx: Strategy context object
            signals: List of signals to restore tracker state from
        """
        # Default implementation - subclasses can override
        pass


@dataclass
class LatencyMetrics:
    """
    Health metrics for system performance.

    All latency values are in milliseconds.
    """

    data_feed: float
    order_submit: float
    order_cancel: float


class IHealthWriter(Protocol):
    """
    Interface for recording health metrics.
    """

    def __call__(self, event_type: str) -> "IHealthWriter":
        """
        Support for context manager usage with event type.

        Args:
            event_type: Type of event being timed

        Returns:
            Self for use in 'with' statement
        """
        ...

    def __enter__(self) -> "IHealthWriter":
        """Enter context for timing measurement"""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and record timing"""
        ...

    def on_data_arrival(self, instrument: Instrument, event_type: str, event_time: dt_64) -> None:
        """
        Record a data arrival time.

        Args:
            instrument: Instrument that data is arrived for
            event_type: Type of event (e.g., "orderbook", "trade")
            event_time: Time of event
        """
        ...

    def record_order_submit_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """
        Record a order submit request time.
        """
        ...

    def record_order_submit_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """
        Record a order submit response time.
        """
        ...

    def record_order_cancel_request(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """
        Record a order cancel request time.
        """
        ...

    def record_order_cancel_response(self, exchange: str, client_id: str, event_time: dt_64) -> None:
        """
        Record a order cancel response time.
        """
        ...

    def set_event_queue_size(self, size: int) -> None:
        """
        Set the current event queue size.

        Args:
            size: Current size of the event queue
        """
        ...

    def set_is_connected(self, exchange: str, is_connected: Callable[[], bool]) -> None:
        """
        Set the is connected callback for an exchange.

        Args:
            exchange: Exchange name
            is_connected: Callback function to check if exchange is connected
        """
        ...

    def watch(self, scope_name: str = "") -> Callable[[Callable], Callable]:
        """Decorator function to time a function execution.

        Args:
            scope_name: Name for the timing scope. If empty string is provided,
                       function's qualified name will be used.

        Returns:
            Decorator function that times the decorated function.
        """
        ...


class IHealthReader(Protocol):
    """
    Interface for reading health metrics about system performance.
    """

    def is_connected(self, exchange: str) -> bool:
        """
        Check if exchange is connected.
        """
        ...

    def get_last_event_time(self, instrument: Instrument, event_type: str) -> dt_64 | None:
        """
        Get the last event time for a specific event type.
        """
        ...

    def get_last_event_times_by_exchange(self, exchange: str) -> dict[str, dt_64]:
        """
        Get all last event times for a specific exchange.

        Args:
            exchange: Exchange name

        Returns:
            Dictionary with event types as keys and last event times as values
        """
        ...

    def is_stale(self, instrument: Instrument, event_type: str, stale_delta: str | td_64 | None = None) -> bool:
        """
        Check if the data is stale.
        """
        ...

    def get_event_frequency(self, instrument: Instrument, event_type: str) -> float:
        """
        Get the events per second for a specific event type.

        Args:
            event_type: Type of event to get frequency for

        Returns:
            Events per second
        """
        ...

    def get_queue_size(self) -> int:
        """
        Get the current event queue size.

        Returns:
            Number of events waiting to be processed
        """
        ...

    def get_data_latency(self, exchange: str, event_type: str, percentile: float = 90) -> float:
        """
        Get latency for a specific data type.

        Args:
            exchange: Exchange name
            event_type: Data type (e.g., "quote", "trade")
            percentile: Optional percentile (0-100) to retrieve (default: 90)

        Returns:
            Latency value in milliseconds
        """
        ...

    def get_data_latencies(self, exchange: str, percentile: float = 90) -> dict[str, float]:
        """
        Get all data latencies.

        Args:
            exchange: Exchange name
            percentile: Optional percentile (0-100) to retrieve (default: 90)

        Returns:
            Dictionary of data latencies with data types as keys and latency values in milliseconds as values
        """
        ...

    def get_order_submit_latency(self, exchange: str, percentile: float = 90) -> float:
        """
        Get order submit latency for an exchange.
        """
        ...

    def get_order_cancel_latency(self, exchange: str, percentile: float = 90) -> float:
        """
        Get order cancel latency for an exchange.
        """
        ...

    def get_execution_latency(self, scope: str, percentile: float = 90) -> float:
        """
        Get execution latency for a specific scope.
        """
        ...

    def get_execution_latencies(self, percentile: float = 90) -> dict[str, float]:
        """
        Get all execution latencies.

        Args:
            percentile: Optional percentile (0-100) to retrieve (default: 90)

        Returns:
            Dictionary of execution latencies with scope names as keys and latency values in milliseconds as values
        """
        ...

    def get_exchange_latencies(self, exchange: str, percentile: float = 90) -> LatencyMetrics:
        """
        Get metrics by exchange.

        Args:
            exchange: Exchange name

        Returns:
            HealthMetrics object for the exchange
        """
        ...


class IHealthMonitor(IHealthWriter, IHealthReader):
    """Interface for health metrics monitoring that combines writing and reading capabilities."""

    def start(self) -> None:
        """Start the health metrics monitor."""
        ...

    def stop(self) -> None:
        """Stop the health metrics monitor."""
        ...

    def subscribe(self, instrument: Instrument, event_type: str) -> None:
        """
        Register active subscription for health tracking.

        Args:
            instrument: The instrument being subscribed to
            event_type: The data type being subscribed to (e.g., 'ohlc', 'quote', 'orderbook')
        """
        ...

    def unsubscribe(self, instrument: Instrument, event_type: str) -> None:
        """
        Remove subscription and cleanup stored metrics.

        Args:
            instrument: The instrument being unsubscribed from
            event_type: The data type being unsubscribed from
        """
        ...


def _unpickle_instance(chain: tuple[type], state: dict):
    """
    chain is a tuple of the *original* classes, e.g. (A, B, C).
    Reconstruct a new ephemeral class that inherits from them.
    """
    name = "_".join(cls.__name__ for cls in chain)
    # Reverse the chain to respect the typical left-to-right MRO
    inst = type(name, chain[::-1], {"__module__": "__main__"})()
    inst.__dict__.update(state)
    return inst


class Mixable(type):
    """
    It's possible to create composite strategies dynamically by adding mixins with functionality.

    NewStrategy = (SignalGenerator + RiskManager + PositionGathering)
    NewStrategy(....) can be used in simulation or live trading.
    """

    def __add__(cls, other_cls):
        # If we already have a _composition, combine them;
        # else treat cls itself as the start of the chain
        cls_chain = getattr(cls, "__composition__", (cls,))
        other_chain = getattr(other_cls, "__composition__", (other_cls,))

        # Combine them into one chain. You can define your own order rules:
        new_chain = cls_chain + other_chain

        # Create ephemeral class
        name = "_".join(c.__name__ for c in new_chain)

        def __reduce__(self):
            # Just return the chain of *original real classes*
            return _unpickle_instance, (new_chain, self.__dict__)

        new_cls = type(
            name,
            new_chain[::-1],
            {"__module__": cls.__module__, "__composition__": new_chain, "__reduce__": __reduce__},
        )
        return new_cls


class StartTimeFinderProtocol(Protocol):
    """Protocol for start time finder functions used in strategy initialization."""

    def __call__(self, time: dt_64, state: RestoredState) -> dt_64:
        """
        Find the start time for a warmup simulation.

        Args:
            time (dt_64): The current time
            state (RestoredState): The restored state from a previous run

        Returns:
            dt_64: The start time for the warmup simulation
        """
        ...


class StateResolverProtocol(Protocol):
    """Protocol for position mismatch resolver functions used in strategy initialization."""

    def __call__(
        self,
        ctx: "IStrategyContext",
        sim_positions: dict[Instrument, Position],
        sim_orders: dict[Instrument, list[Order]],
        sim_active_targets: dict[Instrument, TargetPosition],
    ) -> None:
        """
        Resolve position mismatches between warmup simulation and live trading.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
            sim_active_targets (dict[Instrument, TargetPosition]): Active targets from the simulation
        """
        ...


class IStrategyInitializer:
    """
    Interface for strategy initialization.

    This interface provides methods for configuring various aspects of a strategy
    during initialization, including scheduling, warmup periods, and position
    mismatch resolution.
    """

    def set_base_subscription(self, subscription_type: str) -> None:
        """
        Set the main subscription which should be used for the simulation.

        Args:
            subscription_type: Type of subscription (e.g. DataType.OHLC, DataType.OHLC["1h"])
        """
        ...

    def get_base_subscription(self) -> str | None:
        """
        Get the main subscription which should be used for the simulation.
        """
        ...

    def set_base_live_subscription(self, subscription_type: str) -> None:
        """
        Set the main subscription which should be used for the live trading.
        """
        ...

    def get_base_live_subscription(self) -> str:
        """
        Get the main subscription which should be used for the live trading.
        """
        ...

    def set_auto_subscribe(self, value: bool) -> None:
        """
        Enable or disable automatic subscription of new instruments.

        Args:
            value: True to enable auto-subscription, False to disable
        """
        ...

    def get_auto_subscribe(self) -> bool | None:
        """
        Get whether new instruments are automatically subscribed to existing subscriptions.

        Returns:
            bool: True if auto-subscription is enabled
        """
        ...

    def set_fit_schedule(self, schedule: str) -> None:
        """
        Set the schedule for fitting the strategy model.

        Args:
            schedule (str): A crontab-like schedule string (e.g., "0 0 * * *" for daily at midnight)
                           or a pandas-compatible frequency string (e.g., "1d" for daily).
        """
        ...

    def get_fit_schedule(self) -> str | None:
        """
        Get the schedule for fitting the strategy model.
        """
        ...

    def set_event_schedule(self, schedule: str) -> None:
        """
        Set the schedule for triggering strategy events.

        Args:
            schedule (str): A crontab-like schedule string (e.g., "0 * * * *" for hourly)
                           or a pandas-compatible frequency string (e.g., "1h" for hourly).
        """
        ...

    def get_event_schedule(self) -> str | None:
        """
        Get the schedule for triggering strategy events.
        """
        ...

    def set_warmup(self, period: str, start_time_finder: StartTimeFinderProtocol | None = None) -> None:
        """
        Set the warmup period for the strategy.

        The warmup period is used to initialize the strategy's state before live trading
        by running a simulation for the specified period. This helps avoid cold-start problems
        where the strategy might make suboptimal decisions without historical context.

        Args:
            period (str): A pandas-compatible time period string (e.g., "14d" for 14 days).
            start_time_finder (StartTimeFinder, optional): A function that determines the
                    start time for the warmup simulation.  If None, the current time minus the
                    warmup period is used if there is no restored state. Otherwise, we
                    try to figure out a reasonable start time based on signals from the
                    restored state (defined in TimeFinder.LAST_SIGNAL).

        """
        ...

    def get_warmup(self) -> td_64 | None:
        """
        Get the warmup period for the strategy.
        """
        ...

    def set_start_time_finder(self, finder: StartTimeFinderProtocol) -> None:
        """
        Set the start time finder for the strategy.
        """
        ...

    def get_start_time_finder(self) -> StartTimeFinderProtocol | None:
        """
        Get the start time finder for the strategy.
        """
        ...

    def set_state_resolver(self, resolver: StateResolverProtocol) -> None:
        """
        Set the resolver for handling position mismatches between warmup and live trading.

        When transitioning from warmup simulation to live trading, there may be differences
        between the positions established during simulation and the actual positions in
        the live account. This resolver determines how to handle these mismatches.

        Args:
            resolver (PositionMismatchResolver): A function that resolves position mismatches
                    between simulation and live trading. By default, if position after warmup
                    is less than the reconstructed position, we reduce the position size to
                    the simulated position size. In case simulation position is greater than
                    the reconstructed position, we leave the position size as is without increasing it
                    (defined in StateResolver.REDUCE_ONLY).
        """
        ...

    def get_state_resolver(self) -> StateResolverProtocol | None:
        """
        Get the mismatch resolver for the strategy.
        """
        ...

    def set_config(self, key: str, value: Any) -> None:
        """
        Set an additional configuration value.

        This method allows storing arbitrary configuration values that might be
        needed during strategy initialization but are not covered by the standard
        methods.

        Args:
            key (str): The configuration key
            value (Any): The configuration value
        """
        ...

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get an additional configuration value.

        Args:
            key (str): The configuration key
            default (Any, optional): The default value to return if the key is not found

        Returns:
            Any: The configuration value or the default value if not found
        """
        ...

    @property
    def is_simulation(self) -> bool | None:
        """
        Check if the strategy is running in simulation mode. We need this in on_init stage.
        """
        ...

    def subscribe(self, subscription_type: str, instruments: list[Instrument] | Instrument | None = None) -> None:
        """Subscribe to market data for an instrument.

        Args:
            subscription_type: Type of subscription. If None, the base subscription type is used.
            instruments: A list of instrument of instrument to subscribe to
        """
        ...

    def get_pending_global_subscriptions(self) -> set[str]:
        """
        Get the pending global subscriptions.
        """
        ...

    def get_pending_instrument_subscriptions(self) -> dict[str, set[Instrument]]:
        """
        Get the pending instrument subscriptions.
        """
        ...

    def set_subscription_warmup(self, configs: dict[Any, str]) -> None:
        """
        Set the warmup period for the subscription.

        This method is used to set the warmup period for the subscription in on_init stage.
        Then this config is used to translate warmup configs to StrategyContext

        Args:
            configs: A dictionary of subscription types and their warmup periods.
        """
        ...

    def get_subscription_warmup(self) -> dict[Any, str]:
        """
        Get the warmup period for the subscription.
        """
        ...

    def set_data_cache_config(
        self, enabled: bool = True, prefetch_period: str = "1w", cache_size_mb: int = 1000
    ) -> None:
        """
        Configure CachedPrefetchReader for aux data readers.

        Args:
            enabled: Whether to enable data caching
            prefetch_period: Period to prefetch ahead (e.g., "1w", "2d")
            cache_size_mb: Maximum cache size in MB
        """
        ...

    def get_data_cache_config(self) -> dict[str, Any]:
        """
        Get CachedPrefetchReader configuration.

        Returns:
            Dictionary with cache configuration
        """
        ...

    def schedule(self, cron_schedule: str, method: Callable[["IStrategyContext"], None]) -> None:
        """
        Schedule a custom method to be called at specified times.

        Args:
            cron_schedule: Cron-like schedule string (e.g., "0 0 * * *" for daily at midnight)
            method: Method to call - should accept IStrategyContext as parameter
        """
        ...

    def get_custom_schedules(self) -> dict[str, tuple[str, Callable[["IStrategyContext"], None]]]:
        """
        Get all custom scheduled methods.

        Returns:
            Dictionary mapping schedule IDs to (cron_schedule, method) tuples
        """
        ...

    def set_stale_data_detection(
        self, enabled: bool, detection_period: str | None = None, check_interval: str | None = None
    ) -> None:
        """
        Configure stale data detection settings.

        Args:
            enabled: Whether to enable stale data detection
            detection_period: Period to consider data as stale (e.g., "5Min", "1h"). If None, uses default.
            check_interval: Interval between stale data checks (e.g., "30s", "1Min"). If None, uses default.
        """
        ...

    def get_stale_data_detection_config(self) -> tuple[bool, str | None, str | None]:
        """
        Get current stale data detection configuration.

        Returns:
            tuple: (enabled, detection_period, check_interval)
        """
        ...

    def set_delisting_check_days(self, days: int) -> None:
        """
        Set the number of days ahead to check for delisting.

        When an instrument has a delist_date set, this configuration determines how many
        days before the delisting date the framework should:
        1. Filter the instrument out from universe updates
        2. Close positions and remove the instrument during the daily delisting check

        Args:
            days: Number of days ahead to check for delisting (default: 1)
        """
        ...

    def get_delisting_check_days(self) -> int:
        """
        Get the number of days ahead to check for delisting.

        Returns:
            int: Number of days ahead to check for delisting
        """
        ...

    def set_transfer_manager(self, manager: ITransferManager) -> None:
        """
        Set the transfer manager for handling fund transfers between exchanges.

        This is typically used in live mode to inject a transfer service client.
        In simulation mode, a transfer manager is automatically assigned.

        Args:
            manager: Transfer manager implementation
        """
        ...

    def get_transfer_manager(self) -> ITransferManager | None:
        """
        Get the configured transfer manager.

        Returns:
            ITransferManager | None: The transfer manager if set, None otherwise
        """
        ...


class IStrategy(metaclass=Mixable):
    """Base class for trading strategies."""

    ctx: IStrategyContext

    def __init__(self, **kwargs) -> None:
        set_parameters_to_object(self, **kwargs)

    def on_init(self, initializer: IStrategyInitializer):
        """
        This method is called when strategy is initialized.
        It is useful for setting the base subscription and warmup periods via the subscription manager.
        """
        ...

    def on_start(self, ctx: IStrategyContext):
        """
        This method is called strategy is started. You can already use the market data provider.
        """
        pass

    def on_warmup_finished(self, ctx: IStrategyContext):
        """
        This method is called when the warmup period is finished.
        """
        pass

    def on_fit(self, ctx: IStrategyContext):
        """
        Called when it's time to fit the model.
        """
        return None

    def on_universe_change(
        self, ctx: IStrategyContext, add_instruments: list[Instrument], rm_instruments: list[Instrument]
    ) -> None:
        """
        This method is called when the trading universe is updated.
        """
        return None

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent) -> list[Signal] | Signal | None:
        """Called on strategy events.

        Args:
            ctx: Strategy context.
            event: Trigger event to process.

        Returns:
            List of signals, single signal, or None.
        """
        return None

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> list[Signal] | Signal | None:
        """
        Called when new market data is received.

        Args:
            ctx: Strategy context.
            data: The market data received.

        Returns:
            List of signals, single signal, or None.
        """
        return None

    def on_order_update(self, ctx: IStrategyContext, order: Order) -> list[Signal] | Signal | None:
        """
        Called when an order update is received.

        Args:
            ctx: Strategy context.
            order: The order update.
        """
        return None

    def on_deals(self, ctx: IStrategyContext, instrument: Instrument, deals: list[Deal]) -> None:
        """
        Called when deals are received.

        Args:
            ctx: Strategy context.
            deals: The deals.
        """
        return None

    def on_error(self, ctx: IStrategyContext, error: BaseErrorEvent) -> None:
        """
        Called when an error occurs.

        Args:
            ctx: Strategy context.
            error: The error.
        """
        ...

    def on_stop(self, ctx: IStrategyContext):
        pass

    def tracker(self, ctx: IStrategyContext) -> PositionsTracker | None:
        pass

    def gatherer(self, ctx: IStrategyContext) -> IPositionGathering | None:
        pass

    def get_dashboard_data(self, ctx: IStrategyContext) -> dict[str, Any] | None:
        """
        Provide custom data for the Textual UI dashboard.

        This method is called periodically when running in Textual mode to populate
        custom dashboard panels. Return a dictionary with any JSON-serializable data
        you want to display.

        Args:
            ctx: Strategy context

        Returns:
            Dictionary with custom dashboard data, or None if no custom data

        Example:
            return {
                "signal_strength": self.current_signal,
                "volatility_regime": "high" if self.vol > threshold else "low",
                "open_alerts": len(self.alerts),
            }
        """
        return None


class IMetricEmitter:
    """Interface for emitting metrics to external monitoring systems."""

    def emit(
        self,
        name: str,
        value: float,
        tags: dict[str, Any] | None = None,
        timestamp: dt_64 | None = None,
        instrument: Instrument | None = None,
    ) -> None:
        """
        Emit a metric.

        Args:
            name: Name of the metric
            value: Value of the metric
            tags: Optional dictionary of tags/labels for the metric
            timestamp: Optional timestamp for the metric (may be ignored by some implementations)
            instrument: Optional instrument associated with the metric. If provided, symbol and exchange
                      will be added to the tags.
        """
        pass

    def emit_strategy_stats(self, context: "IStrategyContext") -> None:
        """
        Emit standard strategy statistics.

        This method is called periodically to emit standard statistics about the strategy's
        state, such as total capital, leverage, position information, etc.

        Args:
            context: The strategy context to get statistics from
        """
        pass

    def notify(self, context: "IStrategyContext") -> None:
        """
        Notify the metric emitter of a time update.

        This method is called by the processing manager when time updates.
        Implementations should check if enough time has passed since the last emission
        and emit metrics if necessary.

        Args:
            context: The strategy context to get statistics from
        """
        pass

    def set_context(self, context: "IStrategyContext") -> None:
        """
        Set the strategy context for the metric emitter.

        This method is used to set the context that provides access to time and simulation state.
        The context is used to automatically add is_live tag and get timestamps when no explicit
        timestamp is provided in the emit method.

        Args:
            context: The strategy context to use
        """
        pass

    def emit_signals(
        self,
        time: dt_64,
        signals: list[Signal],
        account: "IAccountViewer",
        target_positions: list["TargetPosition"] | None = None,
    ) -> None:
        """
        Emit signals to the monitoring system.

        This method is called to emit trading signals for monitoring and analysis purposes.

        Args:
            time: Timestamp when the signals were generated
            signals: List of signals to emit
            account: Account viewer to get account information like total capital, leverage, etc.
            target_positions: Optional list of target positions generated from the signals
        """
        pass

    def emit_deals(
        self,
        time: dt_64,
        instrument: Instrument,
        deals: list[Deal],
        account: "IAccountViewer",
    ) -> None:
        """
        Emit deals to the monitoring system.

        This method is called to emit executed deals for monitoring and analysis purposes.
        It has to be manually called by the strategy, context does not call it automatically.

        Args:
            time: Timestamp when the deals were generated
            instrument: Instrument the deals belong to
            deals: List of deals to emit
            account: Account viewer to get account information like total capital, leverage, etc.
        """
        pass


class IStrategyNotifier:
    """Interface for notifying about strategy lifecycle events."""

    def notify_start(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has started.

        Args:
            metadata: Optional dictionary with additional information about the start event
        """
        pass

    def notify_stop(self, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has stopped.

        Args:
            metadata: Optional dictionary with additional information about the stop event
        """
        pass

    def notify_error(self, error: Exception, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has encountered an error.

        Args:
            strategy_name: Name of the strategy that encountered an error
            error: The exception that was raised
            metadata: Optional dictionary with additional information about the error
        """
        pass

    def notify_message(self, message: str, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """
        Notify that a strategy has encountered an error.

        Args:
            message: The message to notify
            metadata: Optional dictionary with additional information about the message
            **kwargs: Additional keyword arguments
        """
        pass
