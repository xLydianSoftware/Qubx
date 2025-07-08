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

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

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
    TriggerEvent,
    dt_64,
    td_64,
)
from qubx.core.errors import BaseErrorEvent
from qubx.core.helpers import set_parameters_to_object
from qubx.core.series import OHLCV, Bar, Quote

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
    def get_balances(self, exchange: str | None = None) -> dict[str, AssetBalance]:
        """Get all currency balances.

        Returns:
            dict[str, AssetBalance]: Dictionary mapping currency codes to AssetBalance objects
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

    @property
    def positions(self) -> dict[Instrument, Position]:
        """[Deprecated: Use get_positions()] Get all current positions.

        Returns:
            dict[Instrument, Position]: Dictionary mapping instruments to their positions
        """
        return self.get_positions()

    def get_orders(self, instrument: Instrument | None = None) -> dict[str, Order]:
        """Get active orders, optionally filtered by instrument.

        Args:
            instrument: Optional instrument to filter orders by

        Returns:
            dict[str, Order]: Dictionary mapping order IDs to Order objects
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


class IBroker:
    """Broker provider interface for managing trading operations.

    Handles account operations, order placement, and position tracking.
    """

    channel: CtrlChannel

    @property
    def is_simulated_trading(self) -> bool:
        """
        Check if the broker is in simulation mode.
        """
        ...

    def send_order(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **optional,
    ) -> Order:
        """Sends an order to the trading service.

        Args:
            instrument: The instrument to trade.
            order_side: Order side ("buy" or "sell").
            order_type: Type of order ("market" or "limit").
            amount: Amount of instrument to trade.
            price: Price for limit orders.
            client_id: Client-specified order ID.
            time_in_force: Time in force for order (default: "gtc").
            **optional: Additional order parameters.

        Returns:
            Order: The created order object.
        """
        raise NotImplementedError("send_order is not implemented")

    def send_order_async(
        self,
        instrument: Instrument,
        order_side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        client_id: str | None = None,
        time_in_force: str = "gtc",
        **optional,
    ) -> None:
        """Sends an order to the trading service.

        Args:
            instrument: The instrument to trade.
            order_side: Order side ("buy" or "sell").
            order_type: Type of order ("market" or "limit").
            amount: Amount of instrument to trade.
            price: Price for limit orders.
            client_id: Client-specified order ID.
            time_in_force: Time in force for order (default: "gtc").
            **optional: Additional order parameters.

        Returns:
            Order: The created order object.
        """
        raise NotImplementedError("send_order_async is not implemented")

    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order (non blocking).

        Args:
            order_id: The ID of the order to cancel.
        """
        raise NotImplementedError("cancel_order is not implemented")

    def cancel_orders(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument.

        Args:
            instrument: The instrument to cancel orders for.
        """
        raise NotImplementedError("cancel_orders is not implemented")

    def update_order(self, order_id: str, price: float | None = None, amount: float | None = None) -> Order:
        """Update an existing order.

        Args:
            order_id: The ID of the order to update.
            price: New price for the order.
            amount: New amount for the order.

        Returns:
            Order: The updated Order object if successful

        Raises:
            NotImplementedError: If the method is not implemented
            OrderNotFound: If the order is not found
            BadRequest: If the request is invalid
        """
        raise NotImplementedError("update_order is not implemented")

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

    def get_quote(self, instrument: Instrument) -> Quote:
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
    ) -> Order:
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

    def close_position(self, instrument: Instrument) -> None:
        """Close position for an instrument.

        Args:
            instrument: The instrument to close position for
        """
        ...

    def close_positions(self, market_type: MarketType | None = None, exchange: str | None = None) -> None:
        """Close all positions."""
        ...

    def cancel_order(self, order_id: str, exchange: str | None = None) -> None:
        """Cancel a specific order.

        Args:
            order_id: ID of the order to cancel
        """
        ...

    def cancel_orders(self, instrument: Instrument) -> None:
        """Cancel all orders for an instrument.

        Args:
            instrument: The instrument to cancel orders for
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

    def emit_signal(self, signal: Signal) -> None:
        """
        Emit a signal for processing
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
):
    strategy: "IStrategy"
    initializer: "IStrategyInitializer"
    account: IAccountProcessor
    emitter: "IMetricEmitter"
    health: "IHealthReader"

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

    def restore_position_from_target(self, ctx: IStrategyContext, target: TargetPosition):
        """
        Restore active position and tracking from the target.

        Args:
            - ctx: Strategy context object.
            - target: Target position to restore from.
        """
        ...


@dataclass
class HealthMetrics:
    """
    Health metrics for system performance.

    All latency values are in milliseconds.
    Dropped events are reported as events per second.
    Queue size is the number of events in the processing queue.
    """

    queue_size: float = 0.0
    drop_rate: float = 0.0

    # Arrival latency statistics
    p50_arrival_latency: float = 0.0
    p90_arrival_latency: float = 0.0
    p99_arrival_latency: float = 0.0

    # Queue latency statistics
    p50_queue_latency: float = 0.0
    p90_queue_latency: float = 0.0
    p99_queue_latency: float = 0.0

    # Processing latency statistics
    p50_processing_latency: float = 0.0
    p90_processing_latency: float = 0.0
    p99_processing_latency: float = 0.0


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

    def record_event_dropped(self, event_type: str) -> None:
        """
        Record that an event was dropped.

        Args:
            event_type: Type of the dropped event
        """
        ...

    def record_data_arrival(self, event_type: str, event_time: dt_64) -> None:
        """
        Record a data arrival time.

        Args:
            event_type: Type of event (e.g., "order_execution")
        """
        ...

    def record_start_processing(self, event_type: str, event_time: dt_64) -> None:
        """
        Record a start processing time.
        """
        ...

    def record_end_processing(self, event_type: str, event_time: dt_64) -> None:
        """
        Record a end processing time.
        """
        ...

    def set_event_queue_size(self, size: int) -> None:
        """
        Set the current event queue size.

        Args:
            size: Current size of the event queue
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

    def get_queue_size(self) -> int:
        """
        Get the current event queue size.

        Returns:
            Number of events waiting to be processed
        """
        ...

    def get_arrival_latency(self, event_type: str, percentile: float = 90) -> float:
        """
        Get latency for a specific event type.

        Args:
            event_type: Type of event (e.g., "quote", "trade")
            percentile: Optional percentile (0-100) to retrieve (default: 90)

        Returns:
            Latency value in milliseconds
        """
        ...

    def get_queue_latency(self, event_type: str, percentile: float = 90) -> float:
        """
        Get queue latency for a specific event type.
        """
        ...

    def get_processing_latency(self, event_type: str, percentile: float = 90) -> float:
        """
        Get processing latency for a specific event type.
        """
        ...

    def get_latency(self, event_type: str, percentile: float = 90) -> float:
        """
        Get end-to-end latency for a specific event type.
        """
        ...

    def get_execution_latency(self, scope: str, percentile: float = 90) -> float:
        """
        Get execution latency for a specific scope.
        """
        ...

    def get_execution_latencies(self) -> dict[str, float]:
        """
        Get all execution latencies.
        """
        ...

    def get_event_frequency(self, event_type: str) -> float:
        """
        Get the events per second for a specific event type.

        Args:
            event_type: Type of event to get frequency for

        Returns:
            Events per second
        """
        ...

    def get_system_metrics(self) -> HealthMetrics:
        """
        Get system-wide metrics.

        Returns:
            HealthMetrics:
            - avg_queue_size: Average queue size in the last window
            - avg_dropped_events: Average number of dropped events per second
            - p50_arrival_latency: Median arrival latency (ms)
            - p90_arrival_latency: 90th percentile arrival latency (ms)
            - p99_arrival_latency: 99th percentile arrival latency (ms)
            - p50_queue_latency: Median queue latency (ms)
            - p90_queue_latency: 90th percentile queue latency (ms)
            - p99_queue_latency: 99th percentile queue latency (ms)
            - p50_processing_latency: Median processing latency (ms)
            - p90_processing_latency: 90th percentile processing latency (ms)
            - p99_processing_latency: 99th percentile processing latency (ms)
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


class IMetricEmitter:
    """Interface for emitting metrics to external monitoring systems."""

    def emit(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
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

    def set_time_provider(self, time_provider: ITimeProvider) -> None:
        """
        Set the time provider for the metric emitter.

        This method is used to set the time provider that will be used to get timestamps
        when no explicit timestamp is provided in the emit method.

        Args:
            time_provider: The time provider to use
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


class IStrategyLifecycleNotifier:
    """Interface for notifying about strategy lifecycle events."""

    def notify_start(self, strategy_name: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has started.

        Args:
            strategy_name: Name of the strategy that started
            metadata: Optional dictionary with additional information about the start event
        """
        pass

    def notify_stop(self, strategy_name: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has stopped.

        Args:
            strategy_name: Name of the strategy that stopped
            metadata: Optional dictionary with additional information about the stop event
        """
        pass

    def notify_error(self, strategy_name: str, error: Exception, metadata: dict[str, Any] | None = None) -> None:
        """
        Notify that a strategy has encountered an error.

        Args:
            strategy_name: Name of the strategy that encountered an error
            error: The exception that was raised
            metadata: Optional dictionary with additional information about the error
        """
        pass
