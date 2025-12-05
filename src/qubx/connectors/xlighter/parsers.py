"""
Parsers for Lighter WebSocket account messages.

This module provides functions to parse Lighter account channel messages
into Qubx data structures (Deal, Order, balance info).
"""

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qubx import logger
from qubx.core.basics import AssetBalance, Deal, FundingPayment, Instrument, Order

from .instruments import LighterInstrumentLoader


@dataclass
class PositionState:
    """
    Position state from Lighter's account_all channel.

    Represents the server's authoritative position data including
    quantity, average entry price, PnL and margin information.
    """

    quantity: float  # Signed quantity (positive for long, negative for short)
    avg_entry_price: float  # Average entry price from server
    unrealized_pnl: float  # Current unrealized PnL
    realized_pnl: float  # Cumulative realized PnL
    allocated_margin: float = 0.0  # Margin allocated to position (from exchange)


def parse_account_tx_message(
    message: dict[str, Any], instrument_loader: LighterInstrumentLoader
) -> list[tuple[Instrument, Deal]]:
    """
    Parse account_tx WebSocket message into list of Deals.

    The account_tx channel provides transaction execution updates for a specific account.
    Each transaction contains an `event_info` JSON string with trade execution details.

    Message structure:
    ```
    {
      "channel": "account_tx:225671",
      "type": "update/account_tx",
      "txs": [
        {
          "type": 14,  # CREATE_ORDER
          "event_info": "{...}",  # JSON string with trade details
          "account_index": 225671,
          "executed_at": 1760285006758,
          ...
        }
      ]
    }
    ```

    event_info structure (JSON string):
    ```
    {
      "m": 24,  # market_index
      "t": {  # trade execution
        "p": 399173,  # price (scaled)
        "s": 100,     # size (scaled)
        "tf": 0,      # taker fee
        "mf": 20      # maker fee
      },
      "mo": {...},  # maker order (counterparty)
      "to": {...},  # taker order (account's order)
      "ae": ""      # additional event info
    }
    ```

    Args:
        message: Raw WebSocket message from account_tx channel
        instrument_loader: Loader with market_id -> Instrument mapping

    Returns:
        List of (Instrument, Deal) tuples. Empty list if:
        - Message is subscription confirmation (type="subscribed/account_tx")
        - No transactions in message
        - Transaction type is not CREATE_ORDER (type 14)
        - Failed to parse event_info

    Note:
        - Creates one Deal per transaction execution
        - Deal.amount is signed: positive for BUY, negative for SELL
        - Deal.aggressive is True (account tx are always the aggressor)
        - Prices and sizes are converted from Lighter's scaled integer format
    """
    from .constants import TX_TYPE_CREATE_ORDER

    deals: list[tuple[Instrument, Deal]] = []

    # Get transactions list
    txs = message.get("txs", [])
    if not txs:
        return deals

    for tx in txs:
        # Only process CREATE_ORDER transactions
        tx_type = tx.get("type")
        if tx_type != TX_TYPE_CREATE_ORDER:
            continue

        # Parse event_info JSON string
        event_info_str = tx.get("event_info", "")
        if not event_info_str:
            logger.warning("Missing event_info in account_tx transaction")
            continue

        try:
            event_info = json.loads(event_info_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse event_info JSON: {e}")
            continue

        # Extract market index and trade details
        market_index = event_info.get("m")
        if market_index is None:
            logger.warning("Missing market index in event_info")
            continue

        # Get instrument
        instrument = instrument_loader.get_instrument_by_market_id(market_index)
        if not instrument:
            logger.warning(f"Unknown market_id: {market_index}")
            continue

        # Extract trade execution details
        trade_info = event_info.get("t", {})
        if not trade_info:
            logger.warning("Missing trade info in event_info")
            continue

        # Get taker order (account's order)
        taker_order = event_info.get("to", {})
        if not taker_order:
            logger.warning("Missing taker order in event_info")
            continue

        # Extract values from trade info
        # Lighter uses scaled integers: actual_value = int_value / (10 ^ decimals)
        price_scaled = trade_info.get("p", 0)
        size_scaled = trade_info.get("s", 0)
        taker_fee = trade_info.get("tf", 0)

        # Convert from scaled integers to float
        price = float(price_scaled) / (10**instrument.price_precision)
        size = float(size_scaled) / (10**instrument.size_precision)

        # Calculate fee amount
        # Fee is in quote currency (USDC)
        fee_amount = float(taker_fee) / 1e6  # USDC has 6 decimals

        # Determine side from is_ask flag
        is_ask = taker_order.get("ia", 0)
        # amount: positive for BUY, negative for SELL
        amount = -size if is_ask else size

        # Get order ID and transaction details
        order_id = str(taker_order.get("i", ""))
        tx_hash = tx.get("hash", "")
        executed_at_ms = tx.get("executed_at", 0)

        # Create Deal object
        deal = Deal(
            id=tx_hash[:16] if tx_hash else str(executed_at_ms),  # Use first 16 chars of hash as trade ID
            order_id=order_id,
            time=pd.Timestamp(executed_at_ms, unit="ms").asm8,
            amount=amount,
            price=price,
            aggressive=True,  # Account transactions are always aggressive (taker)
            fee_amount=fee_amount,
            fee_currency="USDC",  # All Lighter markets settle in USDC
        )

        deals.append((instrument, deal))

    return deals


def parse_account_all_orders_message(
    message: dict[str, Any], instrument_loader: LighterInstrumentLoader
) -> list[Order]:
    """
    Parse account_all_orders WebSocket message into list of Orders.

    The account_all_orders channel provides order updates for all markets.
    Orders are grouped by market_index.

    Message structure:
    ```
    {
      "channel": "account_all_orders:225671",
      "type": "update/account_all_orders",
      "orders": {
        "24": [  # market_index
          {
            "order_index": 7036874567748225,
            "client_order_index": 0,
            "order_id": "7036874567748225",
            "market_index": 24,
            "owner_account_index": 225671,
            "initial_base_amount": "1.00",
            "remaining_base_amount": "0.00",
            "filled_base_amount": "1.00",
            "price": "39.8774",
            "is_ask": true,
            "type": "market",
            "status": "filled",
            "time_in_force": "immediate-or-cancel",
            "created_at": 1760285006,
            "updated_at": 1760285006
          }
        ]
      }
    }
    ```
    """
    orders = []

    # Get orders dict (keyed by market_index)
    orders_by_market = message.get("orders", {})
    if not orders_by_market:
        return orders

    # Iterate through each market's orders
    for market_index_str, market_orders in orders_by_market.items():
        try:
            market_index = int(market_index_str)
        except ValueError:
            logger.warning(f"Invalid market_index: {market_index_str}")
            continue

        # Get instrument
        instrument = instrument_loader.get_instrument_by_market_id(market_index)
        if not instrument:
            logger.warning(f"Unknown market_id: {market_index}")
            continue

        # Parse each order
        for order_data in market_orders:
            try:
                order = _parse_lighter_order(order_data, instrument)
                orders.append(order)
            except Exception as e:
                logger.error(f"Failed to parse order: {e}")
                continue

    return orders


def _parse_lighter_order(order_data: dict[str, Any], instrument: Instrument) -> Order:
    """
    Parse a single Lighter order into Qubx Order.

    Args:
        order_data: Raw order data from Lighter
        instrument: Qubx instrument

    Returns:
        Qubx Order object

    Raises:
        ValueError: If order data is invalid
    """
    # Extract order fields
    order_id = str(order_data.get("order_id", ""))
    client_order_id = order_data.get("client_order_id")
    if client_order_id is not None:
        client_order_id = str(client_order_id)

    # Amounts
    initial_amount = abs(float(order_data.get("initial_base_amount", "0")))
    remaining_amount = abs(float(order_data.get("remaining_base_amount", "0")))
    filled_amount = abs(float(order_data.get("filled_base_amount", "0")))

    # Price (may be "0.0000" for market orders)
    price_str = order_data.get("price", "0")
    price = float(price_str)

    # Side
    is_ask = order_data.get("is_ask", False)
    side = "SELL" if is_ask else "BUY"

    # Order type
    order_type_str = order_data.get("type", "limit").lower()
    if order_type_str == "market":
        order_type = "MARKET"
    elif order_type_str == "limit":
        order_type = "LIMIT"
    elif "stop" in order_type_str:
        order_type = "STOP_MARKET"
    else:
        order_type = "LIMIT"  # Default to limit

    # Status
    status_str = order_data.get("status", "").lower()
    if status_str == "filled":
        status = "CLOSED"
    elif status_str in ("canceled", "cancelled"):
        status = "CANCELED"
    elif status_str in ("open", "new"):
        status = "OPEN"
    elif status_str == "pending":
        status = "PENDING"
    else:
        status = "OPEN"  # Default to open

    # Time in force
    tif_str = order_data.get("time_in_force", "").lower()
    if "immediate" in tif_str or tif_str == "ioc":
        time_in_force = "IOC"
    elif "post" in tif_str:
        time_in_force = "POST_ONLY"
    else:
        time_in_force = "GTC"  # Default

    # Timestamps
    created_at = order_data.get("created_at", 0)
    updated_at = order_data.get("updated_at", created_at)

    # Use updated_at as order time (most recent)
    timestamp = pd.Timestamp(updated_at, unit="s").asm8

    # Build options dict with additional info
    options = {
        "initial_amount": initial_amount,
        "remaining_amount": remaining_amount,
        "filled_amount": filled_amount,
    }

    # Add reduce_only flag if present
    if order_data.get("reduce_only"):
        options["reduceOnly"] = True

    # Add trigger_price if present
    trigger_price_str = order_data.get("trigger_price")
    if trigger_price_str and trigger_price_str != "0.0000":
        options["trigger_price"] = float(trigger_price_str)

    return Order(
        id=order_id,
        type=order_type,
        instrument=instrument,
        time=timestamp,
        quantity=remaining_amount,
        price=price,
        side=side,
        status=status,
        time_in_force=time_in_force,
        client_id=client_order_id,
        options=options,
    )


def parse_account_all_message(
    message: dict[str, Any],
    instrument_loader: LighterInstrumentLoader,
    account_index: int,
) -> tuple[dict[Instrument, PositionState], list[tuple[Instrument, Deal]], dict[Instrument, list[FundingPayment]]]:
    """
    Parse account_all WebSocket message into positions, deals, and funding payments.

    The account_all channel provides comprehensive account updates including
    positions, trades, and funding histories for all markets. This is the
    primary channel for position and trade updates in Lighter.

    Message structure:
    ```
    {
      "account": 225671,
      "channel": "account_all:225671",
      "type": "update/account_all",
      "positions": {
        "24": {
          "market_id": 24,
          "symbol": "HYPE",
          "sign": -1,  # 1 for long, -1 for short
          "position": "1.00",  # magnitude (always positive)
          "avg_entry_price": "40.1342",
          "unrealized_pnl": "0.003900",
          "realized_pnl": "0.000000",
          ...
        }
      },
      "trades": {
        "24": [
          {
            "trade_id": 225067334,
            "market_id": 24,
            "size": "1.00",
            "price": "40.1342",
            "ask_account_id": 225671,
            "bid_account_id": 61127,
            "is_maker_ask": false,
            "maker_fee": 20,
            "taker_fee": 50,
            "timestamp": 1760287839079,
            ...
          }
        ]
      },
      "funding_histories": {
        "24": [
          {
            "timestamp": 1760287839079,
            "market_id": 24,
            "funding_id": 123,
            "change": "-0.005",
            "rate": "0.0001",
            "position_size": "1.00",
            "position_side": "long"
          }
        ]
      }
    }
    ```

    Args:
        message: Raw WebSocket message from account_all channel
        instrument_loader: Loader with market_id -> Instrument mapping
        account_index: The account ID to filter trades for

    Returns:
        Tuple of (positions_dict, deals_list, funding_payments_dict):
        - positions_dict: {Instrument: PositionState} mapping to position state
          including quantity, avg_entry_price, and PnL data from server
        - deals_list: [(Instrument, Deal)] trades to send through channel
        - funding_payments_dict: {Instrument: [FundingPayment]} funding payments

    Note:
        - Position state includes server's authoritative data (quantity, avg price, PnL)
        - Only trades involving this account are returned
        - Deals are created for strategy notification but do not update positions
        - Positions are updated separately from the position state
        - Funding payments are only present when funding occurred
    """
    positions: dict[Instrument, PositionState] = {}
    deals: list[tuple[Instrument, Deal]] = []
    funding_payments: dict[Instrument, list[FundingPayment]] = {}

    # Parse positions
    positions_data = message.get("positions", {})
    for market_id_str, position_data in positions_data.items():
        try:
            market_id = int(market_id_str)
        except ValueError:
            logger.warning(f"Invalid market_id in positions: {market_id_str}")
            continue

        # Get instrument
        instrument = instrument_loader.get_instrument_by_market_id(market_id)
        if not instrument:
            logger.warning(f"Unknown market_id in positions: {market_id}")
            continue

        # Extract position details
        sign = position_data.get("sign", 1)  # 1 for long, -1 for short
        position_magnitude = float(position_data.get("position", "0"))
        avg_entry_price = float(position_data.get("avg_entry_price", "0"))
        unrealized_pnl = float(position_data.get("unrealized_pnl", "0"))
        realized_pnl = float(position_data.get("realized_pnl", "0"))
        allocated_margin = float(position_data.get("allocated_margin", "0"))

        # Calculate signed position
        signed_position = sign * position_magnitude

        # Include ALL positions (including zero) to detect position closures
        # When a position goes from non-zero to zero, we need to know about it
        positions[instrument] = PositionState(
            quantity=signed_position,
            avg_entry_price=avg_entry_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            allocated_margin=allocated_margin,
        )

    # Parse trades
    trades_data = message.get("trades", {})
    for market_id_str, trade_list in trades_data.items():
        try:
            market_id = int(market_id_str)
        except ValueError:
            logger.warning(f"Invalid market_id in trades: {market_id_str}")
            continue

        # Get instrument
        instrument = instrument_loader.get_instrument_by_market_id(market_id)
        if not instrument:
            logger.warning(f"Unknown market_id in trades: {market_id}")
            continue

        # Parse each trade
        for trade_data in trade_list:
            # Determine if this account was buyer or seller
            ask_account = trade_data.get("ask_account_id")
            bid_account = trade_data.get("bid_account_id")

            # Skip if this account wasn't involved (shouldn't happen but be safe)
            if ask_account != account_index and bid_account != account_index:
                continue

            # Extract trade details
            size = float(trade_data.get("size", "0"))
            price = float(trade_data.get("price", "0"))
            trade_id = trade_data.get("trade_id")
            tx_hash = trade_data.get("tx_hash", "")
            timestamp_ms = trade_data.get("timestamp", 0)

            # Determine side and amount
            # If we're the seller (ask side), amount is negative
            # If we're the buyer (bid side), amount is positive
            is_seller = ask_account == account_index
            amount = -size if is_seller else size

            # Check if we were the maker
            is_maker_ask = trade_data.get("is_maker_ask", False)
            is_maker = (is_maker_ask and is_seller) or (not is_maker_ask and not is_seller)

            # Calculate fee based on our role (fees in basis points, e.g., 20 = 0.002 = 0.2%)
            # If maker_fee/taker_fee is absent, it means no fee was paid for that role
            if is_maker:
                fee_bp = trade_data.get("maker_fee", 0)
            else:
                fee_bp = trade_data.get("taker_fee", 0)

            fee_amount = (abs(amount) * price * fee_bp / 100_000) if fee_bp else 0

            # Create Deal object
            deal = Deal(
                id=str(trade_id) if trade_id else tx_hash[:16],
                order_id="",  # Not provided in account_all trades
                time=pd.Timestamp(timestamp_ms, unit="ms").asm8,
                amount=amount,
                price=price,
                aggressive=not is_maker,
                fee_amount=fee_amount,
                fee_currency="USDC",
            )

            deals.append((instrument, deal))

    # Parse funding histories
    funding_histories_data = message.get("funding_histories", {})
    for market_id_str, funding_list in funding_histories_data.items():
        try:
            market_id = int(market_id_str)
        except ValueError:
            logger.warning(f"Invalid market_id in funding_histories: {market_id_str}")
            continue

        # Get instrument
        instrument = instrument_loader.get_instrument_by_market_id(market_id)
        if not instrument:
            logger.warning(f"Unknown market_id in funding_histories: {market_id}")
            continue

        # Parse each funding payment
        payments = []
        for funding_data in funding_list:
            try:
                timestamp_ms = funding_data.get("timestamp", 0)
                rate_str = funding_data.get("rate", "0")

                # Convert rate from string to float
                funding_rate = float(rate_str)

                # Lighter uses hourly funding intervals (1 hour)
                funding_interval_hours = 1

                payment = FundingPayment(
                    time=pd.Timestamp(timestamp_ms, unit="ms").asm8,
                    funding_rate=funding_rate,
                    funding_interval_hours=funding_interval_hours,
                )
                payments.append(payment)

            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse funding history entry: {e}")
                continue

        if payments:
            funding_payments[instrument] = payments

    return positions, deals, funding_payments


def parse_user_stats_message(message: dict[str, Any], exchange: str = "LIGHTER") -> dict[str, AssetBalance]:
    """
    Parse user_stats WebSocket message into balance information.

    The user_stats channel provides account statistics including balances,
    leverage, and margin usage. All Lighter accounts are denominated in USDC.

    Message structure:
    ```
    {
      "channel": "user_stats:225671",
      "type": "update/user_stats",
      "stats": {
        "collateral": "998.888700",
        "portfolio_value": "998.901500",
        "leverage": "0.04",
        "available_balance": "990.920600",
        "margin_usage": "0.80",
        "buying_power": "0",
        "cross_stats": {...},
        "total_stats": {...}
      }
    }
    ```

    Args:
        message: Raw WebSocket message from user_stats channel
        exchange: Exchange name (default: "LIGHTER")

    Returns:
        Dictionary mapping currency to AssetBalance. Empty dict if:
        - No stats in message

    Note:
        - All balances are in USDC (Lighter's settlement currency)
        - free = available_balance (funds available for trading)
        - locked = collateral - available_balance (funds in open orders/positions)
        - total = collateral (total account value)
    """
    balances: dict[str, AssetBalance] = {}

    # Get stats
    stats = message.get("stats", {})
    if not stats:
        return balances

    # Extract balance values (all in USDC)
    collateral = float(stats.get("collateral", "0"))
    available_balance = float(stats.get("available_balance", "0"))

    # Calculate locked amount
    # locked = collateral - available (funds in orders/positions)
    locked = max(0, collateral - available_balance)

    # Create AssetBalance for USDC
    balances["USDC"] = AssetBalance(
        exchange=exchange,
        currency="USDC",
        free=available_balance,
        locked=locked,
        total=collateral,
    )

    return balances
