"""
Unit tests for Lighter account message parsers.

Tests use real captured WebSocket messages from Lighter testnet.
"""

import json
from pathlib import Path

import pytest

from qubx.connectors.xlighter.parsers import (
    PositionState,
    parse_account_all_message,
    parse_account_all_orders_message,
    parse_account_tx_message,
    parse_user_stats_message,
)
from qubx.core.basics import AssetBalance, Deal, Instrument, MarketType, Order

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "account_samples"


@pytest.fixture
def mock_instrument_loader():
    """Create a mock instrument loader with test instrument."""

    class MockInstrumentLoader:
        def __init__(self):
            # Create a test instrument for market_id=24 (LINKUSDC on testnet)
            self.test_instrument = Instrument(
                symbol="LINKUSDC",
                asset_type="CRYPTO",
                market_type=MarketType.SWAP,
                exchange="LIGHTER",
                base="LINK",
                quote="USDC",
                settle="USDC",
                exchange_symbol="LINK",
                tick_size=0.0001,  # 4 decimals
                lot_size=0.01,  # 2 decimals
                min_size=0.01,
                min_notional=5.0,
            )

            self.market_id_to_instrument = {24: self.test_instrument}

        def get_instrument_by_market_id(self, market_id: int):
            return self.market_id_to_instrument.get(market_id)

    return MockInstrumentLoader()


def load_test_message(filename: str) -> dict:
    """Load a test message from JSON file."""
    file_path = TEST_DATA_DIR / filename
    with open(file_path, "r") as f:
        captured_data = json.load(f)
    return captured_data["data"]


class TestParseAccountTxMessage:
    """Tests for parse_account_tx_message()."""

    def test_parse_subscription_confirmation(self, mock_instrument_loader):
        """Test parsing subscription confirmation message (should return empty list)."""
        message = load_test_message("account_tx/sample_01.json")

        deals = parse_account_tx_message(message, mock_instrument_loader)

        assert deals == []

    def test_parse_filled_market_order(self, mock_instrument_loader):
        """Test parsing transaction with filled market order."""
        message = load_test_message("account_tx/sample_02.json")

        deals = parse_account_tx_message(message, mock_instrument_loader)

        # Should have one deal
        assert len(deals) == 1

        instrument, deal = deals[0]

        # Check instrument
        assert instrument.symbol == "LINKUSDC"
        assert instrument.market_type == MarketType.SWAP

        # Check deal structure
        assert isinstance(deal, Deal)
        assert deal.order_id == "7036874567915800"
        assert deal.aggressive is True
        assert deal.fee_currency == "USDC"

        # Check deal values
        # From event_info: "t": {"p": 401342, "s": 100, "tf": 0}
        # price: 401342 / 10^4 = 40.1342
        # size: 100 / 10^2 = 1.00
        # is_ask: 1 (SELL), so amount should be negative
        assert deal.price == pytest.approx(40.1342, rel=1e-4)
        assert deal.amount == pytest.approx(-1.00, rel=1e-4)
        assert deal.fee_amount == 0.0  # tf=0

    def test_parse_empty_txs_list(self, mock_instrument_loader):
        """Test parsing message with empty txs list."""
        message = {"channel": "account_tx:225671", "type": "update/account_tx", "txs": []}

        deals = parse_account_tx_message(message, mock_instrument_loader)

        assert deals == []

    def test_parse_unknown_market_id(self, mock_instrument_loader):
        """Test parsing transaction with unknown market_id (should be skipped)."""
        message = load_test_message("account_tx/sample_02.json")

        # Modify market_id to unknown value
        tx = message["txs"][0]
        event_info = json.loads(tx["event_info"])
        event_info["m"] = 999  # Unknown market
        tx["event_info"] = json.dumps(event_info)

        deals = parse_account_tx_message(message, mock_instrument_loader)

        # Should skip unknown market
        assert deals == []

    def test_parse_multiple_transactions(self, mock_instrument_loader):
        """Test parsing message with multiple transactions."""
        message = load_test_message("account_tx/sample_02.json")

        # Duplicate the transaction
        tx = message["txs"][0]
        message["txs"].append(tx.copy())

        deals = parse_account_tx_message(message, mock_instrument_loader)

        # Should have two deals
        assert len(deals) == 2

        # Both should be for same instrument
        assert deals[0][0] == deals[1][0]


class TestParseAccountAllOrdersMessage:
    """Tests for parse_account_all_orders_message()."""

    def test_parse_subscription_confirmation(self, mock_instrument_loader):
        """Test parsing subscription confirmation message (should return empty list)."""
        message = load_test_message("account_all_orders/sample_01.json")

        orders = parse_account_all_orders_message(message, mock_instrument_loader)

        assert orders == []

    def test_parse_filled_order(self, mock_instrument_loader):
        """Test parsing message with filled market order."""
        message = load_test_message("account_all_orders/sample_02.json")

        orders = parse_account_all_orders_message(message, mock_instrument_loader)

        # Should have one order
        assert len(orders) == 1

        order = orders[0]

        # Check instrument
        assert order.instrument.symbol == "LINKUSDC"
        assert order.instrument.market_type == MarketType.SWAP

        # Check order structure
        assert isinstance(order, Order)
        # For Lighter, order.id uses client_order_id (used for all operations)
        assert order.id == "0"  # client_order_id from sample
        assert order.client_id == "0"
        assert order.type == "MARKET"
        assert order.status == "CLOSED"  # filled
        assert order.time_in_force == "IOC"  # immediate-or-cancel

        # Check order values
        # is_ask: true -> side=SELL
        # quantity is remaining_amount (unsigned), which is 0 for filled orders
        assert order.side == "SELL"
        assert order.quantity == pytest.approx(0.00, rel=1e-4)  # Filled order has 0 remaining
        assert order.price == pytest.approx(40.0843, rel=1e-4)

        # Check options
        assert "initial_amount" in order.options
        assert order.options["initial_amount"] == pytest.approx(1.00, rel=1e-4)
        assert order.options["filled_amount"] == pytest.approx(1.00, rel=1e-4)
        assert order.options["remaining_amount"] == pytest.approx(0.00, rel=1e-4)

    def test_parse_empty_orders_dict(self, mock_instrument_loader):
        """Test parsing message with empty orders dict."""
        message = {"channel": "account_all_orders:225671", "type": "update/account_all_orders", "orders": {}}

        orders = parse_account_all_orders_message(message, mock_instrument_loader)

        assert orders == []

    def test_parse_buy_order(self, mock_instrument_loader):
        """Test parsing BUY order (quantity is unsigned remaining amount)."""
        message = load_test_message("account_all_orders/sample_02.json")

        # Modify order to be a buy
        order_data = message["orders"]["24"][0]
        order_data["is_ask"] = False  # BUY order

        orders = parse_account_all_orders_message(message, mock_instrument_loader)

        assert len(orders) == 1
        order = orders[0]

        assert order.side == "BUY"
        # quantity is always unsigned (remaining_amount), 0 for filled orders
        assert order.quantity >= 0

    def test_parse_limit_order(self, mock_instrument_loader):
        """Test parsing LIMIT order."""
        message = load_test_message("account_all_orders/sample_02.json")

        # Modify order to be a limit order
        order_data = message["orders"]["24"][0]
        order_data["type"] = "limit"

        orders = parse_account_all_orders_message(message, mock_instrument_loader)

        assert len(orders) == 1
        order = orders[0]

        assert order.type == "LIMIT"

    def test_parse_multiple_markets(self, mock_instrument_loader):
        """Test parsing orders from multiple markets."""
        message = load_test_message("account_all_orders/sample_02.json")

        # Add another market with an order
        order_data = message["orders"]["24"][0].copy()
        message["orders"]["25"] = [order_data]

        # Mock loader should only know about market 24
        orders = parse_account_all_orders_message(message, mock_instrument_loader)

        # Should only parse market 24 (market 25 is unknown)
        assert len(orders) == 1


class TestParseUserStatsMessage:
    """Tests for parse_user_stats_message()."""

    def test_parse_subscription_confirmation(self):
        """Test parsing subscription confirmation message (contains initial stats)."""
        message = load_test_message("user_stats/sample_01.json")

        balances = parse_user_stats_message(message)

        # Subscription confirmation contains initial stats, so should parse them
        assert "USDC" in balances
        balance = balances["USDC"]

        assert isinstance(balance, AssetBalance)

        # From sample: collateral=998.931800, available_balance=998.931800
        # free = 998.931800 (available)
        # locked = 0 (collateral - available)
        # total = 998.931800 (collateral)
        assert balance.free == 998.9318
        assert balance.locked == 0
        assert balance.total == 998.9318

    def test_parse_stats_update(self):
        """Test parsing user stats update message."""
        message = load_test_message("user_stats/sample_02.json")

        balances = parse_user_stats_message(message)

        # Should have USDC balance
        assert "USDC" in balances
        balance = balances["USDC"]

        assert isinstance(balance, AssetBalance)

        # From sample: collateral=998.931800, available_balance=990.909640
        # free = 990.909640
        # total = 998.931800
        # locked = 998.931800 - 990.909640 = 8.02216
        assert balance.total == pytest.approx(998.931800, rel=1e-6)
        assert balance.free == pytest.approx(990.909640, rel=1e-6)
        assert balance.locked == pytest.approx(8.02216, rel=1e-4)

    def test_parse_empty_stats(self):
        """Test parsing message with empty stats."""
        message = {"channel": "user_stats:225671", "type": "update/user_stats", "stats": {}}

        balances = parse_user_stats_message(message)

        # Should return empty dict (or dict with 0 balance)
        # Current implementation returns empty dict
        assert balances == {}

    def test_parse_initial_stats(self):
        """Test parsing initial stats (no positions, full balance available)."""
        message = load_test_message("user_stats/sample_01.json")

        # This is a subscription message, so change type to update
        message["type"] = "update/user_stats"

        balances = parse_user_stats_message(message)

        # Should have USDC balance
        assert "USDC" in balances
        balance = balances["USDC"]

        # From sample: collateral=998.931800, available_balance=998.931800
        # locked should be 0 (no positions)
        assert balance.total == pytest.approx(998.931800, rel=1e-6)
        assert balance.free == pytest.approx(998.931800, rel=1e-6)
        assert balance.locked == pytest.approx(0.0, rel=1e-6)

    def test_balance_calculations(self):
        """Test that balance calculations are correct."""
        message = {
            "channel": "user_stats:225671",
            "type": "update/user_stats",
            "stats": {"collateral": "1000.00", "available_balance": "800.00"},
        }

        balances = parse_user_stats_message(message)

        balance = balances["USDC"]

        # total = 1000, free = 800, locked = 200
        assert balance.total == 1000.0
        assert balance.free == 800.0
        assert balance.locked == 200.0


class TestParseAccountAllMessage:
    """Tests for parse_account_all_message()."""

    def test_parse_subscription_with_initial_position(self, mock_instrument_loader):
        """Test parsing subscription message with initial position."""
        message = load_test_message("account_all/sample_01.json")
        account_index = 225671

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        # Subscription message has zero position (position = 0.00)
        # Zero positions are now included to detect position closures
        assert len(positions) == 1
        assert len(deals) == 0
        assert len(funding_payments) == 0

        # Verify it's a zero position
        instrument = mock_instrument_loader.test_instrument
        position_state = positions[instrument]
        assert position_state.quantity == pytest.approx(0.0, abs=1e-8)

    def test_parse_trade_update(self, mock_instrument_loader):
        """Test parsing message with trade only."""
        message = load_test_message("account_all/sample_02.json")
        account_index = 225671

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        # No positions (empty dict)
        assert len(positions) == 0

        # Should have one deal
        assert len(deals) == 1

        # No funding payments
        assert len(funding_payments) == 0

        instrument, deal = deals[0]

        # Check instrument
        assert instrument.symbol == "LINKUSDC"

        # Check deal (account 225671 is ask_account_id, so they sold)
        assert isinstance(deal, Deal)
        assert deal.price == pytest.approx(40.1342, rel=1e-4)
        assert deal.amount == pytest.approx(-1.00, rel=1e-4)  # Negative for sell
        assert deal.fee_currency == "USDC"
        assert deal.id == "225067334"  # trade_id

    def test_parse_short_position_update(self, mock_instrument_loader):
        """Test parsing position update with short position."""
        message = load_test_message("account_all/sample_03.json")
        account_index = 225671

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        # Should have one short position
        assert len(positions) == 1
        assert len(deals) == 0
        assert len(funding_payments) == 0

        instrument = mock_instrument_loader.test_instrument
        assert instrument in positions

        # Check PositionState object
        position_state = positions[instrument]
        assert isinstance(position_state, PositionState)

        # sign=-1, position=1.00 -> signed_position = -1.00
        assert position_state.quantity == pytest.approx(-1.00, rel=1e-4)

        # Check avg_entry_price from sample_03.json
        assert position_state.avg_entry_price == pytest.approx(40.1342, rel=1e-4)

        # Check PnL values
        assert position_state.unrealized_pnl == pytest.approx(0.0039, rel=1e-4)
        assert position_state.realized_pnl == pytest.approx(0.0, rel=1e-4)

    def test_parse_zero_position_included(self, mock_instrument_loader):
        """Test that zero positions ARE included (needed to detect position closures)."""
        message = load_test_message("account_all/sample_05.json")
        account_index = 225671

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        # Position is zero (position=0.00), but should still be included
        # This is important for detecting when positions close
        assert len(positions) == 1
        assert len(deals) == 0
        assert len(funding_payments) == 0

        # Verify it's a zero position
        instrument = mock_instrument_loader.test_instrument
        position_state = positions[instrument]
        assert position_state.quantity == pytest.approx(0.0, abs=1e-8)
        assert position_state.avg_entry_price == pytest.approx(0.0, abs=1e-8)

    def test_parse_trade_closing_position(self, mock_instrument_loader):
        """Test parsing message with trade that closes position."""
        message = load_test_message("account_all/sample_04.json")
        account_index = 225671

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        # No positions after close (empty dict in sample)
        assert len(positions) == 0

        # Should have one deal (closing trade)
        assert len(deals) == 1

        # No funding payments
        assert len(funding_payments) == 0

        instrument, deal = deals[0]

        # Check deal (account 225671 is bid_account_id, so they bought to close short)
        assert deal.price == pytest.approx(40.1397, rel=1e-4)
        assert deal.amount == pytest.approx(1.00, rel=1e-4)  # Positive for buy
        assert deal.id == "225068811"

    def test_parse_account_is_buyer(self, mock_instrument_loader):
        """Test parsing trade where account is the buyer."""
        message = load_test_message("account_all/sample_04.json")
        account_index = 225671  # This account is bid_account_id

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        assert len(deals) == 1
        _, deal = deals[0]

        # bid_account_id = 225671 (buyer), so amount should be positive
        assert deal.amount > 0

    def test_parse_account_is_seller(self, mock_instrument_loader):
        """Test parsing trade where account is the seller."""
        message = load_test_message("account_all/sample_02.json")
        account_index = 225671  # This account is ask_account_id

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        assert len(deals) == 1
        _, deal = deals[0]

        # ask_account_id = 225671 (seller), so amount should be negative
        assert deal.amount < 0

    def test_parse_trade_not_involving_account(self, mock_instrument_loader):
        """Test that trades not involving the account are skipped."""
        message = load_test_message("account_all/sample_02.json")
        account_index = 99999  # Different account

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        # Should skip trade (account not involved)
        assert len(deals) == 0

    def test_parse_unknown_market_id(self, mock_instrument_loader):
        """Test that unknown market_ids are skipped."""
        message = load_test_message("account_all/sample_02.json")
        account_index = 225671

        # Modify market_id to unknown value
        message["trades"]["999"] = message["trades"].pop("24")

        positions, deals, funding_payments = parse_account_all_message(message, mock_instrument_loader, account_index)

        # Should skip unknown market
        assert len(deals) == 0


class TestIntegration:
    """Integration tests with all three parsers."""

    def test_full_trading_flow(self, mock_instrument_loader):
        """
        Test parsing a full trading flow: order creation -> execution -> balance update.
        """
        # 1. Initial balance check
        stats_msg = load_test_message("user_stats/sample_01.json")
        stats_msg["type"] = "update/user_stats"  # Convert to update
        initial_balances = parse_user_stats_message(stats_msg)

        assert "USDC" in initial_balances
        assert initial_balances["USDC"].total == pytest.approx(998.931800, rel=1e-6)

        # 2. Order placed (from account_all_orders)
        orders_msg = load_test_message("account_all_orders/sample_02.json")
        orders = parse_account_all_orders_message(orders_msg, mock_instrument_loader)

        assert len(orders) == 1
        order = orders[0]
        assert order.status == "CLOSED"  # Already filled

        # 3. Order execution (from account_tx)
        tx_msg = load_test_message("account_tx/sample_02.json")
        deals = parse_account_tx_message(tx_msg, mock_instrument_loader)

        assert len(deals) == 1
        _, deal = deals[0]
        assert deal.amount == pytest.approx(-1.00, rel=1e-4)  # Sold 1 LINK

        # 4. Balance after trade
        stats_msg_after = load_test_message("user_stats/sample_02.json")
        final_balances = parse_user_stats_message(stats_msg_after)

        # After selling, available balance decreased (margin locked)
        assert final_balances["USDC"].free < initial_balances["USDC"].free
        assert final_balances["USDC"].locked > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
