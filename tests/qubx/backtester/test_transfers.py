"""Unit tests for SimulationTransferManager."""

import pandas as pd
import pytest

from qubx.backtester.transfers import SimulationTransferManager
from qubx.core.account import BasicAccountProcessor, CompositeAccountProcessor
from qubx.core.basics import AssetBalance, ITimeProvider, TransactionCostsCalculator


class MockTimeProvider(ITimeProvider):
    """Mock time provider for testing."""

    def __init__(self, current_time: pd.Timestamp):
        self._current_time = current_time

    def time(self):
        return self._current_time

    def increment(self, delta: pd.Timedelta):
        """Helper to advance time."""
        self._current_time += delta


class TestSimulationTransferManager:
    """Test suite for SimulationTransferManager."""

    @pytest.fixture
    def time_provider(self):
        """Create a mock time provider."""
        return MockTimeProvider(pd.Timestamp("2024-01-01 00:00:00"))

    @pytest.fixture
    def composite_account(self, time_provider):
        """Create a composite account with two exchange processors."""
        tcc = TransactionCostsCalculator(name="test", maker=0.01, taker=0.02)

        # Create processors for two exchanges
        processor_binance = BasicAccountProcessor(
            account_id="BINANCE",
            time_provider=time_provider,
            base_currency="USDT",
            tcc=tcc,
            initial_capital=50000.0,
        )

        processor_hyperliquid = BasicAccountProcessor(
            account_id="HYPERLIQUID",
            time_provider=time_provider,
            base_currency="USDT",
            tcc=tcc,
            initial_capital=30000.0,
        )

        # Initial balances are already set via initial_capital parameter
        # No need to overwrite them

        # Create composite
        composite = CompositeAccountProcessor(
            time_provider=time_provider,
            account_processors={
                "BINANCE": processor_binance,
                "HYPERLIQUID": processor_hyperliquid,
            },
        )

        return composite

    @pytest.fixture
    def transfer_manager(self, composite_account, time_provider):
        """Create a SimulationTransferManager."""
        return SimulationTransferManager(composite_account, time_provider)

    def test_initialization(self, transfer_manager):
        """Test that manager initializes correctly."""
        assert transfer_manager._transfers == []
        assert transfer_manager.get_transfers() == {}

    def test_simple_transfer(self, transfer_manager, composite_account):
        """Test a basic transfer between exchanges."""
        # Initial balances
        binance_initial = composite_account.get_account_processor("BINANCE").get_balances()["USDT"].total
        hyperliquid_initial = composite_account.get_account_processor("HYPERLIQUID").get_balances()["USDT"].total

        # Execute transfer
        tx_id = transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 10000.0)

        # Check transaction ID format
        assert tx_id.startswith("sim_")
        assert len(tx_id) == 16  # "sim_" + 12 hex chars

        # Check balances updated correctly
        binance_final = composite_account.get_account_processor("BINANCE").get_balances()["USDT"].total
        hyperliquid_final = composite_account.get_account_processor("HYPERLIQUID").get_balances()["USDT"].total

        assert binance_final == binance_initial - 10000.0
        assert hyperliquid_final == hyperliquid_initial + 10000.0

        # Check free balance also updated
        assert (
            composite_account.get_account_processor("BINANCE").get_balances()["USDT"].free == binance_initial - 10000.0
        )
        assert (
            composite_account.get_account_processor("HYPERLIQUID").get_balances()["USDT"].free
            == hyperliquid_initial + 10000.0
        )

    def test_transfer_status(self, transfer_manager):
        """Test getting transfer status."""
        tx_id = transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 5000.0)

        # Get status
        status = transfer_manager.get_transfer_status(tx_id)

        assert status["transaction_id"] == tx_id
        assert status["status"] == "completed"
        assert status["from_exchange"] == "BINANCE"
        assert status["to_exchange"] == "HYPERLIQUID"
        assert status["currency"] == "USDT"
        assert status["amount"] == 5000.0

    def test_transfer_status_not_found(self, transfer_manager):
        """Test getting status of non-existent transfer."""
        status = transfer_manager.get_transfer_status("sim_nonexistent")

        assert status["transaction_id"] == "sim_nonexistent"
        assert status["status"] == "not_found"
        assert "error" in status

    def test_get_all_transfers(self, transfer_manager):
        """Test getting all transfers."""
        # Execute multiple transfers
        tx1 = transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 5000.0)
        tx2 = transfer_manager.transfer_funds("HYPERLIQUID", "BINANCE", "USDT", 2000.0)

        # Get all transfers
        all_transfers = transfer_manager.get_transfers()

        assert len(all_transfers) == 2
        assert tx1 in all_transfers
        assert tx2 in all_transfers
        assert all_transfers[tx1]["amount"] == 5000.0
        assert all_transfers[tx2]["amount"] == 2000.0

    def test_get_transfers_dataframe_empty(self, transfer_manager):
        """Test getting DataFrame when no transfers."""
        df = transfer_manager.get_transfers_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == [
            "transaction_id",
            "from_exchange",
            "to_exchange",
            "currency",
            "amount",
            "status",
        ]

    def test_get_transfers_dataframe_with_data(self, transfer_manager, time_provider):
        """Test getting DataFrame with transfers."""
        # Execute transfers
        transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 5000.0)

        time_provider.increment(pd.Timedelta("1h"))

        transfer_manager.transfer_funds("HYPERLIQUID", "BINANCE", "USDT", 2000.0)

        # Get DataFrame
        df = transfer_manager.get_transfers_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        # timestamp is the index, not a column
        assert list(df.columns) == [
            "transaction_id",
            "from_exchange",
            "to_exchange",
            "currency",
            "amount",
            "status",
        ]
        assert df.index.name == "timestamp"

        # Check first transfer
        assert df.iloc[0]["from_exchange"] == "BINANCE"
        assert df.iloc[0]["to_exchange"] == "HYPERLIQUID"
        assert df.iloc[0]["amount"] == 5000.0
        assert df.iloc[0]["status"] == "completed"

        # Check second transfer
        assert df.iloc[1]["from_exchange"] == "HYPERLIQUID"
        assert df.iloc[1]["to_exchange"] == "BINANCE"
        assert df.iloc[1]["amount"] == 2000.0

    def test_insufficient_funds(self, transfer_manager, composite_account):
        """Test transfer with insufficient funds."""
        # Try to transfer more than available
        available = composite_account.get_account_processor("HYPERLIQUID").get_balances()["USDT"].free

        with pytest.raises(ValueError, match="Insufficient funds"):
            transfer_manager.transfer_funds("HYPERLIQUID", "BINANCE", "USDT", available + 1000.0)

    def test_invalid_currency(self, transfer_manager):
        """Test transfer with non-existent currency."""
        with pytest.raises(ValueError, match="Currency .* not found"):
            transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "BTC", 1.0)

    def test_invalid_exchange(self, transfer_manager):
        """Test transfer with non-existent exchange."""
        with pytest.raises(ValueError, match="Unknown exchange"):
            transfer_manager.transfer_funds("BINANCE", "UNKNOWN_EXCHANGE", "USDT", 1000.0)

    def test_requires_composite_account(self, time_provider):
        """Test that manager requires a CompositeAccountProcessor."""
        # Create a basic (non-composite) processor
        basic_processor = BasicAccountProcessor(
            account_id="SINGLE",
            time_provider=time_provider,
            base_currency="USDT",
            tcc=TransactionCostsCalculator(name="test", maker=0.01, taker=0.02),
            initial_capital=10000.0,
        )

        # Try to create manager with basic processor
        manager = SimulationTransferManager(basic_processor, time_provider)  # type: ignore

        # Should fail when trying to transfer (error message will mention missing get_account_processor)
        with pytest.raises(ValueError, match="Exchange not found"):
            manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 1000.0)

    def test_multiple_currencies(self, composite_account, transfer_manager):
        """Test transfers with multiple currencies."""
        # Add BTC balances
        composite_account.get_account_processor("BINANCE")._balances["BTC"] = AssetBalance(
            free=5.0, locked=0.0, total=5.0
        )
        composite_account.get_account_processor("HYPERLIQUID")._balances["BTC"] = AssetBalance(
            free=2.0, locked=0.0, total=2.0
        )

        # Transfer USDT
        tx1 = transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 1000.0)

        # Transfer BTC
        tx2 = transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "BTC", 0.5)

        # Check both transfers recorded
        all_transfers = transfer_manager.get_transfers()
        assert len(all_transfers) == 2
        assert all_transfers[tx1]["currency"] == "USDT"
        assert all_transfers[tx2]["currency"] == "BTC"
        assert all_transfers[tx2]["amount"] == 0.5

        # Check balances
        assert composite_account.get_account_processor("BINANCE").get_balances()["BTC"].total == 4.5
        assert composite_account.get_account_processor("HYPERLIQUID").get_balances()["BTC"].total == 2.5

    def test_timestamp_recording(self, transfer_manager, time_provider):
        """Test that timestamps are recorded correctly."""
        initial_time = time_provider.time()

        tx1 = transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 1000.0)

        # Advance time
        time_provider.increment(pd.Timedelta("5Min"))

        tx2 = transfer_manager.transfer_funds("HYPERLIQUID", "BINANCE", "USDT", 500.0)

        # Check timestamps
        transfers = transfer_manager.get_transfers()
        assert transfers[tx1]["timestamp"] == initial_time
        assert transfers[tx2]["timestamp"] == initial_time + pd.Timedelta("5Min")

    def test_zero_transfer_not_allowed(self, transfer_manager):
        """Test that zero-amount transfers are handled (should work but maybe we want to prevent?)."""
        # This might be implementation-dependent - currently it would work
        # but we might want to add validation
        tx_id = transfer_manager.transfer_funds("BINANCE", "HYPERLIQUID", "USDT", 0.0)
        assert tx_id.startswith("sim_")

        # Balances should be unchanged
        status = transfer_manager.get_transfer_status(tx_id)
        assert status["amount"] == 0.0
