"""
Tests for the state_resolvers module.

This file contains tests for the StateResolver class and its methods.
"""

from unittest.mock import MagicMock

from qubx.core.basics import Instrument, Position, TargetPosition
from qubx.core.lookups import lookup
from qubx.restarts.state_resolvers import StateResolver


class TestStateResolverBase:
    """Base class for StateResolver tests with common helper methods."""

    def setup_method(self):
        """Set up method that will be called before each test method."""
        self.ctx = MagicMock()

    def _find_instrument(self, exchange: str, symbol: str) -> Instrument:
        """Find an instrument by exchange and symbol."""
        instr = lookup.find_symbol(exchange, symbol)
        assert instr is not None, f"Instrument {symbol} not found on {exchange}"
        return instr


class TestStateResolverReduceOnly(TestStateResolverBase):
    """Tests for the REDUCE_ONLY method of the StateResolver class."""

    def test_reduce_only_with_empty_positions(self):
        """Test REDUCE_ONLY with empty positions."""
        # Setup
        self.ctx.get_positions.return_value = {}
        sim_positions = {}
        sim_orders = {}

        # Execute
        StateResolver.REDUCE_ONLY(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_not_called()

    def test_reduce_only_with_matching_positions(self):
        """Test REDUCE_ONLY with matching positions."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.REDUCE_ONLY(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_not_called()

    def test_reduce_only_with_live_larger_than_sim(self):
        """Test REDUCE_ONLY with live position larger than simulation position."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        btc_position = Position(btc_instrument, quantity=2.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.REDUCE_ONLY(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_called_once_with(btc_instrument, -1.0)

    def test_reduce_only_with_sim_larger_than_live(self):
        """Test REDUCE_ONLY with simulation position larger than live position."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=2.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.REDUCE_ONLY(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_not_called()

    def test_reduce_only_with_opposite_signs(self):
        """Test REDUCE_ONLY with positions having opposite signs."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=-1.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.REDUCE_ONLY(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_called_once_with(btc_instrument, -1.0)

    def test_reduce_only_with_position_in_live_not_in_sim(self):
        """Test REDUCE_ONLY with a position in live but not in simulation."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")

        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)
        eth_position = Position(eth_instrument, quantity=5.0, pos_average_price=3000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position, eth_instrument: eth_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.REDUCE_ONLY(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_called_once_with(eth_instrument, -5.0)

    def test_reduce_only_complex_scenario(self):
        """Test REDUCE_ONLY with a complex scenario involving multiple instruments and different states."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")
        sol_instrument = self._find_instrument("BINANCE.UM", "SOLUSDT")

        # Live positions:
        # BTC: 2.0 (larger than sim)
        # ETH: 5.0 (opposite sign from sim)
        # SOL: 10.0 (not in sim)
        btc_position = Position(btc_instrument, quantity=2.0, pos_average_price=50000.0)
        eth_position = Position(eth_instrument, quantity=5.0, pos_average_price=3000.0)
        sol_position = Position(sol_instrument, quantity=10.0, pos_average_price=100.0)

        self.ctx.get_positions.return_value = {
            btc_instrument: btc_position,
            eth_instrument: eth_position,
            sol_instrument: sol_position,
        }

        # Sim positions:
        # BTC: 1.0 (smaller than live)
        # ETH: -3.0 (opposite sign from live)
        sim_positions = {
            btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0),
            eth_instrument: Position(eth_instrument, quantity=-3.0, pos_average_price=3000.0),
        }
        sim_orders = {}

        # Execute
        StateResolver.REDUCE_ONLY(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()

        # Check that trade was called for each instrument with the correct quantities
        assert self.ctx.trade.call_count == 3

        # Create a dictionary of calls for easier verification
        calls = {call.args[0].symbol: call.args[1] for call in self.ctx.trade.call_args_list}

        assert calls["BTCUSDT"] == -1.0  # Reduce BTC position by 1.0
        assert calls["ETHUSDT"] == -5.0  # Close ETH position due to opposite sign
        assert calls["SOLUSDT"] == -10.0  # Close SOL position not in sim


class TestStateResolverSyncState(TestStateResolverBase):
    """Tests for the SYNC_STATE method of the StateResolver class."""

    def test_sync_state_with_empty_positions(self):
        """Test SYNC_STATE with empty positions."""
        # Setup
        self.ctx.get_positions.return_value = {}
        sim_positions = {}
        sim_orders = {}

        # Execute
        StateResolver.SYNC_STATE(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_not_called()

    def test_sync_state_with_matching_positions(self):
        """Test SYNC_STATE with matching positions."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.SYNC_STATE(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_not_called()

    def test_sync_state_with_different_quantities(self):
        """Test SYNC_STATE with different position quantities."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=2.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.SYNC_STATE(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        # self.ctx.trade.assert_called_once_with(btc_instrument, 1.0)
        self.ctx.emit_signal.assert_called_once()

    def test_sync_state_with_position_in_sim_not_in_live(self):
        """Test SYNC_STATE with a position in simulation but not in live."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")

        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {
            btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0),
            eth_instrument: Position(eth_instrument, quantity=5.0, pos_average_price=3000.0),
        }
        sim_orders = {}

        # Execute
        StateResolver.SYNC_STATE(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        # self.ctx.trade.assert_called_once_with(eth_instrument, 5.0)
        self.ctx.emit_signal.assert_called_once()

    def test_sync_state_with_position_in_live_not_in_sim(self):
        """Test SYNC_STATE with a position in live but not in simulation."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")

        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)
        eth_position = Position(eth_instrument, quantity=5.0, pos_average_price=3000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position, eth_instrument: eth_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.SYNC_STATE(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        # self.ctx.trade.assert_called_once_with(eth_instrument, -5.0)
        self.ctx.emit_signal.assert_called()

    def test_sync_state_complex_scenario(self):
        """Test SYNC_STATE with a complex scenario involving multiple instruments and different states."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")
        sol_instrument = self._find_instrument("BINANCE.UM", "SOLUSDT")

        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)
        eth_position = Position(eth_instrument, quantity=5.0, pos_average_price=3000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position, eth_instrument: eth_position}

        sim_positions = {
            btc_instrument: Position(btc_instrument, quantity=2.0, pos_average_price=50000.0),
            sol_instrument: Position(sol_instrument, quantity=10.0, pos_average_price=100.0),
        }
        sim_orders = {}

        # Execute
        StateResolver.SYNC_STATE(
            self.ctx,
            sim_positions,
            sim_orders,
            {
                sol_instrument: TargetPosition(time="2020-01-01", instrument=sol_instrument, target_position_size=10.0),
            },
        )

        # Verify
        self.ctx.get_positions.assert_called_once()

        # Check that trade was called for each instrument with the correct quantities
        assert self.ctx.emit_signal.call_count == 3
        calls = {call.args[0].instrument.symbol: call.args[0].signal for call in self.ctx.emit_signal.call_args_list}
        assert calls["BTCUSDT"] == 0.0
        assert calls["ETHUSDT"] == 0.0
        assert calls["SOLUSDT"] == 10.0

    def test_sync_state_ignores_small_differences(self):
        """Test SYNC_STATE ignores differences smaller than lot_size."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")

        # Create a position with a quantity just slightly different from the simulation
        # The difference is smaller than lot_size
        btc_position = Position(btc_instrument, quantity=1.0005, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {btc_instrument: Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.SYNC_STATE(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_not_called()  # No trade should be made for small differences


class TestStateResolverCloseAll(TestStateResolverBase):
    """Tests for the CLOSE_ALL method of the StateResolver class."""

    def test_close_all_with_empty_positions(self):
        """Test CLOSE_ALL with empty positions."""
        # Setup
        self.ctx.get_positions.return_value = {}
        sim_positions = {}
        sim_orders = {}

        # Execute
        StateResolver.CLOSE_ALL(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_not_called()

    def test_close_all_with_single_position(self):
        """Test CLOSE_ALL with a single position."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position}

        sim_positions = {}  # Simulation positions don't matter for CLOSE_ALL
        sim_orders = {}

        # Execute
        StateResolver.CLOSE_ALL(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()
        self.ctx.trade.assert_called_once_with(btc_instrument, -1.0)

    def test_close_all_with_multiple_positions(self):
        """Test CLOSE_ALL with multiple positions."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")
        sol_instrument = self._find_instrument("BINANCE.UM", "SOLUSDT")

        btc_position = Position(btc_instrument, quantity=1.0, pos_average_price=50000.0)
        eth_position = Position(eth_instrument, quantity=-5.0, pos_average_price=3000.0)
        sol_position = Position(sol_instrument, quantity=10.0, pos_average_price=100.0)

        self.ctx.get_positions.return_value = {
            btc_instrument: btc_position,
            eth_instrument: eth_position,
            sol_instrument: sol_position,
        }

        # Simulation positions don't matter for CLOSE_ALL
        sim_positions = {btc_instrument: Position(btc_instrument, quantity=2.0, pos_average_price=50000.0)}
        sim_orders = {}

        # Execute
        StateResolver.CLOSE_ALL(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()

        # Check that trade was called for each instrument with the correct quantities
        assert self.ctx.trade.call_count == 3

        # Create a dictionary of calls for easier verification
        calls = {call.args[0].symbol: call.args[1] for call in self.ctx.trade.call_args_list}

        assert calls["BTCUSDT"] == -1.0  # Close BTC position
        assert calls["ETHUSDT"] == 5.0  # Close ETH position (negative quantity)
        assert calls["SOLUSDT"] == -10.0  # Close SOL position

    def test_close_all_ignores_small_positions(self):
        """Test CLOSE_ALL ignores positions smaller than lot_size."""
        # Setup
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")

        # Create a position with a quantity smaller than lot_size
        btc_position = Position(btc_instrument, quantity=0.0005, pos_average_price=50000.0)
        # Create a normal position
        eth_position = Position(eth_instrument, quantity=5.0, pos_average_price=3000.0)

        self.ctx.get_positions.return_value = {btc_instrument: btc_position, eth_instrument: eth_position}

        sim_positions = {}  # Simulation positions don't matter for CLOSE_ALL
        sim_orders = {}

        # Execute
        StateResolver.CLOSE_ALL(self.ctx, sim_positions, sim_orders, {})

        # Verify
        self.ctx.get_positions.assert_called_once()

        # Only the ETH position should be closed
        self.ctx.trade.assert_called_once_with(eth_instrument, -5.0)
