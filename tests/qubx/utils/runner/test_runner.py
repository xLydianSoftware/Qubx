import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from qubx import QubxLogConfig, logger
from qubx.core.account_manager import AccountManager
from qubx.core.basics import Balance, CtrlChannel, DataType, Instrument, LiveTimeProvider, Position, RestoredState
from qubx.core.context import IStrategyContext, StrategyContext
from qubx.core.interfaces import IDataProvider, IStrategy, IStrategyInitializer
from qubx.core.lookups import lookup
from qubx.core.series import Quote
from qubx.data import CsvStorage
from qubx.loggers.inmemory import InMemoryLogsWriter
from qubx.restarts.state_resolvers import StateResolver
from qubx.utils.misc import class_import
from qubx.utils.runner.accounts import AccountConfigurationManager
from qubx.utils.runner.runner import _inject_restored_state, run_strategy_yaml


def _wait_until(condition, timeout: float = 10.0, poll: float = 0.05) -> bool:
    """Poll an outcome predicate — queue size is not a completion signal (the last
    message may still be inside process_data after the queue empties)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll)
    return condition()


class TestRunStrategyYaml:
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="wb") as temp_file:
            config = {
                "strategy": "strategy",
                "parameters": {},
                "aux": None,
                "live": {
                    "exchanges": {
                        "BINANCE.UM": {
                            "connector": "ccxt",
                            "universe": ["BTCUSDT", "ETHUSDT"],
                        }
                    },
                    "logging": {
                        "logger": "InMemoryLogsWriter",
                        "position_interval": "10Sec",
                        "portfolio_interval": "5Min",
                        "heartbeat_interval": "1m",
                    },
                    "warmup": {
                        "data": {
                            "storage": "csv::tests/data/storages/csv",
                        }
                    },
                },
            }
            yaml_content = yaml.dump(config, encoding="utf-8")
            temp_file.write(yaml_content)
            temp_file_path = temp_file.name

        yield Path(temp_file_path)

        os.unlink(temp_file_path)

    @pytest.fixture
    def start_time(self):
        """Return the start time for data loading."""
        return "2023-07-01"

    @pytest.fixture
    def stop_time(self):
        """Return the stop time for data loading."""
        return "2023-08-01"

    @pytest.fixture
    def mock_time_provider(self, start_time):
        """Create a mock time provider that returns a configurable time."""
        mock = MagicMock(spec=LiveTimeProvider)
        # Default time - use the start_time fixture
        mock.current_time = np.datetime64(f"{start_time}T00:00:00")

        # Allow setting the time
        def set_time(new_time):
            mock.current_time = new_time

        # Return the current time as datetime64 object with proper item() method
        def get_time():
            # Create a proper datetime64 object that has an item() method
            dt64 = mock.current_time.astype("datetime64[ns]")
            return dt64

        mock.time.side_effect = get_time
        mock.set_time = set_time
        return mock

    def _find_instrument(self, exchange: str, symbol: str) -> Instrument:
        instr = lookup.find_symbol(exchange, symbol)
        assert instr is not None, f"Instrument {symbol} not found"
        return instr

    @pytest.fixture
    def mock_data_provider(self, mock_time_provider, start_time, stop_time):
        """Create a mock data provider with appropriate implementation of IDataProvider interface."""
        mock = MagicMock(spec=IDataProvider)

        # Set up basic properties
        mock.is_simulation = False

        # Set up time provider
        mock.time_provider = mock_time_provider

        # Set up a real channel
        mock.channel = CtrlChannel("databus")
        # mock.channel = SimulatedCtrlChannel("databus")

        # Set up method returns
        mock.subscribe.return_value = None
        mock.unsubscribe.return_value = None
        mock.has_subscription.return_value = True
        mock.get_subscriptions.return_value = ["ohlc(1h)"]
        mock.get_subscribed_instruments.return_value = []
        mock.exchange.return_value = "BINANCE.UM"
        mock.get_quote.return_value = Quote(
            time=np.datetime64("now"), bid=100000.0, ask=100001.0, bid_size=1.0, ask_size=1.0
        )

        # ldr = loader("BINANCE.UM", "1h", symbols=["BTCUSDT", "ETHUSDT"], source="csv::tests/data/csv_1h/", n_jobs=1)
        # btc_ohlc = ldr.read("BTCUSDT", start=start_time, stop=stop_time, transform=AsBars())
        # eth_ohlc = ldr.read("ETHUSDT", start=start_time, stop=stop_time, transform=AsBars())

        ldr = CsvStorage("tests/data/storages/csv/").get_reader("BINANCE.UM", "SWAP")
        btc_ohlc = ldr.read("BTCUSDT", "ohlc(1h)", start=start_time, stop=stop_time).to_records()
        eth_ohlc = ldr.read("ETHUSDT", "ohlc(1h)", start=start_time, stop=stop_time).to_records()

        btc, eth = self._find_instrument("BINANCE.UM", "BTCUSDT"), self._find_instrument("BINANCE.UM", "ETHUSDT")
        btc_ohlc = [(btc, DataType.OHLC["1h"], bar, False) for bar in btc_ohlc]
        eth_ohlc = [(eth, DataType.OHLC["1h"], bar, False) for bar in eth_ohlc]
        ohlc = sorted([*btc_ohlc, *eth_ohlc], key=lambda x: x[2].time)

        # Store the OHLC data for pushing
        mock.ohlc_data = ohlc
        mock.current_index = 0

        # Implement push method
        def push():
            if mock.current_index >= len(mock.ohlc_data):
                return False  # No more bars to push

            # Get the next bar
            instrument, data_type, bar, is_historical = mock.ohlc_data[mock.current_index]

            # Update the time provider to match the bar's time
            bar_time = np.datetime64(bar.time, "ns")
            mock.time_provider.set_time(bar_time)

            # Send the event through the channel
            logger.info(
                f"Pushing bar {mock.current_index + 1}/{len(mock.ohlc_data)} for {instrument.symbol} at {bar_time}"
            )
            mock.channel.send((instrument, data_type, bar, is_historical))

            # Increment the index
            mock.current_index += 1
            return True

        mock.push = push

        def stream_bars(max_bars: int = 10):
            """
            Helper method to stream test bars through the channel.

            Args:
                context: The strategy context
                mock_data_provider: The mock data provider with a channel and push method
                num_bars: Number of bars to stream (default: 3)
            """
            # Push bars using the data provider's push method
            bars_pushed = 0
            for _ in range(max_bars):
                if not mock.push():
                    logger.info(f"No more bars to push after {bars_pushed} bars")
                    break
                bars_pushed += 1

            logger.info(f"Successfully pushed {bars_pushed} bars to context")

        mock.stream_bars = stream_bars

        return mock

    @pytest.fixture
    def mock_account_manager(self):
        """Create a mock account manager."""
        mock = MagicMock(spec=AccountConfigurationManager)
        mock.get_exchange_settings.return_value = MagicMock(testnet=False, commissions={})
        return mock

    @patch("qubx.utils.runner.runner.LiveTimeProvider")
    @patch("qubx.utils.runner.runner._create_data_provider")
    def test_paper_context_wires_simulated_connector_and_account_manager(
        self,
        mock_create_data_provider,
        mock_live_time_provider_class,
        temp_config_file,
        mock_data_provider,
        mock_time_provider,
    ):
        """Network-free wiring assertion: paper=True builds SimulatedConnectors + a central
        SimulatedAccountManager seeded with the configured initial capital, exposed through
        ctx.account."""
        from qubx.backtester.connector import SimulatedConnector
        from qubx.core.account_manager import SimulatedAccountManager
        from qubx.utils.runner.configs import load_strategy_config_from_yaml
        from qubx.utils.runner.runner import create_strategy_context

        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider

        # - no warmup: focus the test on construction-time wiring (warmup is exercised elsewhere)
        config = load_strategy_config_from_yaml(temp_config_file)
        config.live.warmup = None
        # - non-default AM knob to assert config threading into the constructed AM
        config.live.account_manager.inflight_check_retries = 7

        class MockStrategy(IStrategy):
            def on_init(self, initializer: IStrategyInitializer) -> None:
                initializer.set_base_subscription(DataType.OHLC["1h"])

        acc_manager = MagicMock(spec=AccountConfigurationManager)
        acc_manager.get_exchange_settings.return_value = MagicMock(
            base_currency="USDT", initial_capital=100_000.0, commissions=None, testnet=False
        )

        with (
            patch(
                "qubx.utils.runner.runner.class_import",
                side_effect=lambda arg: MockStrategy if arg == "strategy" else class_import(arg),
            ),
            patch("qubx.utils.runner.runner._create_tcc", return_value=lookup.find_fees("BINANCE.UM", None)),
        ):
            ctx = create_strategy_context(
                config=config,
                account_manager=acc_manager,
                paper=True,
                restored_state=None,
                stg_name="paper_wiring_test",
            )

        assert isinstance(ctx, StrategyContext)

        # - connectors are per-exchange SimulatedConnectors
        assert set(ctx._connectors.keys()) == {"BINANCE.UM"}
        assert isinstance(ctx._connectors["BINANCE.UM"], SimulatedConnector)
        assert ctx.is_paper_trading

        # - central account manager is the simulation variant, seeded with configured capital
        assert isinstance(ctx._account_manager, SimulatedAccountManager)
        assert ctx.account is ctx._account_manager
        balance = ctx.account.get_balance("USDT", exchange="BINANCE.UM")
        assert balance is not None
        assert balance.total == 100_000.0
        assert balance.free == 100_000.0

        # - the AM holds no strategy reference: every callback routes through the PM
        assert not hasattr(ctx._account_manager, "_strategy")

        # - the live.account_manager block is threaded into the AM's config
        assert ctx._account_manager._cfg.inflight_check_retries == 7

    @patch("qubx.utils.runner.runner.LiveTimeProvider")
    @patch("qubx.utils.runner.runner._create_data_provider")
    def test_paper_context_canonicalizes_binance_pm(
        self,
        mock_create_data_provider,
        mock_live_time_provider_class,
        temp_config_file,
        mock_data_provider,
        mock_time_provider,
    ):
        """R13 regression: a BINANCE.PM (portfolio margin) config is canonicalized to
        BINANCE.UM at the runner boundary, so the connector dict and AM states are keyed
        by the exchange the instruments carry — ctx.trade() and on_market_quote route
        instead of KeyError-ing / silently skipping."""
        from qubx.utils.runner.configs import load_strategy_config_from_yaml
        from qubx.utils.runner.runner import create_strategy_context

        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider
        # Faithful to live: the PM data provider self-reports the venue name (the
        # data-side EXCHANGE_MAPPINGS fallback bridges quote lookups by BINANCE.UM).
        mock_data_provider.exchange.return_value = "BINANCE.PM"

        config = load_strategy_config_from_yaml(temp_config_file)
        config.live.warmup = None
        config.live.exchanges = {"BINANCE.PM": config.live.exchanges["BINANCE.UM"]}

        class MockStrategy(IStrategy):
            def on_init(self, initializer: IStrategyInitializer) -> None:
                initializer.set_base_subscription(DataType.OHLC["1h"])

        acc_manager = MagicMock(spec=AccountConfigurationManager)
        acc_manager.get_exchange_settings.return_value = MagicMock(
            base_currency="USDT", initial_capital=100_000.0, commissions=None, testnet=False
        )

        with patch(
            "qubx.utils.runner.runner.class_import",
            side_effect=lambda arg: MockStrategy if arg == "strategy" else class_import(arg),
        ):
            ctx = create_strategy_context(
                config=config,
                account_manager=acc_manager,
                paper=True,
                restored_state=None,
                stg_name="pm_canonical_test",
            )

        assert isinstance(ctx, StrategyContext)

        # - every framework-facing key is canonical; the venue name survives only in
        #   the settings lookups (initial capital / base currency / commissions)
        assert set(ctx._connectors.keys()) == {"BINANCE.UM"}
        assert ctx._connectors["BINANCE.UM"].exchange_name == "BINANCE.UM"
        assert {i.exchange for i in ctx._initial_instruments} == {"BINANCE.UM"}
        acc_manager.get_exchange_settings.assert_any_call("BINANCE.PM")

        # - paper capital seeded into the canonical AM state
        balance = ctx.account.get_balance("USDT", exchange="BINANCE.UM")
        assert balance is not None
        assert balance.total == 100_000.0

        # - ctx.trade() routes: AM.add_order keys by instrument.exchange (KeyError pre-fix)
        btc = self._find_instrument("BINANCE.UM", "BTCUSDT")
        quote = Quote(time=np.datetime64("now"), bid=100_000.0, ask=100_001.0, bid_size=1.0, ask_size=1.0)
        ctx._connectors["BINANCE.UM"].process_market_data(btc, quote)
        order = ctx.trade(btc, 0.01)
        assert order.client_order_id

        # - on_market_quote routes: position marking keys by instrument.exchange
        #   (silently skipped pre-fix; seed_position returned False)
        assert ctx.account.seed_position(Position(btc, quantity=1.0, pos_average_price=90_000.0))
        ctx.account.on_market_quote(btc, quote)
        assert ctx.account.get_position(btc).last_update_price == quote.mid_price()

    @patch("qubx.utils.runner.runner.LiveTimeProvider")
    @patch("qubx.utils.runner.runner._create_data_provider")
    @patch("qubx.utils.runner.runner.CtrlChannel")
    @patch("qubx.utils.runner.runner._restore_state")
    def test_run_strategy_yaml_with_restored_positions(
        self,
        mock_restore_state,
        mock_ctrl_channel_class,
        mock_create_data_provider,
        mock_live_time_provider_class,
        temp_config_file,
        mock_data_provider,
        mock_time_provider,
    ):
        """Test running a strategy from a YAML file with restored positions."""
        # Set up mocks
        channel = mock_data_provider.channel
        assert isinstance(channel, CtrlChannel)

        mock_ctrl_channel_class.return_value = channel
        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider

        # Create mock restored state with positions
        btc_instrument = self._find_instrument("BINANCE.UM", "BTCUSDT")
        eth_instrument = self._find_instrument("BINANCE.UM", "ETHUSDT")

        # Create positions with quantity 2 for both instruments
        btc_position = Position(btc_instrument, quantity=2.0, pos_average_price=50000.0)
        eth_position = Position(eth_instrument, quantity=2.0, pos_average_price=3000.0)

        # Create restored state
        restored_state = RestoredState(
            time=np.datetime64("now"),
            balances=[Balance(exchange="BINANCE.UM", currency="USDT", free=100000.0, locked=0.0, total=100000.0)],
            instrument_to_target_positions={},
            instrument_to_signal_positions={},
            positions={btc_instrument: btc_position, eth_instrument: eth_position},
        )

        # Set up the mock to return our restored state
        mock_restore_state.return_value = restored_state

        QubxLogConfig.set_log_level("INFO")

        class MockStrategy(IStrategy):
            def on_init(self, initializer: IStrategyInitializer) -> None:
                initializer.set_base_subscription(DataType.OHLC["1h"])
                initializer.set_warmup("10d")
                initializer.set_state_resolver(StateResolver.SYNC_STATE)

            def on_start(self, ctx: IStrategyContext) -> None:
                pass

        # Run the function under test
        with patch(
            "qubx.utils.runner.runner.class_import",
            side_effect=lambda arg: MockStrategy if arg == "strategy" else class_import(arg),
        ):
            ctx = run_strategy_yaml(temp_config_file, paper=True, restore=True)

        assert isinstance(ctx, IStrategyContext)
        assert isinstance(ctx.strategy, MockStrategy)

        # Verify that the positions were restored
        assert btc_instrument in ctx.positions
        assert eth_instrument in ctx.positions
        assert ctx.positions[btc_instrument].quantity == 2.0
        assert ctx.positions[eth_instrument].quantity == 2.0

        mock_create_data_provider.assert_called()
        mock_restore_state.assert_called_once()

        # Stream live bars and wait on the asserted outcome (both restored positions
        # closed by SYNC_STATE, executions logged), not on queue size
        QubxLogConfig.set_log_level("INFO")
        assert isinstance(ctx, StrategyContext)
        logs_writer = ctx._logging.logs_writer
        assert isinstance(logs_writer, InMemoryLogsWriter)
        mock_data_provider.stream_bars(max_bars=10)
        assert _wait_until(
            lambda: len(logs_writer.get_executions()) > 0
            and ctx.get_position(btc_instrument).quantity == 0.0
            and ctx.get_position(eth_instrument).quantity == 0.0
        )

        if ctx.is_running():
            ctx.stop()

        # Check executions
        executions = logs_writer.get_executions()
        assert len(executions) > 0

        # Check positions
        pos = ctx.get_position(btc_instrument)
        assert pos.quantity == 0.0
        assert pos.instrument == btc_instrument

        pos = ctx.get_position(eth_instrument)
        assert pos.quantity == 0.0
        assert pos.instrument == eth_instrument

    @patch("qubx.utils.runner.runner.LiveTimeProvider")
    @patch("qubx.utils.runner.runner._create_data_provider")
    @patch("qubx.utils.runner.runner.CtrlChannel")
    def test_run_strategy_yaml_with_warmup(
        self,
        mock_ctrl_channel_class,
        mock_create_data_provider,
        mock_live_time_provider_class,
        temp_config_file,
        mock_data_provider,
        mock_time_provider,
    ):
        """Test running a strategy from a YAML file with warmup."""
        # Set up mocks
        channel = mock_data_provider.channel
        assert isinstance(channel, CtrlChannel)

        mock_ctrl_channel_class.return_value = channel
        mock_create_data_provider.return_value = mock_data_provider
        mock_live_time_provider_class.return_value = mock_time_provider

        QubxLogConfig.set_log_level("DEBUG")

        class MockStrategy(IStrategy):
            def on_init(self, initializer: IStrategyInitializer) -> None:
                initializer.set_base_subscription(DataType.OHLC["1h"])
                initializer.set_warmup("10d")
                initializer.set_state_resolver(StateResolver.SYNC_STATE)

            def on_start(self, ctx: IStrategyContext) -> None:
                instr = sorted(ctx.instruments, key=lambda x: x.symbol)[0]
                logger.info(f"on_start ::: <cyan>Buying {instr.symbol} qty 1</cyan>")
                # ctx.trade(instr, 1)
                ctx.emit_signal(instr.signal(ctx, 1.0))

        # Run the function under test
        with patch(
            "qubx.utils.runner.runner.class_import",
            side_effect=lambda arg: MockStrategy if arg == "strategy" else class_import(arg),
        ):
            ctx = run_strategy_yaml(temp_config_file, paper=True)

        assert isinstance(ctx, IStrategyContext)
        assert isinstance(ctx.strategy, MockStrategy)

        mock_create_data_provider.assert_called()

        # Stream live bars and wait on the asserted outcome (the on_start signal
        # executed), not on queue size
        QubxLogConfig.set_log_level("DEBUG")
        assert isinstance(ctx, StrategyContext)
        logs_writer = ctx._logging.logs_writer
        assert isinstance(logs_writer, InMemoryLogsWriter)
        mock_data_provider.stream_bars(max_bars=10)
        assert _wait_until(lambda: len(logs_writer.get_executions()) >= 1)

        if ctx.is_running():
            ctx.stop()

        # Check executions
        executions = logs_writer.get_executions()
        # - init signal
        assert len(executions) == 1

        # Check positions
        pos = ctx.get_position(sorted(ctx.instruments, key=lambda x: x.symbol)[0])
        assert pos.quantity == 1
        assert pos.instrument == sorted(ctx.instruments, key=lambda x: x.symbol)[0]


def test_inject_restored_state_preserves_accounting_fields():
    # Regression guard: _inject_restored_state seeds the whole persisted Position, so the
    # accounting fields (r_pnl, commissions, cumulative_funding) survive a restart — not
    # just quantity. This lost dedicated coverage when merge_restored_accounting was removed.
    inst = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert inst is not None
    pos = Position(inst, quantity=2.0, pos_average_price=50_000.0)
    pos.r_pnl = 123.0
    pos.commissions = 4.5
    pos.cumulative_funding = -7.0

    am = AccountManager(
        connectors={inst.exchange: MagicMock()},
        base_currencies={inst.exchange: "USDT"},
        time=LiveTimeProvider(),
    )

    restored = RestoredState(
        time=np.datetime64("now"),
        balances=[Balance(exchange=inst.exchange, currency="USDT", free=1000.0, locked=0.0, total=1000.0)],
        instrument_to_target_positions={},
        instrument_to_signal_positions={},
        positions={inst: pos},
    )
    _inject_restored_state(am, restored)

    restored_pos = am.get_position(inst)
    assert restored_pos.quantity == 2.0
    assert restored_pos.r_pnl == 123.0
    assert restored_pos.commissions == 4.5
    assert restored_pos.cumulative_funding == -7.0
    balances = am.get_balances(inst.exchange)
    assert any(b.currency == "USDT" and b.total == 1000.0 for b in balances)
