"""
Test for the run_separate_instruments parameter functionality.
"""

from typing import cast

import pandas as pd

from qubx.backtester.utils import SetupTypes, recognize_simulation_configuration
from qubx.core.basics import Instrument, MarketType
from qubx.core.interfaces import IStrategy
from qubx.core.lookups import lookup
from qubx.trackers.riskctrl import AtrRiskTracker


class TestStrategy(IStrategy):
    """Simple test strategy for testing purposes."""

    pass


class TestRunSeparateInstruments:
    """Test class for run_separate_instruments parameter functionality."""

    def test_strategy_with_run_separate_instruments_false(self):
        """Test that run_separate_instruments=False creates combined setup."""
        # Get test instruments
        instruments = [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "BCHUSDT"]]
        assert all([x and isinstance(x, Instrument) for x in instruments]), "Got wrong instruments"
        instruments = cast(list[Instrument], instruments)

        # Test with run_separate_instruments=False (default)
        setups = recognize_simulation_configuration(
            "TestStrategy",
            TestStrategy(),
            instruments,
            ["BINANCE.UM"],
            10_000,
            "USDT",
            "vip0_usdt",
            "1Min",
            True,
            run_separate_instruments=False,
        )

        # Should have 1 setup with all instruments
        assert len(setups) == 1, "Should have 1 combined setup"
        assert setups[0].setup_type == SetupTypes.STRATEGY, "Should be strategy type"
        assert len(setups[0].instruments) == 2, "Should have both instruments"
        assert setups[0].name == "TestStrategy", "Should have base name"

    def test_strategy_with_run_separate_instruments_true(self):
        """Test that run_separate_instruments=True creates separate setups."""
        # Get test instruments
        instruments = [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "BCHUSDT"]]
        assert all([x and isinstance(x, Instrument) for x in instruments]), "Got wrong instruments"
        instruments = cast(list[Instrument], instruments)

        # Test with run_separate_instruments=True
        setups = recognize_simulation_configuration(
            "TestStrategy",
            TestStrategy(),
            instruments,
            ["BINANCE.UM"],
            10_000,
            "USDT",
            "vip0_usdt",
            "1Min",
            True,
            run_separate_instruments=True,
        )

        # Should have 2 setups, one per instrument
        assert len(setups) == 2, "Should have 2 separate setups"

        # Check first setup (BTCUSDT)
        assert setups[0].setup_type == SetupTypes.STRATEGY, "Should be strategy type"
        assert len(setups[0].instruments) == 1, "Should have only 1 instrument"
        assert setups[0].instruments[0].symbol == "BTCUSDT", "Should be BTCUSDT"
        assert setups[0].name == "TestStrategy/BTCUSDT", "Should have BTCUSDT suffix"

        # Check second setup (BCHUSDT)
        assert setups[1].setup_type == SetupTypes.STRATEGY, "Should be strategy type"
        assert len(setups[1].instruments) == 1, "Should have only 1 instrument"
        assert setups[1].instruments[0].symbol == "BCHUSDT", "Should be BCHUSDT"
        assert setups[1].name == "TestStrategy/BCHUSDT", "Should have BCHUSDT suffix"

    def test_strategy_with_tracker_and_run_separate_instruments(self):
        """Test that strategy+tracker works with run_separate_instruments=True."""
        # Get test instruments
        instruments = [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "ETHUSDT"]]
        assert all([x and isinstance(x, Instrument) for x in instruments]), "Got wrong instruments"
        instruments = cast(list[Instrument], instruments)

        # Test with tracker and run_separate_instruments=True
        setups = recognize_simulation_configuration(
            "TestWithTracker",
            [TestStrategy(), AtrRiskTracker(None, None, "1h", 10)],
            instruments,
            ["BINANCE.UM"],
            10_000,
            "USDT",
            "vip0_usdt",
            "1Min",
            True,
            run_separate_instruments=True,
        )

        # Should have 2 setups with tracker, one per instrument
        assert len(setups) == 2, "Should have 2 separate setups with tracker"

        # Check both setups
        for i, setup in enumerate(setups):
            assert setup.setup_type == SetupTypes.STRATEGY_AND_TRACKER, "Should be strategy+tracker type"
            assert len(setup.instruments) == 1, "Should have only 1 instrument per setup"
            assert setup.tracker is not None, "Should have tracker"
            assert setup.instruments[0] == instruments[i], f"Should have correct instrument {instruments[i].symbol}"
            assert setup.name == f"TestWithTracker/{instruments[i].symbol}", "Should have instrument suffix"

    def test_signals_with_tracker_split_the_signals_per_instrument(self):
        instruments = [
            Instrument(s, MarketType.SWAP, "BINANCE.UM", s[:-4], "USDT", "USDT", s, 0.1, 0.001, 0.001)
            for s in ["BTCUSDT", "ETHUSDT"]
        ]
        signals = pd.DataFrame(
            {"BTCUSDT": [1.0, 0.0], "ETHUSDT": [0.0, -1.0]}, index=pd.date_range("2024-01-01", periods=2, freq="1h")
        )
        tracker = AtrRiskTracker(None, None, "1h", 10)

        setups = recognize_simulation_configuration(
            "Signals", [signals, tracker], instruments, ["BINANCE.UM"], 10_000, "USDT", "vip0_usdt", "1Min", True,
            run_separate_instruments=True,
        )  # fmt: skip

        assert len(setups) == 2
        for setup in sorted(setups, key=lambda s: s.name):
            symbol = setup.instruments[0].symbol
            assert setup.setup_type == SetupTypes.SIGNAL_AND_TRACKER
            assert setup.name == f"Signals/{symbol}"
            assert setup.tracker is tracker
            pd.testing.assert_series_equal(setup.generator, signals[symbol])  # type: ignore[arg-type]

    def test_dict_strategies_with_run_separate_instruments(self):
        """Test that dictionary of strategies works with run_separate_instruments=True."""
        # Get test instruments
        instruments = [lookup.find_symbol("BINANCE.UM", s) for s in ["BTCUSDT", "ETHUSDT"]]
        assert all([x and isinstance(x, Instrument) for x in instruments]), "Got wrong instruments"
        instruments = cast(list[Instrument], instruments)

        # Test with dictionary of strategies and run_separate_instruments=True
        setups = recognize_simulation_configuration(
            "TestDict",
            {"StratA": TestStrategy(), "StratB": TestStrategy()},
            instruments,
            ["BINANCE.UM"],
            10_000,
            "USDT",
            "vip0_usdt",
            "1Min",
            True,
            run_separate_instruments=True,
        )

        # Should have 4 setups (2 strategies * 2 instruments)
        assert len(setups) == 4, "Should have 4 setups (2 strategies * 2 instruments)"

        # Check that we have the right combinations
        expected_names = [
            "TestDict/StratA/BTCUSDT",
            "TestDict/StratA/ETHUSDT",
            "TestDict/StratB/BTCUSDT",
            "TestDict/StratB/ETHUSDT",
        ]
        actual_names = [setup.name for setup in setups]
        for expected_name in expected_names:
            assert expected_name in actual_names, f"Missing expected setup name: {expected_name}"

        # Check that each setup has only one instrument
        for setup in setups:
            assert len(setup.instruments) == 1, "Each setup should have only 1 instrument"
            assert setup.setup_type == SetupTypes.STRATEGY, "Should be strategy type"
