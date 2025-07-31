"""Integration tests for aux config structure and helper functions."""

from unittest.mock import Mock, patch

import pytest

from qubx.data.composite import CompositeReader
from qubx.utils.runner.configs import (
    LiveConfig,
    ReaderConfig,
    SimulationConfig,
    StrategyConfig,
    normalize_aux_config,
    resolve_aux_config,
)
from qubx.utils.runner.runner import _construct_aux_reader


class TestAuxConfigIntegration:
    """Integration tests for aux configuration functionality."""

    def test_normalize_aux_config_none(self):
        """Test normalize_aux_config with None input."""
        result = normalize_aux_config(None)
        assert result == []

    def test_normalize_aux_config_single_reader(self):
        """Test normalize_aux_config with single ReaderConfig."""
        reader_config = ReaderConfig(reader="mqdb", args={"host": "localhost"})
        result = normalize_aux_config(reader_config)

        assert len(result) == 1
        assert result[0] == reader_config

    def test_normalize_aux_config_list_of_readers(self):
        """Test normalize_aux_config with list of ReaderConfig."""
        reader_configs = [
            ReaderConfig(reader="mqdb", args={"host": "localhost"}),
            ReaderConfig(reader="csv", args={"path": "/data"}),
        ]
        result = normalize_aux_config(reader_configs)

        assert len(result) == 2
        assert result == reader_configs

    def test_resolve_aux_config_no_override(self):
        """Test resolve_aux_config when no section override is provided."""
        global_aux = ReaderConfig(reader="mqdb", args={"host": "global"})
        result = resolve_aux_config(global_aux, None)

        assert len(result) == 1
        assert result[0] == global_aux

    def test_resolve_aux_config_with_override(self):
        """Test resolve_aux_config when section override is provided."""
        global_aux = ReaderConfig(reader="mqdb", args={"host": "global"})
        section_aux = ReaderConfig(reader="csv", args={"path": "/data"})

        result = resolve_aux_config(global_aux, section_aux)

        assert len(result) == 1
        assert result[0] == section_aux  # Section should override global

    def test_resolve_aux_config_with_list_override(self):
        """Test resolve_aux_config when section override is a list."""
        global_aux = ReaderConfig(reader="mqdb", args={"host": "global"})
        section_aux = [
            ReaderConfig(reader="csv", args={"path": "/data1"}),
            ReaderConfig(reader="csv", args={"path": "/data2"}),
        ]

        result = resolve_aux_config(global_aux, section_aux)

        assert len(result) == 2
        assert result == section_aux

    def test_resolve_aux_config_both_none(self):
        """Test resolve_aux_config when both global and section are None."""
        result = resolve_aux_config(None, None)
        assert result == []

    @patch("qubx.utils.runner.runner.construct_reader")
    def test_construct_aux_reader_empty_list(self, mock_construct):
        """Test _construct_aux_reader with empty config list."""
        result = _construct_aux_reader([])
        assert result is None
        mock_construct.assert_not_called()

    @patch("qubx.utils.runner.runner.construct_reader")
    def test_construct_aux_reader_single_config(self, mock_construct):
        """Test _construct_aux_reader with single config."""
        mock_reader = Mock()
        mock_construct.return_value = mock_reader

        config = ReaderConfig(reader="mqdb", args={"host": "localhost"})
        result = _construct_aux_reader([config])

        assert result == mock_reader
        mock_construct.assert_called_once_with(config)

    @patch("qubx.utils.runner.runner.construct_reader")
    def test_construct_aux_reader_multiple_configs(self, mock_construct):
        """Test _construct_aux_reader with multiple configs creates CompositeReader."""
        mock_reader1 = Mock()
        mock_reader2 = Mock()
        mock_construct.side_effect = [mock_reader1, mock_reader2]

        configs = [
            ReaderConfig(reader="mqdb", args={"host": "localhost"}),
            ReaderConfig(reader="csv", args={"path": "/data"}),
        ]
        result = _construct_aux_reader(configs)

        assert isinstance(result, CompositeReader)
        assert len(result.readers) == 2
        assert result.readers[0] == mock_reader1
        assert result.readers[1] == mock_reader2

    @patch("qubx.utils.runner.runner.construct_reader")
    def test_construct_aux_reader_with_failures(self, mock_construct):
        """Test _construct_aux_reader handles reader construction failures."""
        mock_reader = Mock()
        mock_construct.side_effect = [Exception("Failed to create reader"), mock_reader]

        configs = [ReaderConfig(reader="invalid", args={}), ReaderConfig(reader="mqdb", args={"host": "localhost"})]
        result = _construct_aux_reader(configs)

        # Should return the single successful reader, not a CompositeReader
        assert result == mock_reader

    @patch("qubx.utils.runner.runner.construct_reader")
    def test_construct_aux_reader_all_failures(self, mock_construct):
        """Test _construct_aux_reader when all reader constructions fail."""
        mock_construct.side_effect = Exception("All readers fail")

        configs = [ReaderConfig(reader="invalid1", args={}), ReaderConfig(reader="invalid2", args={})]
        result = _construct_aux_reader(configs)

        assert result is None

    def test_strategy_config_single_aux_reader(self):
        """Test StrategyConfig can accept single aux reader."""
        config_dict = {
            "name": "test_strategy",
            "strategy": "test.Strategy",
            "aux": {"reader": "mqdb", "args": {"host": "localhost"}},
        }

        config = StrategyConfig(**config_dict)

        assert isinstance(config.aux, ReaderConfig)
        assert config.aux.reader == "mqdb"
        assert config.aux.args["host"] == "localhost"

    def test_strategy_config_multiple_aux_readers(self):
        """Test StrategyConfig can accept list of aux readers."""
        config_dict = {
            "name": "test_strategy",
            "strategy": "test.Strategy",
            "aux": [{"reader": "mqdb", "args": {"host": "db1"}}, {"reader": "csv", "args": {"path": "/data"}}],
        }

        config = StrategyConfig(**config_dict)

        assert isinstance(config.aux, list)
        assert len(config.aux) == 2
        assert config.aux[0].reader == "mqdb"
        assert config.aux[1].reader == "csv"

    def test_simulation_config_aux_override(self):
        """Test SimulationConfig can have aux override."""
        sim_config_dict = {
            "capital": 100000,
            "instruments": ["BTC/USD"],
            "start": "2023-01-01",
            "stop": "2023-12-31",
            "aux": {"reader": "csv", "args": {"path": "/sim_data"}},
        }

        sim_config = SimulationConfig(**sim_config_dict)

        assert isinstance(sim_config.aux, ReaderConfig)
        assert sim_config.aux.reader == "csv"

    def test_live_config_aux_override(self):
        """Test LiveConfig can have aux override."""
        live_config_dict = {
            "exchanges": {"BINANCE": {"connector": "ccxt", "universe": ["BTCUSDT"]}},
            "logging": {"logger": "CsvFileLogsWriter", "position_interval": "5m", "portfolio_interval": "5m"},
            "aux": [
                {"reader": "ccxt", "args": {"exchanges": ["BINANCE"]}},
                {"reader": "mqdb", "args": {"host": "live_db"}},
            ],
        }

        live_config = LiveConfig(**live_config_dict)

        assert isinstance(live_config.aux, list)
        assert len(live_config.aux) == 2
        assert live_config.aux[0].reader == "ccxt"
        assert live_config.aux[1].reader == "mqdb"

    def test_full_strategy_config_with_overrides(self):
        """Test complete StrategyConfig with section overrides."""
        config_dict = {
            "name": "test_strategy",
            "strategy": "test.Strategy",
            "aux": {"reader": "mqdb", "args": {"host": "global_db"}},
            "simulation": {
                "capital": 100000,
                "instruments": ["BTC/USD"],
                "start": "2023-01-01",
                "stop": "2023-12-31",
                "aux": {"reader": "csv", "args": {"path": "/sim_data"}},
            },
            "live": {
                "exchanges": {"BINANCE": {"connector": "ccxt", "universe": ["BTCUSDT"]}},
                "logging": {"logger": "CsvFileLogsWriter", "position_interval": "5m", "portfolio_interval": "5m"},
                "aux": [
                    {"reader": "ccxt", "args": {"exchanges": ["BINANCE"]}},
                    {"reader": "mqdb", "args": {"host": "live_db"}},
                ],
            },
        }

        config = StrategyConfig(**config_dict)

        # Test resolution logic
        sim_aux_configs = resolve_aux_config(config.aux, config.simulation.aux)
        live_aux_configs = resolve_aux_config(config.aux, config.live.aux)

        # Simulation should use override
        assert len(sim_aux_configs) == 1
        assert sim_aux_configs[0].reader == "csv"
        assert sim_aux_configs[0].args["path"] == "/sim_data"

        # Live should use override
        assert len(live_aux_configs) == 2
        assert live_aux_configs[0].reader == "ccxt"
        assert live_aux_configs[1].reader == "mqdb"
        assert live_aux_configs[1].args["host"] == "live_db"

    def test_backward_compatibility(self):
        """Test that old single-reader configs still work."""
        # Old style config (single reader)
        old_config_dict = {
            "name": "old_strategy",
            "strategy": "test.OldStrategy",
            "aux": {"reader": "mqdb", "args": {"host": "old_db"}},
        }

        config = StrategyConfig(**old_config_dict)
        aux_configs = resolve_aux_config(config.aux, None)

        assert len(aux_configs) == 1
        assert aux_configs[0].reader == "mqdb"
        assert aux_configs[0].args["host"] == "old_db"
