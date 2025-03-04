import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qubx.core.interfaces import ITradeDataExport
from qubx.exporters.composite import CompositeExporter
from qubx.exporters.formatters.slack import SlackMessageFormatter
from qubx.exporters.redis_streams import RedisStreamsExporter
from qubx.exporters.slack import SlackExporter
from qubx.utils.runner.configs import load_strategy_config_from_yaml
from qubx.utils.runner.runner import _create_exporters, _resolve_env_vars

CONFIGS_DIR = Path(__file__).parent / "configs"


def test_load_exporters_config():
    """Test loading a configuration with exporters."""
    exporters_yaml = CONFIGS_DIR / "exporters.yaml"
    config = load_strategy_config_from_yaml(exporters_yaml)

    # Check that exporters are loaded correctly
    assert config.exporters is not None
    assert len(config.exporters) == 2

    # Check first exporter (Slack)
    assert config.exporters[0].exporter == "SlackExporter"
    assert config.exporters[0].parameters["signals_webhook_url"] == "env:SLACK_WEBHOOK_URL"
    assert config.exporters[0].parameters["export_signals"] is True
    assert config.exporters[0].parameters["export_targets"] is True
    assert config.exporters[0].parameters["export_position_changes"] is False
    assert config.exporters[0].parameters["strategy_emoji"] == ":rocket:"
    assert config.exporters[0].parameters["include_account_info"] is True

    # Check formatter configuration
    formatter_config = config.exporters[0].parameters["formatter"]
    assert formatter_config["class"] == "SlackMessageFormatter"
    assert formatter_config["args"]["strategy_emoji"] == ":chart_with_upwards_trend:"
    assert formatter_config["args"]["include_account_info"] is True

    # Check second exporter (Redis)
    assert config.exporters[1].exporter == "RedisStreamsExporter"
    assert config.exporters[1].parameters["redis_url"] == "env:REDIS_URL"
    assert config.exporters[1].parameters["signals_stream"] == "strategy_signals"
    assert config.exporters[1].parameters["export_signals"] is True
    assert config.exporters[1].parameters["export_targets"] is False
    assert config.exporters[1].parameters["export_position_changes"] is True
    assert config.exporters[1].parameters["max_stream_length"] == 2000


@patch("qubx.exporters.composite.CompositeExporter")
@patch("qubx.utils.runner.runner.class_import")
@patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test", "REDIS_URL": "redis://localhost:6379/0"})
def test_create_exporters(mock_class_import, mock_composite_class):
    """Test creating exporters from configuration."""
    # Mock the class imports
    mock_slack_exporter = MagicMock(spec=SlackExporter)
    mock_redis_exporter = MagicMock(spec=RedisStreamsExporter)
    mock_slack_formatter = MagicMock(spec=SlackMessageFormatter)
    mock_composite_exporter = MagicMock(spec=CompositeExporter)

    # Set up the mock for CompositeExporter
    mock_composite_class.return_value = mock_composite_exporter

    def side_effect(class_name):
        if class_name == "qubx.exporters.SlackExporter":
            return lambda **kwargs: mock_slack_exporter
        elif class_name == "qubx.exporters.RedisStreamsExporter":
            return lambda **kwargs: mock_redis_exporter
        elif class_name == "qubx.exporters.formatters.SlackMessageFormatter":
            return lambda **kwargs: mock_slack_formatter
        else:
            raise ValueError(f"Unexpected class name: {class_name}")

    mock_class_import.side_effect = side_effect

    # Load the configuration
    exporters_yaml = CONFIGS_DIR / "exporters.yaml"
    config = load_strategy_config_from_yaml(exporters_yaml)

    # Create exporters
    exporter = _create_exporters(config, "TestStrategy")

    # Check that the composite exporter was created
    assert exporter is mock_composite_exporter

    # Check that class_import was called with the correct class names
    mock_class_import.assert_any_call("qubx.exporters.SlackExporter")
    mock_class_import.assert_any_call("qubx.exporters.RedisStreamsExporter")
    mock_class_import.assert_any_call("qubx.exporters.formatters.SlackMessageFormatter")

    # Check that CompositeExporter was called with a list of exporters
    mock_composite_class.assert_called_once()


@patch("qubx.utils.runner.runner.class_import")
@patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"})
def test_create_single_exporter(mock_class_import):
    """Test creating a single exporter from configuration."""
    # Create a mock configuration with a single exporter
    config = load_strategy_config_from_yaml(CONFIGS_DIR / "exporters.yaml")
    config.exporters = [config.exporters[0]]  # Keep only the Slack exporter

    # Mock the class imports
    mock_slack_exporter = MagicMock(spec=SlackExporter)
    mock_slack_formatter = MagicMock(spec=SlackMessageFormatter)

    def side_effect(class_name):
        if class_name == "qubx.exporters.SlackExporter":
            return lambda **kwargs: mock_slack_exporter
        elif class_name == "qubx.exporters.formatters.SlackMessageFormatter":
            return lambda **kwargs: mock_slack_formatter
        else:
            raise ValueError(f"Unexpected class name: {class_name}")

    mock_class_import.side_effect = side_effect

    # Create exporters
    exporter = _create_exporters(config, "TestStrategy")

    # Check that the single exporter was returned directly (not wrapped in a composite)
    assert exporter is mock_slack_exporter

    # Check that class_import was called with the correct class names
    mock_class_import.assert_any_call("qubx.exporters.SlackExporter")
    mock_class_import.assert_any_call("qubx.exporters.formatters.SlackMessageFormatter")


def test_no_exporters_config():
    """Test that None is returned when no exporters are configured."""
    # Create a mock configuration with no exporters
    config = load_strategy_config_from_yaml(CONFIGS_DIR / "exporters.yaml")
    config.exporters = None

    # Create exporters
    exporter = _create_exporters(config, "TestStrategy")

    # Check that None was returned
    assert exporter is None


@patch.dict(os.environ, {"TEST_ENV_VAR": "test_value"})
def test_resolve_env_vars():
    """Test resolving environment variables."""
    # Test with env var
    result = _resolve_env_vars("env:TEST_ENV_VAR")
    assert result == "test_value"

    # Test with regular string
    result = _resolve_env_vars("regular_string")
    assert result == "regular_string"

    # Test with non-string values
    assert _resolve_env_vars(123) == 123
    assert _resolve_env_vars(True) is True
    assert _resolve_env_vars(None) is None

    # Test with dictionary
    test_dict = {"key": "value"}
    result = _resolve_env_vars(test_dict)
    assert result == test_dict
