"""
Unit tests for configuration parsing.
"""

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from qubx.utils.runner.configs import (
    ExchangeConfig,
    LoggingConfig,
    MetricConfig,
    NotifierConfig,
    StrategyConfig,
    load_strategy_config_from_yaml,
)


class TestConfigParsing(unittest.TestCase):
    """Test the configuration parsing."""

    def setUp(self):
        """Set up the test case."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"

    def tearDown(self):
        """Clean up after the test case."""
        self.temp_dir.cleanup()

    def test_load_strategy_config(self):
        """Test loading a strategy configuration from a YAML file."""
        # Create a sample configuration
        config = {
            "strategy": "my_strategy.MyStrategy",
            "parameters": {
                "param1": "value1",
                "param2": 2,
            },
            "exchanges": {
                "BINANCE.UM": {
                    "connector": "ccxt",
                    "universe": ["BTC-USDT", "ETH-USDT"],
                }
            },
            "logging": {
                "logger": "CsvLogger",
                "position_interval": "1m",
                "portfolio_interval": "5m",
            },
            "exporters": [
                {
                    "exporter": "SlackExporter",
                    "parameters": {
                        "strategy_name": "MyStrategy",
                        "signals_webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
                    },
                }
            ],
            "metrics": [
                {
                    "emitter": "PrometheusMetricEmitter",
                    "parameters": {
                        "pushgateway_url": "http://prometheus-pushgateway:9091",
                        "expose_http": True,
                        "http_port": 8000,
                    },
                }
            ],
            "notifiers": [
                {
                    "notifier": "SlackLifecycleNotifier",
                    "parameters": {
                        "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
                        "environment": "production",
                    },
                }
            ],
        }

        # Write the configuration to a file
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Load the configuration
        loaded_config = load_strategy_config_from_yaml(self.config_path)

        # Check that the configuration was loaded correctly
        self.assertEqual(loaded_config.strategy, "my_strategy.MyStrategy")
        self.assertEqual(loaded_config.parameters["param1"], "value1")
        self.assertEqual(loaded_config.parameters["param2"], 2)
        self.assertEqual(loaded_config.exchanges["BINANCE.UM"].connector, "ccxt")
        self.assertEqual(loaded_config.exchanges["BINANCE.UM"].universe, ["BTC-USDT", "ETH-USDT"])
        self.assertEqual(loaded_config.logging.logger, "CsvLogger")
        self.assertEqual(loaded_config.logging.position_interval, "1m")
        self.assertEqual(loaded_config.logging.portfolio_interval, "5m")
        self.assertEqual(loaded_config.exporters[0].exporter, "SlackExporter")
        self.assertEqual(loaded_config.exporters[0].parameters["strategy_name"], "MyStrategy")
        self.assertEqual(
            loaded_config.exporters[0].parameters["signals_webhook_url"],
            "https://hooks.slack.com/services/XXX/YYY/ZZZ",
        )
        self.assertEqual(loaded_config.metrics[0].emitter, "PrometheusMetricEmitter")
        self.assertEqual(loaded_config.metrics[0].parameters["pushgateway_url"], "http://prometheus-pushgateway:9091")
        self.assertEqual(loaded_config.metrics[0].parameters["expose_http"], True)
        self.assertEqual(loaded_config.metrics[0].parameters["http_port"], 8000)
        self.assertEqual(loaded_config.notifiers[0].notifier, "SlackLifecycleNotifier")
        self.assertEqual(
            loaded_config.notifiers[0].parameters["webhook_url"], "https://hooks.slack.com/services/XXX/YYY/ZZZ"
        )
        self.assertEqual(loaded_config.notifiers[0].parameters["environment"], "production")

    def test_load_strategy_config_with_key(self):
        """Test loading a strategy configuration from a YAML file with a key."""
        # Create a sample configuration
        config = {
            "my_strategy": {
                "strategy": "my_strategy.MyStrategy",
                "parameters": {
                    "param1": "value1",
                    "param2": 2,
                },
                "exchanges": {
                    "BINANCE.UM": {
                        "connector": "ccxt",
                        "universe": ["BTC-USDT", "ETH-USDT"],
                    }
                },
                "logging": {
                    "logger": "CsvLogger",
                    "position_interval": "1m",
                    "portfolio_interval": "5m",
                },
            }
        }

        # Write the configuration to a file
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Load the configuration
        loaded_config = load_strategy_config_from_yaml(self.config_path, key="my_strategy")

        # Check that the configuration was loaded correctly
        self.assertEqual(loaded_config.strategy, "my_strategy.MyStrategy")
        self.assertEqual(loaded_config.parameters["param1"], "value1")
        self.assertEqual(loaded_config.parameters["param2"], 2)
        self.assertEqual(loaded_config.exchanges["BINANCE.UM"].connector, "ccxt")
        self.assertEqual(loaded_config.exchanges["BINANCE.UM"].universe, ["BTC-USDT", "ETH-USDT"])
        self.assertEqual(loaded_config.logging.logger, "CsvLogger")
        self.assertEqual(loaded_config.logging.position_interval, "1m")
        self.assertEqual(loaded_config.logging.portfolio_interval, "5m")

    def test_metric_config(self):
        """Test the MetricConfig class."""
        config = MetricConfig(
            emitter="PrometheusMetricEmitter",
            parameters={
                "pushgateway_url": "http://prometheus-pushgateway:9091",
                "expose_http": True,
                "http_port": 8000,
            },
        )

        self.assertEqual(config.emitter, "PrometheusMetricEmitter")
        self.assertEqual(config.parameters["pushgateway_url"], "http://prometheus-pushgateway:9091")
        self.assertEqual(config.parameters["expose_http"], True)
        self.assertEqual(config.parameters["http_port"], 8000)

    def test_notifier_config(self):
        """Test the NotifierConfig class."""
        config = NotifierConfig(
            notifier="SlackLifecycleNotifier",
            parameters={
                "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
                "environment": "production",
            },
        )

        self.assertEqual(config.notifier, "SlackLifecycleNotifier")
        self.assertEqual(config.parameters["webhook_url"], "https://hooks.slack.com/services/XXX/YYY/ZZZ")
        self.assertEqual(config.parameters["environment"], "production")


if __name__ == "__main__":
    unittest.main()
