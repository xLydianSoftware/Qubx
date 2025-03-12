from pathlib import Path

import pytest

from qubx.utils.runner.configs import (
    MetricConfig,
    NotifierConfig,
    load_strategy_config_from_yaml,
)

CONFIGS_DIR = Path(__file__).parent / "configs"


def test_strategy_config_parsing():
    # test basic config
    basic_yaml = CONFIGS_DIR / "basic.yaml"
    config = load_strategy_config_from_yaml(basic_yaml)
    assert config.strategy == "sty.models.portfolio.pigone.TestPig1"
    assert config.parameters["top_capitalization_percentile"] == 2
    assert set(config.exchanges.keys()) == {"BINANCE.UM", "KRAKEN.F"}
    assert config.aux is not None
    assert config.aux.reader == "mqdb::nebula"

    # test config without aux reader (ok)
    no_aux_yaml = CONFIGS_DIR / "no_aux.yaml"
    config = load_strategy_config_from_yaml(no_aux_yaml)
    assert config.aux is None

    # test config without exchanges (throw exception)
    no_exchanges_yaml = CONFIGS_DIR / "no_exchanges.yaml"
    with pytest.raises(ValueError):
        load_strategy_config_from_yaml(no_exchanges_yaml)


def test_metrics_notifiers_config_parsing():
    """Test parsing a config file with metrics and notifiers sections."""
    config_yaml = CONFIGS_DIR / "metrics_notifiers.yaml"
    config = load_strategy_config_from_yaml(config_yaml)

    # Check metrics
    assert config.metrics is not None
    assert len(config.metrics) == 2
    assert config.metrics[0].emitter == "PrometheusMetricEmitter"
    assert config.metrics[0].parameters["pushgateway_url"] == "http://prometheus-pushgateway:9091"
    assert config.metrics[0].parameters["expose_http"] is True
    assert config.metrics[0].parameters["http_port"] == 8000
    assert config.metrics[1].emitter == "NullMetricEmitter"

    # Check notifiers
    assert config.notifiers is not None
    assert len(config.notifiers) == 2
    assert config.notifiers[0].notifier == "SlackLifecycleNotifier"
    assert config.notifiers[0].parameters["webhook_url"] == "https://hooks.slack.com/services/XXX/YYY/ZZZ"
    assert config.notifiers[0].parameters["environment"] == "production"
    assert config.notifiers[1].notifier == "NullLifecycleNotifier"


def test_metric_config():
    """Test the MetricConfig class."""
    config = MetricConfig(
        emitter="PrometheusMetricEmitter",
        parameters={
            "pushgateway_url": "http://prometheus-pushgateway:9091",
            "expose_http": True,
            "http_port": 8000,
        },
    )

    assert config.emitter == "PrometheusMetricEmitter"
    assert config.parameters["pushgateway_url"] == "http://prometheus-pushgateway:9091"
    assert config.parameters["expose_http"] is True
    assert config.parameters["http_port"] == 8000


def test_notifier_config():
    """Test the NotifierConfig class."""
    config = NotifierConfig(
        notifier="SlackLifecycleNotifier",
        parameters={
            "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "environment": "production",
        },
    )

    assert config.notifier == "SlackLifecycleNotifier"
    assert config.parameters["webhook_url"] == "https://hooks.slack.com/services/XXX/YYY/ZZZ"
    assert config.parameters["environment"] == "production"
