from pathlib import Path

from qubx.utils.runner.configs import (
    EmissionConfig,
    EmitterConfig,
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
    assert config.live is not None
    assert set(config.live.exchanges.keys()) == {"BINANCE.UM", "KRAKEN.F"}
    assert config.aux is not None
    assert config.aux.reader == "mqdb::nebula"

    # test config without aux reader (ok)
    no_aux_yaml = CONFIGS_DIR / "no_aux.yaml"
    config = load_strategy_config_from_yaml(no_aux_yaml)
    assert config.aux is None

    # test config without exchanges (loads but live config is None)
    no_exchanges_yaml = CONFIGS_DIR / "no_exchanges.yaml"
    config = load_strategy_config_from_yaml(no_exchanges_yaml)
    assert config.live is None


def test_metrics_notifiers_config_parsing():
    """Test parsing a config file with metrics and notifiers sections."""
    config_yaml = CONFIGS_DIR / "metrics_notifiers.yaml"
    config = load_strategy_config_from_yaml(config_yaml)

    # Check metric_emission
    assert config.live is not None
    assert config.live.emission is not None
    assert len(config.live.emission.emitters) == 1
    assert config.live.emission.emitters[0].emitter == "PrometheusMetricEmitter"
    assert config.live.emission.emitters[0].parameters["pushgateway_url"] == "http://prometheus-pushgateway:9091"
    assert config.live.emission.emitters[0].parameters["expose_http"] is True
    assert config.live.emission.emitters[0].parameters["http_port"] == 8000
    assert config.live.emission.stats_interval == "1m"  # Default value

    # Check notifiers
    assert config.live.notifiers is not None
    assert len(config.live.notifiers) == 2
    assert config.live.notifiers[0].notifier == "SlackLifecycleNotifier"
    assert config.live.notifiers[0].parameters["webhook_url"] == "https://hooks.slack.com/services/XXX/YYY/ZZZ"
    assert config.live.notifiers[0].parameters["environment"] == "production"
    assert config.live.notifiers[1].notifier == "NullLifecycleNotifier"


def test_emtter_config():
    config = EmitterConfig(
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


def test_metric_emission_config():
    """Test the MetricEmissionConfig class."""
    config = EmissionConfig(
        stats_interval="2m",
        stats_to_emit=["total_capital", "net_leverage", "gross_leverage"],
        emitters=[
            EmitterConfig(
                emitter="PrometheusMetricEmitter",
                parameters={
                    "pushgateway_url": "http://prometheus-pushgateway:9091",
                },
            ),
            EmitterConfig(
                emitter="NullMetricEmitter",
                parameters={},
            ),
        ],
    )

    assert config.stats_interval == "2m"
    assert config.stats_to_emit == ["total_capital", "net_leverage", "gross_leverage"]
    assert len(config.emitters) == 2
    assert config.emitters[0].emitter == "PrometheusMetricEmitter"
    assert config.emitters[0].parameters["pushgateway_url"] == "http://prometheus-pushgateway:9091"
    assert config.emitters[1].emitter == "NullMetricEmitter"


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


def test_validate_valid_config():
    """Test validation of a valid configuration."""
    from qubx.utils.runner.configs import validate_strategy_config

    config_yaml = CONFIGS_DIR / "basic.yaml"
    result = validate_strategy_config(config_yaml, check_imports=False)

    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_nonexistent_config():
    """Test validation of a nonexistent file."""
    from qubx.utils.runner.configs import validate_strategy_config

    result = validate_strategy_config(CONFIGS_DIR / "nonexistent.yaml", check_imports=False)

    assert result.valid is False
    assert len(result.errors) == 1
    assert "not found" in result.errors[0].lower()


def test_validate_no_exchanges_config():
    """Test validation of config without exchanges."""
    from qubx.utils.runner.configs import validate_strategy_config

    config_yaml = CONFIGS_DIR / "no_exchanges.yaml"
    result = validate_strategy_config(config_yaml, check_imports=False)

    # Should be valid but may have warnings
    assert result.valid is True
