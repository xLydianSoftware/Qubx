"""Tests for automatic restorer inference from logger config."""

from qubx.utils.runner.configs import LoggingConfig
from qubx.utils.runner.runner import _infer_restorer_from_logger


class TestInferRestorerFromLogger:

    def test_postgres_logger(self):
        logging_config = LoggingConfig(
            logger="PostgresLogsWriter",
            position_interval="10Sec",
            portfolio_interval="5Min",
            args={
                "postgres_uri": "postgresql://user:pass@db:5432/mydb",
                "table_prefix": "my_prefix",
            },
        )
        result = _infer_restorer_from_logger(logging_config, "my_strategy")

        assert result is not None
        assert result.type == "PostgresStateRestorer"
        assert result.parameters["strategy_name"] == "my_strategy"
        assert result.parameters["postgres_uri"] == "postgresql://user:pass@db:5432/mydb"
        assert result.parameters["table_prefix"] == "my_prefix"

    def test_postgres_logger_defaults(self):
        logging_config = LoggingConfig(
            logger="PostgresLogsWriter",
            position_interval="10Sec",
            portfolio_interval="5Min",
        )
        result = _infer_restorer_from_logger(logging_config, "strat")

        assert result.parameters["postgres_uri"] == "postgresql://localhost:5432/qubx_logs"
        assert result.parameters["table_prefix"] == "qubx_logs"

    def test_mongo_logger(self):
        logging_config = LoggingConfig(
            logger="MongoDBLogsWriter",
            position_interval="10Sec",
            portfolio_interval="5Min",
            args={
                "mongo_uri": "mongodb://mongo:27017/",
                "db_name": "mydb",
                "collection_name_prefix": "logs",
            },
        )
        result = _infer_restorer_from_logger(logging_config, "my_strategy")

        assert result is not None
        assert result.type == "MongoDBStateRestorer"
        assert result.parameters["strategy_name"] == "my_strategy"
        assert result.parameters["mongo_uri"] == "mongodb://mongo:27017/"
        assert result.parameters["db_name"] == "mydb"
        assert result.parameters["collection_name_prefix"] == "logs"

    def test_csv_logger(self):
        logging_config = LoggingConfig(
            logger="CsvFileLogsWriter",
            position_interval="10Sec",
            portfolio_interval="5Min",
            args={"log_folder": "/data/logs"},
        )
        result = _infer_restorer_from_logger(logging_config, "my_strategy")

        assert result is not None
        assert result.type == "CsvStateRestorer"
        assert result.parameters["base_dir"] == "/data/logs"
        assert result.parameters["strategy_name"] == "my_strategy"

    def test_csv_logger_defaults(self):
        logging_config = LoggingConfig(
            logger="CsvFileLogsWriter",
            position_interval="10Sec",
            portfolio_interval="5Min",
        )
        result = _infer_restorer_from_logger(logging_config, "strat")

        assert result.parameters["base_dir"] == "logs"

    def test_inmemory_logger_returns_none(self):
        logging_config = LoggingConfig(
            logger="InMemoryLogsWriter",
            position_interval="10Sec",
            portfolio_interval="5Min",
        )
        result = _infer_restorer_from_logger(logging_config, "strat")

        assert result is None

    def test_unknown_logger_returns_none(self):
        logging_config = LoggingConfig(
            logger="SomeCustomLogger",
            position_interval="10Sec",
            portfolio_interval="5Min",
        )
        result = _infer_restorer_from_logger(logging_config, "strat")

        assert result is None
