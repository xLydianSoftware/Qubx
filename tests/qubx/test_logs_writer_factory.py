import pytest

from qubx.loggers import CsvFileLogsWriter, create_logs_writer


def test_unknown_logs_writer_lists_valid_types():
    with pytest.raises(ValueError, match="Unknown logs writer type: MongoDBLogsWriter.*PostgresLogsWriter"):
        create_logs_writer("MongoDBLogsWriter")


def test_known_logs_writer_is_created(tmp_path):
    writer = create_logs_writer(
        "CsvFileLogsWriter", {"account_id": "a", "strategy_id": "s", "run_id": "r", "log_folder": str(tmp_path)}
    )
    assert isinstance(writer, CsvFileLogsWriter)
