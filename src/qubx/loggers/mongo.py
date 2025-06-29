from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Any

from pymongo import MongoClient

from qubx.core.loggers import LogsWriter


class MongoDBLogsWriter(LogsWriter):
    """
    MongoDB implementation of LogsWriter interface.
    Writes log data to a single MongoDB collection asynchronously.
    Supports TTL expiration via index on 'timestamp' field.
    """

    def __init__(
        self,
        account_id: str,
        strategy_id: str,
        run_id: str,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "default_logs_db",
        collection_name_prefix: str = "qubx_logs",
        pool_size: int = 4,
        ttl_seconds: int = 86400,
    ) -> None:
        super().__init__(account_id, strategy_id, run_id)
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.pool = ThreadPool(pool_size)
        self.collection_name_prefix = collection_name_prefix

        # Ensure TTL index exists on the 'timestamp' field
        self.db[f"{collection_name_prefix}_positions"].create_index("timestamp", expireAfterSeconds=ttl_seconds)
        self.db[f"{collection_name_prefix}_portfolio"].create_index("timestamp", expireAfterSeconds=ttl_seconds)
        self.db[f"{collection_name_prefix}_executions"].create_index("timestamp", expireAfterSeconds=ttl_seconds)
        self.db[f"{collection_name_prefix}_signals"].create_index("timestamp", expireAfterSeconds=ttl_seconds)
        self.db[f"{collection_name_prefix}_targets"].create_index("timestamp", expireAfterSeconds=ttl_seconds)
        self.db[f"{collection_name_prefix}_balance"].create_index("timestamp", expireAfterSeconds=ttl_seconds)

    def _attach_metadata(self, data: list[dict[str, Any]], log_type: str) -> list[dict[str, Any]]:
        now = datetime.utcnow()
        return [
            {
                **d,
                "run_id": self.run_id,
                "account_id": self.account_id,
                "strategy_name": self.strategy_id,
                "log_type": log_type,
                "timestamp": now,
            }
            for d in data
        ]

    def _do_write(self, log_type: str, data: list[dict[str, Any]]):
        docs = self._attach_metadata(data, log_type)
        self.db[f"{self.collection_name_prefix}_{log_type}"].insert_many(docs)

    def write_data(self, log_type: str, data: list[dict[str, Any]]):
        if len(data) > 0:
            self.pool.apply_async(
                self._do_write,
                (
                    log_type,
                    data,
                ),
            )

    def flush_data(self):
        pass

    def close(self):
        self.pool.close()
        self.pool.join()
        self.client.close()
