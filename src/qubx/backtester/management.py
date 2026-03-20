"""
Parquet-based backtest storage utilities — schemas, constants, and write helpers.

Used by:
    - qubx.core.metrics.TradingSessionResult  (result model)
    - qubx.backtester.management.BacktestStorage (query interface)
    - qubx.utils.results.SimulationResultsSaver (save / load)
    - qubx.utils.runner.runner.simulate_strategy (cloud detection, tag helpers)

Storage layout (single run)::

    {base_path}/
    └── {yaml.name}/                  # from cfg.name field (required)
        └── {ShortClass}/             # short strategy class name(s), multi joined with '+'
            └── YYYYMMDD_HHMMSS/      # unique per run
                ├── _status.parquet   # written first, updated live during simulation
                ├── _metadata.parquet # written on completion (all perf metrics)
                ├── portfolio.parquet
                ├── executions.parquet
                ├── signals.parquet
                ├── targets.parquet
                └── config.yaml       # attached config file

Storage layout (variation set)::

    {base_path}/
    └── {yaml.name}/
        └── {ShortClass}/
            └── YYYYMMDD_HHMMSS/
                ├── _status.parquet
                ├── _metadata.parquet  # N rows, one per variation — searchable by DuckDB
                ├── var_000/
                │   ├── portfolio.parquet
                │   ├── executions.parquet
                │   ├── signals.parquet
                │   └── targets.parquet
                ├── var_001/
                │   └── ...
                └── config.yaml

DuckDB examples (via BacktestStorage)::

    storage.search("sharpe > 2 AND mdd_pct < 25 AND list_contains(tags, 'momentum')")
    storage.status("running")
    storage.get_portfolio("my_strat/Nimble/20240301_120000", symbol="BTCUSDT", start="2024-01-01")
"""

import pandas as pd

from qubx.core.metrics import TradingSessionResult
from qubx.utils.misc import blue, cyan, green, magenta, red, yellow
from qubx.utils.results import SimulationResultsSaver
from qubx.utils.s3 import S3Client, is_cloud_path


class BacktestStorage:
    """
    Query interface for parquet-based backtest storage.
    Supports local directories and cloud paths (S3, GCS, Azure).

    Uses DuckDB for fast metadata search and data queries across all stored backtests.

    Storage layout (single run)::

        {base_path}/
        └── {yaml.name}/                  # from cfg.name field (required)
            └── {ShortClass}/             # short strategy class name(s)
                └── YYYYMMDD_HHMMSS/
                    ├── _status.parquet   # live progress, written by SimulationResultsSaver
                    ├── _metadata.parquet # completion metrics
                    ├── portfolio.parquet
                    ├── executions.parquet
                    ├── signals.parquet
                    ├── targets.parquet
                    ├── emitter_data.parquet
                    ├── transfers.parquet
                    └── config.yaml

    Examples::

        # - local storage
        storage = BacktestStorage("/backtests/")

        # - S3 storage (creds from env: QUBX_S3_KEY / AWS_ACCESS_KEY_ID)
        storage = BacktestStorage("s3://my-bucket/backtests/")

        # - search: full DuckDB SQL WHERE clause
        df = storage.search("sharpe > 2 AND mdd_pct < 25")
        df = storage.search("list_contains(tags, 'momentum') AND cagr > 0.3")
        df = storage.search("json_extract(parameters, '$.fast_period')::int > 10")
        df = storage.search()  # - all results

        # - live status dashboard
        df = storage.status("running")

        # - load result
        result = storage.load("my_strat/Nimble/20240301_120000")

        # - best variation from a variation set
        result = storage.load_best_variation("my_strat/Nimble/20240301_130000", by="sharpe")
    """

    def __init__(self, base_path: str, storage_options: dict | None = None):
        """
        Initialize BacktestStorage.

        Args:
            base_path: Root path for backtest storage (local dir or cloud URI)
            storage_options: Cloud storage credentials dict. None = uses default_s3_account from settings.
        """
        try:
            import duckdb

            self._duckdb = duckdb
        except ImportError:
            raise ImportError(
                "duckdb is required for BacktestStorage. "
                "Install with: pip install 'qubx[storage]' or pip install duckdb"
            )
        self.base_path = base_path.rstrip("/") + "/"
        self._is_cloud = is_cloud_path(base_path)

        # - for cloud paths: resolve credentials once
        self._storage_options: dict | None = (
            S3Client(storage_options=storage_options).storage_options if self._is_cloud else None
        )
        self._conn = self._duckdb.connect()

        if self._is_cloud:
            self._setup_cloud_duckdb()

    def _setup_cloud_duckdb(self) -> None:
        """Configure DuckDB httpfs extension for cloud storage access."""
        self._conn.execute("INSTALL httpfs; LOAD httpfs;")

        opts = self._storage_options or {}
        if not opts or "key" not in opts:
            return

        # Resolve region
        region = "auto"
        if "client_kwargs" in opts:
            region = opts["client_kwargs"].get("region_name", "auto")
        elif "region" in opts:
            region = opts["region"]

        # Build CREATE SECRET with scope matching our base_path so it takes
        # priority over any pre-existing broader secrets (e.g. from duckdb config).
        scope = self.base_path.rstrip("/")
        endpoint_clause = ""
        url_style_clause = ""
        if "endpoint_url" in opts:
            endpoint = opts["endpoint_url"].removeprefix("https://").removeprefix("http://")
            endpoint_clause = f"ENDPOINT '{endpoint}',"
            url_style_clause = "URL_STYLE 'path',"

        self._conn.execute(f"""
            CREATE OR REPLACE SECRET qubx_s3 (
                TYPE s3,
                KEY_ID '{opts['key']}',
                SECRET '{opts['secret']}',
                {endpoint_clause}
                {url_style_clause}
                REGION '{region}',
                SCOPE '{scope}'
            )
        """)

    def _glob(self, filename: str) -> str:
        """
        Build recursive glob pattern for a filename within base_path.
        """
        return f"{self.base_path}**/{filename}"

    def search(
        self,
        where: str | None = None,
        order_by: str = "sharpe DESC",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Search backtest metadata across all stored results using DuckDB SQL.

        The WHERE clause has full DuckDB SQL power — no restrictions::

            "sharpe > 2 AND mdd_pct < 25 AND author = 'alice'"
            "list_contains(tags, 'momentum') AND cagr > 0.3"
            "json_extract(parameters, '$.fast_period')::int > 10"
            "is_variation = false"
            "strategy_class LIKE '%Nimble%'"
            "start >= '2024-01-01' AND sharpe BETWEEN 1.5 AND 4.0"

        Regular backtests: one row per run.
        Variation sets: N rows per set (one per variation), all with is_variation=true.

        Args:
            where: DuckDB SQL WHERE clause, or None to return all results
            order_by: ORDER BY clause (default: "sharpe DESC")
            limit: Maximum rows to return

        Returns:
            pd.DataFrame with matching metadata rows
        """
        glob = self._glob(SimulationResultsSaver.METADATA_FILE)
        sql = f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        return self._conn.execute(sql).df()

    def status(self, filter_status: str | None = None) -> pd.DataFrame:
        """
        Get status of all simulations (running, completed, failed, pending).

        Reads _status.parquet files written by SimulationResultsSaver.
        Works in real-time — running simulations update their status every 1%.

        Args:
            filter_status: Filter by status value ('running', 'completed', 'failed', 'pending'),
                          or None to return all simulations

        Returns:
            pd.DataFrame with status rows, ordered by started_at DESC
        """
        glob = self._glob(SimulationResultsSaver.STATUS_FILE)
        sql = f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
        if filter_status:
            sql += f" WHERE status = '{filter_status}'"
        sql += " ORDER BY started_at DESC"
        return self._conn.execute(sql).df()

    def _load_from_path(self, run_path: str) -> TradingSessionResult:
        """
        Load a TradingSessionResult from parquet storage (local or cloud).

        Delegates to SimulationResultsSaver.load() which reads data files
        in parallel using a ThreadPoolExecutor.
        """
        return SimulationResultsSaver.load(run_path, self._storage_options)

    def load(self, backtest_id: str) -> TradingSessionResult:
        """
        Load a TradingSessionResult by backtest_id.

        Args:
            backtest_id: Relative path within base_path,
                        e.g. "my_strategy/Nimble/20240301_120000"

        Returns:
            TradingSessionResult with all data loaded from parquet
        """
        return self._load_from_path(f"{self.base_path}{backtest_id.strip('/')}/")

    def load_best_variation(
        self,
        variation_set_id: str,
        by: str = "sharpe",
        ascending: bool = False,
    ) -> TradingSessionResult:
        """
        Load the best-performing variation from a variation set.

        The variation set _metadata.parquet has one row per variation.
        Finds the best row by the given metric, then loads its data.

        Args:
            variation_set_id: Relative path to variation set root,
                             e.g. "my_strategy/Nimble/20240301_130000"
            by: Metric column to rank by (default: "sharpe")
            ascending: If True, load minimum instead of maximum (default: False)

        Returns:
            TradingSessionResult of the best variation
        """
        meta_path = f"{self.base_path}{variation_set_id.strip('/')}/{SimulationResultsSaver.METADATA_FILE}"
        order = "ASC" if ascending else "DESC"
        row = self._conn.execute(f"SELECT * FROM read_parquet('{meta_path}') ORDER BY {by} {order} LIMIT 1").df()

        if row.empty:
            raise ValueError(f"No variations found at '{variation_set_id}'")

        var_id = row["variation_id"].iloc[0]
        run_path = f"{self.base_path}{variation_set_id.strip('/')}/{var_id}/"
        return self._load_from_path(run_path)

    def get_portfolio(
        self,
        backtest_id: str,
        symbol: str | None = None,
        start: str | None = None,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """
        Get portfolio log data for a backtest.

        Portfolio is stored in wide format: one column per symbol metric
        (e.g. "BINANCE.UM:BTCUSDT_PnL", "BINANCE.UM:BTCUSDT_Commission").

        Args:
            backtest_id: Relative path within base_path
            symbol: If set, returns only columns containing this symbol name (case-insensitive)
            start: Start timestamp filter (inclusive)
            stop: Stop timestamp filter (inclusive)

        Returns:
            pd.DataFrame with portfolio data
        """
        path = f"{self.base_path}{backtest_id.strip('/')}/{SimulationResultsSaver.DATA_FILES['portfolio']}"
        return self._query_wide(path, symbol=symbol, start=start, stop=stop)

    def get_executions(
        self,
        backtest_id: str,
        symbol: str | None = None,
        start: str | None = None,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """
        Get execution log data for a backtest.

        Args:
            backtest_id: Relative path within base_path
            symbol: Filter rows by instrument column (case-insensitive match)
            start: Start timestamp filter (inclusive)
            stop: Stop timestamp filter (inclusive)

        Returns:
            pd.DataFrame with execution data
        """
        path = f"{self.base_path}{backtest_id.strip('/')}/{SimulationResultsSaver.DATA_FILES['executions']}"
        return self._query_long(path, symbol=symbol, start=start, stop=stop)

    def get_signals(
        self,
        backtest_id: str,
        symbol: str | None = None,
        start: str | None = None,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """
        Get signals log data for a backtest.

        Args:
            backtest_id: Relative path within base_path
            symbol: Filter rows by instrument column (case-insensitive match)
            start: Start timestamp filter (inclusive)
            stop: Stop timestamp filter (inclusive)

        Returns:
            pd.DataFrame with signals data
        """
        path = f"{self.base_path}{backtest_id.strip('/')}/{SimulationResultsSaver.DATA_FILES['signals']}"
        return self._query_long(path, symbol=symbol, start=start, stop=stop)

    def _query_wide(
        self,
        path: str,
        symbol: str | None = None,
        start: str | None = None,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """
        Query a wide-format parquet (portfolio log) with optional column/time filtering.
        Symbol filtering selects columns containing the symbol string using DuckDB COLUMNS().
        """
        conditions = []
        if start:
            conditions.append(f"timestamp >= '{start}'")
        if stop:
            conditions.append(f"timestamp <= '{stop}'")
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

        if symbol:
            sym = symbol.upper()
            # - DuckDB COLUMNS() lambda: select timestamp + _backtest_id + symbol columns
            sql = f"""
                SELECT COLUMNS(c -> c = 'timestamp' OR c = '_backtest_id'
                               OR contains(upper(c), '{sym}'))
                FROM read_parquet('{path}'){where_clause}
            """
        else:
            sql = f"SELECT * FROM read_parquet('{path}'){where_clause}"

        return self._conn.execute(sql).df()

    def _query_long(
        self,
        path: str,
        symbol: str | None = None,
        start: str | None = None,
        stop: str | None = None,
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Query a long-format parquet (executions, signals) with optional row filtering.
        Symbol filtering matches rows where symbol_col contains the symbol string.
        """
        conditions = []
        if start:
            conditions.append(f"timestamp >= '{start}'")
        if stop:
            conditions.append(f"timestamp <= '{stop}'")
        if symbol:
            conditions.append(f"contains(upper({symbol_col}), '{symbol.upper()}')")
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM read_parquet('{path}'){where_clause}"
        return self._conn.execute(sql).df()

    def print(
        self,
        where: str | None = None,
        order_by: str = "creation_time DESC",
        limit: int | None = None,
        params: bool = False,
    ) -> None:
        """
        Pretty-print a colored list of backtests stored at base_path.

        Matches the style of the old BacktestsResultsManager.list() — header line,
        description, strategy / interval / capital / instruments, full metrics table.

        Args:
            where:    DuckDB WHERE clause to filter (e.g. ``"sharpe > 2"``).  None = all.
            order_by: ORDER BY clause (default: ``"creation_time DESC"``).
            limit:    Maximum number of results to display.
            params:   If True, print strategy parameters below the metrics table.
        """
        df = self.search(where=where, order_by=order_by, limit=limit)

        if df.empty:
            print("No backtests found.")
            return

        _l = lambda v: [] if v is None else list(v)  # noqa: E731 — numpy array → Python list
        _METRIC_COLS = ["gain", "cagr", "sharpe", "qr", "mdd_pct", "mdd_usd", "fees", "execs"]

        for _, row in df.iterrows():
            _id = row.get("backtest_id", "")
            _name = row.get("name", "")
            _cls = str(row.get("strategy_class", "")).split(".")[-1]
            _created = pd.Timestamp(row.get("creation_time")).strftime("%Y-%m-%d %H:%M:%S")
            _author = row.get("author", "")
            _start = pd.Timestamp(row.get("start")).strftime("%Y-%m-%d")
            _stop = pd.Timestamp(row.get("stop")).strftime("%Y-%m-%d")
            _capital = row.get("capital", "")
            _ccy = row.get("base_currency", "")
            _comm = row.get("commissions", "")
            _dscr = row.get("description", "") or ""
            _tags = _l(row.get("tags"))
            _symbols = ", ".join(_l(row.get("symbols")))
            _is_var = row.get("is_variation", False)

            # - header: id :: name ::: created by author
            _s = f"{yellow(_id)} :: {red(_name)}"
            if _is_var:
                _var_id = row.get("variation_id", "")
                _var_params = row.get("variation_params", "") or ""
                _s += f" [{cyan(_var_id)}] {magenta(_var_params)}"
            _s += f" ::: {magenta(_created)} by {cyan(_author)}"

            # - description lines
            if _dscr:
                for _d in _dscr.split("\n"):
                    if _d.strip():
                        _s += f"\n\t{magenta('# ' + _d)}"

            _s += f"\n\tstrategy:    {green(_cls)}"
            _s += f"\n\tinterval:    {blue(_start)} - {blue(_stop)}"
            _s += f"\n\tcapital:     {blue(str(_capital))} {_ccy} ({_comm})"
            _s += f"\n\tinstruments: {blue(_symbols)}"
            if _tags:
                _s += f"\n\ttags:        {cyan(str(_tags))}"

            print(_s)

            # - performance metrics table (red header, cyan values — same as old manager)
            _metrics = {
                c: (int(row.get(c) or 0) if c == "execs" else round(float(row.get(c) or 0.0), 3))
                for c in _METRIC_COLS
                if c in row
            }
            _m_df = pd.DataFrame([_metrics])
            _m_str = _m_df.to_string(index=False)
            _h, _v = _m_str.split("\n")
            print("\t " + red(_h))
            print("\t " + cyan(_v))

            # - optional parameters
            if params:
                import json as _json

                _p = _json.loads(row.get("parameters") or "{}")
                if _p:
                    for k, v in _p.items():
                        print(f"\t   {yellow(k)}: {cyan(str(v))}")

            print()

    def list(
        self,
        where: str | None = None,
        order_by: str = "creation_time DESC",
        limit: int | None = None,
    ) -> list[str]:
        """
        Return a list of backtest IDs matching the given filter.

        Args:
            where:    Optional SQL WHERE clause to filter results.
            order_by: SQL ORDER BY clause (default: ``creation_time DESC``).
            limit:    Maximum number of IDs to return.

        Returns:
            List of backtest_id strings, e.g.
            ``["my_strategy/Nimble/20240301_120000", ...]``
        """
        df = self.search(where=where, order_by=order_by, limit=limit)
        if df.empty or "backtest_id" not in df.columns:
            return []
        return df["backtest_id"].tolist()

    def delete(self, backtest_id: str) -> None:
        """
        Delete a single backtest run directory.

        Args:
            backtest_id: Relative path within base_path,
                        e.g. "my_strategy/Nimble/20240301_120000"
        """
        path = f"{self.base_path}{backtest_id.strip('/')}/"
        if self._is_cloud:
            S3Client(storage_options=self._storage_options).rm(path, recursive=True)
        else:
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        self._reset_conn()

    def delete_group(self, group_path: str) -> None:
        """
        Delete an entire group directory (class-level or config-level).

        Args:
            group_path: Relative path within base_path,
                       e.g. "my_strategy/Nimble" (config) or "Nimble" (class)
        """
        path = f"{self.base_path}{group_path.strip('/')}/"
        if self._is_cloud:
            S3Client(storage_options=self._storage_options).rm(path, recursive=True)
        else:
            import shutil

            shutil.rmtree(path, ignore_errors=True)
        self._reset_conn()

    def _reset_conn(self) -> None:
        """Reset DuckDB connection to clear cached file metadata after mutations."""
        self._conn.close()
        self._conn = self._duckdb.connect()
        if self._is_cloud:
            self._setup_cloud_duckdb()

    def get_log(self, backtest_id: str, config_name: str | None = None) -> str | None:
        """
        Read the log file for a backtest run.

        Args:
            backtest_id: Relative path within base_path.
            config_name: Config name (used for log filename).
                        If None, searches for any .log file.

        Returns:
            Log file content as string, or None if not found.
        """
        run_dir = f"{self.base_path}{backtest_id.strip('/')}/"

        if config_name:
            log_path = f"{run_dir}{config_name}.log"
        else:
            try:
                files_df = self._conn.execute(f"SELECT file FROM glob('{run_dir}*.log')").df()
                if files_df.empty:
                    return None
                log_path = files_df["file"].iloc[0]
            except Exception:
                return None

        if self._is_cloud:
            try:
                from qubx.utils.s3 import strip_scheme

                client = S3Client(storage_options=self._storage_options)
                with client.fs.open_input_stream(strip_scheme(log_path)) as f:
                    return f.read().decode("utf-8")
            except Exception:
                return None
        else:
            from pathlib import Path

            p = Path(log_path)
            return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None

    def export_backtests_to_markdown(self, backtest_id: str, path: str, tags: tuple[str] | None = None):
        if tsr := self.load(backtest_id):
            tsr.to_markdown(path, list(tags) if tags else None)
