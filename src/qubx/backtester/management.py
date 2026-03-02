"""
Parquet-based backtest storage utilities — schemas, constants, and write helpers.

Used by:
    - qubx.core.metrics.TradingSessionResult  (to_storage / from_storage)
    - qubx.backtester.management.BacktestStorage (query interface)
    - qubx.backtester.utils.SimulationStatusWriter (write_parquet_table)
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

import re
import warnings
import zipfile
from collections import defaultdict
from os.path import expanduser
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from tqdm.auto import tqdm

from qubx.backtester.utils import _SIMULATION_STATUS_FILE, is_cloud_path, resolve_s3_storage_options
from qubx.core.metrics import TradingSessionResult
from qubx.utils.misc import blue, cyan, green, magenta, red, yellow

_METADATA_FILE = "_metadata.parquet"

_DATA_FILES = {
    "portfolio": "portfolio.parquet",
    "executions": "executions.parquet",
    "signals": "signals.parquet",
    "targets": "targets.parquet",
    "transfers": "transfers.parquet",
}

_METADATA_SCHEMA = pa.schema(
    [
        ("backtest_id", pa.string()),
        ("name", pa.string()),
        ("config_name", pa.string()),
        ("data_path", pa.string()),
        ("is_variation", pa.bool_()),
        ("variation_id", pa.string()),
        ("variation_name", pa.string()),
        ("variation_params", pa.string()),  # - JSON string for DuckDB json_extract()
        ("strategy_class", pa.string()),
        ("parameters", pa.string()),  # - JSON string for DuckDB json_extract()
        ("start", pa.timestamp("us", tz="UTC")),
        ("stop", pa.timestamp("us", tz="UTC")),
        ("creation_time", pa.timestamp("us", tz="UTC")),
        ("simulation_time_sec", pa.int64()),
        ("capital", pa.float64()),
        ("base_currency", pa.string()),
        ("commissions", pa.string()),
        ("exchanges", pa.list_(pa.string())),
        ("symbols", pa.list_(pa.string())),
        ("author", pa.string()),
        ("qubx_version", pa.string()),
        ("description", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("is_simulation", pa.bool_()),
        # - performance metrics (denormalized — fast DuckDB search without loading data)
        ("sharpe", pa.float64()),
        ("cagr", pa.float64()),
        ("mdd_pct", pa.float64()),
        ("mdd_usd", pa.float64()),
        ("gain", pa.float64()),
        ("qr", pa.float64()),
        ("calmar", pa.float64()),
        ("sortino", pa.float64()),
        ("execs", pa.int64()),
        ("fees", pa.float64()),
        ("daily_turnover", pa.float64()),
    ]
)


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
                    ├── _status.parquet   # live progress, written by SimulationStatusWriter
                    ├── _metadata.parquet # completion metrics
                    ├── portfolio.parquet
                    ├── executions.parquet
                    ├── signals.parquet
                    ├── targets.parquet
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
            storage_options: Cloud storage credentials. None = auto-detect from:
                             QUBX_S3_KEY / AWS_ACCESS_KEY_ID
                             QUBX_S3_SECRET / AWS_SECRET_ACCESS_KEY
                             QUBX_S3_REGION / AWS_DEFAULT_REGION
                             QUBX_S3_ENDPOINT / AWS_ENDPOINT_URL
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
        self._storage_options = storage_options
        self._is_cloud = is_cloud_path(base_path)
        self._resolve_s3 = resolve_s3_storage_options
        self._conn = self._duckdb.connect()

        if self._is_cloud:
            self._setup_cloud_duckdb()

    def _setup_cloud_duckdb(self) -> None:
        """Configure DuckDB httpfs extension for cloud storage access."""
        self._conn.execute("INSTALL httpfs; LOAD httpfs;")

        opts = self._resolve_s3(self._storage_options)
        if "key" in opts:
            self._conn.execute(f"SET s3_access_key_id='{opts['key']}';")
        if "secret" in opts:
            self._conn.execute(f"SET s3_secret_access_key='{opts['secret']}';")
        if "endpoint_url" in opts:
            # - strip protocol prefix — DuckDB expects hostname only
            endpoint = opts["endpoint_url"].removeprefix("https://").removeprefix("http://")
            self._conn.execute(f"SET s3_endpoint='{endpoint}';")
        if "client_kwargs" in opts:
            region = opts["client_kwargs"].get("region_name")
            if region:
                self._conn.execute(f"SET s3_region='{region}';")

    def _glob(self, filename: str) -> str:
        """Build recursive glob pattern for a filename within base_path."""
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
        glob = self._glob(_METADATA_FILE)
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

        Reads _status.parquet files written by SimulationStatusWriter.
        Works in real-time — running simulations update their status every 1%.

        Args:
            filter_status: Filter by status value ('running', 'completed', 'failed', 'pending'),
                          or None to return all simulations

        Returns:
            pd.DataFrame with status rows, ordered by started_at DESC
        """
        glob = self._glob(_SIMULATION_STATUS_FILE)
        sql = f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
        if filter_status:
            sql += f" WHERE status = '{filter_status}'"
        sql += " ORDER BY started_at DESC"
        return self._conn.execute(sql).df()

    def load(self, backtest_id: str) -> TradingSessionResult:
        """
        Load a TradingSessionResult by backtest_id.

        Args:
            backtest_id: Relative path within base_path,
                        e.g. "my_strategy/Nimble/20240301_120000"

        Returns:
            TradingSessionResult with all data loaded from parquet
        """
        path = f"{self.base_path}{backtest_id.strip('/')}/"
        return TradingSessionResult.from_storage(path, self._storage_options)

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
        meta_path = f"{self.base_path}{variation_set_id.strip('/')}/{_METADATA_FILE}"
        order = "ASC" if ascending else "DESC"
        sql = f"SELECT * FROM read_parquet('{meta_path}') ORDER BY {by} {order} LIMIT 1"
        row = self._conn.execute(sql).df()

        if row.empty:
            raise ValueError(f"No variations found at '{variation_set_id}'")

        var_id = row["variation_id"].iloc[0]
        data_path = f"{self.base_path}{variation_set_id.strip('/')}/{var_id}/"
        return TradingSessionResult.from_storage(data_path, self._storage_options)

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
        path = f"{self.base_path}{backtest_id.strip('/')}/{_DATA_FILES['portfolio']}"
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
        path = f"{self.base_path}{backtest_id.strip('/')}/{_DATA_FILES['executions']}"
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
        path = f"{self.base_path}{backtest_id.strip('/')}/{_DATA_FILES['signals']}"
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
            _id      = row.get("backtest_id", "")
            _name    = row.get("name", "")
            _cls     = str(row.get("strategy_class", "")).split(".")[-1]
            _created = pd.Timestamp(row.get("creation_time")).strftime("%Y-%m-%d %H:%M:%S")
            _author  = row.get("author", "")
            _start   = pd.Timestamp(row.get("start")).strftime("%Y-%m-%d")
            _stop    = pd.Timestamp(row.get("stop")).strftime("%Y-%m-%d")
            _capital = row.get("capital", "")
            _ccy     = row.get("base_currency", "")
            _comm    = row.get("commissions", "")
            _dscr    = row.get("description", "") or ""
            _tags    = _l(row.get("tags"))
            _symbols = ", ".join(_l(row.get("symbols")))
            _is_var  = row.get("is_variation", False)

            # - header: id :: name ::: created by author
            _s = f"{yellow(_id)} :: {red(_name)}"
            if _is_var:
                _var_id     = row.get("variation_id", "")
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
                for c in _METRIC_COLS if c in row
            }
            _m_df  = pd.DataFrame([_metrics])
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
        params: bool = False,
    ) -> None:
        """Alias for :meth:`print`."""
        self.print(where=where, order_by=order_by, limit=limit, params=params)


class BacktestsResultsManager:
    """
    Manager class for handling backtesting results stored in zip files.

    .. deprecated::
        BacktestsResultsManager is deprecated. Use :class:`BacktestStorage` instead,
        which provides parquet-based storage with DuckDB querying and cloud support.

    Parameters
    ----------
    path : str
        Path to directory containing backtesting result zip files
    """

    def __init__(self, path: str):
        warnings.warn(
            "BacktestsResultsManager is deprecated and will be removed in a future version. "
            "Use BacktestStorage for parquet-based storage with DuckDB querying.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.path = expanduser(path)
        self.reload()

    def reload(self) -> "BacktestsResultsManager":
        self.results = {}
        self.variations = {}

        _vars = defaultdict(list)
        names = defaultdict(lambda: 0)
        for p in Path(self.path).glob("**/*.zip"):
            with zipfile.ZipFile(p, "r") as zip_ref:
                try:
                    info = yaml.safe_load(zip_ref.read("info.yml"))
                    info["path"] = str(p)
                    n = info.get("name", "")
                    var_set_name = info.get("variation_name", "")

                    # - put variations aside
                    if var_set_name:
                        _vars[var_set_name].append(info)
                        continue

                    _new_name = n if names[n] == 0 else f"{n}.{names[n]}"
                    names[n] += 1
                    info["name"] = _new_name
                    self.results[_new_name] = info
                except Exception:
                    pass

        # - reindex
        _idx = 1
        for n in sorted(self.results.keys()):
            self.results[n]["idx"] = _idx
            _idx += 1

        # - reindex variations at the end
        for n in sorted(_vars.keys()):
            self.variations[_idx] = {
                "name": n,
                "idx": _idx,
                "variations": _vars[n],
                "created": pd.Timestamp(_vars[n][0].get("creation_time", "")).round("1s"),
                "author": _vars[n][0].get("author", ""),
                "description": _vars[n][0].get("description", ""),
            }
            _idx += 1

        return self

    def __getitem__(
        self, name: str | int | list[int] | list[str] | slice
    ) -> TradingSessionResult | list[TradingSessionResult]:
        return self.load(name)

    def load(
        self, name_or_idx: str | int | list[int] | list[str] | slice
    ) -> TradingSessionResult | list[TradingSessionResult]:
        match name_or_idx:
            case list():
                return [self.load(i) for i in name_or_idx]  # type: ignore
            case str():
                return [self.load(i) for i in self._find_indices(name_or_idx)]  # type: ignore
            case slice():
                return [
                    self.load(i)
                    for i in range(name_or_idx.start, name_or_idx.stop, name_or_idx.step if name_or_idx.step else 1)
                ]  # type: ignore
            case int():
                if name_or_idx > len(self.results) and name_or_idx in self.variations:
                    return [
                        TradingSessionResult.from_file(v.get("path", ""))
                        for v in self.variations[name_or_idx].get("variations", [])
                    ]

                # - load by index
                for info in self.results.values():
                    if info.get("idx", -1) == name_or_idx:
                        return TradingSessionResult.from_file(info["path"])

        raise ValueError(f"No result found for '{name_or_idx}' !")

    def load_config(self, name: str | int) -> str:
        """Load the configuration YAML file for a specific backtest result.

        Args:
            name (str | int): The name or index of the backtest result.

        Returns:
            str: The contents of the configuration YAML file as a string.

        Raises:
            ValueError: If no backtest result is found matching the provided name/index.
        """
        p = None
        for info in self.results.values():
            match name:
                case int():
                    if info.get("idx", -1) == name:
                        n = info.get("name", "")
                        p = info.get("path", {})
                        break
                case str():
                    if info.get("name", "") == name:
                        n = info.get("name", "")
                        p = info.get("path", {})
                        break
        if p is None:
            raise ValueError(f"No result found for {name}")

        # - name may have .1, .2, etc. so we need to remove it
        n = n.split(".")[0] if "." in n else n
        with zipfile.ZipFile(p, "r") as zip_ref:
            return zip_ref.read(f"{n}.yaml").decode("utf-8")

    def delete(self, name: str | int | list[int] | list[str] | slice):
        """Delete one or more backtest results.

        Args:
            name: Identifier(s) for the backtest result(s) to delete.
        """

        def _del_idx(idx):
            for info in self.results.values():
                if info.get("idx", -1) == idx:
                    Path(info["path"]).unlink()
                    return info.get("name", idx)
            return None

        match name:
            case str():
                nms = [_del_idx(i) for i in self._find_indices(name)]
                self.reload()
                print(f" -> Deleted {red(', '.join(nms))} ...")
                return

            case list():
                nms = [_del_idx(i) for i in name]
                self.reload()
                print(f" -> Deleted {red(', '.join(nms))} ...")
                return

            case slice():
                nms = [_del_idx(i) for i in range(name.start, name.stop, name.step if name.step else 1)]
                self.reload()
                print(f" -> Deleted {red(', '.join(nms))} ...")
                return

        for info in self.results.values():
            match name:
                case int():
                    if info.get("idx", -1) == name:
                        Path(info["path"]).unlink()
                        print(f" -> Deleted {red(info.get('name', name))} ...")
                        self.reload()
                        return
                case str():
                    if info.get("name", "") == name:
                        Path(info["path"]).unlink()
                        print(f" -> Deleted {red(info.get('name', name))} ...")
                        self.reload()
                        return
        print(f" -> No results found for {red(name)} !")

    def _find_indices(self, regex: str):
        for n in sorted(self.results.keys()):
            info = self.results[n]
            s_cls = info.get("strategy_class", "").split(".")[-1]

            try:
                if not re.match(regex, n, re.IGNORECASE):
                    continue
            except Exception:
                if regex.lower() != n.lower() and regex.lower() != s_cls.lower():
                    continue

            yield info.get("idx", -1)

    def list_variations(self, regex: str = "", detailed=True, sort_by: str | None = "sharpe", ascending=False):
        """
        List only variations of a backtest result.

        Args:
            - regex (str, optional): Regular expression pattern to filter results. Defaults to "".
            - sort_by (str, optional): The criterion to sort the results by. Defaults to "sharpe".
            - ascending (bool, optional): Whether to sort in ascending order. Defaults to False.
            - detailed (bool, optional): Whether to show each variation run. Defaults to True.
        """
        return self.list(
            regex=regex,
            sort_by=sort_by,
            ascending=ascending,
            show_variations=True,
            show_simulations=False,
            show_each_variation_run=detailed,
        )

    def list(
        self,
        regex: str = "",
        with_metrics=True,
        params=False,
        as_table=False,
        pretty_print=False,
        sort_by: str | None = "sharpe",
        ascending=False,
        show_simulations=True,
        show_variations=True,
        show_each_variation_run=True,
    ):
        """List backtesting results with optional filtering and formatting.

        Args:
            - regex (str, optional): Regular expression pattern to filter results. Defaults to "".
            - with_metrics (bool, optional): Whether to include performance metrics. Defaults to True.
            - params (bool, optional): Whether to display strategy parameters. Defaults to False.
            - as_table (bool, optional): Return results as a pandas DataFrame. Defaults to False.
            - sort_by (str, optional): The criterion to sort the results by. Defaults to "sharpe".
            - ascending (bool, optional): Whether to sort in ascending order. Defaults to False.
            - show_simulations (bool, optional): Whether to show simulation results. Defaults to True.
            - show_variations (bool, optional): Whether to show variation results. Defaults to True.
            - show_each_variation_run (bool, optional): Whether to show each variation run. Defaults to True.

        Returns:
            - Optional[pd.DataFrame]: If as_table=True, returns a DataFrame.
        """
        _t_rep = []
        if show_simulations:
            for n in sorted(self.results.keys()):
                info = self.results[n]
                s_cls = info.get("strategy_class", "").split(".")[-1]

                if regex:
                    if not re.match(regex, n, re.IGNORECASE):
                        continue

                name = info.get("name", "")
                smbs = ", ".join(info.get("symbols", list()))
                start = pd.Timestamp(info.get("start", "")).round("1s")
                stop = pd.Timestamp(info.get("stop", "")).round("1s")
                dscr = info.get("description", "")
                created = pd.Timestamp(info.get("creation_time", "")).round("1s")
                metrics = info.get("performance", {})
                author = info.get("author", "")
                _s = f"{yellow(str(info.get('idx')))} - {red(name)} ::: {magenta(created)} by {cyan(author)}"

                _one_line_dscr = ""
                if dscr:
                    dscr = dscr.split("\n")
                    for _d in dscr:
                        _s += f"\n\t{magenta('# ' + _d)}"
                        _one_line_dscr += "\u25cf " + _d + "\n"

                _s += f"\n\tstrategy: {green(s_cls)}"
                _s += f"\n\tinterval: {blue(start)} - {blue(stop)}"
                _s += f"\n\tcapital: {blue(info.get('capital', ''))} {info.get('base_currency', '')} ({info.get('commissions', '')})"
                _s += f"\n\tinstruments: {blue(smbs)}"
                if params:
                    formats = ["{" + f":<{i}" + "}" for i in [50]]
                    _p = pd.DataFrame.from_dict(info.get("parameters", {}), orient="index")
                    for i in _p.to_string(
                        max_colwidth=30,
                        header=False,
                        formatters=[(lambda x: cyan(fmt.format(str(x)))) for fmt in formats],
                        justify="left",
                    ).split("\n"):
                        _s += f"\n\t  |  {yellow(i)}"

                if not as_table:
                    print(_s)

                dd_column = "max_dd_pct" if "max_dd_pct" in metrics else "mdd_pct"
                if with_metrics:
                    _m_repr = (
                        pd.DataFrame.from_dict(metrics, orient="index")
                        .T[["gain", "cagr", "sharpe", "qr", dd_column, "mdd_usd", "fees", "execs"]]
                        .astype(float)
                    )
                    _m_repr = _m_repr.round(3).to_string(index=False)
                    _h, _v = _m_repr.split("\n")
                    if not as_table:
                        print("\t " + red(_h))
                        print("\t " + cyan(_v))

                if not as_table:
                    print()
                else:
                    metrics = {
                        m: round(v, 3)
                        for m, v in metrics.items()
                        if m in ["gain", "cagr", "sharpe", "qr", dd_column, "mdd_usd", "fees", "execs"]
                    }
                    _t_rep.append(
                        {"Index": info.get("idx", ""), "Strategy": name}
                        | metrics
                        | {
                            "start": start,
                            "stop": stop,
                            "Created": created,
                            "Author": author,
                            "Description": _one_line_dscr,
                        },
                    )

        # - variations (only if not as_table for the time being)
        if not as_table and show_variations:
            for _i, vi in self.variations.items():
                n = vi.get("name", "")
                if regex:
                    if not re.match(regex, n, re.IGNORECASE):
                        continue

                _s = f"{yellow(str(_i))} - {red(str(n))} set of {len(vi.get('variations'))} variations ::: {magenta(vi.get('created'))} by {cyan(vi.get('author'))}"

                dscr = vi.get("description", "").split("\n")
                for _d in dscr:
                    _s += f"\n\t{magenta('# ' + _d)}"

                _mtrx = {}
                for v in vi.get("variations", []):
                    _nm = v.get("name", "")
                    _nm = _nm.split("_")[-1].strip("()")
                    _mtrx[_nm] = v.get("performance", {})

                _m_repr = pd.DataFrame.from_dict(_mtrx, orient="index")[
                    ["gain", "cagr", "sharpe", "qr", "mdd_pct", "mdd_usd", "fees", "execs"]
                ].astype(float)
                _m_repr = _m_repr.round(3)
                _m_repr = _m_repr.sort_values(by=sort_by, ascending=ascending) if sort_by else _m_repr
                _m_repr = _m_repr.to_string(index=True)

                print(_s)
                if show_each_variation_run:
                    for _i, _l in enumerate(_m_repr.split("\n")):
                        if _i == 0:
                            print("\t " + red(_l))
                        else:
                            print("\t " + blue(_l))

        if as_table:
            _df = pd.DataFrame.from_records(_t_rep, index="Index")
            _df = _df.sort_values(by=sort_by, ascending=ascending) if sort_by else _df
            if pretty_print:
                from IPython.display import HTML

                return HTML(
                    _df.to_html()
                    .replace("\\n", "<br><hr style='border-color: #005000; '/>")
                    .replace("<td>", '<td align="left" valign="top">')
                )
            return _df

    def variation_plot(self, variation_idx: int, criterion: str = "sharpe", ascending: bool = False, n=3, h=600):
        """
        Plot a variation of a backtest result.

        Args:
            - variation_idx (int): The index of the variation to plot.
            - criterion (str): The criterion to plot (e.g. "sharpe", "mdd_usd", etc.).
            - ascending (bool): Whether to sort the results in ascending order.
            - n (int): The number of decimal places to display.
            - h (int): The height of the plot.

        Returns:
            plotly.graph_objects.Figure: The plot of the variation.
        """
        from itertools import cycle

        import plotly.express as px

        from qubx.utils.misc import string_shortener

        _vars = self.variations.get(variation_idx)
        if not _vars:
            raise ValueError(f"No variations found for index {variation_idx} !")

        variations = _vars.get("variations", [])
        name = _vars.get("name", "") or ""

        _r, _p = {}, {}
        for i, v in enumerate(variations):
            _p[i] = v["parameters"]
        _pp = pd.DataFrame.from_records(_p).T
        # - changed parameters
        _cp = []
        for c in _pp.columns:
            if len(_pp[c].astype(str).unique()) > 1:
                _cp.append(c)

        # - if nothing was actually changed in parameters, raise an error
        if not _cp:
            raise ValueError(f"No variable parameters found for simulation {name} !")

        _ms = max([len(string_shortener(x)) for x in _cp]) + 3
        _h = "".join([string_shortener(x).center(_ms) for x in _cp])

        _sel = lambda ds, _cp: "".join(
            [
                f"<span style='color:{c}'> {str(ds[k]).center(_ms)}</span>"
                for k, c in zip(_cp, cycle(px.colors.qualitative.Plotly))
                if k in k in ds
            ]
        )
        for i, v in enumerate(variations):
            _r[i] = {"name": v["name"], **v["performance"], "parameters": _sel(v["parameters"], _cp)}

        t1 = pd.DataFrame.from_records(_r).T
        if criterion not in t1.columns:
            raise ValueError(f"Criterion {criterion} not found in results: possible values are {t1.columns}")
        t2 = t1.sort_values(criterion, ascending=ascending)

        data = pd.Series([np.nan, *t2[criterion].to_list()], index=[_h, *t2["parameters"].to_list()])

        figure = (
            px.bar(data, orientation="h")
            .update_layout(
                title=dict(
                    text=f"{name} | <span style='color:orange'>{criterion.capitalize()}</span>",
                ),
                xaxis=dict(tickfont=dict(family="monospace", size=10, color="#ff4000")),
                yaxis=dict(
                    tickfont=dict(family="monospace", size=10, color="#40a000"),
                    dtick=1,
                ),
            )
            .update_layout(
                height=h,
                hovermode="x unified",
                showlegend=False,
                hoverdistance=1,
                yaxis={"hoverformat": f".{n}f"},
                dragmode="zoom",
                newshape=dict(line_color="red", line_width=1.0),
                modebar_add=["drawline", "drawopenpath", "drawrect", "eraseshape"],
                hoverlabel=dict(align="auto", bgcolor="rgba(10, 10, 10, 0.5)"),
            )
            .update_xaxes(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#306020",
                spikethickness=1,
                spikedash="dot",
                title=criterion,
            )
            .update_yaxes(
                spikesnap="cursor",
                spikecolor="#306020",
                tickformat=f".{n}f",
                spikethickness=1,
                title="Parameters",
                autorange="reversed",
            )
        )
        return figure

    def export_backtests_to_markdown(self, path: str, tags: tuple[str] | None = None):
        """
        Export backtests to markdown format
        """
        for n, v in tqdm(self.results.items()):
            r = TradingSessionResult.from_file(v.get("path"))
            r.to_markdown(path, list(tags) if tags else None)
