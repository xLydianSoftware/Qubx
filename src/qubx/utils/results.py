"""
Storage utilities and SimulationResultsSaver — single source of truth for all parquet I/O.

- **SimulationResultsSaver**: lifecycle tracking and parquet save/load for simulation results.
    - Status tracking: ``write_pending()`` → ``write_completed()`` / ``write_failed()``
    - :meth:`~SimulationResultsSaver.save` — writes all parquet files for a completed run
    - :meth:`~SimulationResultsSaver.load` — reconstructs a
      :class:`~qubx.core.metrics.TradingSessionResult` from storage
    - :meth:`~SimulationResultsSaver._from_dfs` — shared reconstruction logic
    - :meth:`~SimulationResultsSaver._build_metadata_record` — builds the metadata dict

All I/O uses ``pyarrow.fs.S3FileSystem`` for cloud paths — no dependency on
``s3fs`` / ``aiobotocore`` / ``aiohttp``.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qubx import logger
from qubx.config import S3Account
from qubx.core.metrics import TradingSessionResult
from qubx.utils.time import to_utc

# ── path / tag utilities ─────────────────────────────────────────────────────


def get_short_class_name(strategy_class: str | list[str]) -> str:
    """
    Extract short class name(s) from fully qualified class name(s).
    For multi-class composed strategies, joins short names with '+'.

    Examples::

        "pkg.models.nimble.Nimble" → "Nimble"
        ["pkg.nimble.Nimble", "pkg.risk.AdvRisk"] → "Nimble+AdvRisk"
    """
    if isinstance(strategy_class, list):
        return "+".join(c.split(".")[-1] for c in strategy_class)
    return strategy_class.split(".")[-1]


def normalize_tags(tags: str | list[str] | None) -> list[str]:
    """Normalize tags to a list of strings."""
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    return list(tags)


def is_cloud_path(path: str) -> bool:
    """Check if path is a cloud storage URI (S3, GCS, Azure)."""
    return path.startswith(("s3://", "gs://", "az://", "abfs://"))


def _strip_cloud_scheme(path: str) -> str:
    """Strip cloud URI scheme for use with pyarrow.fs (which wants bucket/key, not s3://bucket/key)."""
    for prefix in ("s3://", "gs://", "az://", "abfs://"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def copy_file_to_storage(
    src: str, dst_dir: str, storage_options: dict | None = None, dst_name: str | None = None
) -> None:
    """
    Copy a local file to a destination directory (local or cloud).

    Args:
        src:             Local path of the file to copy.
        dst_dir:         Destination directory (local path or cloud URI).
        storage_options: Cloud credentials for S3 writes.
        dst_name:        Override the stored filename. Defaults to the source filename.
                         Use this to rename on upload (e.g. temp log files named by OS).
    """
    import shutil

    src_path = Path(src)
    if not src_path.is_file():
        logger.warning(f"[SimulationResultsSaver] Attachment not found, skipping: {src}")
        return

    dst = f"{dst_dir.rstrip('/')}/{dst_name if dst_name else src_path.name}"
    if not is_cloud_path(dst_dir):
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        fs = _make_pa_s3_filesystem(storage_options or {})
        with open(src, "rb") as fin:
            data = fin.read()
        with fs.open_output_stream(_strip_cloud_scheme(dst)) as fout:
            fout.write(data)


# ── S3 filesystem cache ──────────────────────────────────────────────────────

# - module-level cache: one S3FileSystem per unique credential set.
# - pyarrow.fs.S3FileSystem is thread-safe and pools TCP+SSL connections internally.
# - without caching, every write() call re-establishes a new connection to the endpoint
# - which adds ~200-500ms per write — catastrophic for simulations with 10+ status writes.
_PA_S3_FS_CACHE: dict[tuple, Any] = {}


def _make_pa_s3_filesystem(opts: dict):
    """
    Return a cached pyarrow.fs.S3FileSystem for the given storage options.

    The instance is keyed on (key, secret, endpoint_url, region) and reused across
    all calls with identical credentials.  This avoids re-establishing a TCP+SSL
    connection to the S3 endpoint on every parquet write.

    Uses Arrow's native S3 client — no dependency on s3fs, aiobotocore, or aiohttp.
    """
    _region = opts.get("client_kwargs", {}).get("region_name") if opts.get("client_kwargs") else None
    _cache_key = (opts.get("key"), opts.get("secret"), opts.get("endpoint_url"), _region)

    if _cache_key not in _PA_S3_FS_CACHE:
        try:
            import pyarrow.fs as pafs
        except ImportError:
            raise ImportError(
                "pyarrow with S3 support is required for cloud storage. Install with: pip install pyarrow"
            )

        kwargs: dict = {}
        if opts.get("key"):
            kwargs["access_key"] = opts["key"]
        if opts.get("secret"):
            kwargs["secret_key"] = opts["secret"]
        if opts.get("endpoint_url"):
            # - pyarrow.fs.S3FileSystem expects hostname only (no https://)
            kwargs["endpoint_override"] = opts["endpoint_url"].removeprefix("https://").removeprefix("http://")
        if _region:
            kwargs["region"] = _region

        _PA_S3_FS_CACHE[_cache_key] = pafs.S3FileSystem(**kwargs)

    return _PA_S3_FS_CACHE[_cache_key]


# ── parquet I/O ──────────────────────────────────────────────────────────────


def _sanitize_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dict/list object columns to JSON strings before writing to parquet.

    Parquet cannot handle struct types with no child fields, which pyarrow infers
    when a column contains only empty dicts ``{}`` (e.g. Signal.options when unused).
    Converting those columns to JSON strings sidesteps the limitation cleanly.
    """
    import json

    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        # - find first non-null value to check the column type
        first_valid = next(
            (v for v in df[col] if v is not None and not (isinstance(v, float) and pd.isna(v))),
            None,
        )
        if isinstance(first_valid, (dict, list)):
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    return df


def read_parquet(path: str, storage_options: dict | None = None) -> pd.DataFrame:
    """
    Read parquet file supporting local and cloud paths.

    For cloud paths uses ``pyarrow.fs.S3FileSystem`` (Arrow's own S3 client) instead
    of going through ``fsspec → s3fs → aiobotocore → aiohttp``, which avoids
    ``aiohttp.SocketFactoryType`` import errors when package versions are mismatched.

    Args:
        path:            Local path or cloud URI (``s3://bucket/key``).
        storage_options: Resolved storage options from :func:`resolve_s3_storage_options`.

    Returns:
        pandas DataFrame (index is restored from parquet pandas metadata).
    """
    if not is_cloud_path(path):
        return pd.read_parquet(path)

    fs = _make_pa_s3_filesystem(storage_options or {})
    table = pq.read_table(_strip_cloud_scheme(path), filesystem=fs)
    return table.to_pandas()


def write_parquet(df: pd.DataFrame | None, path: str, storage_options: dict | None = None) -> None:
    """
    Write DataFrame to parquet, supporting local and cloud paths.
    Local: creates parent directories automatically.
    Cloud: uses pyarrow.fs.S3FileSystem (no s3fs/aiobotocore dependency).
    Skips write silently when df is None or empty.
    """
    if df is None or df.empty:
        return
    df = _sanitize_df_for_parquet(df)
    if not is_cloud_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True, engine="pyarrow")
    else:
        fs = _make_pa_s3_filesystem(storage_options or {})
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(table, _strip_cloud_scheme(path), filesystem=fs)


def write_parquet_table(table: pa.Table, path: str, storage_options: dict | None = None) -> None:
    """
    Write pyarrow Table to parquet, supporting local and cloud paths.
    Used for schema-enforced writes (status, metadata).
    Cloud: uses pyarrow.fs.S3FileSystem (no s3fs/aiobotocore dependency).
    """
    if not is_cloud_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
    else:
        fs = _make_pa_s3_filesystem(storage_options or {})
        pq.write_table(table, _strip_cloud_scheme(path), filesystem=fs)


def _s3_account_to_opts(acct: "S3Account") -> dict:
    """Convert an S3Account to the storage options dict format."""
    opts: dict = {
        "key": acct.access_key_id,
        "secret": acct.secret_access_key,
    }
    if acct.endpoint_url:
        endpoint = acct.endpoint_url
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        opts["endpoint_url"] = endpoint
    if acct.region:
        opts["client_kwargs"] = {"region_name": acct.region}
    return opts


def resolve_s3_storage_options(explicit: dict | None = None, account: str | None = None) -> dict:
    """
    Resolve S3 storage options from explicit params or named account in settings.

    Priority:
        1. explicit dict (returned as-is)
        2. Named account from settings.s3[account]
        3. settings.default_s3_account (if configured)
        4. Empty dict (default credential chain)
    """
    if explicit is not None:
        return explicit

    from qubx.config import settings

    name = account or settings.default_s3_account
    if name and name in settings.s3:
        return _s3_account_to_opts(settings.s3[name])

    return {}


def write_metadata_parquet(
    meta_records: dict | list[dict],
    path: str,
    storage_options: dict | None = None,
) -> None:
    """
    Build a typed pyarrow Table from one or more metadata record dicts and write to parquet.

    Handles both single-run (single dict) and variation-set (list of dicts) cases.
    Timestamp columns ``start``, ``stop``, ``creation_time`` are coerced to UTC automatically.

    Args:
        meta_records:    Single record dict or list of record dicts.
        path:            Destination path (local or cloud URI).
        storage_options: Cloud credentials from :func:`resolve_s3_storage_options`.
    """
    records = [meta_records] if isinstance(meta_records, dict) else meta_records
    df = pd.DataFrame(records)
    for col in ["start", "stop", "creation_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    table = pa.Table.from_pandas(df, schema=SimulationResultsSaver.METADATA_SCHEMA, preserve_index=False)
    write_parquet_table(table, path, storage_options)


# ── SimulationResultsSaver ───────────────────────────────────────────────────


class SimulationResultsSaver:
    """
    Manages writing simulation results and status to parquet storage.

    Handles the full lifecycle of a simulation run:

    - **Status tracking**: ``write_pending()`` → ``write_completed()`` / ``write_failed()``
    - **Data storage**: :meth:`save` writes all parquet files for a completed run
    - **Data loading**: :meth:`load` / :meth:`_from_dfs` reconstruct
      :class:`~qubx.core.metrics.TradingSessionResult`

    Lifecycle::

        saver = SimulationResultsSaver(run_dir, ...)
        saver.write_pending()                              # fire-and-forget before sim starts
        # ... simulation runs ...
        SimulationResultsSaver.save(result, base_path)    # write data parquets
        saver.write_completed()                           # blocking status write on success
        # — or —
        saver.write_failed(exc)                           # blocking status write on failure
    """

    # ── file name constants ──────────────────────────────────────────────────
    STATUS_FILE = "_status.parquet"
    METADATA_FILE = "_metadata.parquet"
    DATA_FILES = {
        "portfolio": "portfolio.parquet",
        "executions": "executions.parquet",
        "signals": "signals.parquet",
        "targets": "targets.parquet",
        "transfers": "transfers.parquet",
        "emitter": "emitter_data.parquet",
    }

    # ── parquet schemas ──────────────────────────────────────────────────────

    # - _metadata.parquet — written on completion, queried by BacktestStorage.search()
    METADATA_SCHEMA = pa.schema(
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

    # - _status.parquet — written at start/end, queried by BacktestStorage.status()
    _STATUS_SCHEMA = pa.schema(
        [
            ("backtest_id", pa.string()),
            ("name", pa.string()),
            ("strategy_class", pa.string()),
            ("config_name", pa.string()),
            ("status", pa.string()),
            ("progress_pct", pa.float64()),
            ("current_sim_time", pa.timestamp("us", tz="UTC")),
            ("sim_start", pa.timestamp("us", tz="UTC")),
            ("sim_stop", pa.timestamp("us", tz="UTC")),
            ("started_at", pa.timestamp("us", tz="UTC")),
            ("updated_at", pa.timestamp("us", tz="UTC")),
            ("completed_at", pa.timestamp("us", tz="UTC")),
            ("error", pa.string()),
            ("tags", pa.list_(pa.string())),
            ("description", pa.string()),
            ("is_variation", pa.bool_()),
            ("variation_id", pa.string()),
        ]
    )

    # ── constructor ──────────────────────────────────────────────────────────

    def __init__(
        self,
        name: str,
        strategy_class: str | list[str],
        config_name: str,
        sim_start: str | pd.Timestamp,
        sim_stop: str | pd.Timestamp,
        save_path: str | None = None,
        run_id: str | None = None,
        config_file: str | None = None,
        tags: list[str] | None = None,
        description: str | None = None,
        is_variation: bool = False,
        storage_options: dict | None = None,
    ):
        # - compute short class name(s) and run_id from raw inputs
        _short_class = get_short_class_name(strategy_class)
        _run_id = run_id or pd.Timestamp("now").strftime("%Y%m%d_%H%M%S")

        # - strategy_class stored as joined string for composed (multi-class) strategies
        self._strategy_class = " + ".join(strategy_class) if isinstance(strategy_class, list) else strategy_class
        self._backtest_id = f"{_short_class}/{name}/{_run_id}"
        self._name = name
        self._config_name = config_name
        self._save_path = save_path
        self._config_file = config_file
        self._is_variation = is_variation

        # - compute run directory and cloud flag from save_path
        if save_path is not None:
            self._run_dir = f"{save_path.rstrip('/')}/{_short_class}/{name}/{_run_id}/"
            self._is_cloud = is_cloud_path(save_path)
            self._storage_options = (
                resolve_s3_storage_options(storage_options) if self._is_cloud else (storage_options or {})
            )
        else:
            self._run_dir = ""
            self._is_cloud = False
            self._storage_options = {}

        self._status_path = f"{self._run_dir}{SimulationResultsSaver.STATUS_FILE}"
        self._sim_start = to_utc(sim_start)
        self._sim_stop = to_utc(sim_stop)
        self._tags = tags or []
        self._description = description or ""
        self._variation_id = ""  # - saver tracks a variation SET; individual ids are var_000..N
        self._started_at = pd.Timestamp.now(tz="UTC")
        self._pending_thread: threading.Thread | None = None

    @property
    def run_dir(self) -> str:
        """Full path to the run directory where results will be stored."""
        return self._run_dir

    # ── status lifecycle ─────────────────────────────────────────────────────

    def _write_record(self, record: dict) -> None:
        """Write single-row status as parquet, overwriting any previous file."""
        try:
            df = pd.DataFrame([record])
            for col in ["current_sim_time", "sim_start", "sim_stop", "started_at", "updated_at", "completed_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)
            table = pa.Table.from_pandas(df, schema=self._STATUS_SCHEMA, preserve_index=False)
            write_parquet_table(table, self._status_path, self._storage_options)
        except Exception as e:
            logger.warning(f"[SimulationResultsSaver] Failed to write status: {e}")

    def _make_record(
        self,
        status: str,
        progress_pct: float,
        current_sim_time: pd.Timestamp | None = None,
        error: str | None = None,
        completed_at: pd.Timestamp | None = None,
    ) -> dict:
        return {
            "backtest_id": self._backtest_id,
            "name": self._name,
            "strategy_class": self._strategy_class,
            "config_name": self._config_name,
            "status": status,
            "progress_pct": progress_pct,
            "current_sim_time": current_sim_time or self._sim_start,
            "sim_start": self._sim_start,
            "sim_stop": self._sim_stop,
            "started_at": self._started_at,
            "updated_at": pd.Timestamp.now(tz="UTC"),
            "completed_at": completed_at,
            "error": error,
            "tags": self._tags,
            "description": self._description,
            "is_variation": self._is_variation,
            "variation_id": self._variation_id,
        }

    def write_pending(self) -> None:
        """Fire-and-forget: write pending record in a daemon thread (non-blocking)."""
        record = self._make_record("pending", 0.0, self._sim_start)
        self._pending_thread = threading.Thread(
            target=self._write_record, args=(record,), daemon=True, name="SimulationResultsSaver-pending"
        )
        self._pending_thread.start()

    def write_completed(self) -> None:
        """Write completed status. Waits for pending write to finish first."""
        if self._pending_thread and self._pending_thread.is_alive():
            self._pending_thread.join(timeout=10.0)
        self._write_record(
            self._make_record("completed", 100.0, self._sim_stop, completed_at=pd.Timestamp.now(tz="UTC"))
        )

    def write_failed(self, error: Exception, log_file: str | None = None) -> None:
        """Write failed status and optionally upload the log file.

        For cloud runs the log is written to a local temp file during simulation.
        Passing ``log_file`` here ensures the log is uploaded even on failure so
        that it is available for post-mortem debugging.
        """
        import os

        if self._pending_thread and self._pending_thread.is_alive():
            self._pending_thread.join(timeout=10.0)
        self._write_record(self._make_record("failed", 0.0, error=str(error), completed_at=pd.Timestamp.now(tz="UTC")))

        if log_file is not None:
            try:
                copy_file_to_storage(
                    log_file,
                    self._run_dir,
                    self._storage_options,
                    dst_name=f"{self._config_name}.log",
                )
            except Exception as _e:
                logger.warning(f"[SimulationResultsSaver] Failed to upload log file: {_e}")
            try:
                os.unlink(log_file)
            except Exception:
                pass

    def close(self) -> None:
        """Ensure pending write completes (called if write_completed/write_failed not used)."""
        if self._pending_thread and self._pending_thread.is_alive():
            self._pending_thread.join(timeout=10.0)

    # ── store results ────────────────────────────────────────────────────────

    def store_simulation_results(self, test_res: list, sim_time_sec: int, log_file: str | None = None) -> None:
        """
        Persist simulation results to parquet storage and update status to completed.

        Handles both single-run and variation-set layouts::

            Single run:
                {save_path}/{Class}/{name}/{run_id}/
                    portfolio.parquet, executions.parquet, ...
                    _metadata.parquet
                    _status.parquet
                    {config_name}.log          (if log_file provided)

            Variation set:
                {run_dir}/
                    var_000/ ... var_NNN/  (one sub-dir per variation)
                    _metadata.parquet      (N-row combined summary)
                    _status.parquet
                    {config_name}.log      (if log_file provided)

        Args:
            test_res:    List of :class:`~qubx.core.metrics.TradingSessionResult` objects.
            sim_time_sec: Wall-clock simulation time in seconds.
            log_file:    Path to a local log file to upload to ``run_dir`` after saving.
                         Used for cloud runs where the log is written to a temp file
                         during simulation and must be copied to cloud storage.
                         The file is deleted after a successful upload.
                         For local runs the log is written directly to ``run_dir`` so
                         no upload is needed and this should be ``None``.

        When ``save_path`` is None the status is still marked completed so
        ``_status.parquet`` reflects the correct final state.
        """
        import os

        from joblib import Parallel, delayed

        from qubx.utils.misc import green
        from qubx.utils.time import convert_seconds_to_str

        if self._save_path is None:
            # - no storage configured: just mark completed
            self.write_completed()
            return

        def _do_log() -> None:
            """Upload temp log file to run_dir and delete the local copy."""
            if log_file is None:
                return
            try:
                copy_file_to_storage(
                    log_file,
                    self._run_dir,
                    self._storage_options,
                    dst_name=f"{self._config_name}.log",
                )
            except Exception as _e:
                logger.warning(f"[SimulationResultsSaver] Failed to upload log file: {_e}")
            try:
                os.unlink(log_file)
            except Exception:
                pass

        if len(test_res) > 1:
            # ── variation set ────────────────────────────────────────────────
            _variation_name = self._backtest_id

            print(
                f" > Simulation finished in {green(convert_seconds_to_str(sim_time_sec))} "
                f"| Saving {len(test_res)} variations to {green(self._save_path)} ..."
            )

            # - pre-compute ids and stamp results in-memory (no I/O yet)
            _var_ids = [f"var_{k:03d}" for k in range(len(test_res))]
            for t, _var_id in zip(test_res, _var_ids):
                t.variation_name = _variation_name
                t.simulation_time_sec = sim_time_sec

            # - build all metadata records in-memory before I/O
            _meta_records = [
                SimulationResultsSaver._build_metadata_record(
                    t,
                    backtest_id=f"{_variation_name}/{_var_id}",
                    data_path=f"./{_var_id}/",
                    description=self._description,
                    tags=self._tags,
                    config_name=self._config_name,
                    is_variation=True,
                    variation_id=_var_id,
                    variation_name=_variation_name,
                    variation_params=t.parameters,
                )
                for t, _var_id in zip(test_res, _var_ids)
            ]

            # - all variation saves in parallel
            def _save_var(t, _var_id: str, k: int) -> None:
                SimulationResultsSaver.save(
                    t,
                    base_path=self._run_dir,
                    description=self._description,
                    tags=self._tags,
                    config_name=self._config_name,
                    attachments=[str(self._config_file)] if k == 0 and self._config_file else None,
                    storage_options=self._storage_options,
                    is_variation=True,
                    variation_id=_var_id,
                    variation_name=_variation_name,
                )

            Parallel(n_jobs=-1, prefer="threads")(
                delayed(_save_var)(t, _var_id, k) for k, (t, _var_id) in enumerate(zip(test_res, _var_ids))
            )

            # - combined metadata + status + log in parallel (all go to the set root)
            _post_fns: list = [
                lambda: write_metadata_parquet(
                    _meta_records,
                    f"{self._run_dir}{SimulationResultsSaver.METADATA_FILE}",
                    self._storage_options,
                ),
                self.write_completed,
            ]
            if log_file:
                _post_fns.append(_do_log)

            if self._is_cloud:
                Parallel(n_jobs=len(_post_fns), prefer="threads")(delayed(fn)() for fn in _post_fns)
            else:
                for fn in _post_fns:
                    fn()

        else:
            # ── single run ───────────────────────────────────────────────────
            print(
                f" > Simulation finished in {green(convert_seconds_to_str(sim_time_sec))} "
                f"| Saving results to {green(self._save_path)} ..."
            )
            test_res[0].simulation_time_sec = sim_time_sec

            def _do_save() -> None:
                SimulationResultsSaver.save(
                    test_res[0],
                    base_path=self._save_path,
                    # - pass pre-computed run_dir so multi-class composed strategies
                    # - ("Nimble+AdvRisk") don't recompute a mismatched class name from
                    # - result.strategy_class (which is a plain single-class string)
                    run_dir=self._run_dir,
                    description=self._description,
                    tags=self._tags,
                    config_name=self._config_name,
                    attachments=[str(self._config_file)] if self._config_file else None,
                    storage_options=self._storage_options,
                )

            if self._is_cloud:
                # - cloud: overlap data write + status + log upload in parallel
                _fns: list = [_do_save, self.write_completed]
                if log_file:
                    _fns.append(_do_log)
                Parallel(n_jobs=len(_fns), prefer="threads")(delayed(fn)() for fn in _fns)
            else:
                # - local: sequential (no network latency to hide; log already in run_dir)
                _do_save()
                self.write_completed()

    # ── storage: build metadata ──────────────────────────────────────────────

    @staticmethod
    def _build_metadata_record(
        result: TradingSessionResult,
        backtest_id: str,
        data_path: str,
        description: str | None = None,
        tags: list[str] | None = None,
        config_name: str | None = None,
        is_variation: bool = False,
        variation_id: str | None = None,
        variation_name: str | None = None,
        variation_params: dict | None = None,
    ) -> dict:
        """
        Build a metadata record dict suitable for writing to ``_metadata.parquet``.

        Performance metrics are computed and denormalized for fast DuckDB search.
        """
        import json as _json

        _desc = description or result.description or ""
        _tags = tags or []

        _perf: dict = {}
        try:
            _perf = result.performance()
        except Exception:
            pass

        _capital = result.get_total_capital()
        _commissions = (
            _json.dumps(result.commissions) if isinstance(result.commissions, dict) else (result.commissions or "")
        )

        def _ts_utc(ts: str | pd.Timestamp | None) -> pd.Timestamp | None:
            if ts is None:
                return None
            t = pd.Timestamp(ts)
            return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")

        return {
            "backtest_id": backtest_id,
            "name": result.name or "",
            "config_name": config_name or "",
            "data_path": data_path,
            "is_variation": is_variation,
            "variation_id": variation_id or "",
            "variation_name": variation_name or "",
            "variation_params": _json.dumps(variation_params) if variation_params else "",
            "strategy_class": result.strategy_class or "",
            "parameters": _json.dumps(result.parameters),
            "start": _ts_utc(result.start),
            "stop": _ts_utc(result.stop),
            "creation_time": _ts_utc(result.creation_time),
            "simulation_time_sec": int(result.simulation_time_sec or 0),
            "capital": float(_capital),
            "base_currency": result.base_currency or "",
            "commissions": _commissions,
            "exchanges": list(result.exchanges or []),
            "symbols": list(result.symbols or []),
            "author": result.author or "",
            "qubx_version": result.qubx_version or "",
            "description": _desc,
            "tags": list(_tags),
            "is_simulation": bool(result.is_simulation),
            # - performance metrics (denormalized for fast DuckDB search)
            "sharpe": float(_perf.get("sharpe", 0.0)),
            "cagr": float(_perf.get("cagr", 0.0)),
            "mdd_pct": float(_perf.get("mdd_pct", 0.0)),
            "mdd_usd": float(_perf.get("mdd_usd", 0.0)),
            "gain": float(_perf.get("gain", 0.0)),
            "qr": float(_perf.get("qr", 0.0)),
            "calmar": float(_perf.get("calmar", 0.0)),
            "sortino": float(_perf.get("sortino", 0.0)),
            "execs": int(_perf.get("execs", 0)),
            "fees": float(_perf.get("fees", 0.0)),
            "daily_turnover": float(_perf.get("daily_turnover", 0.0)),
        }

    # ── storage: save ────────────────────────────────────────────────────────

    @staticmethod
    def save(
        result: TradingSessionResult,
        base_path: str,
        run_id: str | None = None,
        run_dir: str | None = None,
        description: str | None = None,
        tags: "list[str] | str | None" = None,
        config_name: str | None = None,
        attachments: list[str] | None = None,
        storage_options: dict | None = None,
        is_variation: bool = False,
        variation_id: str | None = None,
        variation_name: str | None = None,
        variation_params: dict | None = None,
    ) -> str:
        """
        Save a TradingSessionResult to parquet-based storage (local or cloud).

        Directory layout (single run)::

            {base_path}/{ShortClass}/{name}/{run_id}/
                ├── _metadata.parquet
                ├── portfolio.parquet
                ├── executions.parquet
                ├── signals.parquet
                ├── targets.parquet
                ├── emitter_data.parquet
                └── config.yaml  (if attachments provided)

        Args:
            run_dir: Optional pre-computed run directory override.  When provided the
                     ``{ShortClass}/{name}/{run_id}`` path computation is skipped entirely.
                     Use this when the caller already holds ``self._run_dir`` (e.g.
                     :meth:`store_simulation_results`) to guarantee consistency for
                     multi-class composed strategies where ``result.strategy_class``
                     (a plain string) may not match the joined ``ShortClass+ShortClass``
                     form used to build ``self._run_dir`` at construction time.

        Returns:
            Full path to the run directory where data was written.

        Raises:
            ValueError: If ``result.name`` is None (required for path construction).
        """
        if not result.name and not (is_variation and variation_id) and run_dir is None:
            raise ValueError(
                "TradingSessionResult.name is required for storage — set 'name:' field in your YAML config"
            )

        _tags = normalize_tags(tags)

        if is_variation and variation_id:
            # - variation set: caller passes the variation set root as base_path;
            # - individual data goes directly into base_path/variation_id/
            # - (no name/class prefix — the parent dir already encodes those)
            _run_dir = f"{base_path.rstrip('/')}/{variation_id}/"
            _backtest_id = f"{variation_name}/{variation_id}" if variation_name else variation_id
        elif run_dir is not None:
            # - caller supplies the pre-computed run directory — skip class/name derivation.
            # - backtest_id is inferred by stripping base_path prefix from run_dir.
            _run_dir = run_dir.rstrip("/") + "/"
            _backtest_id = _run_dir.rstrip("/").removeprefix(base_path.rstrip("/")).strip("/")
        else:
            _short_class = get_short_class_name(result.strategy_class or "Unknown")
            _run_id = run_id or pd.Timestamp(result.creation_time or pd.Timestamp.now()).strftime("%Y%m%d_%H%M%S")
            _run_dir = f"{base_path.rstrip('/')}/{_short_class}/{result.name}/{_run_id}/"
            _backtest_id = f"{_short_class}/{result.name}/{_run_id}"

        def _stamp(df: pd.DataFrame | None) -> pd.DataFrame | None:
            if df is None or df.empty:
                return df
            _df = df.copy()
            _df["_backtest_id"] = _backtest_id
            return _df

        _data_writes = [
            (_stamp(result.portfolio_log), f"{_run_dir}{SimulationResultsSaver.DATA_FILES['portfolio']}"),
            (_stamp(result.executions_log), f"{_run_dir}{SimulationResultsSaver.DATA_FILES['executions']}"),
            (_stamp(result.signals_log), f"{_run_dir}{SimulationResultsSaver.DATA_FILES['signals']}"),
            (_stamp(result.targets_log), f"{_run_dir}{SimulationResultsSaver.DATA_FILES['targets']}"),
        ]
        if result.transfers_log is not None and not result.transfers_log.empty:
            _data_writes.append(
                (_stamp(result.transfers_log), f"{_run_dir}{SimulationResultsSaver.DATA_FILES['transfers']}")
            )
        if result.emitter_data is not None and not result.emitter_data.empty:
            _data_writes.append(
                (_stamp(result.emitter_data), f"{_run_dir}{SimulationResultsSaver.DATA_FILES['emitter']}")
            )

        # - build metadata record in-memory before the parallel I/O round
        meta_record = SimulationResultsSaver._build_metadata_record(
            result,
            backtest_id=_backtest_id,
            data_path="./",
            description=description,
            tags=_tags,
            config_name=config_name,
            is_variation=is_variation,
            variation_id=variation_id,
            variation_name=variation_name,
            variation_params=variation_params,
        )
        _meta_path = f"{_run_dir}{SimulationResultsSaver.METADATA_FILE}"

        # - single parallel round: all data files + metadata together
        _fns = [lambda df=df, p=p: write_parquet(df, p, storage_options) for df, p in _data_writes]
        _fns.append(lambda: write_metadata_parquet(meta_record, _meta_path, storage_options))  # type: ignore

        if _fns:
            if is_cloud_path(_run_dir) and len(_fns) > 1:
                from joblib import Parallel, delayed

                Parallel(n_jobs=len(_fns), prefer="threads")(delayed(fn)() for fn in _fns)
            else:
                [fn() for fn in _fns]

        if attachments:
            for attachment in attachments:
                copy_file_to_storage(attachment, _run_dir, storage_options)

        logger.info(f"Backtest saved to storage: <g>{_run_dir}</g>")
        return _run_dir

    # ── storage: load ────────────────────────────────────────────────────────

    @staticmethod
    def load(path: str, storage_options: dict | None = None) -> TradingSessionResult:
        """
        Load a TradingSessionResult from parquet-based storage (local or cloud).

        Args:
            path: Full path to the run directory (local or cloud URI).
            storage_options: Cloud credentials. None = auto-detect from env.

        Raises:
            FileNotFoundError: If ``_metadata.parquet`` is not found at ``path``.
        """
        _path = path.rstrip("/") + "/"
        _so = storage_options or {}

        def _read_df(filename: str) -> pd.DataFrame:
            try:
                return read_parquet(f"{_path}{filename}", _so)
            except Exception:
                return pd.DataFrame()

        try:
            meta_df = read_parquet(f"{_path}{SimulationResultsSaver.METADATA_FILE}", _so)
        except Exception as e:
            raise FileNotFoundError(f"Cannot load backtest from '{path}': {e}") from e

        if meta_df.empty:
            raise ValueError(f"Metadata at '{_path}{SimulationResultsSaver.METADATA_FILE}' is empty")

        with ThreadPoolExecutor() as pool:
            futures = {
                name: pool.submit(_read_df, filename) for name, filename in SimulationResultsSaver.DATA_FILES.items()
            }
            dfs = {name: f.result() for name, f in futures.items()}

        return SimulationResultsSaver._from_dfs(
            meta=meta_df.iloc[0].to_dict(),
            portfolio=dfs["portfolio"],
            executions=dfs["executions"],
            signals=dfs["signals"],
            targets=dfs["targets"],
            transfers=dfs["transfers"],
            emitter=dfs["emitter"],
        )

    @staticmethod
    def _from_dfs(
        meta: dict,
        portfolio: pd.DataFrame,
        executions: pd.DataFrame,
        signals: pd.DataFrame,
        targets: pd.DataFrame,
        transfers: pd.DataFrame,
        emitter: pd.DataFrame,
    ) -> TradingSessionResult:
        """
        Reconstruct a TradingSessionResult from a metadata dict and DataFrames.

        Single source of truth used by both :meth:`load` (pyarrow.fs reads) and
        ``BacktestStorage._load_from_path()`` (DuckDB reads).
        """
        import json as _json

        from qubx.core.metrics import TradingSessionResult

        # - restore DatetimeIndex where expected
        for df, idx_col in [
            (portfolio, "timestamp"),
            (executions, "timestamp"),
            (signals, "timestamp"),
            (targets, "timestamp"),
        ]:
            if not df.empty and idx_col in df.columns and df.index.name != idx_col:
                df.set_index(idx_col, inplace=True)

        # - drop internal _backtest_id column added during write
        for df in [portfolio, executions, signals, targets, transfers]:
            if "_backtest_id" in df.columns:
                df.drop(columns=["_backtest_id"], inplace=True)

        # - normalize DatetimeIndex to tz-naive: parquet/DuckDB may return UTC-aware timestamps
        # - but the simulator produces tz-naive np.datetime64 — keep consistent
        for df in [portfolio, executions, signals, targets, transfers, emitter]:
            if not df.empty and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        # - normalize scalar timestamps from parquet metadata (stored as pa.timestamp("us", tz="UTC"))
        # - strip tz so they can be used for .loc[] slicing against tz-naive portfolio_log.index
        def _strip_tz(v) -> pd.Timestamp | str:
            if isinstance(v, pd.Timestamp) and v.tzinfo is not None:
                return v.tz_localize(None)
            return v

        # - parquet list columns may come back as numpy object arrays (not Python lists)
        def _to_list(v) -> list:
            if v is None:
                return []
            try:
                return list(v)
            except TypeError:
                return []

        tsr = TradingSessionResult(
            id=meta.get("id", 0),
            name=meta.get("name", ""),
            start=_strip_tz(meta.get("start", "")),
            stop=_strip_tz(meta.get("stop", "")),
            exchanges=_to_list(meta.get("exchanges")),
            instruments=_to_list(meta.get("symbols")),
            capital=float(meta.get("capital", 0.0)),
            base_currency=meta.get("base_currency", "USDT"),
            commissions=meta.get("commissions"),
            portfolio_log=portfolio,
            executions_log=executions,
            signals_log=signals,
            targets_log=targets,
            transfers_log=transfers if not transfers.empty else None,
            emitter_data=emitter if not emitter.empty else None,
            strategy_class=meta.get("strategy_class", ""),
            parameters=_json.loads(meta.get("parameters") or "{}"),
            is_simulation=bool(meta.get("is_simulation", True)),
            creation_time=_strip_tz(meta.get("creation_time")),
            author=meta.get("author"),
            variation_name=meta.get("variation_name") or None,
        )
        tsr.qubx_version = meta.get("qubx_version")
        tsr.description = meta.get("description") or None
        tsr.simulation_time_sec = int(meta.get("simulation_time_sec") or 0)
        return tsr
