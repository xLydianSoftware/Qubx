"""Centralized S3 client with named accounts, retry logic, and unified path handling.

Supports two path formats:
- Cloud URI: ``s3://bucket/key`` (credentials from default account or explicit)
- Account URI: ``account:bucket/key`` (credentials from named account in settings)

Usage::

    from qubx.utils.s3 import S3Client

    client = S3Client("hetzner")
    df = client.read_parquet("frab/spreads/asset=BTC/data.parquet")

    # Or parse account:bucket/key
    client, path = S3Client.from_uri("hetzner:frab/spreads/asset=BTC/data.parquet")
    df = client.read_parquet(path)
"""

import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qubx import logger

# ---------------------------------------------------------------------------
# Module-level filesystem cache (same pattern as before, now owned here)
# ---------------------------------------------------------------------------
_PA_S3_FS_CACHE: dict[tuple, Any] = {}


def _make_filesystem(opts: dict):
    """Return a cached ``pyarrow.fs.S3FileSystem`` for the given storage options."""
    _region = opts.get("client_kwargs", {}).get("region_name") if opts.get("client_kwargs") else None
    if not _region and opts.get("endpoint_url") and ".r2.cloudflarestorage.com" in opts["endpoint_url"]:
        _region = "auto"
    _cache_key = (opts.get("key"), opts.get("secret"), opts.get("endpoint_url"), _region)

    if _cache_key not in _PA_S3_FS_CACHE:
        import pyarrow.fs as pafs

        kwargs: dict = {}
        if opts.get("key"):
            kwargs["access_key"] = opts["key"]
        if opts.get("secret"):
            kwargs["secret_key"] = opts["secret"]
        if opts.get("endpoint_url"):
            kwargs["endpoint_override"] = opts["endpoint_url"].removeprefix("https://").removeprefix("http://")
        if _region:
            kwargs["region"] = _region

        _PA_S3_FS_CACHE[_cache_key] = pafs.S3FileSystem(**kwargs)

    return _PA_S3_FS_CACHE[_cache_key]


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def is_cloud_path(path: str) -> bool:
    """Check if path is a cloud storage URI (S3, GCS, Azure)."""
    return path.startswith(("s3://", "gs://", "az://", "abfs://"))


def strip_scheme(path: str) -> str:
    """Strip cloud URI scheme — pyarrow expects ``bucket/key``, not ``s3://bucket/key``."""
    for prefix in ("s3://", "gs://", "az://", "abfs://"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def parse_uri(uri: str) -> tuple[str, str]:
    """Parse ``account:bucket/key`` into ``(account, bucket_and_key)``.

    Returns:
        (account_name, "bucket/key") tuple.

    Raises:
        ValueError: If format is invalid.
    """
    if ":" not in uri:
        raise ValueError(f"Invalid S3 URI '{uri}'. Expected format: account:bucket/path")
    account, rest = uri.split(":", 1)
    if not account:
        raise ValueError("Account name cannot be empty")
    if not rest:
        raise ValueError("Bucket/path cannot be empty")
    return account, rest


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_TRANSIENT_ERROR_SUBSTRINGS = (
    "NETWORK_CONNECTION",
    "ConnectionReset",
    "ConnectionAborted",
    "BrokenPipe",
    "Timeout",
    "throttl",
    "SlowDown",
    "ServiceUnavailable",
    "InternalError",
    "RequestTimeout",
)


def _is_transient(exc: Exception) -> bool:
    """Check if an exception looks like a transient S3/network error."""
    msg = str(exc)
    return any(s.lower() in msg.lower() for s in _TRANSIENT_ERROR_SUBSTRINGS)


def _retry(func, *, max_attempts: int = 3, base_delay: float = 1.0):
    """Execute func with retries on transient errors using exponential backoff."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if not _is_transient(e) or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"S3 transient error (attempt {attempt + 1}/{max_attempts}), retrying in {delay:.1f}s: {e}")
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# S3Client
# ---------------------------------------------------------------------------

class S3Client:
    """S3 client with named account support, connection caching, and retry logic.

    Args:
        account: Named account from ``~/.qubx/config.json`` (e.g. ``"hetzner"``).
                 If None, uses the default account from settings.
        storage_options: Explicit credentials dict. Overrides account lookup.
        max_retries: Max retry attempts for transient errors.
        retry_delay: Base delay in seconds for exponential backoff.
    """

    def __init__(
        self,
        account: str | None = None,
        storage_options: dict | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._opts = self._resolve_options(storage_options, account)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @staticmethod
    def from_uri(uri: str, **kwargs) -> tuple["S3Client", str]:
        """Create client from ``account:bucket/key`` URI.

        Returns:
            (S3Client, "bucket/key") tuple.
        """
        account, path = parse_uri(uri)
        return S3Client(account=account, **kwargs), path

    @property
    def fs(self):
        """Cached ``pyarrow.fs.S3FileSystem`` instance."""
        return _make_filesystem(self._opts)

    # -- Parquet I/O --------------------------------------------------------

    def read_parquet(self, path: str) -> pd.DataFrame:
        """Read a parquet file from S3 with retry on transient errors.

        Args:
            path: ``bucket/key`` path (no ``s3://`` prefix).
        """
        path = strip_scheme(path)

        def _read():
            table = pq.read_table(path, filesystem=self.fs)
            return table.to_pandas()

        return _retry(_read, max_attempts=self._max_retries, base_delay=self._retry_delay)

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Write a DataFrame to parquet on S3 with retry on transient errors.

        Args:
            df: DataFrame to write.
            path: ``bucket/key`` path (no ``s3://`` prefix).
        """
        if df is None or df.empty:
            return
        path = strip_scheme(path)
        table = pa.Table.from_pandas(df, preserve_index=True)

        def _write():
            pq.write_table(table, path, filesystem=self.fs)

        _retry(_write, max_attempts=self._max_retries, base_delay=self._retry_delay)

    def write_parquet_table(self, table: pa.Table, path: str) -> None:
        """Write a pyarrow Table to parquet on S3 with retry."""
        path = strip_scheme(path)

        def _write():
            pq.write_table(table, path, filesystem=self.fs)

        _retry(_write, max_attempts=self._max_retries, base_delay=self._retry_delay)

    # -- File operations ----------------------------------------------------

    def ls(self, path: str, recursive: bool = False) -> list:
        """List files under a path. Returns list of ``pyarrow.fs.FileInfo``."""
        from pyarrow.fs import FileSelector

        path = strip_scheme(path).rstrip("/")

        def _ls():
            return self.fs.get_file_info(FileSelector(path, recursive=recursive))

        return _retry(_ls, max_attempts=self._max_retries, base_delay=self._retry_delay)

    def rm(self, path: str, recursive: bool = False) -> None:
        """Delete a file or directory on S3."""
        path = strip_scheme(path)

        def _rm():
            if recursive:
                self.fs.delete_dir(path)
            else:
                self.fs.delete_file(path)

        _retry(_rm, max_attempts=self._max_retries, base_delay=self._retry_delay)

    def copy_local_to_s3(self, src: str, dst: str, dst_name: str | None = None) -> None:
        """Copy a local file to S3.

        Args:
            src: Local file path.
            dst: S3 directory path (``bucket/key/``).
            dst_name: Override filename. Defaults to source filename.
        """
        src_path = Path(src)
        if not src_path.is_file():
            logger.warning(f"File not found, skipping: {src}")
            return

        filename = dst_name or src_path.name
        full_dst = f"{strip_scheme(dst).rstrip('/')}/{filename}"

        def _copy():
            with open(src, "rb") as fin:
                data = fin.read()
            with self.fs.open_output_stream(full_dst) as fout:
                fout.write(data)

        _retry(_copy, max_attempts=self._max_retries, base_delay=self._retry_delay)

    def copy_s3(self, src: str, dst: str, recursive: bool = False) -> None:
        """Copy files within the same S3 account."""
        from pyarrow.fs import copy_files

        src = strip_scheme(src)
        dst = strip_scheme(dst)

        def _copy():
            copy_files(src, dst, source_filesystem=self.fs, destination_filesystem=self.fs)

        _retry(_copy, max_attempts=self._max_retries, base_delay=self._retry_delay)

    # -- Internals ----------------------------------------------------------

    @property
    def storage_options(self) -> dict:
        """Resolved storage options dict (for callers that need raw credentials)."""
        return self._opts

    @staticmethod
    def _resolve_options(explicit: dict | None, account: str | None) -> dict:
        """Resolve S3 storage options: explicit → named account → default → empty."""
        if explicit is not None:
            return explicit

        from qubx.config import get_settings

        settings = get_settings()
        name = account or settings.default_s3_account
        if name and name in settings.s3:
            acct = settings.s3[name]
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

        return {}
