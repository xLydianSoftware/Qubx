"""S3 storage operations using named accounts from Qubx settings.

Uses :class:`qubx.utils.s3.S3Client` for all operations.
"""

import click
from pyarrow.fs import FileType, LocalFileSystem, copy_files

from qubx.config import get_settings
from qubx.utils.s3 import S3Client, parse_uri


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:6.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} PB"


def s3_ls(path: str, recursive: bool = False, long: bool = False) -> None:
    try:
        client, s3_path = S3Client.from_uri(path)
    except ValueError as e:
        raise click.BadParameter(str(e))

    try:
        entries = client.ls(s3_path, recursive=recursive)
    except Exception as e:
        click.echo(click.style(f"Error listing {s3_path}: {e}", fg="red"), err=True)
        raise click.Abort()

    if not entries:
        click.echo(click.style(f"No files found under: {s3_path}", fg="yellow"))
        return

    for info in sorted(entries, key=lambda e: e.path):
        if long:
            size = _human_size(info.size) if info.type == FileType.File else "     -"
            kind = "DIR " if info.type == FileType.Directory else "FILE"
            click.echo(f"  {kind}  {size}  {info.path}")
        else:
            click.echo(f"  {info.path}")


def s3_rm(path: str, recursive: bool = False) -> None:
    try:
        client, s3_path = S3Client.from_uri(path)
    except ValueError as e:
        raise click.BadParameter(str(e))

    try:
        if recursive:
            entries = client.ls(s3_path, recursive=True)
            files = [e for e in entries if e.type == FileType.File]
            if not files:
                click.echo(click.style(f"No files found under: {s3_path}", fg="yellow"))
                return
            click.echo(f"Deleting {len(files)} file(s) under {s3_path} ...")
        client.rm(s3_path, recursive=recursive)
        click.echo(click.style(f"✓ Deleted {s3_path}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error deleting {s3_path}: {e}", fg="red"), err=True)
        raise click.Abort()


def s3_cp(src: str, dst: str, recursive: bool = False) -> None:
    src_is_s3 = ":" in src
    dst_is_s3 = ":" in dst

    if src_is_s3 and dst_is_s3:
        try:
            src_account, src_path = parse_uri(src)
            dst_account, dst_path = parse_uri(dst)
        except ValueError as e:
            raise click.BadParameter(str(e))

        if src_account != dst_account:
            click.echo(
                click.style("Cross-account copy is not supported yet.", fg="red"),
                err=True,
            )
            raise click.Abort()
        client = S3Client(account=src_account)
        client.copy_s3(src_path, dst_path, recursive=recursive)
        click.echo(click.style(f"✓ Copied {src_path} → {dst_path}", fg="green"))

    elif src_is_s3 and not dst_is_s3:
        try:
            client, src_path = S3Client.from_uri(src)
        except ValueError as e:
            raise click.BadParameter(str(e))
        copy_files(src_path, dst, source_filesystem=client.fs, destination_filesystem=LocalFileSystem())
        click.echo(click.style(f"✓ Downloaded {src_path} → {dst}", fg="green"))

    elif not src_is_s3 and dst_is_s3:
        try:
            client, dst_path = S3Client.from_uri(dst)
        except ValueError as e:
            raise click.BadParameter(str(e))
        copy_files(src, dst_path, source_filesystem=LocalFileSystem(), destination_filesystem=client.fs)
        click.echo(click.style(f"✓ Uploaded {src} → {dst_path}", fg="green"))

    else:
        click.echo(
            click.style("At least one path must be an S3 path (account:bucket/path).", fg="red"),
            err=True,
        )
        raise click.Abort()


def s3_parquet_stats(path: str, per_column: bool = False) -> None:
    """Print file-level summary + row-group size distribution for one parquet file.

    Accepts either an S3 URI (``account:bucket/key``) or a local filesystem path.
    Uses pyarrow for metadata parsing so no data pages are fetched.
    """
    import pyarrow.parquet as pq
    from pyarrow.fs import LocalFileSystem

    # Heuristic: an S3 URI has "account:..." where account is non-empty and
    # contains no path separator. Fall back to local for anything else.
    is_s3 = ":" in path and "/" not in path.split(":", 1)[0] and not path.startswith("/")
    if is_s3:
        try:
            client, s3_path = S3Client.from_uri(path)
        except ValueError as e:
            raise click.BadParameter(str(e))
        fs = client.fs
        open_path = s3_path
    else:
        fs = LocalFileSystem()
        open_path = path

    try:
        with fs.open_input_file(open_path) as f:
            meta = pq.ParquetFile(f).metadata
    except Exception as e:
        click.echo(click.style(f"Error reading {path}: {e}", fg="red"), err=True)
        raise click.Abort()

    click.echo(click.style(f"File: {path}", fg="cyan", bold=True))
    click.echo(f"  num_rows:       {meta.num_rows:,}")
    click.echo(f"  num_row_groups: {meta.num_row_groups}")
    click.echo(f"  num_columns:    {meta.num_columns}")
    click.echo(f"  format_version: {meta.format_version}")
    if meta.created_by:
        click.echo(f"  created_by:     {meta.created_by}")

    if meta.num_row_groups == 0:
        return

    # Parquet spec: RowGroup.total_byte_size is the SUM of uncompressed
    # column sizes. The compressed on-disk size is the sum of
    # ColumnChunk.total_compressed_size across columns.
    rows: list[int] = []
    per_rg_compressed: list[int] = []
    per_rg_uncompressed: list[int] = []
    compressions: set[str] = set()
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        rows.append(rg.num_rows)
        c_sz = 0
        u_sz = 0
        for c in range(rg.num_columns):
            col = rg.column(c)
            c_sz += col.total_compressed_size
            u_sz += col.total_uncompressed_size
            compressions.add(str(col.compression))
        per_rg_compressed.append(c_sz)
        per_rg_uncompressed.append(u_sz)

    total_compressed = sum(per_rg_compressed)
    total_uncompressed = sum(per_rg_uncompressed)
    click.echo()
    click.echo(click.style("Row groups:", fg="cyan", bold=True))
    click.echo(f"  rows per rg:   min={min(rows):,}  max={max(rows):,}  avg={sum(rows)//len(rows):,}")
    click.echo(
        f"  cmp per rg:    min={_human_size(min(per_rg_compressed))}  "
        f"max={_human_size(max(per_rg_compressed))}  avg={_human_size(sum(per_rg_compressed) // len(per_rg_compressed))}"
    )
    click.echo(f"  total:         compressed={_human_size(total_compressed)}  uncompressed={_human_size(total_uncompressed)}")
    if total_uncompressed > 0:
        ratio = 100 * total_compressed / total_uncompressed
        click.echo(f"  compression:   {', '.join(sorted(compressions))}  ({ratio:.1f}% of uncompressed)")
    else:
        click.echo(f"  compression:   {', '.join(sorted(compressions))}")

    if per_column:
        click.echo()
        click.echo(click.style("Columns (row group 0):", fg="cyan", bold=True))
        rg0 = meta.row_group(0)
        click.echo(f"  {'path':30s}  {'type':12s}  {'compressed':>11s}  {'uncompressed':>12s}   pct  compression")
        for c in range(rg0.num_columns):
            col = rg0.column(c)
            pct = (100 * col.total_compressed_size / col.total_uncompressed_size) if col.total_uncompressed_size else 0.0
            click.echo(
                f"  {col.path_in_schema:30s}  {str(col.physical_type):12s}  "
                f"{_human_size(col.total_compressed_size):>11s}  {_human_size(col.total_uncompressed_size):>12s}  "
                f"{pct:5.1f}%  {col.compression}"
            )


def s3_accounts() -> None:
    s = get_settings()
    if not s.s3:
        click.echo("No S3 accounts configured.")
        return

    click.echo(click.style("Configured S3 accounts:", fg="cyan", bold=True))
    for name, acct in s.s3.items():
        default_marker = " (default)" if name == s.default_s3_account else ""
        click.echo(f"  {click.style(name, fg='yellow')}{default_marker}")
        click.echo(f"    endpoint: {acct.endpoint_url}")
        if acct.region:
            click.echo(f"    region:   {acct.region}")
