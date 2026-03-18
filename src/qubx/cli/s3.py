"""S3 storage operations using named accounts from Qubx settings.

Uses pyarrow.fs.S3FileSystem — no dependency on s3fs/aiobotocore.
"""

from __future__ import annotations

import click
from pyarrow.fs import FileSelector, FileType, LocalFileSystem, copy_files

from qubx.config import get_s3_account, get_settings
from qubx.utils.results import _make_pa_s3_filesystem, _s3_account_to_opts


def _parse_path(path: str) -> tuple[str, str, str]:
    """Parse ``account:bucket/key`` into (account, bucket, key).

    Supports formats:
        account:bucket/path/to/key
        account:bucket

    Returns:
        (account_name, bucket, key)  — *key* may be empty string.
    """
    if ":" not in path:
        raise click.BadParameter(
            f"Invalid path '{path}'. Expected format: account:bucket/path"
        )
    account, rest = path.split(":", 1)
    if not account:
        raise click.BadParameter("Account name cannot be empty")
    if not rest:
        raise click.BadParameter("Bucket/path cannot be empty")

    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return account, bucket, key


def _get_fs(account: str):
    """Build a pyarrow.fs.S3FileSystem from a named Qubx account."""
    acct = get_s3_account(account)
    opts = _s3_account_to_opts(acct)
    return _make_pa_s3_filesystem(opts)


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:6.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} PB"


def _full_path(bucket: str, key: str) -> str:
    return f"{bucket}/{key}".rstrip("/") if key else bucket


def s3_ls(path: str, recursive: bool = False, long: bool = False) -> None:
    account, bucket, key = _parse_path(path)
    fs = _get_fs(account)
    full = _full_path(bucket, key)

    try:
        entries = fs.get_file_info(FileSelector(full, recursive=recursive))
    except Exception as e:
        click.echo(click.style(f"Error listing {full}: {e}", fg="red"), err=True)
        raise click.Abort()

    if not entries:
        click.echo(click.style(f"No files found under: {full}", fg="yellow"))
        return

    for info in sorted(entries, key=lambda e: e.path):
        if long:
            size = _human_size(info.size) if info.type == FileType.File else "     -"
            kind = "DIR " if info.type == FileType.Directory else "FILE"
            click.echo(f"  {kind}  {size}  {info.path}")
        else:
            click.echo(f"  {info.path}")


def s3_rm(path: str, recursive: bool = False) -> None:
    account, bucket, key = _parse_path(path)
    fs = _get_fs(account)
    full = _full_path(bucket, key)

    try:
        if recursive:
            entries = fs.get_file_info(FileSelector(full, recursive=True))
            files = [e for e in entries if e.type == FileType.File]
            if not files:
                click.echo(click.style(f"No files found under: {full}", fg="yellow"))
                return
            click.echo(f"Deleting {len(files)} file(s) under {full} ...")
            fs.delete_dir(full)
        else:
            fs.delete_file(full)
        click.echo(click.style(f"✓ Deleted {full}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error deleting {full}: {e}", fg="red"), err=True)
        raise click.Abort()


def s3_cp(src: str, dst: str, recursive: bool = False) -> None:
    src_is_s3 = ":" in src
    dst_is_s3 = ":" in dst

    if src_is_s3 and dst_is_s3:
        src_account, src_bucket, src_key = _parse_path(src)
        dst_account, dst_bucket, dst_key = _parse_path(dst)
        if src_account != dst_account:
            click.echo(
                click.style("Cross-account copy is not supported yet.", fg="red"),
                err=True,
            )
            raise click.Abort()
        fs = _get_fs(src_account)
        src_path = _full_path(src_bucket, src_key)
        dst_path = _full_path(dst_bucket, dst_key)
        copy_files(src_path, dst_path, source_filesystem=fs, destination_filesystem=fs)
        click.echo(click.style(f"✓ Copied {src_path} → {dst_path}", fg="green"))

    elif src_is_s3 and not dst_is_s3:
        account, bucket, key = _parse_path(src)
        s3_fs = _get_fs(account)
        src_path = _full_path(bucket, key)
        copy_files(src_path, dst, source_filesystem=s3_fs, destination_filesystem=LocalFileSystem())
        click.echo(click.style(f"✓ Downloaded {src_path} → {dst}", fg="green"))

    elif not src_is_s3 and dst_is_s3:
        account, bucket, key = _parse_path(dst)
        s3_fs = _get_fs(account)
        dst_path = _full_path(bucket, key)
        copy_files(src, dst_path, source_filesystem=LocalFileSystem(), destination_filesystem=s3_fs)
        click.echo(click.style(f"✓ Uploaded {src} → {dst_path}", fg="green"))

    else:
        click.echo(
            click.style("At least one path must be an S3 path (account:bucket/path).", fg="red"),
            err=True,
        )
        raise click.Abort()


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
