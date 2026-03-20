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
