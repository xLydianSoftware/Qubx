"""Registry placeholder for a storage whose optional extra is not installed.

Without it, `StorageRegistry.get("iceberg::r2")` on an install lacking the extra
says "Unknown storage type", which reads like a typo rather than a missing package.
"""

from qubx.data.registry import StorageRegistry


def register_missing_extra(name: str, extra: str, error: ImportError, requires: str = "") -> None:
    """`requires` names the pinned package, so a version mismatch reads as one too."""
    if StorageRegistry.is_registered(name):
        return

    class _MissingExtraStorage:
        def __init__(self, *args, **kwargs):
            needs = f"{extra} ({requires})" if requires else extra
            raise ImportError(f"storage {name!r} needs the optional dependency {needs}: uv add '{extra}'") from error

    StorageRegistry.register(name)(_MissingExtraStorage)
