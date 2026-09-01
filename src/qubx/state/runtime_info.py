"""
Static runtime facts written once at startup to ``state:{bot_id}:info``.

The blob is the platform's only runtime source for package versions and
exported streams (see xlydian-platform
docs/superpowers/specs/2026-09-01-bot-versions-and-dependency-graph-design.md).
It is computed once — importlib.metadata.distributions() is too slow for the
5s snapshot cadence — and everything in it is plain JSON types.
"""

import sys
from importlib import metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qubx.core.interfaces import ITradeDataExport

RUNTIME_INFO_KEY = "info"
RUNTIME_INFO_SCHEMA_VERSION = 1


def _distribution_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").strip() if dist.metadata else ""
        if name:
            out[name.lower()] = dist.version
    return out


def _strategy_identity(strategy_class: type, packages: dict[str, str]) -> dict:
    """Resolve the strategy's distribution name + version from its top module."""
    top_module = strategy_class.__module__.split(".")[0]
    package: str = top_module
    version: str | None = None
    try:
        for dist_name in metadata.packages_distributions().get(top_module) or []:
            if dist_name.lower() in packages:
                package = dist_name.lower()
                version = packages[dist_name.lower()]
                break
    except Exception:  # packages_distributions can choke on broken metadata
        pass
    if version is None and top_module.lower() in packages:
        # metadata.packages_distributions() reads top_level.txt / RECORD entries
        # that uv editable installs (root project run from source, e.g. `qubx`
        # itself or a strategy repo installed with `-e .`) don't always populate,
        # so the loop above finds nothing even though the module name IS the
        # distribution name. Fall back to a direct name match against the
        # already-collected distributions before giving up on a version.
        package = top_module.lower()
        version = packages[top_module.lower()]
    return {
        "package": package,
        "version": version,
        "class": f"{strategy_class.__module__}.{strategy_class.__name__}",
    }


def build_runtime_info(strategy: object, exporter: "ITradeDataExport | None", timestamp: str) -> dict:
    packages = _distribution_versions()
    exports: dict[str, list[str]] = {}
    if exporter is not None:
        try:
            exports = exporter.get_export_info()
        except Exception:
            exports = {}
    return {
        "schema_version": RUNTIME_INFO_SCHEMA_VERSION,
        "started_at": timestamp,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "qubx": packages.get("qubx", "Dev"),
        "strategy": _strategy_identity(type(strategy), packages),
        "packages": packages,
        "exports": exports,
    }
