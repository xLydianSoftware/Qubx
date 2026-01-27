"""
Plugin loader for Qubx.

This module provides functionality to load plugins from:
1. Local paths (for development) - scans .py files and imports them
2. Installed modules (for pip packages) - imports modules to trigger decorators
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qubx import logger

if TYPE_CHECKING:
    from qubx.utils.runner.configs import PluginsConfig

# Track which plugins have been loaded to avoid duplicate loading
_loaded_plugins: set[str] = set()


def load_plugins(config: "PluginsConfig | None") -> None:
    """
    Load plugins from the configuration.

    This function:
    1. Adds paths to sys.path
    2. Auto-discovers and imports .py files from those paths
    3. Imports explicit modules (triggers decorators)

    Args:
        config: Plugin configuration with paths and modules
    """
    if config is None:
        return

    # Load plugins from paths (local development)
    for path_str in config.paths:
        _load_plugins_from_path(path_str)

    # Load plugins from modules (installed packages)
    for module_name in config.modules:
        _load_plugin_module(module_name)


def _load_plugins_from_path(path_str: str) -> None:
    """
    Load all .py files from a path.

    This adds the path to sys.path and imports all Python files found in it.
    Subdirectories are also scanned recursively.

    Args:
        path_str: Path to scan for .py files
    """
    path = Path(os.path.expanduser(path_str)).resolve()

    if not path.exists():
        logger.warning(f"Plugin path does not exist: {path}")
        return

    if not path.is_dir():
        logger.warning(f"Plugin path is not a directory: {path}")
        return

    # Add to sys.path if not already there
    path_str_resolved = str(path)
    if path_str_resolved not in sys.path:
        sys.path.insert(0, path_str_resolved)
        logger.debug(f"Added plugin path to sys.path: {path_str_resolved}")

    # Find and import all .py files
    py_files = list(path.rglob("*.py"))
    loaded_count = 0

    for py_file in py_files:
        # Skip __pycache__ directories
        if "__pycache__" in str(py_file):
            continue

        # Skip __init__.py files (they'll be imported when importing the package)
        if py_file.name == "__init__.py":
            continue

        # Create a unique key for this file
        file_key = str(py_file.resolve())
        if file_key in _loaded_plugins:
            continue

        try:
            # Calculate module name relative to the plugin path
            rel_path = py_file.relative_to(path)
            # Convert path to module name (e.g., foo/bar.py -> foo.bar)
            module_parts = list(rel_path.parts)
            module_parts[-1] = module_parts[-1][:-3]  # Remove .py extension
            module_name = ".".join(module_parts)

            # Generate a unique module name to avoid conflicts
            unique_module_name = f"qubx_plugin_{path.name}_{module_name}"

            # Load the module
            spec = importlib.util.spec_from_file_location(unique_module_name, py_file)
            if spec is None or spec.loader is None:
                logger.warning(f"Failed to create module spec for: {py_file}")
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_module_name] = module
            spec.loader.exec_module(module)

            _loaded_plugins.add(file_key)
            loaded_count += 1
            logger.debug(f"Loaded plugin file: {py_file}")

        except Exception as e:
            logger.warning(f"Failed to load plugin file {py_file}: {e}")

    if loaded_count > 0:
        logger.info(f"Loaded {loaded_count} plugin file(s) from {path}")


def _load_plugin_module(module_name: str) -> None:
    """
    Import a module to trigger its decorators.

    This is used for pip-installed packages that use @storage(), @reader(),
    @data_provider(), etc. decorators.

    Args:
        module_name: Fully qualified module name to import
    """
    if module_name in _loaded_plugins:
        return

    try:
        importlib.import_module(module_name)
        _loaded_plugins.add(module_name)
        logger.info(f"Loaded plugin module: {module_name}")
    except ImportError as e:
        logger.warning(f"Failed to import plugin module {module_name}: {e}")
    except Exception as e:
        logger.warning(f"Error loading plugin module {module_name}: {e}")


def reset_loaded_plugins() -> None:
    """
    Reset the set of loaded plugins.

    This is primarily useful for testing.
    """
    global _loaded_plugins
    _loaded_plugins = set()
