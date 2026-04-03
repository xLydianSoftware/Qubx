import ast
import getpass
import os
import platform
import re
import shutil
import urllib.request
from base64 import b64encode
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin

import yaml
from git import Repo

from qubx import logger
from qubx.utils.misc import (
    cyan,
    generate_name,
    green,
    magenta,
    makedirs,
    red,
    white,
    yellow,
)
from qubx.utils.runner.configs import ExchangeConfig, LoggingConfig, StrategyConfig, load_strategy_config_from_yaml

from .misc import (
    PyClassInfo,
    find_git_root,
    find_pyproject_root,
    scan_py_classes_in_directory,
)

Import = namedtuple("Import", ["module", "name", "alias"])
DEFAULT_CFG_NAME = "config.yml"


class ImportResolutionError(Exception):
    """Raised when import resolution fails."""

    pass


class DependencyResolutionError(Exception):
    """Raised when dependency file resolution fails."""

    pass


@dataclass
class ReleaseInfo:
    tag: str
    commit: str
    user: str
    time: datetime
    commited_files: list[str]


@dataclass
class StrategyInfo:
    name: str
    classes: list[PyClassInfo]
    config: StrategyConfig


def resolve_relative_import(relative_module: str, file_path: str, project_root: str) -> str:
    """
    Resolve a relative import to an absolute module path.

    Args:
        relative_module: The relative module string (e.g., "..utils", ".helper")
        file_path: Absolute path to the file containing the relative import
        project_root: Root directory of the project

    Returns:
        Absolute module path string

    Raises:
        ImportResolutionError: If the relative import cannot be resolved
    """
    # Get the relative path from project root to the file
    rel_file_path = os.path.relpath(file_path, project_root)

    # Get the directory containing the file (remove filename)
    file_dir = os.path.dirname(rel_file_path)

    # Convert file directory path to module path
    if file_dir:
        current_module_parts = file_dir.replace(os.sep, ".").split(".")
        # Remove 'src' prefix if present (common Python project structure)
        if current_module_parts[0] == "src" and len(current_module_parts) > 1:
            current_module_parts = current_module_parts[1:]
    else:
        current_module_parts = []

    # Parse the relative import
    level = 0
    module_name = relative_module

    # Count leading dots to determine level
    while module_name.startswith("."):
        level += 1
        module_name = module_name[1:]

    # Calculate the target module parts
    if level == 0:
        # Not actually a relative import
        return module_name

    # For relative imports, we need to go up from the current package
    # level-1 because level=1 means "same package", level=2 means "parent package"
    if level == 1:
        # from .module -> current package + module
        parent_parts = current_module_parts
    else:
        # from ..module -> parent package + module
        levels_up = level - 1
        if levels_up > len(current_module_parts):
            raise ImportResolutionError(f"Relative import '{relative_module}' goes beyond project root in {file_path}")
        parent_parts = current_module_parts[:-levels_up] if levels_up > 0 else current_module_parts

    # Combine parent with the remaining module name
    if module_name:
        resolved_parts = parent_parts + module_name.split(".")
    else:
        resolved_parts = parent_parts

    return ".".join(resolved_parts) if resolved_parts else ""


def get_imports(
    path: str, what_to_look: list[str] = ["my_strategy"], project_root: str | None = None
) -> Generator[Import, None, None]:
    """
    Get imports from the given file.

    Args:
        path: Path to Python file to analyze
        what_to_look: List of module prefixes to filter for (empty list = no filter)
        project_root: Root directory for resolving relative imports (optional)

    Yields:
        Import namedtuples for each matching import statement

    Raises:
        SyntaxError: If the Python file has syntax errors
        FileNotFoundError: If the file doesn't exist
        ImportResolutionError: If relative imports cannot be resolved
    """
    with open(path) as fh:
        root = ast.parse(fh.read(), path)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            # Handle direct imports like: import module, import module.submodule
            for n in node.names:
                module_parts = n.name.split(".")
                # Apply filter if provided
                if not what_to_look or module_parts[0] in what_to_look:
                    yield Import(module_parts, module_parts[-1:], n.asname)

        elif isinstance(node, ast.ImportFrom):
            # Handle from imports like: from module import name
            level = getattr(node, "level", 0)

            if level > 0:
                # This is a relative import (has dots)
                if project_root is None:
                    # Skip relative imports if no project root provided
                    continue

                # Build the relative module string
                relative_module = "." * level
                if node.module:
                    relative_module += node.module

                try:
                    # Resolve relative import to absolute module path
                    resolved_module = resolve_relative_import(relative_module, path, project_root)
                    if resolved_module:
                        module_parts = resolved_module.split(".")
                        # Apply filter if provided
                        if not what_to_look or (module_parts and module_parts[0] in what_to_look):
                            for n in node.names:
                                yield Import(module_parts, n.name.split("."), n.asname)
                except ImportResolutionError as e:
                    # Log the error but don't fail completely
                    logger.warning(f"Failed to resolve relative import: {e}")
                    continue
            else:
                # Regular from import (no dots)
                if node.module:
                    module_parts = node.module.split(".")
                    # Apply filter if provided
                    if not what_to_look or module_parts[0] in what_to_look:
                        for n in node.names:
                            yield Import(module_parts, n.name.split("."), n.asname)


def get_project_package_name(pyproject_root: str) -> str | None:
    """
    Extract package name from pyproject.toml.

    Args:
        pyproject_root: Path to directory containing pyproject.toml

    Returns:
        Package name or None if not found
    """
    try:
        import toml

        pyproject_path = os.path.join(pyproject_root, "pyproject.toml")
        if not os.path.exists(pyproject_path):
            return None

        with open(pyproject_path, "r") as f:
            data = toml.load(f)

        # Try PEP 621 format first
        if "project" in data and "name" in data["project"]:
            return data["project"]["name"]

        # Fall back to Poetry format
        if "tool" in data and "poetry" in data["tool"]:
            return data["tool"]["poetry"].get("name", None)

        return None
    except Exception:
        return None


def extract_external_dependencies(strategy_config: StrategyConfig, current_package: str | None) -> list[str]:
    """
    Extract external package dependencies from strategy config.

    Args:
        strategy_config: The strategy configuration
        current_package: Name of the current project's package (e.g., "my_strategy")

    Returns:
        List of external package names
    """
    strategies = strategy_config.strategy
    if isinstance(strategies, str):
        strategies = [strategies]
    elif not isinstance(strategies, list):
        strategies = []

    external_packages = set()
    for strat in strategies:
        if isinstance(strat, str):
            package = strat.split(".")[0]
            # Only include if it's NOT the current project's package
            # Normalize hyphens/underscores for comparison (PEP 503)
            if package.replace("-", "_") != (current_package or "").replace("-", "_"):
                external_packages.add(package)

    return sorted(external_packages)


def ls_strats(path: str) -> None:
    """
    List all strategies in the given directory.
    """
    classes = scan_py_classes_in_directory(path)
    for si in classes:
        if not si.is_strategy:
            continue
        strs = ""
        descr = (": " + green(si.docstring.replace("\n", " ").strip('" '))) if si.docstring else ""
        _p_str = ""
        if si.parameters:
            _max_l = max(map(len, si.parameters.keys())) + 2
            for k, c in si.parameters.items():
                _p_str += f"\t{red(':')}  {cyan(k.ljust(_max_l))}: {yellow(str(c))}\n"
            strs += f"\t{red('.--(')} {white(si.name)} {descr} \n{_p_str}\n"

        rst = f""" - {magenta(si.path)} -
{strs}"""
        print(rst)


def _find_class_by_name(classes: list[PyClassInfo], class_name: str) -> list[PyClassInfo]:
    """
    Filter classes by name. If there are multiple classes with the same name, the first one is returned.

    Args:
        classes: list[PyClassInfo] - classes to filter
        class_name: str - class name to filter

    Returns:
        list[PyClassInfo] - filtered classes, each tuple contains (path, class info)
    """
    _filt_strats = []

    for si in classes:
        _s_name = si.name
        if _s_name.lower() == class_name.lower():
            _filt_strats.append(si)

    return _filt_strats


def is_config_file(path: str) -> bool:
    """
    Check if the given path is a YAML config file.

    Args:
        path: Path to check

    Returns:
        True if the path is a YAML config file, False otherwise
    """
    config_path = Path(os.path.expanduser(path))
    return config_path.exists() and config_path.is_file() and (config_path.suffix.lower() in [".yml", ".yaml"])


def find_class_by_name(directory: str, class_name: str) -> PyClassInfo:
    """
    Find a class by name in the given directory.

    Args:
        directory: Directory to scan for classes
        class_name: Class name to find

    Returns:
        PyClassInfo for the found class

    Raises:
        ValueError: If no class is found or multiple classes are found
    """
    classes = scan_py_classes_in_directory(directory)
    matching_classes = _find_class_by_name(classes, class_name)

    if not matching_classes:
        raise ValueError(f"Class {class_name} not found in {directory}")

    if len(matching_classes) > 1:
        class_paths = [si.path for si in matching_classes]
        class_paths_str = "\n".join([f"\t - {path}" for path in class_paths])
        raise ValueError(f"Multiple classes found for {class_name}:\n{class_paths_str}")

    return matching_classes[0]


def generate_default_config(
    stg_info: PyClassInfo, exchange: str, connector: str, instruments: list[str]
) -> StrategyConfig:
    """
    Generate a default configuration for the strategy.

    Args:
        stg_info: Strategy information
        exchange: Default exchange
        connector: Default connector
        instruments: Default instruments

    Returns:
        Default StrategyConfig
    """
    # Create exchange config
    exchange_config = ExchangeConfig(connector=connector, universe=instruments)

    # Create logging config
    logging_config = LoggingConfig(logger="CsvFileLogsWriter", position_interval="10Sec", portfolio_interval="5Min")

    # Generate the import path for the strategy
    strategy_path = stg_info.path

    # Get the pyproject root directory
    pyproject_root = find_pyproject_root(strategy_path)
    package_name = os.path.basename(pyproject_root)

    # Get the relative path from the pyproject root to the strategy file
    rel_path = os.path.relpath(strategy_path, pyproject_root)

    # Convert file path to module path
    # Remove .py extension and replace path separators with dots
    module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")

    # Construct the full import path
    import_path = f"{module_path}.{stg_info.name}"

    # If the module path starts with the package name, we don't need to repeat it
    if import_path.startswith(f"{package_name}."):
        import_path = import_path
    else:
        import_path = f"{package_name}.{import_path}"

    # Create strategy config
    strategy_config = StrategyConfig(
        strategy=import_path,
        parameters=stg_info.parameters,
        exchanges={exchange: exchange_config},
        logging=logging_config,
    )

    return strategy_config


def load_strategy_from_config(config_path: Path, directory: str) -> StrategyInfo:
    """
    Load strategy information from a config file.

    Args:
        config_path: Path to the config file
        directory: Directory to scan for strategies

    Returns:
        StrategyInfo object
    """
    try:
        # Load the config without resolving env vars (not needed for release)
        strategy_config = load_strategy_config_from_yaml(os.path.expanduser(config_path), resolve_env=False)

        # Extract strategy name from config
        strategy_class_names = strategy_config.strategy
        if not strategy_class_names:
            raise ValueError("Strategy description not found in config file !")

        strategy_class_names = (
            strategy_class_names if isinstance(strategy_class_names, list) else [strategy_class_names]
        )

        # - get all classes in the specified directory
        classes = scan_py_classes_in_directory(directory)

        # - find all strategy components
        _found_classes = []
        _name_leader = ""
        for s in strategy_class_names:
            s_name = s.split(".")[-1]
            for c in classes:
                if c.name == s_name:
                    _found_classes.append(c)
                    if c.is_strategy and not _name_leader:
                        _name_leader = c.name + "_"

        strat_name = "".join([x.name for x in _found_classes])
        # - cut the name if it's too long and there are multiple components
        if len(strat_name) > 16 and len(_found_classes) > 1:
            strat_name = _name_leader + generate_name(strat_name, 8)

        return StrategyInfo(name=strat_name, classes=_found_classes, config=strategy_config)

    except Exception as e:
        logger.opt(colors=False).error(f"Error loading strategy from config file: {e}")
        raise


def release_strategy(
    directory: str,
    config_file: str,
    tag: str | None,
    message: str | None,
    commit: bool,
    output_dir: str,
) -> None:
    """
    Release strategy to zip file from given directory or config file

    Args:
        directory: str - directory to scan for strategies
        config_file: str - path to config file
        tag: str - additional tag for this release
        message: str - release message
        commit: bool - commit changes and create tag in repo
        output_dir: str - output directory to put zip file
    """
    from qubx import QubxLogConfig

    QubxLogConfig.set_log_level("INFO")

    try:
        # - determine if strategy_name is a config file or a strategy name
        if not is_config_file(config_file):
            raise ValueError("Try using yaml config file path")

        # - load strategy from config file
        logger.info(f"Loading strategy from config file: {config_file}")
        stg_info = load_strategy_from_config(Path(config_file), directory)

        # - process pyproject.toml and git repo (if available)
        repo_path = None
        if stg_info.classes:
            # Normal path: use class file locations
            repos_paths = set()
            pyproject_roots = set()
            for sc in stg_info.classes:
                try:
                    repos_paths.add(find_git_root(sc.path))
                except ValueError:
                    pass
                pyproject_roots.add(find_pyproject_root(sc.path))

            # - check if all strategy components are in the same repo and pyproject.toml
            if len(repos_paths) > 1:
                raise ValueError(
                    " >>> Multiple repositories found for the strategy - try to put all strategies components into one repository"
                )

            if len(pyproject_roots) > 1:
                raise ValueError(
                    " >>> Multiple pyproject.toml files found for the strategy - try to put all strategies components into one project"
                )

            repo_path = repos_paths.pop() if repos_paths else None
            pyproject_root = pyproject_roots.pop()
        else:
            # Empty classes: use config file location
            logger.info("No custom strategy classes found - using config file location for release")
            config_path = os.path.abspath(os.path.expanduser(config_file))
            try:
                repo_path = find_git_root(config_path)
            except ValueError:
                pass
            pyproject_root = find_pyproject_root(config_path)

        # - process git repo (or create minimal release info if no git)
        if repo_path is not None:
            _skip_tag, _skip_commit = not commit, not commit
            _git_info = process_git_repo(
                repo_path=repo_path,
                subdir=pyproject_root,
                strategy_name_id=stg_info.name,
                tag_sfx=tag,
                message=message,
                skip_tag=_skip_tag,
                skip_commit=_skip_commit,
            )
        else:
            logger.info("No git repository found — skipping git operations")
            _git_info = ReleaseInfo(
                tag=generate_tag(stg_info.name, tag),
                commit="none",
                user=getpass.getuser(),
                time=datetime.now(),
                commited_files=[],
            )

        # - create zipped pack
        create_released_pack(
            stg_info=stg_info,
            git_info=_git_info,
            pyproject_root=pyproject_root,
            output_dir=output_dir,
            config_file=config_file,
        )
    except ValueError as e:
        logger.opt(colors=False).error(str(e))
    except Exception as e:
        logger.opt(colors=False).error(f"Error releasing strategy: {e}")


def _find_source_root(pyproject_root: str, project_name: str) -> str | None:
    """Find the source package directory for a project.

    Searches for src-layout first (e.g. src/my_strategy), then flat layout (e.g. my_strategy/).

    Args:
        pyproject_root: Root directory containing pyproject.toml
        project_name: Project name (e.g. "my_strategy")

    Returns:
        Absolute path to the source package directory, or None if not found.
    """
    src_dir_name = project_name.replace("-", "_")

    # Try src layout first
    candidate = os.path.join(pyproject_root, "src", src_dir_name)
    if os.path.isdir(candidate):
        return candidate

    # Also walk to find nested src/ dirs
    for root, dirs, _files in os.walk(pyproject_root):
        if src_dir_name in dirs:
            c = os.path.join(root, src_dir_name)
            if os.path.sep + "src" + os.path.sep in c:
                return c

    # Try flat layout
    candidate = os.path.join(pyproject_root, src_dir_name)
    if os.path.isdir(candidate):
        return candidate

    return None


def _collect_all_imports(strategy_files: list[str], project_root: str) -> set[str]:
    """
    Collect all top-level import module names from strategy source files.

    Scans each file's AST for import/from-import statements and returns
    the set of top-level (first component) module names, excluding stdlib
    and the project's own package.

    Args:
        strategy_files: List of absolute paths to strategy .py files
        project_root: Project root for resolving relative imports

    Returns:
        Set of top-level import names (e.g. {"numpy", "cachetools", "qubx"})
    """
    top_level_modules: set[str] = set()

    for file_path in strategy_files:
        try:
            # Pass empty what_to_look to get ALL imports (no filter)
            for imp in get_imports(file_path, what_to_look=[], project_root=project_root):
                if imp.module:
                    top_level_modules.add(imp.module[0])
        except Exception as e:
            logger.warning(f"Failed to scan imports from {file_path}: {e}")

    return top_level_modules


def _parse_uv_lock(uv_lock_path: str) -> dict[str, str]:
    """
    Parse uv.lock (TOML format) and return {normalized_name: version} dict.

    Args:
        uv_lock_path: Path to uv.lock file

    Returns:
        Dict mapping package name (lowercase, normalized) to pinned version
    """
    import toml

    if not os.path.exists(uv_lock_path):
        logger.warning(f"uv.lock not found at {uv_lock_path}")
        return {}

    with open(uv_lock_path) as f:
        lock_data = toml.load(f)

    versions: dict[str, str] = {}
    for pkg in lock_data.get("package", []):
        name = pkg.get("name", "")
        version = pkg.get("version", "")
        if name and version:
            versions[name.lower().replace("-", "_")] = version
    return versions


def _scan_strategy_deps(
    strategy_files: list[str],
    pyproject_root: str,
    lock_versions: dict[str, str],
    pyproject_data: dict,
    external_imports: set[str] | None = None,
) -> list[str]:
    """
    Scan strategy source files for external imports, map to packages, pin versions from lock.

    For each dependency declared in pyproject.toml [project.dependencies]:
    1. Extract the package name from the dep spec
    2. Look up its top-level import names via importlib.metadata
    3. Check if any of those import names appear in the strategy's imports
    4. If yes, include that dep with the exact version from uv.lock

    Args:
        strategy_files: List of absolute paths to strategy .py files
        pyproject_root: Root directory containing pyproject.toml
        lock_versions: Pre-parsed {normalized_name: version} from uv.lock
        pyproject_data: Parsed pyproject.toml dict
        external_imports: Pre-computed set of external top-level import names.
            If provided, skips internal import collection.

    Returns:
        List of pinned dependency specs like ["cachetools==6.2.5", "Qubx==1.0.1.dev1"]
    """
    import importlib.metadata

    # Step 1: collect all imports from strategy files (or use pre-computed)
    if external_imports is not None:
        strategy_imports = external_imports
    else:
        strategy_imports = _collect_all_imports(strategy_files, pyproject_root)
    logger.info(f"Strategy imports (top-level modules): {sorted(strategy_imports)}")

    # Step 3: gather all declared deps (regular + optional)
    all_deps: list[str] = list(pyproject_data.get("project", {}).get("dependencies", []))
    for group_deps in pyproject_data.get("project", {}).get("optional-dependencies", {}).values():
        all_deps.extend(group_deps)

    # Step 4: for each declared dep, check if the strategy uses it
    scanned_deps: list[str] = []
    for dep_spec in all_deps:
        # Parse dep spec like "qubx[connectors,db,k8,tui]==1.0.3" or "cachetools>=6.2.1,<7"
        # Extract: package name, extras (if any), version specifiers
        match = re.match(r"^([A-Za-z0-9_.-]+)(\[[^\]]*\])?", dep_spec)
        pkg_name = (
            match.group(1).strip()
            if match
            else dep_spec.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        )
        extras = match.group(2) or "" if match else ""
        pkg_name_normalized = pkg_name.lower().replace("-", "_")

        # Determine import names for this package
        import_names: set[str] = set()
        try:
            dist = importlib.metadata.distribution(pkg_name)
            top_level_text = dist.read_text("top_level.txt")
            if top_level_text:
                for line in top_level_text.strip().splitlines():
                    import_names.add(line.strip())
            else:
                # Fallback: check RECORD for package directories
                if dist.files:
                    for f in dist.files:
                        parts = str(f).split("/")
                        if len(parts) > 1 and parts[0] and not parts[0].endswith(".dist-info"):
                            import_names.add(parts[0].replace("-", "_"))
                if not import_names:
                    import_names.add(pkg_name_normalized)
        except importlib.metadata.PackageNotFoundError:
            # Package not installed — use fallback name mapping
            import_names.add(pkg_name_normalized)

        # Check if any import name is used by strategy
        if import_names & strategy_imports:
            # Pin version from lock file, preserving extras
            version = lock_versions.get(pkg_name_normalized)
            if version:
                pinned = f"{pkg_name}{extras}=={version}"
            else:
                # Fallback to declared spec
                pinned = dep_spec
            scanned_deps.append(pinned)
            logger.debug(f"  Matched dep: {pinned} (imports: {import_names & strategy_imports})")

    logger.info(f"Scanned strategy deps ({len(scanned_deps)}): {scanned_deps}")
    return scanned_deps


def _build_strategy_wheel(
    pyproject_root: str,
    release_dir: str,
) -> str | None:
    """
    Build a compiled wheel from the project's source tree.

    Runs `uv build --wheel .` directly from the project root, using the
    project's own build system (hatchling, setuptools, etc.). The wheel
    includes the entire project package with all its dependencies declared
    in pyproject.toml.

    Args:
        pyproject_root: Root directory of the source project
        release_dir: Release output directory

    Returns:
        Wheel filename (e.g. "my_strategy-0.3.0-cp312-linux_x86_64.whl") or None on failure
    """
    import subprocess

    wheels_dir = os.path.join(release_dir, "wheels")
    os.makedirs(wheels_dir, exist_ok=True)

    logger.info("Building strategy wheel...")
    try:
        result = subprocess.run(
            ["uv", "build", "--wheel", ".", "--out-dir", wheels_dir],
            cwd=pyproject_root,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(f"uv build stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.opt(colors=False).error(f"Failed to build strategy wheel:\n{e.stderr}")
        return None

    # Find the built wheel
    for fname in os.listdir(wheels_dir):
        if fname.endswith(".whl"):
            logger.info(f"Built strategy wheel: {fname}")
            return fname

    logger.error("Wheel build completed but no wheel file found")
    return None


def _generate_release_pyproject(
    release_dir: str,
    strategy_wheel_name: str | None,
    has_strategy_code: bool,
    plugin_deps: list[str] | None = None,
    external_deps: list[str] | None = None,
    pyproject_data: dict | None = None,
) -> None:
    """
    Generate a minimal pyproject.toml for the release package.

    For strategy-code releases:
        dependencies = ["my_strategy==0.3.0"]  (the strategy wheel)

    For external-deps-only releases:
        dependencies = ["quantkit>=1.3.0", ...]  (the external packages)

    Always includes find-links=./wheels and package=false.

    Args:
        release_dir: Release output directory
        strategy_wheel_name: Filename of the strategy wheel (e.g. "my_strategy-0.3.0-cp312-linux.whl")
        has_strategy_code: Whether this release includes custom strategy code
        plugin_deps: Plugin dependency specs to include
        external_deps: External package dependency specs (for no-code configs)
        pyproject_data: Original pyproject data for preserving index config
    """
    import toml

    deps: list[str] = []

    if has_strategy_code and strategy_wheel_name:
        # Extract package name and version from wheel filename
        # Wheel format: {name}-{version}(-{build})?-{python}-{abi}-{platform}.whl
        parts = strategy_wheel_name.split("-")
        pkg_name = parts[0].replace("_", "-")
        pkg_version = parts[1]
        deps.append(f"{pkg_name}=={pkg_version}")

    if external_deps:
        deps.extend(external_deps)

    if plugin_deps:
        for pdep in plugin_deps:
            pdep_pkg = pdep.split(">=")[0].split("==")[0].split("<")[0].strip().lower()
            already = any(d.split(">=")[0].split("==")[0].split("<")[0].strip().lower() == pdep_pkg for d in deps)
            if not already:
                deps.append(pdep)

    uv_config: dict = {
        "package": False,
        "prerelease": "allow",
    }

    # Only add find-links if wheels/ directory exists and has files
    wheels_dir = os.path.join(release_dir, "wheels")
    if os.path.isdir(wheels_dir) and any(f.endswith(".whl") for f in os.listdir(wheels_dir)):
        uv_config["find-links"] = ["./wheels"]

    if pyproject_data:
        source_uv = pyproject_data.get("tool", {}).get("uv", {})
        source_indexes = source_uv.get("index", [])
        if source_indexes:
            uv_config["index"] = source_indexes

        # Propagate index-based sources for packages in our deps so uv lock can resolve them
        dep_names = {d.split(">=")[0].split("==")[0].split("<")[0].strip().lower() for d in deps}
        sources = source_uv.get("sources", {})
        index_sources = {k: v for k, v in sources.items() if "index" in v and k.lower() in dep_names}
        if index_sources:
            uv_config["sources"] = index_sources

    release_pyproject: dict = {
        "project": {
            "name": "strategy-release",
            "version": "0.1.0",
            "requires-python": ">=3.12",
            "dependencies": deps,
        },
        "tool": {
            "uv": uv_config,
        },
    }

    pyproject_path = os.path.join(release_dir, "pyproject.toml")
    with open(pyproject_path, "w") as f:
        toml.dump(release_pyproject, f)

    logger.info(f"Generated release pyproject.toml with deps: {deps}")


def create_released_pack(
    stg_info: StrategyInfo,
    git_info: ReleaseInfo,
    pyproject_root: str,
    output_dir: str,
    config_file: str,
):
    """
    Create a release package for a strategy.

    Flow:
    1. Build compiled strategy wheel from the full project (if custom code exists)
    2. Detect plugin deps
    3. Detect external strategy package deps
    4. Bundle private/local dependency wheels
    5. Generate minimal release pyproject.toml
    6. Generate uv.lock + create zip

    Args:
        stg_info: Strategy information
        git_info: Git release information
        pyproject_root: Path to the pyproject.toml root directory
        output_dir: Output directory for the release package
        config_file: Path to the original config file to copy
    """
    import toml

    logger.info(f"Creating release pack for {git_info.tag} ...")

    # Setup directory structure
    stg_name = stg_info.name
    release_dir = makedirs(output_dir, git_info.tag)
    has_strategy_code = bool(stg_info.classes)

    # Read source pyproject.toml once
    pyproject_src = os.path.join(pyproject_root, "pyproject.toml")
    with open(pyproject_src) as f:
        pyproject_data = toml.load(f)

    sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})
    for src_pkg, source in sources.items():
        if "index" in source:
            username, password = _resolve_index_credentials(source["index"])
            if not username or not password:
                idx_env = source["index"].upper().replace("-", "_")
                logger.warning(
                    f"No credentials found for private index '{source['index']}'. "
                    f"Set UV_INDEX_{idx_env}_USERNAME/PASSWORD or {idx_env}_KEY."
                )

    # Parse uv.lock once as the single source of truth for versions.
    # If missing, generate it first.
    uv_lock_path = os.path.join(pyproject_root, "uv.lock")
    if not os.path.exists(uv_lock_path):
        logger.info("uv.lock not found in source project, generating...")
        _generate_lock_file(pyproject_root)
    lock_versions = _parse_uv_lock(uv_lock_path)

    # --- Step 1: Build strategy wheel (if custom code) ---
    strategy_wheel_name: str | None = None
    if has_strategy_code:
        strategy_wheel_name = _build_strategy_wheel(pyproject_root, release_dir)
        if not strategy_wheel_name:
            raise RuntimeError("Failed to build strategy wheel")

    # --- Step 2: Detect plugin deps ---
    plugin_deps: list[str] = []
    if stg_info.config:
        plugin_deps = _get_plugin_deps(stg_info.config, pyproject_data, lock_versions)
        if plugin_deps:
            logger.info(f"Plugin dependencies from config: {plugin_deps}")

    # --- Step 3: Detect external strategy package deps ---
    # Always check for external strategy packages (e.g. quantkit.universe.basics.TopNUniverse)
    # These need to be included as dependencies whether or not there's also local strategy code.
    current_package = get_project_package_name(pyproject_root)
    ext_pkgs = extract_external_dependencies(stg_info.config, current_package)
    external_deps: list[str] | None = None
    if ext_pkgs:
        external_deps = []
        for pkg in ext_pkgs:
            pkg_norm = pkg.lower().replace("-", "_")
            ver = lock_versions.get(pkg_norm)
            if ver:
                external_deps.append(f"{pkg}=={ver}")
            else:
                logger.warning(f"  {pkg} not found in uv.lock, using unversioned spec")
                external_deps.append(pkg)
        logger.info(f"External strategy packages: {external_deps}")

    # --- Step 4: Bundle private/local dependency wheels ---
    # Build required_packages from ALL project dependencies (the wheel carries its own deps)
    all_project_deps = list(pyproject_data.get("project", {}).get("dependencies", []))
    required_packages: set[str] = set()
    for dep in all_project_deps:
        name = dep.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip().lower()
        required_packages.add(name)
    for pdep in plugin_deps:
        name = pdep.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip().lower()
        required_packages.add(name)

    bundled_packages: list[str] = []
    if required_packages:
        logger.info("Resolving private/local source dependencies...")
        bundled_packages = _bundle_source_overrides(
            pyproject_data,
            pyproject_root,
            release_dir,
            required_packages,
            lock_versions,
        )
        if bundled_packages:
            logger.info(f"Bundled {len(bundled_packages)} package(s): {', '.join(bundled_packages)}")

    # Log all wheels included in the release
    wheels_dir = os.path.join(release_dir, "wheels")
    if os.path.isdir(wheels_dir):
        wheel_files = [f for f in os.listdir(wheels_dir) if f.endswith(".whl")]
        if wheel_files:
            logger.info(f"Wheels included in release ({len(wheel_files)}):")
            for whl in sorted(wheel_files):
                logger.info(f"  {whl}")

    _generate_release_pyproject(
        release_dir=release_dir,
        strategy_wheel_name=strategy_wheel_name,
        has_strategy_code=has_strategy_code,
        plugin_deps=plugin_deps,
        external_deps=external_deps,
        pyproject_data=pyproject_data,
    )

    # --- Save config + metadata ---
    _save_strategy_config(config_file, release_dir)
    _create_metadata(stg_name, git_info, release_dir)

    # --- Step 6: Generate lock file + zip ---
    _generate_lock_file(release_dir)
    _strip_private_index_config(release_dir)
    _create_zip_archive(output_dir, release_dir, git_info.tag)

    logger.info(f"Created release pack: {os.path.join(output_dir, git_info.tag)}.zip")


def _save_strategy_config(config_file: str, release_dir: str) -> None:
    """Copy the original strategy configuration file to preserve comments and structure."""
    config_path = os.path.join(release_dir, "config.yml")
    shutil.copy2(config_file, config_path)
    logger.debug(f"Copied strategy config to {config_path}")


def _create_metadata(stg_name: str, git_info: ReleaseInfo, release_dir: str) -> None:
    """Create metadata files for the release."""
    # Create meta info file
    with open(os.path.join(release_dir, f"{stg_name}.info"), "wt") as fs:
        yaml.safe_dump(
            {
                "tag": git_info.tag,
                "date": git_info.time.isoformat(),
                "author": git_info.user,
            },
            fs,
            sort_keys=False,
        )

    # Create a README.md file
    with open(os.path.join(release_dir, "README.md"), "wt") as fs:
        fs.write(f"# {stg_name}\n\n")
        fs.write("## Git Info\n\n")
        fs.write(f"Tag: {git_info.tag}\n")
        fs.write(f"Date: {git_info.time.isoformat()}\n")
        fs.write(f"Author: {git_info.user}\n")
        fs.write(f"Commit: {git_info.commit}\n")


def _version_exists_on_pypi(pkg_name: str, version: str) -> bool:
    """Check if a specific package version is available on public PyPI."""
    try:
        url = f"https://pypi.org/pypi/{pkg_name}/{version}/json"
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def _get_index_url(index_name: str, pyproject_data: dict) -> str | None:
    """Get URL for a named uv index from [[tool.uv.index]] entries."""
    indexes = pyproject_data.get("tool", {}).get("uv", {}).get("index", [])
    for idx in indexes:
        if idx.get("name") == index_name:
            return idx.get("url")
    return None


def _resolve_index_credentials(index_name: str) -> tuple[str, str]:
    """Resolve credentials for a private index from environment variables.

    Checks UV_INDEX_{NAME}_USERNAME/PASSWORD first, falls back to {NAME}_KEY
    with _json_key_base64 username. Returns (username, password) tuple,
    either or both may be empty if no credentials are found.
    """
    env_name = index_name.upper().replace("-", "_")
    username = os.environ.get(f"UV_INDEX_{env_name}_USERNAME", "")
    password = os.environ.get(f"UV_INDEX_{env_name}_PASSWORD", "")
    if not username or not password:
        key = os.environ.get(f"{env_name}_KEY", "")
        if key:
            return "_json_key_base64", key
    return username, password


def _download_wheel_from_index(
    pkg_name: str,
    pkg_ver: str,
    index_name: str,
    index_url: str,
    wheels_dir: str,
) -> None:
    """Download a wheel from a private Simple API index using HTTP Basic auth.

    Fetches the package's Simple API page, finds the matching wheel link,
    and downloads it to wheels_dir.
    """

    username, password = _resolve_index_credentials(index_name)
    auth_header = None
    if username and password:
        credentials = b64encode(f"{username}:{password}".encode()).decode()
        auth_header = f"Basic {credentials}"

    def _make_request(url: str) -> urllib.request.Request:
        req = urllib.request.Request(url)
        if auth_header:
            req.add_header("Authorization", auth_header)
        return req

    def _wheel_filename(link: str) -> str:
        return link.rsplit("/", 1)[-1].split("#")[0]

    pkg_normalized = re.sub(r"[-_.]+", "-", pkg_name).lower()
    simple_url = f"{index_url.rstrip('/')}/{pkg_normalized}/"

    with urllib.request.urlopen(_make_request(simple_url), timeout=30) as resp:
        page = resp.read().decode()

    wheel_links = re.findall(r'href="([^"]*\.whl[^"]*)"', page, re.IGNORECASE)
    if not wheel_links:
        raise RuntimeError(f"No wheels found at {simple_url}")

    norm_name = re.sub(r"[-_.]+", "[-_.]+", pkg_name.lower())
    norm_ver = re.sub(r"[-_.]+", "[-_.]+", pkg_ver)
    wheel_re = re.compile(rf"{norm_name}-{norm_ver}-.*\.whl", re.IGNORECASE)

    matching = [link for link in wheel_links if wheel_re.search(_wheel_filename(link))]
    if not matching:
        raise RuntimeError(f"No wheel found for {pkg_name}=={pkg_ver} at {simple_url}")

    # Prefer pure Python wheel, then current platform
    machine = platform.machine().lower()
    chosen = None
    for link in matching:
        fname = _wheel_filename(link)
        if "none-any" in fname:
            chosen = link
            break
        if machine in fname or "manylinux" in fname:
            chosen = link
    if not chosen:
        chosen = matching[0]

    wheel_url = urljoin(simple_url, chosen).split("#")[0]
    dest_path = os.path.join(wheels_dir, _wheel_filename(chosen))
    with urllib.request.urlopen(_make_request(wheel_url), timeout=120) as whl_resp:
        with open(dest_path, "wb") as f:
            f.write(whl_resp.read())


def _resolve_local_package_version(local_path: str, pkg_norm: str) -> str | None:
    """Resolve version from a local package's _version.py or pyproject.toml."""
    import re as _re

    import toml

    # Try _version.py (hatch-vcs style)
    src_dir = os.path.join(local_path, "src", pkg_norm)
    if not os.path.isdir(src_dir):
        src_dir = os.path.join(local_path, pkg_norm)
    version_file = os.path.join(src_dir, "_version.py")
    if os.path.exists(version_file):
        with open(version_file) as vf:
            m = _re.search(r"__version__\s*=.*?['\"]([^'\"]+)['\"]", vf.read())
            if m:
                return m.group(1)

    # Try pyproject.toml static version
    pyproject_path = os.path.join(local_path, "pyproject.toml")
    if os.path.exists(pyproject_path):
        with open(pyproject_path) as f:
            data = toml.load(f)
        ver = data.get("project", {}).get("version")
        if ver:
            return ver

    return None


def _bundle_source_overrides(
    pyproject_data: dict,
    pyproject_root: str,
    release_dir: str,
    required_packages: set[str],
    lock_versions: dict[str, str],
) -> list[str]:
    """
    For each [tool.uv.sources] entry that is required by this release:
    - path source + version on public PyPI → skip (will resolve from PyPI)
    - path source + NOT on public PyPI → build wheel from local path and bundle in wheels/
    - index source (private registry) → download wheel from that index and bundle in wheels/

    Uses uv.lock as the single source of truth for package versions.

    Returns list of bundled package names (lowercase).
    """
    import subprocess

    sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})
    if not sources:
        return []

    wheels_dir = os.path.join(release_dir, "wheels")
    bundled: list[str] = []

    for pkg_name, source in sources.items():
        if pkg_name.lower() not in required_packages:
            continue

        pkg_norm = pkg_name.lower().replace("-", "_")
        pkg_ver = lock_versions.get(pkg_norm)

        if "path" in source:
            local_path = os.path.normpath(os.path.join(pyproject_root, source["path"]))

            # For editable/path sources, uv.lock may not store a version.
            # Resolve from the local package's _version.py or pyproject.toml.
            if not pkg_ver:
                pkg_ver = _resolve_local_package_version(local_path, pkg_norm)

            if not pkg_ver:
                logger.warning(f"  {pkg_name} version not found in uv.lock or local package, skipping bundle")
                continue

            if _version_exists_on_pypi(pkg_name, pkg_ver):
                logger.info(f"  {pkg_name}=={pkg_ver} found on public PyPI, will resolve from registry")
                continue

            logger.info(f"  Bundling {pkg_name}=={pkg_ver} from local path {local_path} ...")
            os.makedirs(wheels_dir, exist_ok=True)
            try:
                subprocess.run(
                    ["uv", "build", "--wheel", ".", "--out-dir", wheels_dir],
                    cwd=local_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for whl in os.listdir(wheels_dir):
                    if whl.lower().startswith(pkg_norm) and "none-any" not in whl:
                        logger.warning(
                            f"  {whl} is platform-specific. "
                            "Ensure the container architecture matches the build machine."
                        )
                bundled.append(pkg_name.lower())
                logger.info(f"  Bundled {pkg_name}")
            except subprocess.CalledProcessError as e:
                logger.opt(colors=False).warning(f"  Failed to build wheel for {pkg_name}: {e.stderr or e}")

        elif "index" in source:
            if not pkg_ver:
                logger.warning(f"  {pkg_name} not found in uv.lock, skipping bundle")
                continue

            index_name = source["index"]
            index_url = _get_index_url(index_name, pyproject_data)
            if not index_url:
                logger.warning(f"  Index '{index_name}' not found in [[tool.uv.index]] (needed for {pkg_name})")
                continue

            if _version_exists_on_pypi(pkg_name, pkg_ver):
                logger.info(f"  {pkg_name}=={pkg_ver} found on public PyPI, will resolve from registry")
                continue

            logger.info(f"  Downloading {pkg_name}=={pkg_ver} from private index '{index_name}' ...")
            os.makedirs(wheels_dir, exist_ok=True)
            try:
                _download_wheel_from_index(pkg_name, pkg_ver, index_name, index_url, wheels_dir)
                bundled.append(pkg_name.lower())
                logger.info(f"  Downloaded {pkg_name}=={pkg_ver}")
            except Exception as e:
                logger.opt(colors=False).warning(f"  Failed to download wheel for {pkg_name}: {e}")

    return bundled


def _get_plugin_deps(
    stg_config: "StrategyConfig",
    pyproject_data: dict,
    lock_versions: dict[str, str],
) -> list[str]:
    """
    Extract package dependency specs for plugin modules listed in the strategy config.

    Maps plugin module names (e.g. qubx_lighter) to package specs
    (e.g. qubx-lighter==0.1.0) by looking in [project.optional-dependencies].
    Falls back to uv.lock version if not found there.

    Returns list of dep specs like ["qubx-lighter==0.1.0"].
    """
    if not stg_config.plugins or not stg_config.plugins.modules:
        return []

    optional_deps: dict[str, list[str]] = pyproject_data.get("project", {}).get("optional-dependencies", {})

    plugin_deps: list[str] = []
    for module_name in stg_config.plugins.modules:
        pkg_name = module_name.replace("_", "-")

        # Search in optional-dependency groups first
        found = False
        for group_name, group_deps in optional_deps.items():
            for dep in group_deps:
                dep_pkg = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
                if dep_pkg.lower() == pkg_name.lower():
                    plugin_deps.append(dep)
                    logger.info(f"  Plugin '{module_name}' -> adding '{dep}' (from optional-deps [{group_name}])")
                    found = True
                    break
            if found:
                break

        if not found:
            pkg_norm = pkg_name.lower().replace("-", "_")
            ver = lock_versions.get(pkg_norm)
            if ver:
                spec = f"{pkg_name}=={ver}"
                plugin_deps.append(spec)
                logger.warning(f"  Plugin '{module_name}' not in optional-deps; adding '{spec}' from uv.lock")
            else:
                logger.warning(
                    f"  Plugin '{module_name}' ({pkg_name}) not found in optional-deps or uv.lock - skipping"
                )

    return plugin_deps


def _generate_lock_file(release_dir: str) -> None:
    """
    Generate a uv.lock file.

    Args:
        release_dir: Directory where the uv.lock file should be generated
    """
    import subprocess

    try:
        logger.info("Generating uv.lock file...")

        # Check if we're already in an active virtual environment
        in_venv = "VIRTUAL_ENV" in os.environ

        lock_cmd = ["uv", "lock"]
        env = os.environ.copy()
        if in_venv:
            # Force uv to use a clean environment even if we're in an active one
            logger.debug("Detected active virtual environment. Using a clean environment for lock generation.")
            for var in ["VIRTUAL_ENV"]:
                if var in env:
                    del env[var]

        result = subprocess.run(lock_cmd, cwd=release_dir, check=False, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            logger.opt(colors=False).error(f"uv lock stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, lock_cmd)

    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to generate uv.lock: {e}")
        raise e


def _strip_private_index_config(release_dir: str) -> None:
    """Remove private index and sources config from release pyproject.toml.

    After uv lock has resolved dependencies, the lock file pins everything.
    Stripping index/sources ensures deploy doesn't require private registry auth —
    bundled wheels in find-links are used instead.
    """
    import toml

    pyproject_path = os.path.join(release_dir, "pyproject.toml")
    with open(pyproject_path) as f:
        data = toml.load(f)

    uv_config = data.get("tool", {}).get("uv", {})
    uv_config.pop("index", None)
    uv_config.pop("sources", None)

    with open(pyproject_path, "w") as f:
        toml.dump(data, f)


def _create_zip_archive(output_dir: str, release_dir: str, tag: str) -> None:
    """Create a zip archive of the release package."""
    logger.debug("Creating zip file...")
    file_path = os.path.join(output_dir, tag)
    shutil.make_archive(file_path, "zip", release_dir)
    shutil.rmtree(release_dir)


def generate_tag(strategy_name_id: str, tag_sfx: str | None) -> str:
    _tn = datetime.now()
    tag_s = f".{tag_sfx}" if tag_sfx else ""
    _strategy_name_id = strategy_name_id.replace(",", "_")
    tag = f"R_{_strategy_name_id}_{_tn.strftime('%Y%m%d%H%M%S')}{tag_s}"
    return tag


def make_tag_in_repo(repo: Repo, strategy_name_id: str, user: str, tag: str) -> str:
    """
    Create an annotated tag in the repository and push it to the remote.

    Args:
        repo: Git repository object
        strategy_name_id: Strategy name identifier
        user: Username
        tag: Tag name

    Returns:
        str: The tag name
    """
    repo.config_writer().set_value("push", "followTags", "true").release()

    _tn = datetime.now()
    _ = repo.create_tag(
        tag,
        message=f"Release of '{strategy_name_id}' at {_tn.strftime('%Y-%b-%d %H:%M:%S')} by {user}",
    )
    repo.remote("origin").push(f"refs/tags/{tag}")
    return tag


def process_git_repo(
    repo_path: str,
    subdir: str,
    strategy_name_id: str,
    tag_sfx: str | None = None,
    message: str | None = None,
    skip_tag: bool = False,
    skip_commit: bool = False,
) -> ReleaseInfo:
    """
    Process the git repository to get the release information and commit the changes.

    Args:
        repo_path: str - path to the git repository
        subdir: str - subdirectory to scan for changes
        strategy_name_id: str - strategy name id
        tag_sfx: str | None - tag suffix
        message: str | None - message
        skip_tag: bool - skip tag
        skip_commit: bool - skip commit

    Returns:
        ReleaseInfo - release information
    """
    repo = Repo(repo_path)
    _to_add: list[str] = []

    # Convert subdir to absolute path for comparison
    abs_subdir = os.path.abspath(subdir)

    if diffs := repo.index.diff(None):
        logger.info(f"- Found {len(diffs)} modified files in the repo:")
        for d in diffs:
            # Convert the file path to absolute path for comparison
            file_path = str(d.a_path) if d.a_path is not None else ""
            abs_file_path = os.path.abspath(os.path.join(repo_path, file_path))
            # Check if the file is under subdir
            if abs_file_path.startswith(abs_subdir):
                logger.info(f"\t<r>{file_path}</r>")
                _to_add.append(file_path)

    if untr := repo.untracked_files:
        logger.info(f"- Found {len(untr)} untracked files in the repo:")
        for d in untr:
            # Convert the file path to absolute path for comparison
            file_path = str(d)
            abs_file_path = os.path.abspath(os.path.join(repo_path, file_path))
            # Check if the file is under subdir
            if abs_file_path.startswith(abs_subdir):
                logger.info(f"\t<r>{file_path}</r>")
                _to_add.append(file_path)

    user = getpass.getuser()
    tag = None
    commit_sha = None
    _tn = datetime.now()
    if _to_add:
        if not skip_commit:
            # - add changed files
            logger.info(f"Commiting changes for {len(_to_add)} files ... ")
            try:
                repo.index.add(_to_add)
                cmt = repo.index.commit(
                    f"Changes before release of '{strategy_name_id}' at {_tn.strftime('%Y-%b-%d %H:%M:%S')} by {user}."
                    f"{'' + message if message else ''}"
                )
                commit_sha = cmt.hexsha
                _ilist = repo.remotes[0].push()
                for i in _ilist:
                    logger.debug(f"\t{yellow(i.summary)}")
            except Exception as e:
                logger.error(f"Error committing changes: {e}")
                raise e

    if commit_sha is None:
        # get latest commit sha
        commit_sha = repo.head.commit.hexsha
        if _to_add and skip_commit:
            logger.info("<y>Commiting changes is skipped due to --skip-commit option</y>")

    # - add annotated tag
    tag = generate_tag(strategy_name_id, tag_sfx)
    if not skip_tag:
        tag = make_tag_in_repo(repo, strategy_name_id, user, tag)
    else:
        logger.info("<y>Creating git tag is skipped due to --skip-tag option</y>")

    return ReleaseInfo(tag=tag, commit=commit_sha, user=user, time=_tn, commited_files=_to_add)
