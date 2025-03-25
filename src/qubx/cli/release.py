import ast
import getpass
import os
import shutil
import sys
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator

import yaml
from git import Repo

from qubx import logger
from qubx.utils.misc import (
    cyan,
    generate_name,
    green,
    load_qubx_resources_as_text,
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


def get_imports(path: str, what_to_look: list[str] = ["xincubator"]) -> Generator[Import, None, None]:
    """
    Get imports from the given file.
    """
    with open(path) as fh:
        root = ast.parse(fh.read(), path)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module = []
        elif isinstance(node, ast.ImportFrom):
            module = node.module.split(".") if node.module else []
        else:
            continue
        for n in node.names:
            if module and what_to_look and module[0] in what_to_look:
                yield Import(module, n.name.split("."), n.asname)


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
        # Load the config using the utility function
        strategy_config = load_strategy_config_from_yaml(os.path.expanduser(config_path))

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
        logger.error(f"Error loading strategy from config file: {e}")
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

        # - process git repo and pyproject.toml for each strategy component
        repos_paths = set()
        pyproject_roots = set()
        for sc in stg_info.classes:
            repos_paths.add(find_git_root(sc.path))
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

        repo_path = repos_paths.pop()
        pyproject_root = pyproject_roots.pop()

        # - process git repo
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

        # - create zipped pack
        create_released_pack(
            stg_info=stg_info,
            git_info=_git_info,
            pyproject_root=pyproject_root,
            output_dir=output_dir,
        )
    except ValueError as e:
        logger.error(f"<r>{str(e)}</r>")
    except Exception as e:
        logger.error(f"<r>Error releasing strategy: {e}</r>")


def create_released_pack(
    stg_info: StrategyInfo,
    git_info: ReleaseInfo,
    pyproject_root: str,
    output_dir: str,
):
    """
    Create a release package for a strategy.

    Args:
        stg_info: Strategy information
        git_info: Git release information
        pyproject_root: Path to the pyproject.toml root directory
        output_dir: Output directory for the release package
        strategy_config: Strategy configuration
    """
    logger.info(f"Creating release pack for {git_info.tag} ...")

    # Setup directory structure
    stg_name = stg_info.name
    release_dir = makedirs(output_dir, git_info.tag)

    # Copy strategy files and dependencies
    for sc in stg_info.classes:
        _copy_strategy_file(sc.path, pyproject_root, release_dir)
        _copy_dependencies(sc.path, pyproject_root, release_dir)

    # Save configuration
    _save_strategy_config(stg_name, stg_info.config, release_dir)

    # Create metadata
    _create_metadata(stg_name, git_info, release_dir)

    # Handle project files
    _handle_project_files(pyproject_root, release_dir)

    # Create zip archive
    _create_zip_archive(output_dir, release_dir, git_info.tag)

    logger.info(f"Created release pack: {os.path.join(output_dir, git_info.tag)}.zip")


def _save_strategy_config(stg_name: str, strategy_config: StrategyConfig, release_dir: str) -> None:
    """Save the strategy configuration to the release directory."""
    config_path = os.path.join(release_dir, "config.yml")
    with open(config_path, "wt") as fs:
        # Convert to dict and save directly (not under 'config' key)
        config_dict = strategy_config.model_dump()
        yaml.safe_dump(config_dict, fs, sort_keys=False, indent=2)
    logger.debug(f"Saved strategy config to {config_path}")


def _copy_strategy_file(strategy_path: str, pyproject_root: str, release_dir: str) -> None:
    """Copy the strategy file to the release directory."""
    rel_path = os.path.relpath(strategy_path, pyproject_root)
    dest_file_path = os.path.join(release_dir, rel_path)

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

    # Copy strategy file
    logger.debug(f"Copying strategy file from {strategy_path} to {dest_file_path}")
    shutil.copy2(strategy_path, dest_file_path)


def _try_copy_file(src_file: str, dest_dir: str, pyproject_root: str) -> None:
    """Try to copy the file to the release directory."""
    if os.path.exists(src_file):
        # Get the relative path from pyproject_root
        _rel_import_path = os.path.relpath(src_file, pyproject_root)
        _dest_import_path = os.path.join(dest_dir, _rel_import_path)

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(_dest_import_path), exist_ok=True)

        # Copy the import file
        logger.debug(f"Copying import from {src_file} to {_dest_import_path}")
        shutil.copy2(src_file, _dest_import_path)


def _copy_dependencies(strategy_path: str, pyproject_root: str, release_dir: str) -> None:
    """Copy all dependencies required by the strategy."""
    _src_dir = os.path.basename(pyproject_root)
    _imports = _get_imports(strategy_path, pyproject_root, [_src_dir])
    # find inside of the pyproject_root a folder with the same name as the _src_dir
    # for instance it could be like macd_crossover/src/macd_crossover
    # or macd_crossover/macd_crossover
    # and assign this folder to _src_root
    _src_root = None
    for root, dirs, files in os.walk(pyproject_root):
        if _src_dir in dirs:
            _src_root = os.path.join(root, _src_dir)
            break

    if _src_root is None:
        raise ValueError(f"Could not find the source root for {_src_dir} in {pyproject_root}")

    for _imp in _imports:
        # Construct source path
        _base = os.path.join(_src_root, *[s for s in _imp.module if s != _src_dir])

        # - try to copy all available files for satisfying the import
        if os.path.isdir(_base):
            _try_copy_file(os.path.join(_base, "__init__.py"), release_dir, pyproject_root)
        else:
            _try_copy_file(_base + ".py", release_dir, pyproject_root)
            _try_copy_file(_base + ".pyx", release_dir, pyproject_root)
            _try_copy_file(_base + ".pyi", release_dir, pyproject_root)
            _try_copy_file(_base + ".pxd", release_dir, pyproject_root)


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


def _modify_pyproject_toml(pyproject_path: str, package_name: str) -> None:
    """
    Modify the pyproject.toml file to include the project package as a dependency.

    Args:
        pyproject_path: Path to the pyproject.toml file
        package_name: Name of the package to add as a dependency
    """
    try:
        from importlib.metadata import version

        import toml

        # Read the existing pyproject.toml
        with open(pyproject_path, "r") as f:
            pyproject_data = toml.load(f)

        # Add the package as a local dependency
        if "tool" in pyproject_data and "poetry" in pyproject_data["tool"]:
            # Ensure dependencies section exists
            if "dependencies" not in pyproject_data["tool"]["poetry"]:
                pyproject_data["tool"]["poetry"]["dependencies"] = {}

            # Add Python as a dependency if not already present
            if "python" not in pyproject_data["tool"]["poetry"]["dependencies"]:
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                pyproject_data["tool"]["poetry"]["dependencies"]["python"] = f"^{python_version}"

            # Special case when we have dev dependencies for Qubx or QuantKit
            deps = pyproject_data["tool"]["poetry"]["dependencies"]
            for d in deps:
                if d.lower().startswith("qubx") or d.lower().startswith("quantkit"):
                    if "develop" in deps[d] and deps[d]["develop"]:
                        deps[d] = f">={version(d)}"

            # Replace the packages section with the new one
            # pyproject_data["tool"]["poetry"]["packages"] = [{"include": package_name}]

            # Check if build section exists
            if "build" not in pyproject_data["tool"]["poetry"]:
                pyproject_data["tool"]["poetry"]["build"] = {
                    "script": "build.py",
                    "generate-setup-file": False,
                }
                # Add build-system section
                pyproject_data["build-system"] = {
                    "requires": [
                        "poetry-core",
                        "setuptools",
                        "numpy>=1.26.3",
                        "cython==3.0.8",
                        "toml>=0.10.2",
                        "qubx>=0.6.0",
                    ],
                    "build-backend": "poetry.core.masonry.api",
                }

            # Write the updated pyproject.toml
            with open(pyproject_path, "w") as f:
                toml.dump(pyproject_data, f)

            logger.debug(f"Updated pyproject.toml to include {package_name} as a local dependency")
    except Exception as e:
        logger.warning(f"Failed to update pyproject.toml: {e}")


def _generate_poetry_lock(release_dir: str) -> None:
    """
    Generate a poetry.lock file without creating a virtual environment.

    Args:
        release_dir: Directory where the poetry.lock file should be generated
    """
    import subprocess

    try:
        # Configure Poetry settings for lock generation
        logger.debug("Configuring Poetry settings for lock generation without venv creation")

        # Set virtualenvs.create=false to prevent venv creation during lock generation
        subprocess.run(
            ["poetry", "config", "virtualenvs.create", "false", "--local"],
            cwd=release_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        # Set virtualenvs.in-project=true for when the venv is eventually created during deployment
        subprocess.run(
            ["poetry", "config", "virtualenvs.in-project", "true", "--local"],
            cwd=release_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        # Check if we're already in a Poetry shell
        in_poetry_env = "POETRY_ACTIVE" in os.environ or "VIRTUAL_ENV" in os.environ

        logger.info("Generating poetry.lock file without creating virtual environment...")

        # If we're in a Poetry shell, we need to be more explicit about avoiding environment creation
        # Add --no-interaction to prevent any prompts
        lock_cmd = ["poetry", "lock", "--no-interaction"]
        if in_poetry_env:
            # Force Poetry to use a clean environment even if we're in an active one
            logger.info("Detected active Poetry environment. Using a clean environment for lock generation.")
            env = os.environ.copy()
            # Temporarily unset Poetry environment variables to avoid interference
            for var in ["POETRY_ACTIVE", "VIRTUAL_ENV"]:
                if var in env:
                    del env[var]

            subprocess.run(lock_cmd, cwd=release_dir, check=True, capture_output=False, text=True, env=env)
        else:
            # Normal case - not in a Poetry shell
            subprocess.run(lock_cmd, cwd=release_dir, check=True, capture_output=True, text=True)

        # After lock generation, reset the virtualenvs.create setting to true for deployment
        # This ensures that when the package is deployed, the venv can be created
        subprocess.run(
            ["poetry", "config", "virtualenvs.create", "true", "--local"],
            cwd=release_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        logger.warning(f"Failed to generate poetry.lock: {e}")
        raise e


def _handle_project_files(pyproject_root: str, release_dir: str) -> None:
    """Handle project files like pyproject.toml and generate lock file."""
    # Copy pyproject.toml if it exists
    pyproject_src = os.path.join(pyproject_root, "pyproject.toml")
    if not os.path.exists(pyproject_src):
        raise FileNotFoundError(f"pyproject.toml not found in {pyproject_root}")

    pyproject_dest = os.path.join(release_dir, "pyproject.toml")
    logger.debug(f"Copying pyproject.toml from {pyproject_src} to {pyproject_dest}")
    shutil.copy2(pyproject_src, pyproject_dest)

    # Copy build.py if it exists
    build_src = os.path.join(pyproject_root, "build.py")
    if not os.path.exists(build_src):
        logger.info(f"build.py not found in {pyproject_root} using default one")
        build_src = load_qubx_resources_as_text("_build.py")

        # - setup project's name in default build.py
        prj_name = os.path.basename(pyproject_root)
        build_src = build_src.replace("<<PROJECT_NAME>>", prj_name)

        with open(os.path.join(release_dir, "build.py"), "wt") as fs:
            fs.write(build_src)
    else:
        build_dest = os.path.join(release_dir, "build.py")
        logger.debug(f"Copying build.py from {build_src} to {build_dest}")
        shutil.copy2(build_src, build_dest)

    # Get the basename of the pyproject_root as the package name
    package_name = os.path.basename(pyproject_root)

    # Modify the pyproject.toml to include the project package
    _modify_pyproject_toml(pyproject_dest, package_name)

    # Generate the poetry.lock file
    _generate_poetry_lock(release_dir)


def _create_zip_archive(output_dir: str, release_dir: str, tag: str) -> None:
    """Create a zip archive of the release package."""
    logger.debug("Creating zip file...")
    file_path = os.path.join(output_dir, tag)
    shutil.make_archive(file_path, "zip", release_dir)
    shutil.rmtree(release_dir)


def _get_imports(file_name: str, current_directory: str, what_to_look: list[str]) -> list[Import]:
    imports = list(get_imports(file_name, what_to_look))
    current_dirname = os.path.basename(current_directory)
    for i in imports:
        try:
            base = os.path.join(*[current_directory, *[s for s in i.module if s != current_dirname]])

            # - first try to find a .py file
            if not os.path.exists(f1_py := base + ".py"):
                f1_py = os.path.join(base, "__init__") + ".py"

            imports.extend(_get_imports(f1_py, current_directory, what_to_look))
        except Exception:
            pass
    return imports


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
