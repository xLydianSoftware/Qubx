import ast
import getpass
import importlib
import os
import shutil
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import Generator

import yaml
from git import Remote, Repo

from qubx import logger
from qubx.utils.misc import Struct, cyan, green, magenta, makedirs, red, white, yellow

from .misc import (
    StrategyInfo,
    ask_y_n,
    copy_file_to_dir,
    copy_to_dir,
    find_git_root,
    find_pyproject_root,
    generate_dependency_file,
    scan_strategies_in_directory,
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


def get_imports(path, what_to_look=["xincubator"]) -> Generator[Import, None, None]:
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
    strats = scan_strategies_in_directory(path)
    for si in strats:
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


def _find_strategies(strats: list[StrategyInfo], strategy_name: str) -> list[StrategyInfo]:
    """
    Filter strategies by name. If there are multiple strategies with the same name, the first one is returned.

    Args:
        strats: dict[str, list[tuple[str, str, dict]]] - strategies to filter
        strategy_name: str - strategy name to filter

    Returns:
        list[tuple[str, StrategyInfo]] - filtered strategies, each tuple contains (path, strategy info)
    """
    _filt_strats = []

    for si in strats:
        _s_name = si.name
        if _s_name.lower() == strategy_name.lower():
            _filt_strats.append(si)

    return _filt_strats


def release_strategy(
    directory: str,
    strategy_name: str,
    tag: str | None,
    message: str | None,
    commit: bool,
    output_dir: str,
    skip_confirmation: bool,
) -> None:
    """
    Release strategy to zip file from given directory

    Args:
        directory: str - directory to scan for strategies
        strategy_name: str - strategy name to release
        tag: str - additional tag for this release
        message: str - release message
        commit: bool - commit changes and create tag in repo
        output_dir: str - output directory to put zip file
        skip_confirmation: bool - skip confirmation
    """
    strats = scan_strategies_in_directory(directory)
    _to_release = _find_strategies(strats, strategy_name)
    _strat_name_ids = [si.name for si in _to_release]
    _strategy_name_id = ",".join(_strat_name_ids)

    # - check if we have any strategies to release
    if len(_to_release) > 1:
        _stg_paths = [si.path for si in _to_release]
        _stg_path_str = "\n".join([f"\t - {stg_path}" for stg_path in _stg_paths])
        logger.warning(f"<r>Multiple strategies found for {strategy_name}:\n{_stg_path_str}</r>")
        return
    elif len(_to_release) == 0:
        logger.info(f"<r>Strategy {strategy_name} not found in {directory}</r>")
        return

    # - get repo root path
    stg_path, stg_info = _to_release[0].path, _to_release[0]
    repo_path = find_git_root(stg_path)
    pyproject_root = find_pyproject_root(stg_path)
    src_dir = os.path.split(pyproject_root)[1]

    # - processing repo
    _skip_tag, _skip_commit = not commit, not commit

    _git_info = process_git_repo(
        repo_path=repo_path,
        subdir=pyproject_root,
        strategy_name_id=_strategy_name_id,
        tag_sfx=tag,
        message=message,
        skip_tag=_skip_tag,
        skip_commit=_skip_commit,
    )

    # - create zipped pack
    create_released_pack(
        src_dir=src_dir,
        stg_info=stg_info,
        git_info=_git_info,
        pyproject_root=pyproject_root,
        output_dir=output_dir,
        skip_confirmation=skip_confirmation,
    )


def generate_py_deps(req_output_file: str):
    with open(req_output_file, "wt") as f:
        for p in sorted(list(importlib.metadata.distributions()), key=lambda x: x.metadata["Name"]):  # type: ignore
            print(f"{p.metadata['Name']}=={p.metadata['Version']}", file=f)


def create_released_pack(
    src_dir: str,
    stg_info: StrategyInfo,
    git_info: ReleaseInfo,
    pyproject_root: str,
    output_dir: str,
    skip_confirmation: bool,
):
    logger.info(f"Creating release pack for {git_info.tag} ...")

    stg_name = stg_info.name
    release_dir = makedirs(output_dir, git_info.tag)
    src_dest_dir = makedirs(output_dir, git_info.tag, src_dir)

    # Get the relative path of the strategy file within the src_dir
    rel_path = os.path.relpath(stg_info.path, pyproject_root)
    dest_file_path = os.path.join(release_dir, src_dir, rel_path)

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)

    # Copy strategy file
    logger.debug(f"Copying strategy file from {stg_info.path} to {dest_file_path}")
    shutil.copy2(stg_info.path, dest_file_path)

    # Process imports strategy requires
    imports = _get_imports(stg_info.path, pyproject_root, [src_dir])
    for i in imports:
        # Construct source path
        src_i = os.path.join(pyproject_root, *[s for s in i.module if s != src_dir]) + ".py"
        if not os.path.exists(src_i):
            src_i = os.path.join(pyproject_root, *[s for s in i.module if s != src_dir], "__init__.py")
            if not os.path.exists(src_i):
                logger.warning(f"Import file not found: {src_i}")
                continue

        # Get the relative path from pyproject_root
        rel_import_path = os.path.relpath(src_i, pyproject_root)
        dest_import_path = os.path.join(release_dir, src_dir, rel_import_path)

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(dest_import_path), exist_ok=True)

        # Copy the import file
        logger.debug(f"Copying import from {src_i} to {dest_import_path}")
        shutil.copy2(src_i, dest_import_path)

    # Check default config file
    src_cfg = os.path.join(os.path.dirname(stg_info.path), DEFAULT_CFG_NAME)
    if os.path.exists(src_cfg):
        copy_file_to_dir(src_cfg, release_dir, f"{stg_name}.yml")
    else:
        # Generate default config from strategy file
        _save_config(src_dir, release_dir, stg_info.path, stg_name, stg_info.parameters)

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

    # Generate requirements.txt
    requirements_path = os.path.join(release_dir, "requirements.txt")
    generate_dependency_file(output_file=requirements_path, project_name=stg_name, save_as_requirements=True)

    # Copy pyproject.toml if it exists
    pyproject_src = os.path.join(pyproject_root, "pyproject.toml")
    if os.path.exists(pyproject_src):
        pyproject_dest = os.path.join(release_dir, "pyproject.toml")
        logger.debug(f"Copying pyproject.toml from {pyproject_src} to {pyproject_dest}")
        shutil.copy2(pyproject_src, pyproject_dest)

    # Generate pip.lock (empty file as placeholder)
    with open(os.path.join(release_dir, "pip.lock"), "w") as f:
        f.write("{}")

    # Make empty .venv file
    makedirs(release_dir, ".venv")

    # Zip all files
    logger.debug("Creating zip file ...")
    file_path = os.path.join(output_dir, git_info.tag)
    shutil.make_archive(file_path, "zip", release_dir)
    if not skip_confirmation:
        if ask_y_n(f"Do you want to clean temp release directory: {release_dir} ?"):
            shutil.rmtree(release_dir)
    logger.info(f"Created release pack: {file_path}.zip")


def _save_config(current_directory: str, dest_dir: str, f: str, s_name: str, s_params: dict):
    s_class = PurePath(os.path.join(current_directory, f.split(".py")[0])).parts
    s_class = ".".join([*s_class, s_name])
    cfg = {"config": {"strategy": s_class, "parameters": s_params}}
    with open(os.path.join(dest_dir, f"{s_name}.yml"), "wt") as fs:
        yaml.safe_dump(cfg, fs, sort_keys=False)


def _get_imports(file_name: str, current_directory: str, what_to_look: list[str]) -> list[Import]:
    imports = list(get_imports(file_name, what_to_look))
    current_dirname = os.path.basename(current_directory)
    for i in imports:
        try:
            f1 = os.path.join(*[current_directory, *[s for s in i.module if s != current_dirname]]) + ".py"
            if not os.path.exists(f1):
                f1 = (
                    os.path.join(*[current_directory, *[s for s in i.module if s != current_dirname], "__init__"])
                    + ".py"
                )
            imports.extend(_get_imports(f1, current_directory, what_to_look))
        except Exception as e:
            pass
    return imports


def generate_tag(strategy_name_id: str, tag_sfx: str | None) -> str:
    _tn = datetime.now()
    tag_s = f".{tag_sfx}" if tag_sfx else ""
    _strategy_name_id = strategy_name_id.replace(",", "_")
    tag = f"R_{_strategy_name_id}_{_tn.strftime('%Y%m%d%H%M%S')}{tag_s}"
    return tag


def make_tag_in_repo(repo: Repo, strategy_name_id: str, user: str, tag: str) -> str:
    repo.config_writer().set_value("push", "followTags", "true").release()

    _tn = datetime.now()
    ref_an = repo.create_tag(
        tag,
        message=f"Release of '{strategy_name_id}' at {_tn.strftime('%Y-%b-%d %H:%M:%S')} by {user}",
    )
    repo.remote("origin").push(ref_an)
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
        logger.info(" - Modified files -")
        for d in diffs:
            # Convert the file path to absolute path for comparison
            file_path = str(d.a_path) if d.a_path is not None else ""
            abs_file_path = os.path.abspath(os.path.join(repo_path, file_path))
            # Check if the file is under subdir
            if abs_file_path.startswith(abs_subdir):
                logger.info(f"  - {red(file_path)}")
                _to_add.append(file_path)

    if untr := repo.untracked_files:
        logger.info(" - Untracked files -")
        for d in untr:
            # Convert the file path to absolute path for comparison
            file_path = str(d)
            abs_file_path = os.path.abspath(os.path.join(repo_path, file_path))
            # Check if the file is under subdir
            if abs_file_path.startswith(abs_subdir):
                logger.info(f"  - {red(file_path)}")
                _to_add.append(file_path)

    user = getpass.getuser()
    tag = None
    commit_sha = None
    _tn = datetime.now()
    if _to_add:
        if not skip_commit:
            # - add changed files
            logger.info(green(f"Commiting changes for {len(_to_add)} files ... "))
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
            logger.warning("<r> >> Commiting changes is skipped due to --skip-commit option</r>")

    # - add annotated tag
    tag = generate_tag(strategy_name_id, tag_sfx)
    if not skip_tag:
        tag = make_tag_in_repo(repo, strategy_name_id, user, tag)
    else:
        logger.warning("<r> >> Creating git tag is skipped due to --skip-tag option</r>")

    return ReleaseInfo(tag=tag, commit=commit_sha, user=user, time=_tn, commited_files=_to_add)
