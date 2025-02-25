import ast
import getpass
import importlib
import os
import shutil
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path, PurePath

import yaml
from git import Remote, Repo

from qubx import logger
from qubx.utils.misc import Struct, cyan, green, magenta, makedirs, red, white, yellow

from .misc import (
    ask_y_n,
    copy_file_to_dir,
    copy_to_dir,
    find_git_root,
    generate_dependency_file,
    scan_strategies_in_directory,
)

Import = namedtuple("Import", ["module", "name", "alias"])
DEFAULT_CFG_NAME = "config.yml"


def get_imports(path, what_to_look=["xincubator"]):
    with open(path) as fh:
        root = ast.parse(fh.read(), path)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module = []
        elif isinstance(node, ast.ImportFrom):
            module = node.module.split(".")
        else:
            continue
        for n in node.names:
            if module and what_to_look and module[0] in what_to_look:
                yield Import(module, n.name.split("."), n.asname)


def ls_strats(path: str):
    strats = scan_strategies_in_directory(path)
    for f, sd in strats.items():
        strs = ""
        for sn, descr, pars in sd:
            descr = (": " + green(descr.replace("\n", " ").strip('" '))) if descr else ""
            _p_str = ""
            if pars:
                _max_l = max(map(len, pars.keys())) + 2
                for k, c in pars.items():
                    _p_str += f"\t{red(':')}  {cyan(k.ljust(_max_l))}: {yellow(str(c))}\n"
            strs += f"\t{red('.--(')} {white(sn)} {descr} \n{_p_str}\n"

        rst = f""" - {magenta(f)} -
{strs}"""
        print(rst)


def release_strategy(
    directory: str,
    strategy_name: str,
    tag: str | None,
    comment: str | None,
    skip_tag: bool,
    out_directory: str,
    skip_confirmation: bool,
):
    """
    Release strategy to zip file from given directory

    Args:
        directory: str - directory to scan for strategies
        strategy_name: str - strategy name to release
        tag: str - additional tag for this release
        comment: str - release comment
        skip_tag: bool - skip creating tag in repo
        out_directory: str - output directory to put zip file
        skip_confirmation: bool - skip confirmation
    """
    strats = scan_strategies_in_directory(directory)
    _to_release = []

    _strat_name_ids = []
    for k, v in strats.items():
        _s_name = v[0][0]
        if strategy_name == "*" or _s_name.lower() == strategy_name.lower():
            print(_s_name)
            _to_release.append((k, v))
            _strat_name_ids.append(_s_name)

    if _to_release:
        if len(_to_release) > 1 and not skip_confirmation:
            if not ask_y_n(
                f"Found {len(_to_release)} matches, are you sure you want to release all these strategies ? "
            ):
                return
    else:
        logger.info(f"<r>Strategy {strategy_name} not found in {directory}</r>")
        return

    # - get repo root path
    root_path = find_git_root(_to_release[0][0])
    os.chdir(root_path)

    # - processing repo
    g_info = process_git_repo(".", ",".join(_strat_name_ids), tag, comment, skip_tag, skip_tag)

    # - create zipped pack
    xsrc = os.path.split(os.getcwd())[1]
    create_released_pack(xsrc, _to_release, g_info, out_directory, skip_confirmation)


def generate_py_deps(req_output_file: str):
    with open(req_output_file, "wt") as f:
        for p in sorted(list(importlib.metadata.distributions()), key=lambda x: x.metadata["Name"]):  # type: ignore
            print(f"{p.metadata['Name']}=={p.metadata['Version']}", file=f)


def create_released_pack(
    current_directory: str,
    release_info: list[tuple[str, list]],
    git_info: Struct | None,
    out_directory: str,
    skip_confirmation: bool,
):
    logger.info(f"Creating release pack for {git_info.tag} ...")
    r_dir = makedirs(out_directory, git_info.tag)
    dest = makedirs(out_directory, git_info.tag, current_directory)

    for f, strat_info in release_info:
        file_int_dir = os.path.dirname(f.split(current_directory)[1])
        # - copy strategy file
        copy_file_to_dir(f, dest + file_int_dir)  # @dmitry - is it correct fix, because f is the file?
        s_name = strat_info[0][0]

        # - processing imports strategy requires
        imports = _get_imports(f, current_directory, [current_directory])
        for i in imports:
            dest_i = os.path.join(*[out_directory, git_info.tag, current_directory])
            src_i = os.path.join(*[*[s for s in i.module if s != current_directory]]) + ".py"
            if not os.path.exists(src_i):
                src_i = os.path.join(*[*[s for s in i.module if s != current_directory], "__init__"]) + ".py"
            copy_to_dir(src_i, dest_i)

        # - check default config file
        src_cfg = os.path.join(os.path.dirname(f), DEFAULT_CFG_NAME)
        # logger.info(src_cfg)

        if os.path.exists(src_cfg):
            copy_file_to_dir(src_cfg, r_dir, f"{s_name}.yml")
        else:
            # generate default config from strategy file
            _save_config(current_directory, r_dir, f, s_name, strat_info[0][2])

        # - create meta info file
        meta = {
            "tag": git_info.tag,  # type: ignore
            "date": git_info.time.isoformat(),  # type: ignore
            "author": git_info.user,  # type: ignore
        }
        with open(os.path.join(r_dir, f"{s_name}.info"), "wt") as fs:
            yaml.safe_dump(meta, fs, sort_keys=False)

        # - collect packages deps
        # generate_py_deps(os.path.join(r_dir, "requirements.txt"))
        generate_dependency_file(
            output_file=os.path.join(r_dir, "requirements.txt"), project_name=s_name, save_as_requirements=True
        )

        # - make empty .venv file
        makedirs(r_dir, ".venv")

    # - zip all files
    logger.debug("Creating zip file ...")
    file_path = os.path.join(out_directory, git_info.tag)
    shutil.make_archive(file_path, "zip", r_dir)  # type: ignore
    if not skip_confirmation:
        if ask_y_n(f"Do you want to clean temp release directory: {r_dir} ?"):
            shutil.rmtree(r_dir)
    logger.info(f"Created release pack: {file_path}.zip")


def _save_config(current_directory: str, dest_dir: str, f: str, s_name: str, s_params: dict):
    s_class = PurePath(os.path.join(current_directory, f.split(".py")[0])).parts
    s_class = ".".join([*s_class, s_name])
    cfg = {"config": {"strategy": s_class, "parameters": s_params}}
    with open(os.path.join(dest_dir, f"{s_name}.yml"), "wt") as fs:
        yaml.safe_dump(cfg, fs, sort_keys=False)


def _get_imports(file_name: str, current_directory: str, what_to_look: list[str]) -> list[Import]:
    imports = list(get_imports(file_name, what_to_look))  # type: ignore
    for i in imports:
        try:
            f1 = os.path.join(*[*[s for s in i.module if s != current_directory]]) + ".py"
            if not os.path.exists(f1):
                f1 = os.path.join(*[*[s for s in i.module if s != current_directory], "__init__"]) + ".py"
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
    strategy_name_id: str,
    tag_sfx: str | None,
    comment: str | None,
    skip_tag: bool,
    skip_commit: bool,
) -> Struct | None:
    repo = Repo(repo_path)

    _to_add = []
    if diffs := repo.index.diff(None):
        logger.debug(" - Modified files -")
        for d in diffs:
            logger.debug(f"  - {red(d.a_path)}")
            _to_add.append(d.a_path)

    if untr := repo.untracked_files:
        logger.debug(" - Untracked files -")
        for d in untr:
            logger.debug(f"  - {red(d)}")
            _to_add.append(d)

    user = getpass.getuser()
    tag = ""
    commit_sha = ""
    _tn = datetime.now()
    if _to_add:
        logger.info(green(f"Commiting changes for {len(_to_add)} files ... "))
        if not skip_commit:
            # - add changed files
            try:
                repo.index.add(_to_add)
                cmt = repo.index.commit(
                    f"Changes before release of '{strategy_name_id}' at {_tn.strftime('%Y-%b-%d %H:%M:%S')} by {user}."
                    f"{'' + comment if comment else ''}"
                )
                commit_sha = cmt.hexsha
                _ilist = repo.remotes[0].push()
                for i in _ilist:
                    logger.debug(f"\t{yellow(i.summary)}")
            except Exception as e:
                logger.error(f"Error committing changes: {e}")
                return None

    # - add annotated tag
    tag = generate_tag(strategy_name_id, tag_sfx)
    if not skip_tag:
        tag = make_tag_in_repo(repo, strategy_name_id, user, tag)
    else:
        logger.warning(red(" >> Creating git tag is skipped due to --skip-tag option"))

    return Struct(tag=tag, commit=commit_sha, user=user, time=_tn, commited_files=_to_add)
