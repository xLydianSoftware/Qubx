import ast
import fnmatch
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
import yaml

from qubx import logger
from qubx.utils.misc import green


@dataclass
class PyClassInfo:
    name: str
    path: str
    docstring: str
    parameters: dict[str, Any]
    is_strategy: bool


def strategies_root() -> Path:
    """
    Get the root directory for strategies.
    """
    user_home = Path.home()
    strats_dir = user_home / "strategies"
    if not strats_dir.exists():
        strats_dir.mkdir()
    return strats_dir


def search_file(file_path: str | Path) -> Path | None:
    """
    Search for a file in the current directory and all parent directories.
    """
    file_path = Path(file_path)
    if file_path.exists():
        return file_path
    elif file_path.name == str(file_path):  # if only file name is provided
        curr_dir = Path.cwd()
        file_path = curr_dir / file_path
        while not file_path.exists():
            if curr_dir == curr_dir.parent:
                return None
            curr_dir = curr_dir.parent
            file_path = curr_dir / file_path.name
        return file_path
    return None


def zipdir(path: str | Path, zipname: str | Path) -> None:
    def _zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file_name in files:
                ziph.write(
                    os.path.join(root, file_name),
                    os.path.relpath(os.path.join(root, file_name), os.path.join(path, "..")),
                )

    logger.debug(zipname)
    with zipfile.ZipFile(f"{zipname}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        _zipdir(path, zipf)


def flatten(decl_list: list):
    for d in decl_list:
        try:
            yield d.id
        except AttributeError:
            try:
                yield d.func.id
            except AttributeError:
                yield None


def is_decorated(decorator_list, decorator_name) -> bool:
    for x in flatten(decorator_list):
        if x == decorator_name:
            return True
    return False


def copy_to_dir(src: str, dest_dir: str):
    try:
        d = Path(dest_dir) / src
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(src, d)
        logger.debug(f"  Copying <g>{src}</g> -> <c>{d}</c>")
    except:  # noqa: E722
        logger.error(f" - Error copying {src} -> {d}")


def copy_file_to_dir(src: str, dest_dir: str, new_name: str | None = None):
    try:
        if new_name:
            d = Path(dest_dir) / new_name
        else:
            d = Path(dest_dir) / os.path.split(src)[-1]
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(src, d)
        logger.debug(f"  Copying <g>{src}</g> -> <c>{d}</c>")
    except:  # noqa: E722
        logger.error(f" - Error copying {src} -> {d}")


def ask_y_n(question: str) -> bool:
    user_input = input(f" -> {green(question)} (yes/no, default=yes): ")
    return user_input.lower() in ["yes", "y", ""]


def get_current_python_version() -> str:
    """
    Retrieves the current Python version (major.minor) as a string.
    """
    try:
        output = subprocess.check_output(["python", "--version"], text=True)
        version = output.strip().split()[1]
        major_minor = ".".join(version.split(".")[:2])  # Extract major.minor
        return major_minor
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error retrieving Python version: {e}")


def find_file_by_mask(mask: str | Path, root_dir: str | Path) -> list[Path]:
    """
    Finds files matching the given mask starting from the root directory.

    Args:
        mask (str): The file mask to search for (e.g., "qubx/utils/runner.py").
        root_dir (str): The root directory to start the search.

    Returns:
        list: A list of full paths to files matching the mask.
    """
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, os.path.basename(mask)):
            if mask in os.path.join(root, filename):  # TODO: WTF ???
                matches.append(Path(os.path.join(root, filename)))
    return matches


def scan_py_classes_in_directory(
    root_path: str | Path, skip_directories: tuple[str, ...] = ("dist",)
) -> list[PyClassInfo]:
    """
    Scan Python classes in the given directory. Additionaly it will extract parameters from the class and mark it as a strategy.

    Args:
        root_path: The root directory to scan for classes
        skip_directories: Tuple of directory names to skip during scanning. Defaults to ("dist",)

    Returns:
        A list of PyClassInfo objects.
    """
    classes = []
    root_path = os.path.expanduser(root_path)

    for file_path in glob.glob(os.path.join(root_path, "**/*.py"), recursive=True):
        # Skip files in directories that should be skipped
        if any(skip_dir in file_path.split(os.sep) for skip_dir in skip_directories):
            continue

        name = os.path.splitext(os.path.basename(file_path))[0]

        # Ignore __ files
        if name.startswith("__"):
            continue

        with open(file_path, "r") as f:
            src = f.read()

        try:
            class_node = ast.parse(src)
        except:  # noqa: E722
            continue

        nodes = [node for node in ast.walk(class_node) if isinstance(node, ast.ClassDef)]
        for n in nodes:
            if len(n.bases) > 0:
                for b in n.bases:
                    if hasattr(b, "id") and getattr(b, "id", None) in ["IStrategy"]:
                        # - when class inherits IStrategy, it is a strategy
                        classes.append(
                            PyClassInfo(
                                name=n.name,
                                path=file_path,
                                docstring=ast.get_docstring(n) or "",
                                parameters=iter_class_parameters(n),
                                is_strategy=True,
                            )
                        )
                        break
                else:
                    # - otherwise we just add it as a class but still extract parameters etc
                    classes.append(
                        PyClassInfo(
                            name=n.name,
                            path=file_path,
                            docstring=ast.get_docstring(n) or "",
                            parameters=iter_class_parameters(n),
                            is_strategy=False,
                        )
                    )
                    continue
                break
            else:
                classes.append(
                    PyClassInfo(
                        name=n.name,
                        path=file_path,
                        docstring=ast.get_docstring(n) or "",
                        parameters=iter_class_parameters(n),
                        is_strategy=False,
                    )
                )
    return classes


def iter_class_parameters(class_node: ast.AST) -> dict[str, Any]:
    """
    Iterate over the parameters of a class.
    """
    decls = {}
    for n in ast.iter_child_nodes(class_node):
        if isinstance(n, ast.AnnAssign):
            if not n.target.id.startswith("_"):
                if hasattr(n, "value"):
                    try:
                        decls[n.target.id] = None
                        if n.value:
                            if isinstance(n.value.value, ast.Name):
                                decls[n.target.id] = "???"
                            else:
                                decls[n.target.id] = n.value.value

                    except Exception as _:
                        decls[n.target.id] = None
    return dict(reversed(list(decls.items())))


def find_runner_file(root_path: str | Path) -> Path:
    """
    Find qubx runner file in the given root directory.
    """
    if runner_files := find_file_by_mask(mask="qubx/utils/runner.py", root_dir=Path(root_path)):
        assert len(runner_files) == 1, f"Multiple runner files found in {root_path}"
        return runner_files[0]
    raise FileNotFoundError(f"Runner file not found in {root_path}")


def find_strategy_data(strategy_name: str, start_dir: str | Path) -> tuple[Path | None, Path | None, Path | None]:
    """
    Find root directory of the strategy by its configuration files / directories
    """
    start_dir = Path(start_dir).expanduser()
    assert start_dir.exists(), f"Start directory {start_dir} does not exist"

    for root, dirs, files in os.walk(start_dir):
        for file_name in files:
            if file_name.endswith(".yaml") or file_name.endswith(".yml"):
                with open(os.path.join(root, file_name), "r") as f:
                    try:
                        yaml_data = yaml.safe_load(f)
                        if yaml_data["config"]["strategy"].split(".")[-1].lower() == strategy_name.lower():
                            if runner_files := find_file_by_mask(mask="qubx/utils/runner.py", root_dir=Path(root)):
                                assert len(runner_files) == 1, f"Multiple runner files found in {root}"
                                return Path(root), runner_files[0], Path(os.path.join(root, file_name))
                            else:
                                logger.error(f"Runner qubx file not found in {root}. Skipping...")
                    except Exception as e:
                        logger.error(f"Error processing {file_name}: {str(e)}")
    return None, None, None


def find_git_root(file_path: str) -> str:
    """
    Find the root directory of a Git repository from a file path.

    :param file_path: Path to a file within the repository.
    :return: The root directory of the Git repository.
    :raises ValueError: If the file is not inside a Git repository.
    """
    try:
        dir_path = os.path.abspath(os.path.dirname(file_path))

        git_root = subprocess.check_output(
            ["git", "-C", dir_path, "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT, text=True
        ).strip()

        return git_root
    except subprocess.CalledProcessError as e:
        raise ValueError(f"The file '{file_path}' is not inside a Git repository.") from e


def find_pyproject_root(file_path: str) -> str:
    """
    Find the root directory of a pyproject.toml file from a file path.
    """
    try:
        dir_path = os.path.abspath(os.path.dirname(file_path))
        while not os.path.exists(os.path.join(dir_path, "pyproject.toml")):
            dir_path = os.path.dirname(dir_path)
        return dir_path
    except Exception as e:
        raise ValueError(f"The file '{file_path}' is not inside a pyproject.toml file.") from e


def locate_poetry(paths: list[str] | None = None, silent: bool = False) -> Path | None:
    """
    Locate the Poetry executable.

    :param paths: Optional list of paths to search for the Poetry executable.
    :return: The path to the Poetry executable.
    :raises FileNotFoundError: If the Poetry executable is not found.
    """
    poetry_executable = shutil.which("poetry") or shutil.which("poetry.exe")
    if poetry_executable is None:
        poetry_paths = [
            Path.home() / ".poetry" / "bin" / "poetry",
            Path.home() / ".local" / "bin" / "poetry",
            Path("/usr/local/bin/poetry"),
            Path("/usr/bin/poetry"),
        ]
        poetry_executable = next(
            (path for path in poetry_paths if path.is_file() or Path(f"{path}.exe").is_file()), None
        )

    if not poetry_executable:
        if not silent:
            raise FileNotFoundError("Poetry executable not found. Ensure Poetry is installed and in the PATH.")
        else:
            logger.error("Poetry executable not found. Ensure Poetry is installed and in the PATH.")
            return None
    return poetry_executable


def _is_package_platform_compatible(package: dict[str, Any]) -> bool:
    files = package.get("files", [])
    current_platform = sys.platform
    if "win" in current_platform:
        current_platforms = ["win32", "win_amd64", "win_arm64", "any"]
    elif "linux" in current_platform:
        current_platforms = ["linux", "any"]
    else:
        current_platforms = ["any"]
    for file_data in files:
        if file_data["file"].endswith(".whl"):
            if any(platform in file_data["file"] for platform in current_platforms):
                return True
    return len(files) == 0


def generate_dependency_file(
    lock_file: str | Path = "poetry.lock",
    output_file: str | Path = "pyproject.toml",
    project_name: str = "generated-project",
    save_as_requirements: bool = False,
):
    try:
        with open(lock_file, "r") as lock:
            lock_data = toml.load(lock)

        if save_as_requirements:
            requirements = []
            for package in lock_data.get("package", []):
                dep_name = package["name"]
                version = package["version"]
                if not _is_package_platform_compatible(package):
                    continue
                requirements.append(f"{dep_name}=={version}")

            with open(output_file, "w") as requirements_file:
                requirements_file.write("\n".join(requirements))

            print(f"requirements.txt has been generated and saved to {output_file}.")
        else:
            pyproject = {
                "tool": {
                    "poetry": {
                        "name": project_name,
                        "version": "0.1.0",
                        "description": "Generated from poetry.lock",
                        "dependencies": {},
                        "dev-dependencies": {},
                    }
                }
            }

            for package in lock_data.get("package", []):
                dep_name = package["name"]
                version = package["version"]
                category = package.get("category", "main")
                if not _is_package_platform_compatible(package):
                    continue

                if category == "main":
                    pyproject["tool"]["poetry"]["dependencies"][dep_name] = version
                elif category == "dev":
                    pyproject["tool"]["poetry"]["dev-dependencies"][dep_name] = version

            with open(output_file, "w") as pyproject_file:
                toml.dump(pyproject, pyproject_file)

            logger.debug(f"pyproject.toml has been generated and saved to {output_file}.")
    except Exception as e:
        print(f"An error occurred during making dependencies file: {e}")
