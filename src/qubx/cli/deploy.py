import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from qubx import logger


def validate_zip_file(zip_file: str) -> bool:
    """
    Validates that the provided file is a zip file.

    Args:
        zip_file: Path to the zip file to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not zip_file.endswith(".zip"):
        logger.error("The file must be a zip file with .zip extension")
        return False
    return True


def determine_output_directory(zip_file: str, output_dir: str | None) -> str:
    """
    Determines the output directory for the deployment.

    Args:
        zip_file: Path to the zip file to deploy
        output_dir: User-specified output directory or None

    Returns:
        str: The resolved output directory path
    """
    zip_path = Path(zip_file)
    zip_name = zip_path.stem

    if output_dir is None:
        return str(zip_path.parent / zip_name)

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    # If output_dir is a directory that exists, create a subdirectory with the zip name
    if os.path.isdir(output_dir):
        return os.path.join(output_dir, zip_name)

    return output_dir


def prepare_output_directory(output_dir: str, force: bool) -> bool:
    """
    Prepares the output directory, handling existing directories based on the force flag.

    Args:
        output_dir: The output directory path
        force: Whether to force overwrite if the directory exists

    Returns:
        bool: True if successful, False otherwise
    """
    if os.path.exists(output_dir):
        if not force:
            logger.error(f"Output directory {output_dir} already exists. Use --force to overwrite.")
            return False
        logger.warning(f"Removing existing directory {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    return True


def extract_zip_file(zip_file: str, output_dir: str) -> bool:
    """
    Extracts the zip file to the output directory.

    Args:
        zip_file: Path to the zip file to extract
        output_dir: Directory to extract to

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Unpacking {zip_file} to {output_dir}")
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        return True
    except zipfile.BadZipFile:
        logger.error(f"The file {zip_file} is not a valid zip file")
        return False
    except Exception as e:
        logger.error(f"Failed to unpack zip file: {e}")
        return False


def _detect_package_manager(output_dir: str) -> str:
    """Detect whether the release was built with uv or poetry.

    Returns "uv" or "poetry" based on which lock file is present.
    Defaults to "uv" if neither is found.
    """
    if os.path.exists(os.path.join(output_dir, "uv.lock")):
        return "uv"
    if os.path.exists(os.path.join(output_dir, "poetry.lock")):
        return "poetry"
    return "uv"


def ensure_lock_exists(output_dir: str) -> bool:
    """
    Ensures that a uv.lock file exists in the output directory.
    If not, attempts to generate one.

    Args:
        output_dir: The directory to check/generate in

    Returns:
        bool: True if successful, False otherwise
    """
    uv_lock_path = os.path.join(output_dir, "uv.lock")
    if not os.path.exists(uv_lock_path):
        logger.warning("uv.lock not found in the zip file. Attempting to generate it.")
        try:
            subprocess.run(["uv", "lock"], cwd=output_dir, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate uv.lock: {e.stderr}")
            return False
    return True


def setup_uv_environment(output_dir: str) -> bool:
    """
    Sets up the uv virtual environment in the output directory.

    Args:
        output_dir: The directory to set up the environment in

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Creating virtual environment with uv")
    try:
        # Check if we're already in an active virtual environment
        in_venv = "VIRTUAL_ENV" in os.environ

        if in_venv:
            logger.debug(
                "Detected active virtual environment. "
                "Will explicitly create a new environment for the deployed strategy."
            )

        # Install dependencies with uv sync
        logger.info("Installing dependencies")

        install_cmd = ["uv", "sync"]
        if in_venv:
            # Force uv to create a new environment even if we're in an active one
            env = os.environ.copy()
            # Temporarily unset environment variables to avoid interference
            for var in ["VIRTUAL_ENV"]:
                if var in env:
                    del env[var]

            subprocess.run(install_cmd, cwd=output_dir, check=True, capture_output=False, text=True, env=env)
        else:
            # Normal case - not in a venv
            subprocess.run(install_cmd, cwd=output_dir, check=True, capture_output=False, text=True)

        # Verify that the virtual environment was created
        venv_path = os.path.join(output_dir, ".venv")
        if not os.path.exists(venv_path):
            logger.warning(
                "Virtual environment directory (.venv) not found. "
                "You may need to run 'cd %s && uv venv' to create it manually.",
                output_dir,
            )

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set up uv environment: {e.stderr}")
        return False


# --- Legacy Poetry support (temporary) ---
# TODO: Remove Poetry deploy functions once all old releases are re-released with uv


def _ensure_poetry_lock_exists(output_dir: str) -> bool:
    """Ensures that a poetry.lock file exists in the output directory.

    Legacy support for old Poetry-based releases.
    """
    poetry_lock_path = os.path.join(output_dir, "poetry.lock")
    if not os.path.exists(poetry_lock_path):
        logger.warning("poetry.lock not found in the zip file. Attempting to generate it.")
        try:
            subprocess.run(["poetry", "lock"], cwd=output_dir, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate poetry.lock: {e.stderr}")
            return False
    return True


def _setup_poetry_environment(output_dir: str) -> bool:
    """Sets up the Poetry virtual environment in the output directory.

    Legacy support for old Poetry-based releases.
    """
    logger.info("Creating Poetry virtual environment")
    try:
        subprocess.run(
            ["poetry", "config", "virtualenvs.in-project", "true", "--local"],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        in_poetry_env = "POETRY_ACTIVE" in os.environ or "VIRTUAL_ENV" in os.environ

        logger.info("Installing dependencies")

        install_cmd = ["poetry", "install"]
        if in_poetry_env:
            env = os.environ.copy()
            for var in ["POETRY_ACTIVE", "VIRTUAL_ENV"]:
                if var in env:
                    del env[var]
            subprocess.run(install_cmd, cwd=output_dir, check=True, capture_output=False, text=True, env=env)
        else:
            subprocess.run(install_cmd, cwd=output_dir, check=True, capture_output=False, text=True)

        venv_path = os.path.join(output_dir, ".venv")
        if not os.path.exists(venv_path):
            logger.warning(
                "Virtual environment directory (.venv) not found. "
                "You may need to run 'cd %s && poetry env use python' to create it manually.",
                output_dir,
            )

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set up Poetry environment: {e.stderr}")
        return False


# --- End legacy Poetry support ---


def create_strategy_runners(output_dir: str, pkg_manager: str = "uv"):
    """
    Creates a strategy runner script in the output_dir
    """
    import sys

    run_prefix = "uv run" if pkg_manager == "uv" else "poetry run"

    if sys.platform == "win32":
        _pfx = ""
        _f_name = os.path.join(output_dir, "run_paper.bat")
    else:
        _pfx = "#!/bin/bash\n"
        _f_name = os.path.join(output_dir, "run_paper.sh")

    logger.info(f"Creating strategy paper runner script: {_f_name}")

    try:
        with open(_f_name, "w") as f:
            f.write(f"{_pfx}{run_prefix} qubx run config.yml --paper -j")
        os.chmod(_f_name, 0o755)
    except Exception as e:
        logger.error(f"Failed to create strategy paper runner script: {e}")


def install_system_wheels(output_dir: str) -> bool:
    """
    Install all wheels from wheels/ directory into system site-packages.

    This is the Docker deployment path: no venv, no uv, just pip install.

    Args:
        output_dir: Directory containing the extracted release (with wheels/ subdirectory)

    Returns:
        bool: True if successful, False otherwise
    """
    wheels_dir = os.path.join(output_dir, "wheels")
    if not os.path.isdir(wheels_dir):
        logger.warning("No wheels/ directory found — skipping system wheel install")
        return True

    wheel_files = [os.path.join(wheels_dir, f) for f in os.listdir(wheels_dir) if f.endswith(".whl")]
    if not wheel_files:
        logger.warning("No .whl files found in wheels/ — skipping system wheel install")
        return True

    logger.info(f"Installing {len(wheel_files)} wheel(s) into system site-packages...")
    pip_exe = shutil.which("pip") or shutil.which("pip3")
    if not pip_exe:
        logger.error("pip not found — cannot install wheels into system site-packages")
        return False

    try:
        subprocess.run(
            [pip_exe, "install", *wheel_files],
            check=True,
            capture_output=False,
            text=True,
        )
        logger.info("System wheel install complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install wheels: {e}")
        return False


def deploy_strategy(zip_file: str, output_dir: str | None, force: bool, system: bool = False) -> bool:
    """
    Deploys a strategy from a zip file created by the release command.

    This function:
    1. Unpacks the zip file to the specified output directory
    2. Auto-detects the package manager (uv or poetry) based on the lock file
    3. Creates a virtual environment and installs dependencies (default)
       OR installs wheels into system site-packages (--system mode for Docker)

    Supports both uv-based releases, legacy Poetry-based releases, and system mode.

    Args:
        zip_file: Path to the zip file to deploy
        output_dir: Output directory to unpack the zip file. If None, uses the directory containing the zip file.
        force: Whether to force overwrite if the output directory already exists
        system: If True, install wheels directly into system site-packages (Docker mode)

    Returns:
        bool: True if deployment was successful, False otherwise
    """
    # Validate the zip file
    if not validate_zip_file(zip_file):
        return False

    # Determine the output directory
    resolved_output_dir = determine_output_directory(zip_file, output_dir)

    # Prepare the output directory
    if not prepare_output_directory(resolved_output_dir, force):
        return False

    # Extract the zip file
    if not extract_zip_file(zip_file, resolved_output_dir):
        return False

    if system:
        # Docker path: pip install wheels directly into system site-packages
        if not install_system_wheels(resolved_output_dir):
            return False
        logger.info(f"Strategy deployed (system mode) to {resolved_output_dir}")
        return True

    # Default path: venv-based install
    # Detect package manager
    pkg_manager = _detect_package_manager(resolved_output_dir)
    logger.info(f"Detected package manager: {pkg_manager}")
    if pkg_manager == "poetry":
        logger.warning(
            "This is a legacy Poetry-based release. "
            "Consider re-releasing with the latest qubx version (uv-based)."
        )

    # Ensure lock file exists
    if pkg_manager == "uv":
        if not ensure_lock_exists(resolved_output_dir):
            return False
    else:
        if not _ensure_poetry_lock_exists(resolved_output_dir):
            return False

    # Set up environment
    if pkg_manager == "uv":
        if not setup_uv_environment(resolved_output_dir):
            return False
    else:
        if not _setup_poetry_environment(resolved_output_dir):
            return False

    # Create the strategy runners
    create_strategy_runners(resolved_output_dir, pkg_manager)

    # Success messages
    logger.info(f"Strategy deployed successfully to {resolved_output_dir}")
    logger.info(f" -> To run the strategy (in paper mode): <cyan>cd {resolved_output_dir} && ./run_paper.sh</cyan>")
    return True
