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


def ensure_poetry_lock_exists(output_dir: str) -> bool:
    """
    Ensures that a poetry.lock file exists in the output directory.
    If not, attempts to generate one.

    Args:
        output_dir: The directory to check/generate in

    Returns:
        bool: True if successful, False otherwise
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


def setup_poetry_environment(output_dir: str) -> bool:
    """
    Sets up the Poetry virtual environment in the output directory.

    Args:
        output_dir: The directory to set up the environment in

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Creating Poetry virtual environment")
    try:
        # Configure Poetry to create a virtual environment in the .venv directory
        logger.info("Configuring Poetry")
        subprocess.run(
            ["poetry", "config", "virtualenvs.in-project", "true", "--local"],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        # Check if we're already in a Poetry shell
        in_poetry_env = "POETRY_ACTIVE" in os.environ or "VIRTUAL_ENV" in os.environ

        if in_poetry_env:
            logger.debug(
                "Detected active Poetry environment. "
                "Will explicitly create a new environment for the deployed strategy."
            )

        # Install dependencies
        logger.info("Installing dependencies")

        # If we're in a Poetry shell, we need to be more explicit about creating a new environment
        install_cmd = ["poetry", "install"]
        if in_poetry_env:
            # Force Poetry to create a new environment even if we're in an active one
            env = os.environ.copy()
            # Temporarily unset Poetry environment variables to avoid interference
            for var in ["POETRY_ACTIVE", "VIRTUAL_ENV"]:
                if var in env:
                    del env[var]

            subprocess.run(install_cmd, cwd=output_dir, check=True, capture_output=False, text=True, env=env)
        else:
            # Normal case - not in a Poetry shell
            subprocess.run(install_cmd, cwd=output_dir, check=True, capture_output=False, text=True)

        # Verify that the virtual environment was created
        venv_path = os.path.join(output_dir, ".venv")
        if not os.path.exists(venv_path):
            logger.warning(
                "Virtual environment directory (.venv) not found. "
                "This might happen if you're already in a Poetry shell. "
                "You may need to run 'cd %s && poetry env use python' to create it manually.",
                output_dir,
            )

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set up Poetry environment: {e.stderr}")
        return False


def create_strategy_runners(output_dir: str):
    """
    Creates a strategy runner script in the output_dir
    """
    import sys

    if sys.platform == "win32":
        _pfx = ""
        _f_name = os.path.join(output_dir, "run_paper.bat")
    else:
        _pfx = "#!/bin/bash\n"
        _f_name = os.path.join(output_dir, "run_paper.sh")

    logger.info(f"Creating strategy paper runner script: {_f_name}")

    try:
        with open(_f_name, "w") as f:
            f.write(f"{_pfx}poetry run qubx run config.yml --paper -j")
        os.chmod(_f_name, 0o755)
    except Exception as e:
        logger.error(f"Failed to create strategy paper runner script: {e}")


def deploy_strategy(zip_file: str, output_dir: str | None, force: bool) -> bool:
    """
    Deploys a strategy from a zip file created by the release command.

    This function:
    1. Unpacks the zip file to the specified output directory
    2. Creates a Poetry virtual environment in the .venv folder
    3. Installs dependencies from the poetry.lock file

    Args:
        zip_file: Path to the zip file to deploy
        output_dir: Output directory to unpack the zip file. If None, uses the directory containing the zip file.
        force: Whether to force overwrite if the output directory already exists

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

    # Ensure poetry.lock exists
    if not ensure_poetry_lock_exists(resolved_output_dir):
        return False

    # Set up the Poetry environment
    if not setup_poetry_environment(resolved_output_dir):
        return False

    # Create the strategy runners
    create_strategy_runners(resolved_output_dir)

    # Success messages
    logger.info(f"Strategy deployed successfully to {resolved_output_dir}")
    logger.info(f" -> To run the strategy (in paper mode): <cyan>cd {resolved_output_dir} && ./run_paper.sh</cyan>")
    return True
