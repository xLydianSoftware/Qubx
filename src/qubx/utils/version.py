import os
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import toml

from qubx import logger
from qubx.utils.misc import this_project_root


class VersionPart(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


def read_current_version() -> Tuple[int, int, int]:
    """Read the current version from pyproject.toml."""
    try:
        project_root = this_project_root()
        if not project_root:
            raise FileNotFoundError("Could not find pyproject.toml in any parent directory")

        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path, "r") as f:
            pyproject_data = toml.load(f)

        version_str = pyproject_data["tool"]["poetry"]["version"]
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)

        if not match:
            raise ValueError(f"Invalid version format: {version_str}")

        major, minor, patch = map(int, match.groups())
        return major, minor, patch

    except Exception as e:
        logger.error(f"Error reading current version: {e}")
        raise


def update_version(part: str = "patch") -> str:
    """
    Update the version in pyproject.toml based on semantic versioning.

    Args:
        part: Which part of the version to increment ("major", "minor", or "patch")

    Returns:
        The new version string

    Raises:
        ValueError: If the part is invalid or the version cannot be updated
    """
    try:
        # Validate part
        try:
            version_part = VersionPart(part.lower())
        except ValueError:
            raise ValueError(f"Invalid version part: {part}. Must be one of: major, minor, patch")

        project_root = this_project_root()
        if not project_root:
            raise FileNotFoundError("Could not find pyproject.toml in any parent directory")

        pyproject_path = project_root / "pyproject.toml"

        major, minor, patch = read_current_version()

        # Update version based on the specified part
        if version_part == VersionPart.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif version_part == VersionPart.MINOR:
            minor += 1
            patch = 0
        elif version_part == VersionPart.PATCH:
            patch += 1

        new_version = f"{major}.{minor}.{patch}"

        # Update pyproject.toml by loading it as TOML, modifying the version, and writing it back
        with open(pyproject_path, "r") as f:
            pyproject_data = toml.load(f)

        # Update the version
        pyproject_data["tool"]["poetry"]["version"] = new_version

        # Write the updated TOML back to the file
        with open(pyproject_path, "w") as f:
            toml.dump(pyproject_data, f)

        logger.info(f"Updated version to {new_version}")
        return new_version

    except Exception as e:
        logger.error(f"Error updating version: {e}")
        raise


def git_commit_and_tag(version: str) -> bool:
    """
    Commit changes and create a tag with the new version.

    Args:
        version: The new version string

    Returns:
        True if successful, False otherwise
    """
    try:
        project_root = this_project_root()
        if not project_root:
            logger.error("Could not find project root")
            return False

        os.chdir(project_root)

        # Check if there are changes to commit
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)

        has_changes = bool(result.stdout.strip())

        if has_changes:
            # Stage all changes
            subprocess.run(["git", "add", "."], check=True)

            # Commit with version message
            commit_message = f"Bump version to {version}"
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            logger.info(f"Committed changes with message: {commit_message}")
        else:
            logger.info("No changes to commit")

        # Create tag
        tag_name = f"v{version}"
        subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Version {version}"], check=True)
        logger.info(f"Created tag: {tag_name}")

        # Push commit and tag
        subprocess.run(["git", "push", "origin", "HEAD"], check=True)
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        logger.info("Pushed commit and tag to origin")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error in git operations: {e}")
        return False


def update_project_version(part: str = "patch") -> bool:
    """
    Main function to update version, commit, tag and push.

    Args:
        part: Which part of the version to increment ("major", "minor", or "patch")

    Returns:
        True if successful, False otherwise
    """
    try:
        # Update version
        new_version = update_version(part)

        # Commit, tag and push
        success = git_commit_and_tag(new_version)

        return success

    except Exception as e:
        logger.error(f"Error updating version: {e}")
        return False
