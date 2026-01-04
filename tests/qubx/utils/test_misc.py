"""Tests for qubx.utils.misc module."""

import sys

import pytest

from qubx.utils.misc import add_project_to_system_path


@pytest.fixture(autouse=True)
def cleanup_sys_path():
    """Remove test paths from sys.path after each test."""
    original_path = sys.path.copy()
    yield
    sys.path[:] = original_path


class TestAddProjectToSystemPath:
    """Tests for add_project_to_system_path function."""

    def test_add_project_with_pyproject_and_src(self, tmp_path):
        """Should add src/ folder when project has pyproject.toml and src/ subfolder."""
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "pyproject.toml").touch()
        (project / "src").mkdir()

        add_project_to_system_path(project)  # Pass Path directly

        assert (project / "src").as_posix() in sys.path

    def test_add_project_with_pyproject_no_src(self, tmp_path):
        """Should add project root when pyproject.toml exists but no src/ folder."""
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "pyproject.toml").touch()

        add_project_to_system_path(str(project))  # Pass string

        assert project.as_posix() in sys.path

    def test_scans_subdirs_for_pyproject(self, tmp_path):
        """Should scan subdirs and add those with pyproject.toml."""
        parent = tmp_path / "projects"
        parent.mkdir()

        # Create subproject with pyproject.toml + src
        subproj = parent / "subproject"
        subproj.mkdir()
        (subproj / "pyproject.toml").touch()
        (subproj / "src").mkdir()

        add_project_to_system_path(parent)

        # Parent added, and subproject's src/ added
        assert parent.as_posix() in sys.path
        assert (subproj / "src").as_posix() in sys.path
