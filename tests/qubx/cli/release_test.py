import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import qubx.pandaz.ta as pta
from qubx.backtester.simulator import simulate
from qubx.cli.misc import PyClassInfo, find_pyproject_root
from qubx.cli.release import (
    ReleaseInfo,
    StrategyInfo,
    _bundle_source_overrides,
    _find_uv_workspace_root,
    _find_workspace_member_for_package,
    _generate_release_pyproject,
    _lock_constraint_dependencies,
    _resolve_source_lockfile,
    create_released_pack,
)
from qubx.core.series import OHLCV
from qubx.data import CsvStorage
from qubx.utils.runner.configs import ExchangeConfig, LoggingConfig, ReleaseSourceConfig, StrategyConfig

# Add tests/strategies to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "tests" / "strategies" / "macd_crossover" / "src"))

from pytest import approx

from tests.strategies.macd_crossover.src.macd_crossover.indicators.macd import macd
from tests.strategies.macd_crossover.src.macd_crossover.models.macd_crossover import MacdCrossoverStrategy

N = lambda x, r=1e-4: approx(x, rel=r, nan_ok=True)

_CSV_STORAGE = "tests/data/storages/csv/"


class TestMacdCrossoverSimulation:
    def test_macd_indicator(self):
        r = CsvStorage(_CSV_STORAGE).get_reader("BINANCE.UM", "SWAP")

        ohlc = r.read("ETHUSDT", "ohlc(1h)", start="2023-06-01", stop="+30d").to_ohlc()  # type: ignore
        assert isinstance(ohlc, OHLCV)
        _macd = macd(ohlc.close).to_series().dropna()
        expected_macd = pta.macd(ohlc.close.pd()).dropna()
        assert N(_macd[-50:]) == expected_macd[-50:]

    def test_macd_crossover_simulation(self):
        ld = CsvStorage(_CSV_STORAGE)
        test0 = simulate(
            MacdCrossoverStrategy(),
            ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-07-01",
            stop="2023-07-10",
            debug="INFO",
            n_jobs=1,
        )
        sim = test0[0]
        assert len(sim.executions_log) > 1


class TestCreateReleasedPack:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_git_info(self):
        """Create a mock ReleaseInfo object."""
        from datetime import datetime

        return ReleaseInfo(
            tag="R_MacdCrossoverStrategy_20240101000000",
            commit="abcdef1234567890",
            user="test_user",
            time=datetime.now(),
            commited_files=["tests/strategies/macd_crossover/src/macd_crossover/models/macd_crossover.py"],
        )

    @pytest.fixture
    def mock_strategy_info(self):
        """Create a mock PyClassInfo object for the MACD strategy."""
        # Get the actual path to the MACD strategy
        strategy_path = Path("tests/strategies/macd_crossover/src/macd_crossover/models/macd_crossover.py").absolute()
        return PyClassInfo(
            path=str(strategy_path),
            name="MacdCrossoverStrategy",
            docstring="MACD Crossover Strategy.",
            parameters={"timeframe": "1h", "leverage": 1.0, "fast_period": 12, "slow_period": 26, "signal_period": 9},
            is_strategy=True,
        )

    @pytest.fixture
    def mock_config_file(self):
        """Get the path to the MACD strategy config file."""
        return str(Path("tests/strategies/macd_crossover/config.yml").absolute())

    @pytest.fixture
    def mock_strategy_config(self, mock_strategy_info):
        """Create a mock StrategyConfig object."""
        from qubx.utils.runner.configs import LiveConfig

        # Create exchange config
        exchange_config = ExchangeConfig(connector="ccxt", universe=["BTCUSDT"])

        # Create logging config
        logging_config = LoggingConfig(logger="CsvFileLogsWriter", position_interval="10Sec", portfolio_interval="5Min")

        # Create live config
        live_config = LiveConfig(
            exchanges={"BINANCE.UM": exchange_config},
            logging=logging_config,
        )

        # Create strategy config
        return StrategyConfig(
            strategy=mock_strategy_info.name,
            parameters=mock_strategy_info.parameters,
            live=live_config,
        )

    def mock_create_zip_archive(self, output_dir, release_dir, tag):
        """Mock version of _create_zip_archive that doesn't remove the directory."""
        file_path = os.path.join(output_dir, tag)
        shutil.make_archive(file_path, "zip", release_dir)
        # Don't remove the release_dir so we can check its contents

    @patch("subprocess.run")
    @patch("qubx.cli.release._create_zip_archive")
    @patch("qubx.cli.release._build_strategy_wheel")
    @patch("qubx.cli.release._generate_lock_file")
    def test_create_released_pack_basic(
        self,
        mock_generate_lock,
        mock_build_wheel,
        mock_zip_archive,
        mock_subprocess,
        temp_dir,
        mock_git_info,
        mock_strategy_info,
        mock_strategy_config,
        mock_config_file,
    ):
        """Test basic functionality of create_released_pack."""
        # Mock subprocess (for _bundle_source_overrides)
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock the zip archive creation to not delete the directory
        mock_zip_archive.side_effect = self.mock_create_zip_archive

        # Mock wheel build to return a fake wheel name
        mock_build_wheel.return_value = "macd_crossover-0.1.0-cp312-cp312-linux_x86_64.whl"

        # Mock lock file generation (no actual uv lock needed in test)
        mock_generate_lock.return_value = None

        # Get project root using the find_pyproject_root function
        project_root = find_pyproject_root(mock_strategy_info.path)

        # Create a StrategyInfo instance for testing
        strategy_info = StrategyInfo(
            name=mock_strategy_info.name, classes=[mock_strategy_info], config=mock_strategy_config
        )

        # Call the function
        create_released_pack(
            stg_info=strategy_info,
            git_info=mock_git_info,
            pyproject_root=project_root,
            output_dir=temp_dir,
            config_file=mock_config_file,
        )

        # Check that the zip file was created
        zip_path = os.path.join(temp_dir, mock_git_info.tag + ".zip")
        assert os.path.exists(zip_path), f"Zip file not created at {zip_path}"

        # Check that the release directory was created and not removed
        release_dir = os.path.join(temp_dir, mock_git_info.tag)
        assert os.path.exists(release_dir), f"Release directory not created at {release_dir}"

        # Check that the metadata file was created
        metadata_path = os.path.join(release_dir, f"{mock_strategy_info.name}.info")
        assert os.path.exists(metadata_path), f"Metadata file not created at {metadata_path}"

        # Check that the config file was created
        config_path = os.path.join(release_dir, "config.yml")
        assert os.path.exists(config_path), f"Config file not created at {config_path}"

        # Check that pyproject.toml was generated (not copied)
        pyproject_path = os.path.join(release_dir, "pyproject.toml")
        assert os.path.exists(pyproject_path), f"pyproject.toml not created at {pyproject_path}"

        # Verify the wheel build was called
        mock_build_wheel.assert_called_once()

    @patch("qubx.cli.release.process_git_repo")
    @patch("qubx.cli.release._create_zip_archive")
    @patch("subprocess.run")
    @patch("qubx.cli.release.release_strategy")
    def test_release_cli_command(
        self, mock_release_strategy, mock_subprocess, mock_zip_archive, mock_process_git, temp_dir
    ):
        """Test the full flow of the release command using CliRunner."""
        # Setup mocks
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_zip_archive.side_effect = self.mock_create_zip_archive

        # Mock the git repo processing to avoid actual git operations
        from datetime import datetime

        mock_git_info = ReleaseInfo(
            tag="R_MacdCrossoverStrategy_20240101000000",
            commit="abcdef1234567890",
            user="test_user",
            time=datetime.now(),
            commited_files=["tests/strategies/macd_crossover/models/macd_crossover.py"],
        )
        mock_process_git.return_value = mock_git_info

        # Create a runner
        from qubx.cli.commands import release as release_command

        runner = CliRunner()

        # Create a temporary directory for the output
        output_dir = os.path.join(temp_dir, "releases")
        os.makedirs(output_dir, exist_ok=True)

        # Run the command
        with patch("qubx.cli.release.makedirs", return_value=os.path.join(temp_dir, mock_git_info.tag)):
            result = runner.invoke(
                release_command,
                [
                    "--config",
                    "tests/strategies/macd_crossover/config.yml",
                    "--output-dir",
                    output_dir,
                    "--tag",
                    "test",
                    "--message",
                    "Test release",
                    "tests/strategies",
                ],
            )

        # Check that the command executed successfully
        assert result.exit_code == 0, f"Command failed with: {result.output}"

        # Verify that release_strategy was called with the correct arguments
        mock_release_strategy.assert_called_once()
        args, kwargs = mock_release_strategy.call_args

        # Check the keyword arguments passed to release_strategy
        assert kwargs.get("directory").endswith("tests" + os.sep + "strategies"), "Directory not passed correctly"
        assert kwargs.get("tag") == "test", "Tag not passed correctly"
        assert kwargs.get("message") == "Test release", "Message not passed correctly"
        assert kwargs.get("output_dir") == output_dir, "Output directory not passed correctly"
        assert kwargs.get("commit") is False, "Commit flag not passed correctly"


class TestBundleSourceOverrides:
    """Verify [tool.uv.sources] git source bundling — including monorepo `subdirectory`.

    `uv build` refuses to run inside uv's own cache dir (where a cache-hit git
    checkout lives), so the git-source path must build from a fresh temp copy
    of the checkout rather than the checkout itself (see issue #398 follow-up).
    """

    def _make_pyproject(self, *, subdirectory: str | None = None) -> dict:
        source: dict = {
            "git": "https://github.com/example/monorepo.git",
            "tag": "pkg/v0.2.0",
        }
        if subdirectory is not None:
            source["subdirectory"] = subdirectory
        return {"tool": {"uv": {"sources": {"sample-pkg": source}}}}

    @staticmethod
    def _build_tmp_root(cwd: str) -> Path:
        """Walk up from the build cwd to the `qubx-release-build-*` temp root."""
        for parent in Path(cwd).parents:
            if parent.name.startswith("qubx-release-build-"):
                return parent
        raise AssertionError(f"cwd {cwd!r} was not built under a qubx-release-build- temp dir")

    @patch("qubx.cli.release._find_uv_git_checkout")
    @patch("subprocess.run")
    def test_git_source_with_subdirectory_builds_from_subdir(self, mock_run, mock_find_checkout, tmp_path):
        """Git source with `subdirectory` must run `uv build` from a temp copy's <subdirectory>."""
        checkout_root = tmp_path / "cache" / "deadbeef"
        (checkout_root / "qubx-xdata").mkdir(parents=True)
        (checkout_root / "qubx-xdata" / "pyproject.toml").write_text('[project]\nname = "sample-pkg"\n')
        mock_find_checkout.return_value = str(checkout_root)

        # The temp copy is cleaned up in a `finally` before _bundle_source_overrides
        # returns, so inspect it from inside the mocked build call itself.
        seen: dict = {}

        def fake_run(cmd, cwd=None, **kwargs):
            seen["cwd"] = cwd
            seen["pyproject_copied"] = os.path.isfile(os.path.join(cwd, "pyproject.toml"))
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _bundle_source_overrides(
            pyproject_data=self._make_pyproject(subdirectory="qubx-xdata"),
            pyproject_root=str(tmp_path),
            release_dir=str(release_dir),
            required_packages={"sample-pkg"},
            lock_versions={"sample_pkg": "0.2.0"},
            git_commits={"sample_pkg": "deadbeefcafebabe"},
        )

        assert mock_run.called, "uv build should be invoked for the git source"

        # Must NOT build from the original uv-cache checkout path.
        assert seen["cwd"] != str(checkout_root / "qubx-xdata")
        assert seen["cwd"].endswith(os.sep + "qubx-xdata")

        # Temp copy must actually contain the subdir's contents (a real copy,
        # not just a matching path).
        assert seen["pyproject_copied"] is True

    @patch("qubx.cli.release._find_uv_git_checkout")
    @patch("subprocess.run")
    def test_git_source_without_subdirectory_builds_from_root_copy(self, mock_run, mock_find_checkout, tmp_path):
        """Without `subdirectory`, `uv build` must run from a temp copy of the checkout root."""
        checkout_root = tmp_path / "cache" / "deadbeef"
        checkout_root.mkdir(parents=True)
        (checkout_root / "pyproject.toml").write_text('[project]\nname = "sample-pkg"\n')
        mock_find_checkout.return_value = str(checkout_root)

        seen: dict = {}

        def fake_run(cmd, cwd=None, **kwargs):
            seen["cwd"] = cwd
            seen["pyproject_copied"] = os.path.isfile(os.path.join(cwd, "pyproject.toml"))
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _bundle_source_overrides(
            pyproject_data=self._make_pyproject(subdirectory=None),
            pyproject_root=str(tmp_path),
            release_dir=str(release_dir),
            required_packages={"sample-pkg"},
            lock_versions={"sample_pkg": "0.2.0"},
            git_commits={"sample_pkg": "deadbeefcafebabe"},
        )

        assert seen["cwd"] != str(checkout_root)
        assert seen["pyproject_copied"] is True

    @patch("qubx.cli.release._find_uv_git_checkout")
    @patch("subprocess.run")
    def test_git_source_build_copies_git_metadata_and_cleans_up(self, mock_run, mock_find_checkout, tmp_path):
        """`.git` must be copied (hatch-vcs-style build backends need it), and the temp
        copy must be removed again once the build finishes."""
        checkout_root = tmp_path / "cache" / "deadbeef"
        checkout_root.mkdir(parents=True)
        (checkout_root / "pyproject.toml").write_text('[project]\nname = "sample-pkg"\n')
        (checkout_root / ".git").mkdir()
        (checkout_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        mock_find_checkout.return_value = str(checkout_root)

        # copytree runs for real (only subprocess.run for the build itself is
        # mocked) — capture whether .git made it into the copy at the moment
        # `uv build` would have run, before the try/finally cleans it up.
        seen_git_dir_present = {}

        def fake_run(cmd, cwd=None, **kwargs):
            seen_git_dir_present["value"] = os.path.isdir(os.path.join(cwd, ".git"))
            seen_git_dir_present["head"] = os.path.isfile(os.path.join(cwd, ".git", "HEAD"))
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _bundle_source_overrides(
            pyproject_data=self._make_pyproject(subdirectory=None),
            pyproject_root=str(tmp_path),
            release_dir=str(release_dir),
            required_packages={"sample-pkg"},
            lock_versions={"sample_pkg": "0.2.0"},
            git_commits={"sample_pkg": "deadbeefcafebabe"},
        )

        assert seen_git_dir_present.get("value") is True, ".git must be present in the temp build copy"
        assert seen_git_dir_present.get("head") is True

        # The source checkout itself must be untouched.
        assert os.path.isdir(checkout_root / ".git")

        # The temp build root must be cleaned up (try/finally) after the call
        # returns.
        kwargs = mock_run.call_args.kwargs
        build_tmp_root = self._build_tmp_root(kwargs["cwd"])
        assert not build_tmp_root.exists(), f"{build_tmp_root} should have been removed after the build"

    @patch("qubx.cli.release._find_uv_git_checkout")
    @patch("subprocess.run")
    def test_git_source_build_failure_still_cleans_up_temp_copy(self, mock_run, mock_find_checkout, tmp_path):
        """A failed build must not leak the temp copy."""
        import subprocess as subprocess_mod

        checkout_root = tmp_path / "cache" / "deadbeef"
        checkout_root.mkdir(parents=True)
        (checkout_root / "pyproject.toml").write_text('[project]\nname = "sample-pkg"\n')
        mock_find_checkout.return_value = str(checkout_root)
        mock_run.side_effect = subprocess_mod.CalledProcessError(1, "uv build", stderr="boom")

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        bundled = _bundle_source_overrides(
            pyproject_data=self._make_pyproject(subdirectory=None),
            pyproject_root=str(tmp_path),
            release_dir=str(release_dir),
            required_packages={"sample-pkg"},
            lock_versions={"sample_pkg": "0.2.0"},
            git_commits={"sample_pkg": "deadbeefcafebabe"},
        )

        assert bundled == []
        kwargs = mock_run.call_args.kwargs
        build_tmp_root = self._build_tmp_root(kwargs["cwd"])
        assert not build_tmp_root.exists()


class TestReleaseSourceConfigSubdirectory:
    """Tests for the optional `subdirectory` field on ``ReleaseSourceConfig``.

    The field lets monorepo / uv-workspace source repos point ``qubx release
    --from-sources`` at the workspace member that owns the strategy's
    pyproject.toml.
    """

    def test_subdirectory_optional_defaults_to_none(self):
        """Existing single-package configs (no subdirectory) parse unchanged."""
        cfg = ReleaseSourceConfig(repo="xLydianSoftware/foo", ref="main")
        assert cfg.subdirectory is None

    def test_subdirectory_accepts_relative_path(self):
        """Relative subpath is preserved verbatim after normalisation."""
        cfg = ReleaseSourceConfig(
            repo="xLydianSoftware/exchanges",
            ref="main",
            subdirectory="e2e-driver",
        )
        assert cfg.subdirectory == "e2e-driver"

    def test_subdirectory_accepts_nested_relative_path(self):
        cfg = ReleaseSourceConfig(
            repo="xLydianSoftware/exchanges",
            ref="main",
            subdirectory="packages/strategy",
        )
        # normpath uses os.sep; on POSIX this is "/", which is what we expect.
        assert cfg.subdirectory == os.path.normpath("packages/strategy")

    def test_subdirectory_normalises_trailing_slash(self):
        """`e2e-driver/` should be stored as `e2e-driver` via normpath."""
        cfg = ReleaseSourceConfig(
            repo="xLydianSoftware/exchanges",
            ref="main",
            subdirectory="e2e-driver/",
        )
        assert cfg.subdirectory == "e2e-driver"

    def test_subdirectory_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="relative"):
            ReleaseSourceConfig(
                repo="xLydianSoftware/exchanges",
                ref="main",
                subdirectory="/abs/path",
            )

    def test_subdirectory_rejects_dotdot_escape(self):
        with pytest.raises(ValueError, match=r"\.\."):
            ReleaseSourceConfig(
                repo="xLydianSoftware/exchanges",
                ref="main",
                subdirectory="../escape",
            )

    def test_subdirectory_rejects_nested_dotdot_escape(self):
        with pytest.raises(ValueError, match=r"\.\."):
            ReleaseSourceConfig(
                repo="xLydianSoftware/exchanges",
                ref="main",
                subdirectory="ok/../../escape",
            )


class TestWorkspaceHelpers:
    """Tests for `_find_uv_workspace_root` and `_find_workspace_member_for_package`."""

    @staticmethod
    def _write_workspace_root(tmp_path: Path, members: list[str]) -> None:
        members_repr = ", ".join(f'"{m}"' for m in members)
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "ws-root"\nversion = "0.0.0"\n\n[tool.uv.workspace]\nmembers = [{members_repr}]\n'
        )

    @staticmethod
    def _write_member(member_dir: Path, project_name: str) -> None:
        member_dir.mkdir(parents=True, exist_ok=True)
        (member_dir / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\nversion = "0.1.0"\n')

    def test_find_uv_workspace_root_walks_up_from_member(self, tmp_path: Path):
        self._write_workspace_root(tmp_path, ["pkg-a"])
        member = tmp_path / "pkg-a"
        self._write_member(member, "pkg-a")

        # Walk up from the member directory
        result = _find_uv_workspace_root(str(member))
        assert result == str(tmp_path.resolve())

    def test_find_uv_workspace_root_walks_up_from_nested_subdir(self, tmp_path: Path):
        self._write_workspace_root(tmp_path, ["pkg-a"])
        member = tmp_path / "pkg-a"
        self._write_member(member, "pkg-a")
        nested = member / "src" / "pkg_a"
        nested.mkdir(parents=True)

        result = _find_uv_workspace_root(str(nested))
        assert result == str(tmp_path.resolve())

    def test_find_uv_workspace_root_returns_none_when_no_workspace(self, tmp_path: Path):
        # A bare pyproject.toml without [tool.uv.workspace] should not match.
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "standalone"\nversion = "0.1.0"\n')
        sub = tmp_path / "src"
        sub.mkdir()
        result = _find_uv_workspace_root(str(sub))
        assert result is None

    def test_find_workspace_member_resolves_by_exact_name(self, tmp_path: Path):
        self._write_workspace_root(tmp_path, ["pkg-a"])
        member = tmp_path / "pkg-a"
        self._write_member(member, "pkg-a")

        result = _find_workspace_member_for_package(str(tmp_path), "pkg-a")
        assert result == str(member.resolve())

    def test_find_workspace_member_handles_underscore_normalisation(self, tmp_path: Path):
        self._write_workspace_root(tmp_path, ["pkg-a"])
        member = tmp_path / "pkg-a"
        self._write_member(member, "pkg-a")

        # Search with underscored name should still resolve
        result = _find_workspace_member_for_package(str(tmp_path), "pkg_a")
        assert result == str(member.resolve())

    def test_find_workspace_member_returns_none_for_unknown_package(self, tmp_path: Path):
        self._write_workspace_root(tmp_path, ["pkg-a"])
        self._write_member(tmp_path / "pkg-a", "pkg-a")

        result = _find_workspace_member_for_package(str(tmp_path), "missing-pkg")
        assert result is None

    def test_find_workspace_member_handles_glob_patterns(self, tmp_path: Path):
        self._write_workspace_root(tmp_path, ["pkgs/*"])
        member = tmp_path / "pkgs" / "sub"
        self._write_member(member, "sub")

        result = _find_workspace_member_for_package(str(tmp_path), "sub")
        assert result == str(member.resolve())

    def test_find_workspace_member_returns_none_when_no_members_listed(self, tmp_path: Path):
        # workspace section but empty members list
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "ws-root"\nversion = "0.0.0"\n\n[tool.uv.workspace]\nmembers = []\n'
        )
        result = _find_workspace_member_for_package(str(tmp_path), "anything")
        assert result is None


class TestBundleSourceOverridesWorkspace:
    """Verify [tool.uv.sources] `workspace = true` bundling."""

    @patch("qubx.cli.release._version_exists_on_pypi", return_value=False)
    @patch("subprocess.run")
    def test_workspace_source_builds_from_member_dir(self, mock_run, _mock_pypi, tmp_path: Path):
        # Set up a workspace with a member that owns the package being bundled.
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "ws-root"\nversion = "0.0.0"\n\n[tool.uv.workspace]\nmembers = ["pkg-a", "consumer"]\n'
        )

        member_dir = tmp_path / "pkg-a"
        member_dir.mkdir()
        (member_dir / "pyproject.toml").write_text('[project]\nname = "pkg-a"\nversion = "0.2.0"\n')

        # Consumer is the strategy's own pyproject (the pyproject_root passed to
        # _bundle_source_overrides). It declares pkg-a as a workspace source.
        consumer_dir = tmp_path / "consumer"
        consumer_dir.mkdir()
        (consumer_dir / "pyproject.toml").write_text('[project]\nname = "consumer"\nversion = "0.1.0"\n')

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        bundled = _bundle_source_overrides(
            pyproject_data={"tool": {"uv": {"sources": {"pkg-a": {"workspace": True}}}}},
            pyproject_root=str(consumer_dir),
            release_dir=str(release_dir),
            required_packages={"pkg-a"},
            lock_versions={"pkg_a": "0.2.0"},
            git_commits={},
        )

        assert bundled == ["pkg-a"]
        assert mock_run.called
        kwargs = mock_run.call_args.kwargs
        assert kwargs["cwd"] == str(member_dir.resolve())
        # uv build invoked with wheel + out-dir
        args = mock_run.call_args.args[0]
        assert args[:4] == ["uv", "build", "--wheel", "."]

    @patch("qubx.cli.release._version_exists_on_pypi", return_value=False)
    @patch("subprocess.run")
    def test_workspace_source_skips_when_no_workspace_root(self, mock_run, _mock_pypi, tmp_path: Path):
        # A consumer with no surrounding workspace root.
        consumer_dir = tmp_path / "consumer"
        consumer_dir.mkdir()
        (consumer_dir / "pyproject.toml").write_text('[project]\nname = "consumer"\nversion = "0.1.0"\n')

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        bundled = _bundle_source_overrides(
            pyproject_data={"tool": {"uv": {"sources": {"pkg-a": {"workspace": True}}}}},
            pyproject_root=str(consumer_dir),
            release_dir=str(release_dir),
            required_packages={"pkg-a"},
            lock_versions={"pkg_a": "0.2.0"},
            git_commits={},
        )

        assert bundled == []
        assert not mock_run.called

    @patch("qubx.cli.release._version_exists_on_pypi", return_value=True)
    @patch("subprocess.run")
    def test_workspace_source_skips_when_version_on_public_pypi(self, mock_run, _mock_pypi, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "ws-root"\nversion = "0.0.0"\n\n[tool.uv.workspace]\nmembers = ["pkg-a", "consumer"]\n'
        )
        member_dir = tmp_path / "pkg-a"
        member_dir.mkdir()
        (member_dir / "pyproject.toml").write_text('[project]\nname = "pkg-a"\nversion = "0.2.0"\n')
        consumer_dir = tmp_path / "consumer"
        consumer_dir.mkdir()
        (consumer_dir / "pyproject.toml").write_text('[project]\nname = "consumer"\nversion = "0.1.0"\n')

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        bundled = _bundle_source_overrides(
            pyproject_data={"tool": {"uv": {"sources": {"pkg-a": {"workspace": True}}}}},
            pyproject_root=str(consumer_dir),
            release_dir=str(release_dir),
            required_packages={"pkg-a"},
            lock_versions={"pkg_a": "0.2.0"},
            git_commits={},
        )

        # Found on PyPI → resolved from registry, not bundled.
        assert bundled == []
        assert not mock_run.called


class TestResolveSourceLockfile:
    """Tests for `_resolve_source_lockfile` workspace-aware lockfile resolution."""

    def test_single_package_uses_member_local_lock(self, tmp_path: Path):
        # Single-package project: uv.lock at pyproject_root, no workspace.
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "single-pkg"\nversion = "0.1.0"\n')
        member_lock = tmp_path / "uv.lock"
        member_lock.write_text("# placeholder\n")

        result = _resolve_source_lockfile(str(tmp_path))
        assert result == str(member_lock)

    def test_workspace_member_uses_workspace_root_lock(self, tmp_path: Path):
        # Workspace root has uv.lock; member dir does NOT.
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "ws-root"\nversion = "0.0.0"\n\n[tool.uv.workspace]\nmembers = ["pkg-a"]\n'
        )
        ws_lock = tmp_path / "uv.lock"
        ws_lock.write_text("# placeholder\n")

        member_dir = tmp_path / "pkg-a"
        member_dir.mkdir()
        (member_dir / "pyproject.toml").write_text('[project]\nname = "pkg-a"\nversion = "0.1.0"\n')

        result = _resolve_source_lockfile(str(member_dir))
        # Should pick the workspace-root lock, not the missing member lock.
        assert result == str(ws_lock)
        assert not (member_dir / "uv.lock").exists()

    def test_workspace_member_falls_through_to_generate_then_finds_workspace_lock(self, tmp_path: Path):
        # Both locks missing initially. `uv lock` invocation is mocked to
        # simulate uv writing the lock at the workspace root (typical
        # behaviour for a workspace member).
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "ws-root"\nversion = "0.0.0"\n\n[tool.uv.workspace]\nmembers = ["pkg-a"]\n'
        )

        member_dir = tmp_path / "pkg-a"
        member_dir.mkdir()
        (member_dir / "pyproject.toml").write_text('[project]\nname = "pkg-a"\nversion = "0.1.0"\n')

        ws_lock = tmp_path / "uv.lock"

        def fake_generate(pyproject_root: str) -> None:
            # Simulate `uv lock` running from the member: writes lock at the
            # workspace root, not the member dir.
            ws_lock.write_text("# placeholder\n")

        with patch("qubx.cli.release._generate_lock_file", side_effect=fake_generate) as mock_gen:
            result = _resolve_source_lockfile(str(member_dir))

        assert mock_gen.called
        # Resolution should still land on the workspace-root lock after generation.
        assert result == str(ws_lock)
        assert not (member_dir / "uv.lock").exists()

    def test_single_package_missing_lock_generates_and_returns_member_lock(self, tmp_path: Path):
        # No workspace. Initial lock missing → generate writes member-local lock.
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "single-pkg"\nversion = "0.1.0"\n')
        member_lock = tmp_path / "uv.lock"

        def fake_generate(pyproject_root: str) -> None:
            member_lock.write_text("# placeholder\n")

        with patch("qubx.cli.release._generate_lock_file", side_effect=fake_generate) as mock_gen:
            result = _resolve_source_lockfile(str(tmp_path))

        assert mock_gen.called
        assert result == str(member_lock)


class TestLockConstraintDependencies:
    """Tests for `_lock_constraint_dependencies` (issue #398).

    Covers the registry-only rule (issue #398 follow-up): git/editable/path
    sources are pinned by ref or bundled wheel already, and an `==` constraint
    for them is unsatisfiable since uv can only satisfy `==` from an index.
    """

    def test_single_version_registry_packages_become_constraints(self, tmp_path: Path):
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(
            "\n".join(
                [
                    '[[package]]',
                    'name = "foo"',
                    'version = "1.2.3"',
                    'source = { registry = "https://pypi.org/simple" }',
                    "",
                    '[[package]]',
                    'name = "bar"',
                    'version = "0.5.0"',
                    'source = { registry = "https://pypi.org/simple" }',
                    "",
                    # "baz" appears with two versions (e.g. a platform-forked
                    # resolution) — must be skipped entirely.
                    '[[package]]',
                    'name = "baz"',
                    'version = "1.0.0"',
                    'source = { registry = "https://pypi.org/simple" }',
                    "",
                    '[[package]]',
                    'name = "baz"',
                    'version = "2.0.0"',
                    'source = { registry = "https://pypi.org/simple" }',
                ]
            )
        )

        result = _lock_constraint_dependencies(str(lock_path))

        assert result == ["bar==0.5.0", "foo==1.2.3"]

    def test_git_sourced_package_excluded(self, tmp_path: Path):
        """A git-sourced dep (e.g. quantkit pinned by tag) must not get an == constraint."""
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(
            "\n".join(
                [
                    '[[package]]',
                    'name = "foo"',
                    'version = "1.2.3"',
                    'source = { registry = "https://pypi.org/simple" }',
                    "",
                    '[[package]]',
                    'name = "quantkit"',
                    'version = "4.1.1"',
                    'source = { git = "https://github.com/example/quantkit?tag=v4.1.1#deadbeef" }',
                ]
            )
        )

        result = _lock_constraint_dependencies(str(lock_path))

        assert result == ["foo==1.2.3"]

    def test_editable_sourced_package_excluded(self, tmp_path: Path):
        """An editable/local dep (e.g. the strategy package itself) must not get an == constraint."""
        lock_path = tmp_path / "uv.lock"
        lock_path.write_text(
            "\n".join(
                [
                    '[[package]]',
                    'name = "foo"',
                    'version = "1.2.3"',
                    'source = { registry = "https://pypi.org/simple" }',
                    "",
                    '[[package]]',
                    'name = "factors"',
                    'version = "0.1.0"',
                    'source = { editable = "." }',
                ]
            )
        )

        result = _lock_constraint_dependencies(str(lock_path))

        assert result == ["foo==1.2.3"]

    def test_missing_lock_file_returns_empty_list(self, tmp_path: Path):
        missing_path = tmp_path / "does-not-exist" / "uv.lock"
        assert _lock_constraint_dependencies(str(missing_path)) == []


class TestGenerateReleasePyprojectConstraints:
    """Tests for `constraint_dependencies` wiring in `_generate_release_pyproject` (issue #398)."""

    def test_constraint_dependencies_written_to_pyproject(self, tmp_path: Path):
        import toml

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _generate_release_pyproject(
            release_dir=str(release_dir),
            strategy_wheel_name=None,
            has_strategy_code=False,
            external_deps=["quantkit>=1.3.0"],
            constraint_dependencies=["foo==1.2.3"],
        )

        with open(release_dir / "pyproject.toml") as f:
            data = toml.load(f)

        assert data["tool"]["uv"]["constraint-dependencies"] == ["foo==1.2.3"]

    def test_no_constraint_dependencies_key_absent(self, tmp_path: Path):
        import toml

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _generate_release_pyproject(
            release_dir=str(release_dir),
            strategy_wheel_name=None,
            has_strategy_code=False,
            external_deps=["quantkit>=1.3.0"],
            constraint_dependencies=None,
        )

        with open(release_dir / "pyproject.toml") as f:
            data = toml.load(f)

        assert "constraint-dependencies" not in data["tool"]["uv"]

    def test_empty_constraint_dependencies_key_absent(self, tmp_path: Path):
        import toml

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _generate_release_pyproject(
            release_dir=str(release_dir),
            strategy_wheel_name=None,
            has_strategy_code=False,
            external_deps=["quantkit>=1.3.0"],
            constraint_dependencies=[],
        )

        with open(release_dir / "pyproject.toml") as f:
            data = toml.load(f)

        assert "constraint-dependencies" not in data["tool"]["uv"]


class TestGenerateReleasePyprojectDeployTarget:
    """The wrapper only ever runs inside the deploy image (python:3.12-slim, linux/amd64).

    `uv lock` must resolve for exactly that target, not universally — otherwise
    it fails (or silently drops) on Python versions/platforms the cp312-linux
    bundled wheels were never built for.
    """

    def test_requires_python_pinned_to_deploy_image_version(self, tmp_path: Path):
        import toml

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _generate_release_pyproject(
            release_dir=str(release_dir),
            strategy_wheel_name=None,
            has_strategy_code=False,
            external_deps=["quantkit>=1.3.0"],
        )

        with open(release_dir / "pyproject.toml") as f:
            data = toml.load(f)

        assert data["project"]["requires-python"] == "==3.12.*"

    def test_environments_scoped_to_linux_x86_64(self, tmp_path: Path):
        import toml

        release_dir = tmp_path / "release"
        release_dir.mkdir()

        _generate_release_pyproject(
            release_dir=str(release_dir),
            strategy_wheel_name=None,
            has_strategy_code=False,
            external_deps=["quantkit>=1.3.0"],
        )

        with open(release_dir / "pyproject.toml") as f:
            data = toml.load(f)

        assert data["tool"]["uv"]["environments"] == ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
