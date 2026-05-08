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
from qubx.cli.release import ReleaseInfo, StrategyInfo, _bundle_source_overrides, create_released_pack
from qubx.core.series import OHLCV
from qubx.data import CsvStorage
from qubx.utils.runner.configs import ExchangeConfig, LoggingConfig, StrategyConfig

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
    """Verify [tool.uv.sources] git source bundling — including monorepo `subdirectory`."""

    def _make_pyproject(self, *, subdirectory: str | None = None) -> dict:
        source: dict = {
            "git": "https://github.com/example/monorepo.git",
            "tag": "pkg/v0.2.0",
        }
        if subdirectory is not None:
            source["subdirectory"] = subdirectory
        return {"tool": {"uv": {"sources": {"sample-pkg": source}}}}

    @patch("qubx.cli.release._find_uv_git_checkout")
    @patch("subprocess.run")
    def test_git_source_with_subdirectory_builds_from_subdir(
        self, mock_run, mock_find_checkout, tmp_path
    ):
        """Git source with `subdirectory` must run `uv build` from <checkout>/<subdirectory>."""
        checkout_root = tmp_path / "cache" / "deadbeef"
        checkout_root.mkdir(parents=True)
        mock_find_checkout.return_value = str(checkout_root)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

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
        kwargs = mock_run.call_args.kwargs
        expected_cwd = str(checkout_root / "qubx-xdata")
        assert kwargs["cwd"] == expected_cwd, (
            f"Expected build cwd={expected_cwd!r}, got {kwargs['cwd']!r}"
        )

    @patch("qubx.cli.release._find_uv_git_checkout")
    @patch("subprocess.run")
    def test_git_source_without_subdirectory_builds_from_root(
        self, mock_run, mock_find_checkout, tmp_path
    ):
        """Without `subdirectory`, `uv build` must run from the checkout root (unchanged)."""
        checkout_root = tmp_path / "cache" / "deadbeef"
        checkout_root.mkdir(parents=True)
        mock_find_checkout.return_value = str(checkout_root)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

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

        kwargs = mock_run.call_args.kwargs
        assert kwargs["cwd"] == str(checkout_root)
