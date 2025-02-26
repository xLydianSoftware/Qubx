import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

import qubx.pandaz.ta as pta
import tests.qubx.ta.utils_for_testing as test
from qubx.backtester.simulator import simulate
from qubx.cli.commands import release
from qubx.cli.misc import StrategyInfo, find_pyproject_root
from qubx.cli.release import ReleaseInfo, create_released_pack
from qubx.core.series import OHLCV
from qubx.data import loader
from qubx.data.readers import AsOhlcvSeries, CsvStorageDataReader
from qubx.utils.runner.configs import ExchangeConfig, LoggingConfig, StrategyConfig

# Add tests/strategies to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "tests" / "strategies"))

from tests.strategies.macd_crossover.indicators.macd import macd
from tests.strategies.macd_crossover.models.macd_crossover import MacdCrossoverStrategy


class TestMacdCrossoverSimulation:
    def test_macd_indicator(self):
        r = CsvStorageDataReader("tests/data/csv/")
        ohlc = r.read("SOLUSDT", start="2024-04-01", stop="+5h", transform=AsOhlcvSeries("1Min", "ms"))
        assert isinstance(ohlc, OHLCV)
        _macd = macd(ohlc.close).to_series().dropna()
        expected_macd = pta.macd(ohlc.close.pd()).dropna()
        assert test.N(_macd[-50:]) == expected_macd[-50:]

    def test_macd_crossover_simulation(self):
        ld = loader("BINANCE.UM", "1h", source="csv::tests/data/csv_1h/", n_jobs=1)
        test0 = simulate(
            MacdCrossoverStrategy(),
            ld,
            capital=100_000,
            instruments=["BINANCE.UM:BTCUSDT"],
            commissions="vip0_usdt",
            start="2023-06-01",
            stop="2023-06-10",
            debug="INFO",
            n_jobs=1,
        )
        sim = test0[0]
        assert len(sim.executions_log) > 1

    def test_release(self):
        # runner = CliRunner()
        # result = runner.invoke(release, ["--strategy", "test_strategy", "--tag", "test_tag", "--comment", "test_comment"])
        pass


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
            commited_files=["tests/strategies/macd_crossover/models/macd_crossover.py"],
        )

    @pytest.fixture
    def mock_strategy_info(self):
        """Create a mock StrategyInfo object for the MACD strategy."""
        # Get the actual path to the MACD strategy
        strategy_path = Path("tests/strategies/macd_crossover/models/macd_crossover.py").absolute()
        return StrategyInfo(
            path=str(strategy_path),
            name="MacdCrossoverStrategy",
            docstring="MACD Crossover Strategy.",
            parameters={"timeframe": "1h", "leverage": 1.0, "fast_period": 12, "slow_period": 26, "signal_period": 9},
        )

    @pytest.fixture
    def mock_strategy_config(self, mock_strategy_info):
        """Create a mock StrategyConfig object."""
        # Create exchange config
        exchange_config = ExchangeConfig(connector="ccxt", universe=["BTCUSDT"])

        # Create logging config
        logging_config = LoggingConfig(logger="CsvFileLogsWriter", position_interval="10Sec", portfolio_interval="5Min")

        # Create strategy config
        return StrategyConfig(
            strategy=mock_strategy_info.name,
            parameters=mock_strategy_info.parameters,
            exchanges={"BINANCE.UM": exchange_config},
            logging=logging_config,
        )

    def mock_create_zip_archive(self, output_dir, release_dir, tag):
        """Mock version of _create_zip_archive that doesn't remove the directory."""
        file_path = os.path.join(output_dir, tag)
        shutil.make_archive(file_path, "zip", release_dir)
        # Don't remove the release_dir so we can check its contents

    @patch("subprocess.run")
    @patch("qubx.cli.release._create_zip_archive")
    def test_create_released_pack_basic(
        self, mock_zip_archive, mock_subprocess, temp_dir, mock_git_info, mock_strategy_info, mock_strategy_config
    ):
        """Test basic functionality of create_released_pack."""
        # Mock the poetry lock command
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock the zip archive creation to not delete the directory
        mock_zip_archive.side_effect = self.mock_create_zip_archive

        # Get project root using the find_pyproject_root function
        project_root = find_pyproject_root(mock_strategy_info.path)

        # Call the function
        create_released_pack(
            stg_info=mock_strategy_info,
            git_info=mock_git_info,
            pyproject_root=project_root,
            output_dir=temp_dir,
            strategy_config=mock_strategy_config,
        )

        # Get the src_dir (basename of project_root)
        src_dir = os.path.basename(project_root)

        # Check that the zip file was created
        zip_path = os.path.join(temp_dir, mock_git_info.tag + ".zip")
        assert os.path.exists(zip_path), f"Zip file not created at {zip_path}"

        # Check that the release directory was created and not removed
        release_dir = os.path.join(temp_dir, mock_git_info.tag)
        assert os.path.exists(release_dir), f"Release directory not created at {release_dir}"

        # Check that the strategy file was copied
        strategy_rel_path = os.path.relpath(mock_strategy_info.path, project_root)
        strategy_dest_path = os.path.join(release_dir, src_dir, strategy_rel_path)
        assert os.path.exists(strategy_dest_path), f"Strategy file not copied to {strategy_dest_path}"

        # Check that the metadata file was created
        metadata_path = os.path.join(release_dir, f"{mock_strategy_info.name}.info")
        assert os.path.exists(metadata_path), f"Metadata file not created at {metadata_path}"

        # Check that the config file was created
        config_path = os.path.join(release_dir, f"{mock_strategy_info.name}.yml")
        assert os.path.exists(config_path), f"Config file not created at {config_path}"

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
                    "--strategy",
                    "MacdCrossoverStrategy",
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
        assert kwargs.get("directory").endswith("tests/strategies"), "Directory not passed correctly"
        assert kwargs.get("strategy_name") == "MacdCrossoverStrategy", "Strategy name not passed correctly"
        assert kwargs.get("tag") == "test", "Tag not passed correctly"
        assert kwargs.get("message") == "Test release", "Message not passed correctly"
        assert kwargs.get("output_dir") == output_dir, "Output directory not passed correctly"
        assert kwargs.get("commit") is False, "Commit flag not passed correctly"
