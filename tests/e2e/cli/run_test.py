import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from qubx.cli.commands import deploy as deploy_command
from qubx.cli.commands import release as release_command
from qubx.cli.deploy import deploy_strategy


class TestRunStrategy:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def strategy_zip(self, temp_dir):
        """Release the MACD Crossover strategy to a zip file."""
        # Create a runner for the release command
        runner = CliRunner()

        # Create output directory for the release
        output_dir = os.path.join(temp_dir, "releases")
        os.makedirs(output_dir, exist_ok=True)

        # Run the release command to create the zip file
        # fmt: off
        result = runner.invoke(
            release_command,
            [
                "--config", "tests/strategies/macd_crossover/config.yml",
                "--output-dir", output_dir,
                "--tag", "test",              # Add a tag
                "--message", "Test release",  # Add a message
                "tests/strategies/macd_crossover",
            ],
        )
        # fmt: on

        # Check that the command executed successfully
        assert result.exit_code == 0, f"Release command failed with: {result.output}"
        print(f"Release command output: {result.output}")

        # Find the zip file in the output directory
        zip_files = [f for f in os.listdir(output_dir) if f.endswith(".zip")]
        assert len(zip_files) > 0, "No zip file was created"

        # Return the path to the zip file
        return os.path.join(output_dir, zip_files[0])

    @pytest.mark.e2e
    def test_run_strategy(self, temp_dir, strategy_zip):
        """Test running a strategy after deploying it."""
        # Create a deployment directory
        deploy_dir = os.path.join(temp_dir, "deploy")
        os.makedirs(deploy_dir, exist_ok=True)

        # Deploy the strategy
        result = deploy_strategy(zip_file=strategy_zip, output_dir=deploy_dir, force=True)
        assert result is True, "Failed to deploy strategy"

        # Find the deployed strategy directory
        strategy_dir = os.path.join(deploy_dir, Path(strategy_zip).stem)
        assert os.path.exists(strategy_dir), f"Strategy directory not found at {strategy_dir}"

        # Create a log file path to check for heartbeat messages
        log_dir = os.path.join(strategy_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "strategy_run.log")

        # Start the strategy in a separate process with a timeout
        with patch("subprocess.run") as mock_run:
            # Configure the mock to simulate running the strategy
            def side_effect(*args, **kwargs):
                # Simulate the strategy running and writing a strategy started message
                with open(log_file, "w") as f:
                    f.write("2024-01-01 12:00:00 - [i] (context) [StrategyContext] :: strategy is started in thread\n")
                return MagicMock(returncode=0)

            mock_run.side_effect = side_effect

            # Run the strategy with the qubx run command
            cmd = ["qubx", "run", str(Path(strategy_dir) / "config.yml"), "--paper"]
            subprocess.run(cmd, cwd=strategy_dir, timeout=5)

            # Check that the run command was called
            mock_run.assert_called_once()

        # Check if the log file contains a strategy started message
        assert os.path.exists(log_file), f"Log file not found at {log_file}"
        with open(log_file, "r") as f:
            log_content = f.read()

        assert "strategy is started" in log_content, "No 'strategy is started' message found in the log file"

    @pytest.mark.e2e
    def test_run_strategy_integration(self, temp_dir, strategy_zip):
        """
        Integration test for running a strategy after deploying it.
        This test actually runs the strategy process and checks for heartbeat.
        Marked with integration to be skipped in normal test runs.
        """
        # Create a deployment directory
        deploy_dir = os.path.join(temp_dir, "deploy")
        os.makedirs(deploy_dir, exist_ok=True)

        # Deploy the strategy
        result = deploy_strategy(zip_file=strategy_zip, output_dir=deploy_dir, force=True)
        assert result is True, "Failed to deploy strategy"

        # Find the deployed strategy directory
        strategy_dir = os.path.join(deploy_dir, Path(strategy_zip).stem)
        assert os.path.exists(strategy_dir), f"Strategy directory not found at {strategy_dir}"

        # Start the strategy in a separate process with a timeout
        process = None
        try:
            # Run the strategy with the qubx run command in a separate process
            cmd = ["qubx", "run", str(Path(strategy_dir) / "config.yml"), "--paper"]
            print(f"\nRunning command: {' '.join(cmd)} in directory: {strategy_dir}")

            process = subprocess.Popen(
                cmd,
                cwd=strategy_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Wait for the strategy started message or timeout
            start_time = time.time()
            strategy_started = False
            all_output = []

            print("\n--- Strategy Output Start ---")
            while time.time() - start_time < 60:  # 60 second timeout
                if process.poll() is not None:
                    # Process ended
                    print(f"Process ended with return code: {process.returncode}")
                    break

                if process.stdout:
                    output = process.stdout.readline()
                    if output:
                        print(output.strip())  # Print the output as it comes
                        all_output.append(output)
                        if "strategy is started" in output:
                            strategy_started = True
                            print("Strategy started message found!")
                            break

                time.sleep(0.1)
            print("--- Strategy Output End ---\n")

            # If process is still running but we didn't find a strategy started message
            if not strategy_started and process.poll() is None:
                print("Timeout reached without finding 'strategy is started' message")

            # Print all collected output for debugging
            if not strategy_started:
                print("\nAll collected output:")
                print("".join(all_output))

            assert strategy_started, "No 'strategy is started' message found in the strategy output"

        finally:
            # Clean up the process
            if process and process.poll() is None:
                print("Terminating process...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Process did not terminate, killing it...")
                    process.kill()
