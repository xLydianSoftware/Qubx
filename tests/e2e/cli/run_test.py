"""E2E tests for the ``qubx run`` CLI: boot a real paper-mode strategy subprocess.

Paper mode uses the full live wiring (StrategyContext, ProcessingManager, CtrlChannel,
threads) with SimulatedConnector/SimulatedAccountManager and REAL market data via the
ccxt data provider — so these tests only need internet access to Binance, no credentials.
"""

import shutil
import signal
import subprocess
import time
from pathlib import Path
from threading import Thread

import pytest
from click.testing import CliRunner

from qubx.cli.commands import validate as validate_command

STRATEGY_SRC_DIR = Path(__file__).parents[2] / "strategies" / "macd_crossover"

STARTUP_TIMEOUT = 90.0
SHUTDOWN_TIMEOUT = 30.0


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    """Copy the MACD crossover strategy to a temp dir so the run leaves no artifacts in the repo."""
    target = tmp_path / "macd_crossover"
    shutil.copytree(STRATEGY_SRC_DIR, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


class TestRunCli:
    @pytest.mark.e2e
    def test_validate_config(self, monkeypatch: pytest.MonkeyPatch):
        """The shipped test config must parse against the current schema and its strategy must import."""
        monkeypatch.syspath_prepend(str(STRATEGY_SRC_DIR / "src"))
        result = CliRunner().invoke(validate_command, [str(STRATEGY_SRC_DIR / "config.yml")])
        assert result.exit_code == 0, f"Config validation failed:\n{result.output}"

    @pytest.mark.e2e
    def test_run_strategy_paper(self, strategy_dir: Path):
        """``qubx run config.yml --paper`` boots the strategy and shuts down cleanly on SIGINT."""
        cmd = ["qubx", "run", "config.yml", "--paper", "--no-color"]
        process = subprocess.Popen(
            cmd,
            cwd=strategy_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines: list[str] = []

        def _pump():
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)

        reader = Thread(target=_pump, daemon=True)
        reader.start()

        try:
            deadline = time.time() + STARTUP_TIMEOUT
            started = False
            while time.time() < deadline:
                if any("strategy is started" in line for line in lines):
                    started = True
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.2)

            output = "".join(lines)
            assert process.poll() is None, f"Process exited prematurely (rc={process.returncode}):\n{output}"
            assert started, f"No 'strategy is started' within {STARTUP_TIMEOUT}s:\n{output}"

            process.send_signal(signal.SIGINT)
            process.wait(timeout=SHUTDOWN_TIMEOUT)
            reader.join(timeout=5)

            output = "".join(lines)
            assert process.returncode == 0, f"Non-zero exit code {process.returncode} after SIGINT:\n{output}"
            assert "Traceback" not in output, f"Traceback in strategy output:\n{output}"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
