"""Test runner for connector verification — runs each test case via run_strategy."""

from __future__ import annotations

import multiprocessing
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from qubx.testing.connectors.assertions import AssertionResult, run_assertion
from qubx.testing.connectors.spec import ConnectorTestSpec, TestCaseSpec


TEST_RUNS_DIR = Path("test_runs")


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_sec: float
    assertion_results: list[AssertionResult]
    error: str | None = None
    total_events: int = 0
    log_dir: str | None = None


def run_test_suite(spec: ConnectorTestSpec, account_file: Path | None) -> list[TestResult]:
    """Run all test cases, each in a separate process."""
    # Create test run directory
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = TEST_RUNS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[TestResult] = []
    max_parallel = spec.settings.max_parallel

    # Run tests in batches of max_parallel
    for i in range(0, len(spec.tests), max_parallel):
        batch = spec.tests[i : i + max_parallel]
        batch_results = _run_batch(spec, batch, account_file, run_dir)
        results.extend(batch_results)

    return results


def _run_batch(
    spec: ConnectorTestSpec,
    test_cases: list[TestCaseSpec],
    account_file: Path | None,
    run_dir: Path,
) -> list[TestResult]:
    """Run a batch of test cases in parallel processes."""
    processes: list[tuple[TestCaseSpec, multiprocessing.Process, multiprocessing.Queue]] = []

    for test_case in test_cases:
        # Create per-test log directory
        test_log_dir = run_dir / test_case.name
        test_log_dir.mkdir(parents=True, exist_ok=True)

        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_run_single_test_process,
            args=(spec.connector, spec.exchange, test_case, account_file, result_queue, str(test_log_dir)),
            name=f"test-{test_case.name}",
        )
        processes.append((test_case, p, result_queue))
        p.start()

    results = []
    for test_case, p, result_queue in processes:
        # Wait with timeout
        from qubx.utils.time import to_timedelta

        duration_sec = to_timedelta(test_case.duration).total_seconds()
        warmup_sec = to_timedelta(test_case.warmup).total_seconds() if test_case.warmup else 0
        timeout = (duration_sec + warmup_sec + 60) * spec.settings.timeout_multiplier

        p.join(timeout=timeout)

        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            results.append(
                TestResult(
                    name=test_case.name,
                    passed=False,
                    duration_sec=timeout,
                    assertion_results=[],
                    error=f"Test timed out after {timeout:.0f}s",
                )
            )
        elif not result_queue.empty():
            results.append(result_queue.get())
        else:
            results.append(
                TestResult(
                    name=test_case.name,
                    passed=False,
                    duration_sec=0,
                    assertion_results=[],
                    error=f"Test process exited with code {p.exitcode} without returning results",
                )
            )

    return results


def _run_single_test_process(
    connector: str,
    exchange: str,
    test_case: TestCaseSpec,
    account_file: Path | None,
    result_queue: multiprocessing.Queue,
    log_dir: str,
) -> None:
    """Entry point for each test process. Runs the full Qubx pipeline and evaluates assertions."""
    start = time.monotonic()
    try:
        result = _run_single_test(connector, exchange, test_case, account_file, log_dir)
        result.duration_sec = time.monotonic() - start
        result_queue.put(result)
    except Exception as e:
        result_queue.put(
            TestResult(
                name=test_case.name,
                passed=False,
                duration_sec=time.monotonic() - start,
                assertion_results=[],
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
        )


def _run_single_test(
    connector: str,
    exchange: str,
    test_case: TestCaseSpec,
    account_file: Path | None,
    log_dir: str,
) -> TestResult:
    """Run a single test case using the full Qubx pipeline."""
    import qubx.connectors  # noqa: F401 - register connectors

    from qubx.testing.connectors.collector import EventCollector
    from qubx.testing.connectors.strategy import ConnectorVerificationStrategy
    from qubx.utils.runner.accounts import AccountConfigurationManager
    from qubx.utils.runner.configs import ExchangeConfig, LiveConfig, LoggingConfig, StorageConfig, StrategyConfig, WarmupConfig
    from qubx.utils.runner.runner import run_strategy
    from qubx.utils.time import to_timedelta

    collector = EventCollector()
    spec_ref = test_case

    # Create a bound strategy class that captures collector and spec
    class _BoundStrategy(ConnectorVerificationStrategy):
        def __init__(self, **kwargs):
            super().__init__(spec=spec_ref, collector=collector)

    # Build strategy config programmatically
    warmup_config = None
    if test_case.warmup:
        warmup_config = WarmupConfig(data=StorageConfig(storage=connector))

    config = StrategyConfig(
        name=f"connector_test_{test_case.name}",
        strategy=_BoundStrategy,
        live=LiveConfig(
            exchanges={
                exchange: ExchangeConfig(
                    connector=connector,
                    universe=test_case.instruments,
                )
            },
            logging=LoggingConfig(
                logger="CsvFileLogsWriter",
                position_interval="1Min",
                portfolio_interval="1Min",
                args={"log_folder": log_dir},
            ),
            warmup=warmup_config,
        ),
    )

    # Account manager
    acc_manager = AccountConfigurationManager(account_file, Path("."), search_qubx_dir=True)

    # Run strategy (non-blocking)
    ctx = run_strategy(
        config,
        acc_manager,
        paper=True,
        blocking=False,
        no_emission=True,
        no_notifiers=True,
        no_exporters=True,
    )

    # Wait for the specified duration
    duration_sec = to_timedelta(test_case.duration).total_seconds()
    time.sleep(duration_sec)

    # Stop strategy
    ctx.stop()

    # Evaluate assertions
    assertion_results = []
    for name, params in test_case.parsed_assertions():
        ar = run_assertion(name, collector, **params)
        assertion_results.append(ar)

    all_passed = all(ar.passed for ar in assertion_results)

    return TestResult(
        name=test_case.name,
        passed=all_passed,
        duration_sec=0,  # filled by caller
        assertion_results=assertion_results,
        total_events=len(collector.bar_events),
        log_dir=log_dir,
    )
