"""Console report formatting for connector verification results."""

from __future__ import annotations

from qubx.testing.connectors.runner import TestResult


def print_report(results: list[TestResult], verbose: bool = False) -> None:
    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed
    total_time = sum(r.duration_sec for r in results)

    print()
    print("=" * 60)
    print("Connector Verification Results")
    print("=" * 60)
    print()

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        marker = "+" if r.passed else "x"
        print(f"  [{marker}] {r.name:40s} {status} ({r.duration_sec:.1f}s, {r.total_events} events)")

        if r.log_dir:
            print(f"      logs: {r.log_dir}")

        if r.error:
            print(f"      ERROR: {r.error.splitlines()[0]}")
            if verbose:
                for line in r.error.splitlines()[1:]:
                    print(f"        {line}")

        for ar in r.assertion_results:
            a_marker = "+" if ar.passed else "x"
            print(f"      [{a_marker}] {ar.name:36s} {ar.message}")

        print()

    print("-" * 60)
    print(f"  {n_passed} passed, {n_failed} failed ({total_time:.1f}s total)")
    print()
