"""Lint gate: no unbounded blocking wait anywhere under `qubx/connectors/ccxt/`.

A single timeout-less `concurrent.futures.Future.result()` on the exchange event loop froze a live
bot for 18 days. This makes the pattern unrepresentable rather than merely fixed.

Covered: `.result(...)`, `.exception(...)` and `run_sync(...)` — the three ways this package blocks
a thread on the exchange loop. A literal `None` timeout counts as unbounded, because that is what
`Future.result(timeout=None)` actually does. Not covered (and not currently used here to wait on the
exchange loop): `Event.wait`, `Thread.join`, `concurrent.futures.wait`.

A call that genuinely cannot take a timeout (an already-resolved asyncio Future, whose `result()`
has no timeout parameter at all) must say so with an explicit `unbounded-result-ok:` comment on its
line.
"""

import ast
import pathlib

import qubx.connectors.ccxt as ccxt_package

CCXT_ROOT = pathlib.Path(ccxt_package.__file__).parent
OPT_OUT_MARKER = "unbounded-result-ok"
BLOCKING_ATTRS = ("result", "exception")


def _is_none_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_bounded(node: ast.Call) -> bool:
    """A call is bounded when it is given a timeout that is not a literal ``None``."""
    for kw in node.keywords:
        if kw.arg == "timeout":
            return not _is_none_literal(kw.value)
    if node.args:
        return not _is_none_literal(node.args[0])
    return False


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _unbounded_calls(path: pathlib.Path) -> list[tuple[int, str]]:
    source = path.read_text()
    lines = source.splitlines()
    findings = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name == "run_sync":
            # run_sync's own timeout defaults to None, so an omitted timeout is unbounded too.
            bounded = any(kw.arg == "timeout" and not _is_none_literal(kw.value) for kw in node.keywords)
        elif name in BLOCKING_ATTRS:
            bounded = _is_bounded(node)
        else:
            continue
        if bounded:
            continue
        # A multi-line call reports at its first line; scan the whole span for the opt-out.
        span = lines[node.lineno - 1 : (node.end_lineno or node.lineno)]
        if any(OPT_OUT_MARKER in line for line in span):
            continue
        findings.append((node.end_lineno or node.lineno, lines[node.lineno - 1].strip()))
    return findings


def test_no_unbounded_waits_in_ccxt_connector():
    offenders = []
    for path in sorted(CCXT_ROOT.rglob("*.py")):
        for lineno, line in _unbounded_calls(path):
            offenders.append(f"{path.relative_to(CCXT_ROOT.parents[2])}:{lineno}: {line}")

    assert not offenders, (
        "unbounded blocking waits found - pass a timeout, or annotate the call with "
        f"`# {OPT_OUT_MARKER}: <reason>` if the future genuinely cannot take one:\n  " + "\n  ".join(offenders)
    )


def test_the_gate_actually_detects_the_pattern(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(future, other, third, fourth, loop, coro):\n"
        "    future.result()\n"
        "    other.result(5)\n"
        "    third.result(timeout=1)\n"
        "    fourth.result(timeout=None)\n"
        "    future.result(None)\n"
        "    future.exception()\n"
        "    logger.exception(err)\n"
        "    loop.run_sync(coro)\n"
        "    loop.run_sync(coro, timeout=None)\n"
        "    loop.run_sync(coro, timeout=3)\n"
        f"    future.result()  # {OPT_OUT_MARKER}: already done\n"
    )

    assert [lineno for lineno, _ in _unbounded_calls(sample)] == [2, 5, 6, 7, 9, 10]


def test_the_gate_finds_the_opt_out_on_a_multiline_call(tmp_path):
    sample = tmp_path / "multiline.py"
    sample.write_text(
        "def f(loop, coro):\n"
        f"    return loop.submit(  # {OPT_OUT_MARKER}: reason\n"
        "        coro\n"
        "    ).result()\n"
    )

    assert _unbounded_calls(sample) == []
