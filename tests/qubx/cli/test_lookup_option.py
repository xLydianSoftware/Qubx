import os
import subprocess
import sys
import textwrap

import pytest
from click.testing import CliRunner

from qubx.cli.commands import main
from qubx.config import ENV_INSTRUMENT_LOOKUP_PATH, ENV_INSTRUMENT_LOOKUP_TYPE
from qubx.core.basics import Instrument, MarketType
from qubx.core.lookups import FileInstrumentsLookupWithCCXT

_SCRIPT = """
import multiprocessing as mp

from click.testing import CliRunner

from qubx.cli.commands import main


def found_in_child(out):
    from qubx.core.lookups import lookup

    out.put(lookup.find_symbol("YAHOO", "AAPL") is not None)


if __name__ == "__main__":
    # - validate never reads instruments; it is used here only to apply the option
    CliRunner().invoke(main, ["validate", "config.yaml", "--lookup", "file:instruments"])
    from qubx.core.lookups import lookup

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    child = ctx.Process(target=found_in_child, args=(out,))
    child.start()
    child.join()
    print(lookup.find_symbol("YAHOO", "AAPL") is not None, out.get(timeout=60))
"""


def test_lookup_option_serves_the_folder_instruments_here_and_in_child_processes(tmp_path):
    (tmp_path / "instruments").mkdir()
    aapl = Instrument("AAPL", MarketType.SPOT, "YAHOO", "AAPL", "USD", "USD", "AAPL", 0.01, 1.0, 1.0)
    FileInstrumentsLookupWithCCXT.__new__(FileInstrumentsLookupWithCCXT)._save_to_json(
        str(tmp_path / "instruments" / "yahoo.json"), [aapl]
    )
    (tmp_path / "run.py").write_text(textwrap.dedent(_SCRIPT))
    env = {k: v for k, v in os.environ.items() if k not in (ENV_INSTRUMENT_LOOKUP_TYPE, ENV_INSTRUMENT_LOOKUP_PATH)}

    done = subprocess.run(
        [sys.executable, "run.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split()[-2:] == ["True", "True"]


def _lookup_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in (ENV_INSTRUMENT_LOOKUP_TYPE, ENV_INSTRUMENT_LOOKUP_PATH)}


@pytest.mark.parametrize("spec", ["nope", "file", "file:", "file:no/such/folder", "service:http://localhost"])
@pytest.mark.parametrize("command", ["simulate", "validate"])
def test_lookup_option_rejects_anything_but_an_existing_folder(command, spec):
    before = _lookup_env()
    result = CliRunner().invoke(main, [command, "config.yaml", "--lookup", spec])
    assert result.exit_code == 2
    assert "expected file:<existing folder>" in result.output
    assert _lookup_env() == before
