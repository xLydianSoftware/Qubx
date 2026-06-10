"""Render the qubx init strategy templates and pin them to the current callback contract."""

import importlib.util
from pathlib import Path

import pytest

from qubx.core.interfaces import IStrategy
from qubx.core.mixins.processing import validate_account_callback_signatures
from qubx.templates.base import TemplateManager


def _load_strategy_class(strategy_file: Path, module_name: str) -> type[IStrategy]:
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, IStrategy) and obj is not IStrategy
    ]
    assert len(classes) == 1, f"expected exactly one IStrategy subclass in {strategy_file}"
    return classes[0]


@pytest.mark.parametrize(
    "template_name,strategy_relpath",
    [
        ("simple", "strategy.py"),
        ("project", "src/my_strategy/strategy.py"),
        ("repo", "src/my_strategy/strategy.py"),
    ],
)
def test_rendered_strategy_passes_callback_signature_check(tmp_path, template_name, strategy_relpath):
    # A template shipping a stale callback arity (e.g. the old 2-arg on_order_update) would
    # make every scaffolded strategy's callback die in a swallowed TypeError at dispatch.
    out = TemplateManager().generate_strategy(template_name=template_name, output_dir=tmp_path, name="my_strategy")
    cls = _load_strategy_class(out / strategy_relpath, f"rendered_{template_name}_strategy")
    validate_account_callback_signatures(cls())
