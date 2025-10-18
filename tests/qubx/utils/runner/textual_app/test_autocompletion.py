"""Tests for autocompletion functionality."""

import pytest

from qubx.utils.runner.textual.kernel import IPyKernel


@pytest.mark.asyncio
async def test_kernel_completion():
    """Test that kernel completion works."""
    kernel = IPyKernel()

    try:
        await kernel.start()

        # Execute some code to create variables
        kernel.execute("import math")
        kernel.execute("my_var = 42")

        # Wait a bit for execution
        import asyncio
        await asyncio.sleep(0.5)

        # Test completion on 'math.'
        completions = await kernel.complete("math.", 5)
        assert len(completions) > 0, "Should get completions for math module"
        assert any("sqrt" in c for c in completions), "Should have sqrt in math completions"

        # Test completion on 'my_'
        completions = await kernel.complete("my_", 3)
        assert len(completions) > 0, "Should get completions for my_ prefix"
        assert any("my_var" in c for c in completions), "Should complete my_var"

    finally:
        await kernel.stop()


@pytest.mark.asyncio
async def test_kernel_completion_empty():
    """Test that kernel returns empty list for invalid completion."""
    kernel = IPyKernel()

    try:
        await kernel.start()

        # Test completion on invalid/empty input
        completions = await kernel.complete("", 0)
        assert isinstance(completions, list), "Should return a list"

    finally:
        await kernel.stop()
