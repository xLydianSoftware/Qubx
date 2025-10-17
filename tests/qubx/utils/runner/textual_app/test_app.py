"""Tests for Textual strategy runner app."""

import pytest
from pathlib import Path
from textual.widgets import Footer

from qubx.utils.runner.textual.app import TextualStrategyApp


@pytest.fixture
def config_file(tmp_path):
    """Create a minimal config file for testing."""
    config = tmp_path / "config.yml"
    config.write_text("""
strategy: qubx.core.strategy.Strategy
parameters: {}
live:
  read_only: true
  exchanges:
    TEST:
      connector: ccxt
      universe:
        - BTCUSDT
""")
    return config


@pytest.mark.skip(reason="Textual app tests temporarily skipped")
@pytest.mark.asyncio
async def test_app_structure(config_file):
    """Test that the app has correct widget structure."""
    app = TextualStrategyApp(config_file, None, paper=True, restore=False, test_mode=True)

    async with app.run_test() as pilot:
        await pilot.pause()

        # Query all top-level widgets
        widgets = list(app.children)
        widget_names = [w.__class__.__name__ for w in widgets]

        print(f"\nTop-level widgets: {widget_names}")

        # Check that key widgets are present in the app tree
        from textual.widgets import Header, Footer
        from textual.containers import Horizontal, Vertical

        assert app.query_one(Header) is not None, "Header should be present"
        assert app.query_one(Footer) is not None, "Footer should be present"
        assert len(app.query(Horizontal)) > 0, "Horizontal container should be present"
        assert len(app.query(Vertical)) > 0, "Vertical container should be present"


@pytest.mark.asyncio
async def test_app_bindings():
    """Test that app has correct bindings."""
    config_file = Path("/tmp/test_config.yml")
    config_file.write_text("strategy: qubx.core.strategy.Strategy\nparameters: {}")

    app = TextualStrategyApp(config_file, None, paper=True, restore=False, test_mode=True)

    binding_keys = [b.key for b in app.BINDINGS]
    binding_actions = [b.action for b in app.BINDINGS]

    print(f"\nBindings: {list(zip(binding_keys, binding_actions))}")

    assert "q" in binding_keys, "Should have quit binding"
    assert "p" in binding_keys, "Should have positions binding"
    assert "o" in binding_keys, "Should have orders binding"
    assert "m" in binding_keys, "Should have market binding"

    # All bindings should be shown in footer
    for binding in app.BINDINGS:
        assert binding.show is True, f"Binding {binding.key} should be visible in footer"


@pytest.mark.skip(reason="Textual app tests temporarily skipped")
@pytest.mark.asyncio
async def test_footer_visibility(config_file):
    """Test that Footer is actually rendered and visible."""
    app = TextualStrategyApp(config_file, None, paper=True, restore=False, test_mode=True)

    async with app.run_test() as pilot:
        # Wait for app to be ready
        await pilot.pause()

        # Query for Footer
        footer = app.query_one(Footer)
        assert footer is not None, "Footer should be queryable"

        # Check Footer is in the widget tree
        assert footer.is_mounted, "Footer should be mounted"

        # Check Footer styles
        print(f"\nFooter styles: {footer.styles}")
        print(f"Footer display: {footer.styles.display}")
        print(f"Footer height: {footer.styles.height}")
        print(f"Footer dock: {footer.styles.dock}")

        # Footer should not be hidden
        assert footer.display is True or footer.styles.display != "none", "Footer should be visible"


@pytest.mark.skip(reason="Textual app tests temporarily skipped")
def test_footer_snapshot(config_file, snap_compare):
    """Test Footer appearance with snapshot testing."""
    app = TextualStrategyApp(config_file, None, paper=True, restore=False, test_mode=True)

    # Take snapshot - this will show if Footer is rendering
    # snap_compare will handle running the app itself
    assert snap_compare(app, terminal_size=(80, 24))


if __name__ == "__main__":
    # Run a quick check
    import asyncio

    config = Path("/tmp/test_config.yml")
    config.write_text("strategy: qubx.core.strategy.Strategy\nparameters: {}")

    async def check():
        app = TextualStrategyApp(config, None, paper=True, restore=False, test_mode=True)
        async with app.run_test() as pilot:
            await pilot.pause()

            print("\n=== Widget Tree ===")
            for widget in app.query("*"):
                print(f"{widget} (visible={widget.display}, mounted={widget.is_mounted})")

            print("\n=== Footer Check ===")
            try:
                footer = app.query_one(Footer)
                print(f"Footer found: {footer}")
                print(f"Footer display: {footer.display}")
                print(f"Footer styles.display: {footer.styles.display}")
                print(f"Footer region: {footer.region}")
                print(f"Footer size: {footer.size}")
                print(f"Footer background: {footer.styles.background}")
                print(f"Footer color: {footer.styles.color}")

                print("\n=== FooterKey styles ===")
                for i, key in enumerate(app.query("FooterKey")):
                    print(f"  Key {i}: {key}")
                    print(f"    background: {key.styles.background}")
                    print(f"    color: {key.styles.color}")
                    print(f"    render: {key.render()}")
                    if i >= 2:  # Just check first 3
                        break

            except Exception as e:
                print(f"Footer query failed: {e}")

    asyncio.run(check())
