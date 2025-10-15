"""Test Footer rendering with real app and progressive CSS."""

import shutil
from pathlib import Path

import pytest

from qubx.utils.runner.textual.app import TextualStrategyApp


class ProgressiveRealApp(TextualStrategyApp):
    """Real app with overrideable CSS path."""

    def __init__(self, config_file, account_file, paper, restore, test_mode, css_path=None, css=None):
        super().__init__(config_file, account_file, paper, restore, test_mode)
        self.CSS_PATH = css_path
        self.CSS = css


@pytest.mark.skip(reason="Skipping minimal CSS test")
def test_real_app_minimal_css(snap_compare):
    """Test real app with minimal CSS - just Footer dock."""
    css_path = Path(__file__).parent / "styles_minimal.tcss"
    app = ProgressiveRealApp(None, None, paper=True, restore=False, test_mode=True, css_path=css_path)
    assert snap_compare(app, terminal_size=(80, 24))


@pytest.mark.skip(reason="Skipping minimal CSS test")
def test_real_app_with_overflow_hidden(snap_compare):
    """Test real app with overflow: hidden on content-wrapper."""
    # Create CSS with overflow hidden
    css_content = """
Screen {
    layout: vertical;
}

#content-wrapper {
    overflow: hidden;
}

Footer {
    dock: bottom;
}
"""

    app = ProgressiveRealApp(None, None, paper=True, restore=False, test_mode=True, css=css_content)

    assert snap_compare(app, terminal_size=(80, 40))
