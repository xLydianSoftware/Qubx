"""
Unit tests for strategy lifecycle notifiers.
"""

from unittest.mock import ANY, MagicMock, patch

import pytest

from qubx.core.interfaces import IStrategyLifecycleNotifier
from qubx.notifications.composite import CompositeLifecycleNotifier


class TestCompositeLifecycleNotifier:
    """Test the CompositeLifecycleNotifier class."""

    @pytest.fixture
    def notifiers(self):
        """Create mock notifiers."""
        notifier1 = MagicMock(spec=IStrategyLifecycleNotifier)
        notifier2 = MagicMock(spec=IStrategyLifecycleNotifier)
        return notifier1, notifier2

    @pytest.fixture
    def composite(self, notifiers):
        """Create a CompositeLifecycleNotifier instance."""
        return CompositeLifecycleNotifier(list(notifiers))

    def test_notify_start(self, composite, notifiers):
        """Test that notify_start delegates to all notifiers."""
        notifier1, notifier2 = notifiers
        metadata = {"key": "value"}
        composite.notify_start("test_strategy", metadata)
        notifier1.notify_start.assert_called_once_with("test_strategy", metadata)
        notifier2.notify_start.assert_called_once_with("test_strategy", metadata)

    def test_notify_stop(self, composite, notifiers):
        """Test that notify_stop delegates to all notifiers."""
        notifier1, notifier2 = notifiers
        metadata = {"key": "value"}
        composite.notify_stop("test_strategy", metadata)
        notifier1.notify_stop.assert_called_once_with("test_strategy", metadata)
        notifier2.notify_stop.assert_called_once_with("test_strategy", metadata)

    def test_notify_error(self, composite, notifiers):
        """Test that notify_error delegates to all notifiers."""
        notifier1, notifier2 = notifiers
        error = Exception("Test error")
        metadata = {"key": "value"}
        composite.notify_error("test_strategy", error, metadata)
        notifier1.notify_error.assert_called_once_with("test_strategy", error, metadata)
        notifier2.notify_error.assert_called_once_with("test_strategy", error, metadata)

    def test_notify_start_with_exception(self, composite, notifiers):
        """Test that notify_start continues even if one notifier raises an exception."""
        notifier1, notifier2 = notifiers
        notifier1.notify_start.side_effect = Exception("Test exception")
        metadata = {"key": "value"}
        # This should not raise an exception
        composite.notify_start("test_strategy", metadata)
        notifier1.notify_start.assert_called_once_with("test_strategy", metadata)
        notifier2.notify_start.assert_called_once_with("test_strategy", metadata)

    def test_notify_stop_with_exception(self, composite, notifiers):
        """Test that notify_stop continues even if one notifier raises an exception."""
        notifier1, notifier2 = notifiers
        notifier1.notify_stop.side_effect = Exception("Test exception")
        metadata = {"key": "value"}
        # This should not raise an exception
        composite.notify_stop("test_strategy", metadata)
        notifier1.notify_stop.assert_called_once_with("test_strategy", metadata)
        notifier2.notify_stop.assert_called_once_with("test_strategy", metadata)

    def test_notify_error_with_exception(self, composite, notifiers):
        """Test that notify_error continues even if one notifier raises an exception."""
        notifier1, notifier2 = notifiers
        notifier1.notify_error.side_effect = Exception("Test exception")
        error = Exception("Test error")
        metadata = {"key": "value"}
        # This should not raise an exception
        composite.notify_error("test_strategy", error, metadata)
        notifier1.notify_error.assert_called_once_with("test_strategy", error, metadata)
        notifier2.notify_error.assert_called_once_with("test_strategy", error, metadata)


class TestSlackLifecycleNotifier:
    """Test the SlackLifecycleNotifier class."""

    @pytest.fixture
    def notifier(self):
        """Create a SlackLifecycleNotifier instance."""
        from qubx.notifications.slack import SlackLifecycleNotifier

        webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        return SlackLifecycleNotifier(webhook_url=webhook_url, environment="test")

    @pytest.fixture
    def webhook_url(self):
        """Return the webhook URL."""
        return "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"

    @patch("requests.post")
    def test_notify_start(self, mock_post, notifier, webhook_url):
        """Test that notify_start sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        metadata = {"key": "value"}
        notifier.notify_start("test_strategy", metadata)

        # Verify the calls
        mock_post.assert_called_once_with(webhook_url, json=ANY)
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        assert "test_strategy" in str(kwargs["json"])
        # Check that the message contains the metadata
        assert "key" in str(kwargs["json"])
        assert "value" in str(kwargs["json"])

    @patch("requests.post")
    def test_notify_stop(self, mock_post, notifier, webhook_url):
        """Test that notify_stop sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        metadata = {"key": "value"}
        notifier.notify_stop("test_strategy", metadata)

        # Verify the calls
        mock_post.assert_called_once_with(webhook_url, json=ANY)
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        assert "test_strategy" in str(kwargs["json"])
        # Check that the message contains the metadata
        assert "key" in str(kwargs["json"])
        assert "value" in str(kwargs["json"])

    @patch("requests.post")
    def test_notify_error(self, mock_post, notifier, webhook_url):
        """Test that notify_error sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        error = Exception("Test error")
        metadata = {"key": "value"}
        notifier.notify_error("test_strategy", error, metadata)

        # Verify the calls
        mock_post.assert_called_once_with(webhook_url, json=ANY)
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        assert "test_strategy" in str(kwargs["json"])
        # Check that the message contains the error
        assert "Test error" in str(kwargs["json"])
        # Check that the message contains the metadata
        assert "key" in str(kwargs["json"])
        assert "value" in str(kwargs["json"])

    @patch("requests.post")
    def test_notify_start_with_request_exception(self, mock_post, notifier, webhook_url):
        """Test that notify_start handles request exceptions."""
        # Set up the mock
        from requests.exceptions import RequestException

        mock_post.side_effect = RequestException("Test exception")

        # Call the method
        metadata = {"key": "value"}
        # This should not raise an exception
        notifier.notify_start("test_strategy", metadata)

        # Verify the calls
        mock_post.assert_called_once_with(webhook_url, json=ANY)
