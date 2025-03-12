"""
Unit tests for strategy lifecycle notifiers.
"""

import unittest
from unittest.mock import ANY, MagicMock, patch

from qubx.core.interfaces import IStrategyLifecycleNotifier
from qubx.notifications.composite import CompositeLifecycleNotifier


class TestCompositeLifecycleNotifier(unittest.TestCase):
    """Test the CompositeLifecycleNotifier class."""

    def setUp(self):
        """Set up the test case."""
        self.notifier1 = MagicMock(spec=IStrategyLifecycleNotifier)
        self.notifier2 = MagicMock(spec=IStrategyLifecycleNotifier)
        self.composite = CompositeLifecycleNotifier([self.notifier1, self.notifier2])

    def test_notify_start(self):
        """Test that notify_start delegates to all notifiers."""
        metadata = {"key": "value"}
        self.composite.notify_start("test_strategy", metadata)
        self.notifier1.notify_start.assert_called_once_with("test_strategy", metadata)
        self.notifier2.notify_start.assert_called_once_with("test_strategy", metadata)

    def test_notify_stop(self):
        """Test that notify_stop delegates to all notifiers."""
        metadata = {"key": "value"}
        self.composite.notify_stop("test_strategy", metadata)
        self.notifier1.notify_stop.assert_called_once_with("test_strategy", metadata)
        self.notifier2.notify_stop.assert_called_once_with("test_strategy", metadata)

    def test_notify_error(self):
        """Test that notify_error delegates to all notifiers."""
        error = Exception("Test error")
        metadata = {"key": "value"}
        self.composite.notify_error("test_strategy", error, metadata)
        self.notifier1.notify_error.assert_called_once_with("test_strategy", error, metadata)
        self.notifier2.notify_error.assert_called_once_with("test_strategy", error, metadata)

    def test_notify_start_with_exception(self):
        """Test that notify_start continues even if one notifier raises an exception."""
        self.notifier1.notify_start.side_effect = Exception("Test exception")
        metadata = {"key": "value"}
        # This should not raise an exception
        self.composite.notify_start("test_strategy", metadata)
        self.notifier1.notify_start.assert_called_once_with("test_strategy", metadata)
        self.notifier2.notify_start.assert_called_once_with("test_strategy", metadata)

    def test_notify_stop_with_exception(self):
        """Test that notify_stop continues even if one notifier raises an exception."""
        self.notifier1.notify_stop.side_effect = Exception("Test exception")
        metadata = {"key": "value"}
        # This should not raise an exception
        self.composite.notify_stop("test_strategy", metadata)
        self.notifier1.notify_stop.assert_called_once_with("test_strategy", metadata)
        self.notifier2.notify_stop.assert_called_once_with("test_strategy", metadata)

    def test_notify_error_with_exception(self):
        """Test that notify_error continues even if one notifier raises an exception."""
        self.notifier1.notify_error.side_effect = Exception("Test exception")
        error = Exception("Test error")
        metadata = {"key": "value"}
        # This should not raise an exception
        self.composite.notify_error("test_strategy", error, metadata)
        self.notifier1.notify_error.assert_called_once_with("test_strategy", error, metadata)
        self.notifier2.notify_error.assert_called_once_with("test_strategy", error, metadata)


@patch("requests.post")
class TestSlackLifecycleNotifier(unittest.TestCase):
    """Test the SlackLifecycleNotifier class."""

    def setUp(self):
        """Set up the test case."""
        from qubx.notifications.slack import SlackLifecycleNotifier

        self.webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        self.notifier = SlackLifecycleNotifier(webhook_url=self.webhook_url, environment="test")

    def test_notify_start(self, mock_post):
        """Test that notify_start sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        metadata = {"key": "value"}
        self.notifier.notify_start("test_strategy", metadata)

        # Verify the calls
        mock_post.assert_called_once_with(self.webhook_url, json=ANY)
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        self.assertIn("test_strategy", str(kwargs["json"]))
        # Check that the message contains the metadata
        self.assertIn("key", str(kwargs["json"]))
        self.assertIn("value", str(kwargs["json"]))

    def test_notify_stop(self, mock_post):
        """Test that notify_stop sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        metadata = {"key": "value"}
        self.notifier.notify_stop("test_strategy", metadata)

        # Verify the calls
        mock_post.assert_called_once_with(self.webhook_url, json=ANY)
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        self.assertIn("test_strategy", str(kwargs["json"]))
        # Check that the message contains the metadata
        self.assertIn("key", str(kwargs["json"]))
        self.assertIn("value", str(kwargs["json"]))

    def test_notify_error(self, mock_post):
        """Test that notify_error sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        error = Exception("Test error")
        metadata = {"key": "value"}
        self.notifier.notify_error("test_strategy", error, metadata)

        # Verify the calls
        mock_post.assert_called_once_with(self.webhook_url, json=ANY)
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        self.assertIn("test_strategy", str(kwargs["json"]))
        # Check that the message contains the error
        self.assertIn("Test error", str(kwargs["json"]))
        # Check that the message contains the metadata
        self.assertIn("key", str(kwargs["json"]))
        self.assertIn("value", str(kwargs["json"]))

    def test_notify_start_with_request_exception(self, mock_post):
        """Test that notify_start handles request exceptions."""
        # Set up the mock
        from requests.exceptions import RequestException

        mock_post.side_effect = RequestException("Test exception")

        # Call the method
        metadata = {"key": "value"}
        # This should not raise an exception
        self.notifier.notify_start("test_strategy", metadata)

        # Verify the calls
        mock_post.assert_called_once_with(self.webhook_url, json=ANY)


if __name__ == "__main__":
    unittest.main()
