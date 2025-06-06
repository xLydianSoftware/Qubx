"""
Unit tests for strategy lifecycle notifiers.
"""

from unittest.mock import ANY, MagicMock, patch

import pytest

from qubx.core.interfaces import IStrategyLifecycleNotifier
from qubx.notifications.composite import CompositeLifecycleNotifier
from qubx.notifications.throttler import IMessageThrottler, TimeWindowThrottler


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


class TestSlackLifecycleNotifierWithThrottling:
    """Test the SlackLifecycleNotifier class with throttling."""

    @pytest.fixture
    def mock_throttler(self):
        """Create a mock throttler."""
        throttler = MagicMock(spec=IMessageThrottler)
        throttler.should_send.return_value = True
        return throttler

    @pytest.fixture
    def notifier_with_throttler(self, mock_throttler):
        """Create a SlackLifecycleNotifier instance with the mock throttler."""
        from qubx.notifications.slack import SlackLifecycleNotifier

        webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        return SlackLifecycleNotifier(webhook_url=webhook_url, environment="test", throttler=mock_throttler)

    @pytest.fixture
    def webhook_url(self):
        """Return the webhook URL."""
        return "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"

    @patch("requests.post")
    def test_throttler_used_for_error_notifications(self, mock_post, mock_throttler, notifier_with_throttler):
        """Test that the throttler is used for error notifications."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        error = Exception("Test error")
        notifier_with_throttler.notify_error("test_strategy", error)

        # Verify that the throttler was consulted
        mock_throttler.should_send.assert_called_once()
        # Verify that the throttler was updated
        mock_throttler.register_sent.assert_called_once()

    @patch("requests.post")
    def test_throttled_error_notification_not_sent(self, mock_post, mock_throttler, notifier_with_throttler):
        """Test that throttled error notifications are not sent."""
        # Set up the mock to indicate that the message should be throttled
        mock_throttler.should_send.return_value = False

        # Call the method
        error = Exception("Test error")
        notifier_with_throttler.notify_error("test_strategy", error)

        # Verify that no post request was made
        mock_post.assert_not_called()
        # Verify that the throttler was consulted
        mock_throttler.should_send.assert_called_once()
        # The message wasn't sent, so register_sent shouldn't be called
        mock_throttler.register_sent.assert_not_called()

    @patch("requests.post")
    def test_throttler_not_used_for_start_notifications(self, mock_post, mock_throttler, notifier_with_throttler):
        """Test that the throttler is not used for start notifications."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        notifier_with_throttler.notify_start("test_strategy")

        # Verify that the post request was made
        mock_post.assert_called_once()
        # Verify that the throttler was not consulted
        mock_throttler.should_send.assert_not_called()
        # Verify that the throttler was not updated
        mock_throttler.register_sent.assert_not_called()

    @patch("requests.post")
    def test_throttler_not_used_for_stop_notifications(self, mock_post, mock_throttler, notifier_with_throttler):
        """Test that the throttler is not used for stop notifications."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        notifier_with_throttler.notify_stop("test_strategy")

        # Verify that the post request was made
        mock_post.assert_called_once()
        # Verify that the throttler was not consulted
        mock_throttler.should_send.assert_not_called()
        # Verify that the throttler was not updated
        mock_throttler.register_sent.assert_not_called()

    @patch("requests.post")
    def test_integration_with_time_window_throttler(self, mock_post):
        """Test integration with a real TimeWindowThrottler."""
        from qubx.notifications.slack import SlackLifecycleNotifier

        # Create a throttler with a small window
        throttler = TimeWindowThrottler(window_seconds=0.5)

        # Create the notifier
        webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        notifier = SlackLifecycleNotifier(webhook_url=webhook_url, environment="test", throttler=throttler)

        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # First error notification should be sent
        error1 = Exception("Test error 1")

        # Directly test the throttling logic without using the executor
        throttle_key = f"error:test_strategy:{type(error1).__name__}"

        # First message should be allowed
        assert throttler.should_send(throttle_key) is True
        notifier._post_to_slack_impl("Test message 1", ":rotating_light:", "#FF0000", None, throttle_key)
        # Since we're calling _post_to_slack_impl directly (bypassing _post_to_slack),
        # we need to manually register that the message was sent
        throttler.register_sent(throttle_key)

        # Second message should be throttled
        assert throttler.should_send(throttle_key) is False

        # Wait for the throttling window to expire
        import time

        time.sleep(0.6)

        # Third message should be allowed again
        assert throttler.should_send(throttle_key) is True
        notifier._post_to_slack_impl("Test message 3", ":rotating_light:", "#FF0000", None, throttle_key)
        # Register the third message as sent as well
        throttler.register_sent(throttle_key)

        # Two post requests should have been made
        assert mock_post.call_count == 2
