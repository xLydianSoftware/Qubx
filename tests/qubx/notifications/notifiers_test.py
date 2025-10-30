"""
Unit tests for strategy notifiers.
"""

from unittest.mock import ANY, MagicMock, patch

import pytest

from qubx.core.interfaces import IStrategyNotifier
from qubx.notifications.composite import CompositeNotifier
from qubx.notifications.throttler import IMessageThrottler, TimeWindowThrottler


class TestCompositeNotifier:
    """Test the CompositeNotifier class."""

    @pytest.fixture
    def notifiers(self):
        """Create mock notifiers."""
        notifier1 = MagicMock(spec=IStrategyNotifier)
        notifier2 = MagicMock(spec=IStrategyNotifier)
        return notifier1, notifier2

    @pytest.fixture
    def composite(self, notifiers):
        """Create a CompositeNotifier instance."""
        return CompositeNotifier(list(notifiers))

    def test_notify_start(self, composite, notifiers):
        """Test that notify_start delegates to all notifiers."""
        notifier1, notifier2 = notifiers
        metadata = {"key": "value"}
        composite.notify_start(metadata)
        notifier1.notify_start.assert_called_once_with(metadata)
        notifier2.notify_start.assert_called_once_with(metadata)

    def test_notify_stop(self, composite, notifiers):
        """Test that notify_stop delegates to all notifiers."""
        notifier1, notifier2 = notifiers
        metadata = {"key": "value"}
        composite.notify_stop(metadata)
        notifier1.notify_stop.assert_called_once_with(metadata)
        notifier2.notify_stop.assert_called_once_with(metadata)

    def test_notify_error(self, composite, notifiers):
        """Test that notify_error delegates to all notifiers."""
        notifier1, notifier2 = notifiers
        error = Exception("Test error")
        metadata = {"key": "value"}
        composite.notify_error(error, metadata)
        notifier1.notify_error.assert_called_once_with(error, metadata)
        notifier2.notify_error.assert_called_once_with(error, metadata)

    def test_notify_start_with_exception(self, composite, notifiers):
        """Test that notify_start continues even if one notifier raises an exception."""
        notifier1, notifier2 = notifiers
        notifier1.notify_start.side_effect = Exception("Test exception")
        metadata = {"key": "value"}
        # This should not raise an exception
        composite.notify_start(metadata)
        notifier1.notify_start.assert_called_once_with(metadata)
        notifier2.notify_start.assert_called_once_with(metadata)

    def test_notify_stop_with_exception(self, composite, notifiers):
        """Test that notify_stop continues even if one notifier raises an exception."""
        notifier1, notifier2 = notifiers
        notifier1.notify_stop.side_effect = Exception("Test exception")
        metadata = {"key": "value"}
        # This should not raise an exception
        composite.notify_stop(metadata)
        notifier1.notify_stop.assert_called_once_with(metadata)
        notifier2.notify_stop.assert_called_once_with(metadata)

    def test_notify_error_with_exception(self, composite, notifiers):
        """Test that notify_error continues even if one notifier raises an exception."""
        notifier1, notifier2 = notifiers
        notifier1.notify_error.side_effect = Exception("Test exception")
        error = Exception("Test error")
        metadata = {"key": "value"}
        # This should not raise an exception
        composite.notify_error(error, metadata)
        notifier1.notify_error.assert_called_once_with(error, metadata)
        notifier2.notify_error.assert_called_once_with(error, metadata)

    def test_notify_message(self, composite, notifiers):
        """Test that notify_message delegates to all notifiers."""
        notifier1, notifier2 = notifiers
        message = "Test message"
        metadata = {"key": "value"}
        composite.notify_message(message, metadata, channel="#custom")
        notifier1.notify_message.assert_called_once_with(message, metadata, channel="#custom")
        notifier2.notify_message.assert_called_once_with(message, metadata, channel="#custom")

    def test_notify_message_with_exception(self, composite, notifiers):
        """Test that notify_message continues even if one notifier raises an exception."""
        notifier1, notifier2 = notifiers
        notifier1.notify_message.side_effect = Exception("Test exception")
        message = "Test message"
        metadata = {"key": "value"}
        # This should not raise an exception
        composite.notify_message(message, metadata)
        notifier1.notify_message.assert_called_once_with(message, metadata)
        notifier2.notify_message.assert_called_once_with(message, metadata)


class TestSlackNotifier:
    """Test the SlackNotifier class."""

    @pytest.fixture
    def notifier(self):
        """Create a SlackNotifier instance."""
        from qubx.notifications.slack import SlackNotifier

        bot_token = "xoxb-test-token"
        return SlackNotifier(strategy_name="test_strategy", bot_token=bot_token, environment="test")

    @patch("requests.post")
    def test_notify_start(self, mock_post, notifier):
        """Test that notify_start sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        metadata = {"key": "value"}
        notifier.notify_start(metadata)

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify the calls - using assert_called since executor runs in background
        assert mock_post.called
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        assert "test_strategy" in str(kwargs["json"])
        # Check that the message contains the metadata
        assert "key" in str(kwargs["json"])
        assert "value" in str(kwargs["json"])

    @patch("requests.post")
    def test_notify_stop(self, mock_post, notifier):
        """Test that notify_stop sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        metadata = {"key": "value"}
        notifier.notify_stop(metadata)

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify the calls
        assert mock_post.called
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        assert "test_strategy" in str(kwargs["json"])
        # Check that the message contains the metadata
        assert "key" in str(kwargs["json"])
        assert "value" in str(kwargs["json"])

    @patch("requests.post")
    def test_notify_error(self, mock_post, notifier):
        """Test that notify_error sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        error = Exception("Test error")
        metadata = {"key": "value"}
        notifier.notify_error(error, metadata)

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify the calls
        assert mock_post.called
        # Check that the message contains the strategy name
        args, kwargs = mock_post.call_args
        assert "test_strategy" in str(kwargs["json"])
        # Check that the message contains the error
        assert "Test error" in str(kwargs["json"])
        # Check that the message contains the metadata
        assert "key" in str(kwargs["json"])
        assert "value" in str(kwargs["json"])

    @patch("requests.post")
    def test_notify_start_with_request_exception(self, mock_post, notifier):
        """Test that notify_start handles request exceptions."""
        # Set up the mock
        from requests.exceptions import RequestException

        mock_post.side_effect = RequestException("Test exception")

        # Call the method
        metadata = {"key": "value"}
        # This should not raise an exception
        notifier.notify_start(metadata)

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify the calls
        assert mock_post.called

    @patch("requests.post")
    def test_notify_message(self, mock_post, notifier):
        """Test that notify_message sends a message to Slack."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        message = "Custom message"
        metadata = {"key": "value"}
        notifier.notify_message(message, metadata)

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify the calls
        assert mock_post.called
        # Check that the message contains the custom message
        args, kwargs = mock_post.call_args
        assert "Custom message" in str(kwargs["json"])
        # Check that the message contains the metadata
        assert "key" in str(kwargs["json"])
        assert "value" in str(kwargs["json"])

    @patch("requests.post")
    def test_notify_message_with_custom_channel(self, mock_post, notifier):
        """Test that notify_message can use a custom channel."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method with a custom channel
        message = "Custom message"
        notifier.notify_message(message, channel="#custom-channel")

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify the calls
        assert mock_post.called
        # Check that the custom channel was used
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["channel"] == "#custom-channel"


class TestSlackNotifierWithThrottling:
    """Test the SlackNotifier class with throttling."""

    @pytest.fixture
    def mock_throttler(self):
        """Create a mock throttler."""
        throttler = MagicMock(spec=IMessageThrottler)
        throttler.should_send.return_value = True
        return throttler

    @pytest.fixture
    def notifier_with_throttler(self, mock_throttler):
        """Create a SlackNotifier instance with the mock throttler."""
        from qubx.notifications.slack import SlackNotifier

        bot_token = "xoxb-test-token"
        return SlackNotifier(
            strategy_name="test_strategy", bot_token=bot_token, environment="test", throttler=mock_throttler
        )

    @patch("requests.post")
    def test_throttler_used_for_error_notifications(self, mock_post, mock_throttler, notifier_with_throttler):
        """Test that the throttler is used for error notifications."""
        # Set up the mock
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        # Call the method
        error = Exception("Test error")
        notifier_with_throttler.notify_error(error)

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

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
        notifier_with_throttler.notify_error(error)

        # Wait a bit to ensure nothing was queued
        import time
        time.sleep(0.1)

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
        notifier_with_throttler.notify_start()

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify that the post request was made
        assert mock_post.called
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
        notifier_with_throttler.notify_stop()

        # Wait a bit for the executor to process
        import time
        time.sleep(0.1)

        # Verify that the post request was made
        assert mock_post.called
        # Verify that the throttler was not consulted
        mock_throttler.should_send.assert_not_called()
        # Verify that the throttler was not updated
        mock_throttler.register_sent.assert_not_called()

    @patch("qubx.notifications.slack.SlackClient")
    def test_integration_with_time_window_throttler(self, mock_slack_client_class):
        """Test integration with a real TimeWindowThrottler."""
        from qubx.notifications.slack import SlackNotifier

        # Create a mock instance
        mock_client_instance = MagicMock()
        mock_slack_client_class.return_value = mock_client_instance

        # Create a throttler with a small window
        throttler = TimeWindowThrottler(window_seconds=0.5)

        # Create the notifier
        bot_token = "xoxb-test-token"
        notifier = SlackNotifier(strategy_name="test_strategy", bot_token=bot_token, environment="test", throttler=throttler)

        # First error notification should be sent
        error1 = Exception("Test error 1")

        # Directly test the throttling logic
        throttle_key = f"error:test_strategy:{type(error1).__name__}"

        # First message should be allowed
        assert throttler.should_send(throttle_key) is True
        notifier._post_to_slack("Test message 1", ":rotating_light:", "#FF0000", None, throttle_key)
        # The throttle_key is now handled inside _post_to_slack, so it should be registered

        # Second message should be throttled (won't be sent)
        notifier._post_to_slack("Test message 2", ":rotating_light:", "#FF0000", None, throttle_key)

        # Wait for the throttling window to expire
        import time

        time.sleep(0.6)

        # Third message should be allowed again
        notifier._post_to_slack("Test message 3", ":rotating_light:", "#FF0000", None, throttle_key)

        # Two post requests should have been made (message 1 and 3, message 2 was throttled)
        assert mock_client_instance.post_message_async.call_count == 2
