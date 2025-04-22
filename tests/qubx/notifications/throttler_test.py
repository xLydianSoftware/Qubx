"""Tests for the throttler classes in the notifications package."""

import time

from qubx.notifications.throttler import CountBasedThrottler, NoThrottling, TimeWindowThrottler


class TestTimeWindowThrottler:
    """Tests for the TimeWindowThrottler class."""

    def test_should_allow_first_message(self):
        """The first message for a key should always be allowed."""
        throttler = TimeWindowThrottler(window_seconds=1.0)
        assert throttler.should_send("test_key") is True

    def test_should_throttle_subsequent_messages_within_window(self):
        """Messages within the time window should be throttled."""
        throttler = TimeWindowThrottler(window_seconds=1.0)
        assert throttler.should_send("test_key") is True
        throttler.register_sent("test_key")
        assert throttler.should_send("test_key") is False

    def test_should_allow_messages_after_window(self):
        """Messages after the time window should be allowed."""
        throttler = TimeWindowThrottler(window_seconds=0.1)
        assert throttler.should_send("test_key") is True
        throttler.register_sent("test_key")
        assert throttler.should_send("test_key") is False
        time.sleep(0.2)  # Wait longer than the window
        assert throttler.should_send("test_key") is True

    def test_different_keys_are_throttled_independently(self):
        """Different keys should be throttled independently."""
        throttler = TimeWindowThrottler(window_seconds=1.0)
        throttler.register_sent("key1")
        assert throttler.should_send("key1") is False
        assert throttler.should_send("key2") is True


class TestCountBasedThrottler:
    """Tests for the CountBasedThrottler class."""

    def test_should_allow_messages_up_to_count(self):
        """Should allow messages up to the max count."""
        throttler = CountBasedThrottler(max_count=2, window_seconds=1.0)
        # First message
        assert throttler.should_send("test_key") is True
        throttler.register_sent("test_key")
        # Second message (reaches max_count)
        assert throttler.should_send("test_key") is True
        throttler.register_sent("test_key")
        # Third message (exceeds max_count)
        assert throttler.should_send("test_key") is False

    def test_should_reset_count_after_window(self):
        """Count should reset after the time window."""
        throttler = CountBasedThrottler(max_count=1, window_seconds=0.1)
        assert throttler.should_send("test_key") is True
        throttler.register_sent("test_key")
        assert throttler.should_send("test_key") is False
        time.sleep(0.2)  # Wait longer than the window
        assert throttler.should_send("test_key") is True

    def test_different_keys_counted_independently(self):
        """Different keys should be counted independently."""
        throttler = CountBasedThrottler(max_count=1, window_seconds=1.0)
        throttler.register_sent("key1")
        assert throttler.should_send("key1") is False
        assert throttler.should_send("key2") is True


class TestNoThrottling:
    """Tests for the NoThrottling class."""

    def test_should_always_allow_messages(self):
        """Should always allow messages regardless of frequency."""
        throttler = NoThrottling()
        for _ in range(10):
            assert throttler.should_send("test_key") is True
            throttler.register_sent("test_key") 