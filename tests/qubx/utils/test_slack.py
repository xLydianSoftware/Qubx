"""
Unit tests for the Slack Client.

These tests use mocks to simulate Slack API responses.
"""

import json
import time
from unittest.mock import patch

import pytest
import requests

from qubx.utils.slack import SlackClient


class MockResponse:
    """Mock response for requests.post."""

    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self.json_data = json_data or {"ok": True}
        self.text = json.dumps(self.json_data)

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"HTTP Error: {self.status_code}")


class TestSlackClient:
    """Unit tests for the SlackClient."""

    @patch("requests.post")
    def test_post_message_basic(self, mock_post):
        """Test posting a basic message without emoji or metadata."""
        # Configure mock
        mock_post.return_value = MockResponse(200, {"ok": True})

        # Create client
        client = SlackClient(bot_token="xoxb-test-token")

        # Post message
        client.post_message_async(
            message="Test message",
            channel="#test-channel",
        )

        # Wait for background thread
        time.sleep(0.1)

        # Verify post was called
        assert mock_post.call_count == 1
        call_args = mock_post.call_args[1]

        # Check headers
        assert "headers" in call_args
        assert call_args["headers"]["Authorization"] == "Bearer xoxb-test-token"

        # Check payload
        json_data = call_args["json"]
        assert json_data["text"] == "Test message"
        assert json_data["channel"] == "#test-channel"
        assert "attachments" not in json_data  # No attachments for basic message

    @patch("requests.post")
    def test_post_message_with_emoji(self, mock_post):
        """Test posting a message with emoji."""
        mock_post.return_value = MockResponse(200, {"ok": True})

        client = SlackClient(bot_token="xoxb-test-token")

        client.post_message_async(
            message="Test message",
            channel="#test-channel",
            emoji=":rocket:",
        )

        time.sleep(0.1)

        assert mock_post.call_count == 1
        json_data = mock_post.call_args[1]["json"]
        assert json_data["text"] == ":rocket: Test message"

    @patch("requests.post")
    def test_post_message_with_metadata(self, mock_post):
        """Test posting a message with metadata fields."""
        mock_post.return_value = MockResponse(200, {"ok": True})

        client = SlackClient(bot_token="xoxb-test-token")

        metadata = {
            "Environment": "production",
            "Status": "running",
            "Long Value": "This is a very long value that should not be short in the attachment",
        }

        client.post_message_async(
            message="Test message",
            channel="#test-channel",
            metadata=metadata,
        )

        time.sleep(0.1)

        assert mock_post.call_count == 1
        json_data = mock_post.call_args[1]["json"]

        # Check attachments
        assert "attachments" in json_data
        assert len(json_data["attachments"]) == 1

        attachment = json_data["attachments"][0]
        assert "fields" in attachment
        assert len(attachment["fields"]) == 3

        # Check fields
        fields_dict = {f["title"]: f for f in attachment["fields"]}
        assert "Environment" in fields_dict
        assert fields_dict["Environment"]["value"] == "production"
        assert fields_dict["Environment"]["short"] is True  # Short value

        assert "Long Value" in fields_dict
        assert fields_dict["Long Value"]["short"] is False  # Long value

    @patch("requests.post")
    def test_post_message_with_color(self, mock_post):
        """Test posting a message with color."""
        mock_post.return_value = MockResponse(200, {"ok": True})

        client = SlackClient(bot_token="xoxb-test-token")

        client.post_message_async(
            message="Test message",
            channel="#test-channel",
            color="#FF0000",
        )

        time.sleep(0.1)

        assert mock_post.call_count == 1
        json_data = mock_post.call_args[1]["json"]

        # Check attachments
        assert "attachments" in json_data
        attachment = json_data["attachments"][0]
        assert attachment["color"] == "#FF0000"
        assert "footer" in attachment  # Footer should be present

    @patch("requests.post")
    def test_post_message_full_options(self, mock_post):
        """Test posting a message with all options."""
        mock_post.return_value = MockResponse(200, {"ok": True})

        client = SlackClient(bot_token="xoxb-test-token")

        client.post_message_async(
            message="Test message",
            channel="#test-channel",
            emoji=":warning:",
            color="#FFD700",
            metadata={"Key": "Value"},
        )

        time.sleep(0.1)

        assert mock_post.call_count == 1
        json_data = mock_post.call_args[1]["json"]

        # Check all parts
        assert json_data["text"] == ":warning: Test message"
        assert json_data["channel"] == "#test-channel"
        assert "attachments" in json_data

        attachment = json_data["attachments"][0]
        assert attachment["color"] == "#FFD700"
        assert len(attachment["fields"]) == 1
        assert attachment["fields"][0]["title"] == "Key"

    @patch("requests.post")
    def test_error_handling(self, mock_post):
        """Test error handling when posting fails."""
        # Configure mock to raise an error
        mock_post.side_effect = requests.RequestException("Connection error")

        client = SlackClient(bot_token="xoxb-test-token")

        # This should not raise an exception
        client.post_message_async(
            message="Test message",
            channel="#test-channel",
        )

        time.sleep(0.1)

        # Verify post was attempted
        assert mock_post.call_count == 1

    @patch("requests.post")
    def test_multiple_workers(self, mock_post):
        """Test posting with multiple workers."""
        mock_post.return_value = MockResponse(200, {"ok": True})

        # Create client with 3 workers
        client = SlackClient(bot_token="xoxb-test-token", max_workers=3)

        # Post multiple messages
        for i in range(5):
            client.post_message_async(
                message=f"Message {i}",
                channel="#test-channel",
            )

        # Wait for all background threads
        time.sleep(0.2)

        # All messages should be posted
        assert mock_post.call_count == 5

    @patch("requests.post")
    def test_cleanup(self, mock_post):
        """Test that executor is cleaned up properly."""
        mock_post.return_value = MockResponse(200, {"ok": True})

        client = SlackClient(bot_token="xoxb-test-token")
        client.post_message_async(message="Test", channel="#test")

        # Delete the client (should trigger cleanup)
        del client

        # No way to test executor shutdown directly, but this shouldn't raise
        time.sleep(0.1)
