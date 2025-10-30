"""
Slack client for posting messages using bot token API.

This module provides a reusable client for sending messages to Slack channels
using the chat.postMessage API with bot tokens.
"""

import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from qubx import logger


class SlackClient:
    """
    Client for posting messages to Slack using bot token API.

    This client handles asynchronous message posting with proper thread management
    and error handling.
    """

    SLACK_API_URL = "https://slack.com/api/chat.postMessage"

    def __init__(self, bot_token: str, max_workers: int = 1, environment: str | None = None):
        """
        Initialize the Slack Client.

        Args:
            bot_token: Slack bot token for authentication
            max_workers: Maximum number of worker threads for posting messages
            environment: Optional environment name to include in message footers
        """
        self._bot_token = bot_token
        self._environment = environment
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="slack_client")
        logger.debug(f"Initialized with {max_workers} workers")

    def post_message_async(
        self,
        message: str,
        channel: str,
        emoji: str | None = None,
        color: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Queue a message to be posted to Slack asynchronously.

        Args:
            message: Main message text
            channel: Slack channel to post to (e.g., "#channel-name")
            emoji: Optional emoji to prepend to the message (e.g., ":rocket:")
            color: Optional color for the message attachment (e.g., "#36a64f")
            metadata: Optional dictionary with additional fields to include
        """
        try:
            self._executor.submit(self._post_message_impl, message, channel, emoji, color, metadata)
        except Exception as e:
            logger.error(f"Failed to queue Slack message: {e}")

    def post_payload_async(self, payload: dict[str, Any], channel: str) -> None:
        """
        Queue a raw payload to be posted to Slack asynchronously.

        This method allows posting custom Slack messages with blocks or other advanced formatting.

        Args:
            payload: Raw Slack API payload (e.g., with "blocks", "attachments", etc.)
            channel: Slack channel to post to (e.g., "#channel-name")
        """
        try:
            self._executor.submit(self._post_payload_impl, payload, channel)
        except Exception as e:
            logger.error(f"Failed to queue Slack payload: {e}")

    def _post_message_impl(
        self,
        message: str,
        channel: str,
        emoji: str | None = None,
        color: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Implementation that actually posts to Slack (called from worker thread).

        Args:
            message: Main message text
            channel: Slack channel to post to
            emoji: Optional emoji to prepend to the message
            color: Optional color for the message attachment
            metadata: Optional dictionary with additional fields to include

        Returns:
            bool: True if the post was successful, False otherwise
        """
        try:
            # Build fields from metadata
            fields = []
            if metadata:
                for key, value in metadata.items():
                    fields.append({"title": key, "value": str(value), "short": len(str(value)) < 50})

            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Build message text with optional emoji
            text_message = f"{emoji} {message}" if emoji else message

            # Build payload
            payload: dict[str, Any] = {
                "text": text_message,
            }

            # Add attachment if we have fields, color, or environment
            if fields or color or self._environment:
                attachment: dict[str, Any] = {}
                if color:
                    attachment["color"] = color
                if fields:
                    attachment["fields"] = fields

                # Build footer with environment and time
                if self._environment:
                    attachment["footer"] = f"Environment: {self._environment} | Time: {timestamp}"
                else:
                    attachment["footer"] = f"Time: {timestamp}"

                payload["attachments"] = [attachment]

            # Use the payload implementation to actually send the message
            return self._post_payload_impl(payload, channel)
        except Exception as e:
            logger.error(f"Failed to build message payload: {e}")
            return False

    def _post_payload_impl(self, payload: dict[str, Any], channel: str) -> bool:
        """
        Implementation that posts a raw payload to Slack (called from worker thread).

        Args:
            payload: Raw Slack API payload
            channel: Slack channel to post to

        Returns:
            bool: True if the post was successful, False otherwise
        """
        try:
            # Add channel to payload
            data = {**payload, "channel": channel}

            # Post to Slack
            response = requests.post(
                SlackClient.SLACK_API_URL,
                json=data,
                headers={
                    "Authorization": f"Bearer {self._bot_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
            )
            response.raise_for_status()

            logger.debug(f"Successfully posted payload to {channel}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to post payload to Slack: {e}")
            return False

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            self._executor.shutdown(wait=False)
        except:
            pass
