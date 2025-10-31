"""
Slack client for posting/updating messages with Block Kit (bot token API).
- Pass "blocks" directly (advanced), or
- Pass simple key/value "metadata" and it will auto-build a nice block payload.
- Provide a "key" to upsert (create then update the same message).
"""

import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pandas as pd
import requests

from qubx import logger


class SlackClient:
    SLACK_POST_URL = "https://slack.com/api/chat.postMessage"
    SLACK_UPDATE_URL = "https://slack.com/api/chat.update"

    def __init__(self, bot_token: str, max_workers: int = 1, environment: str | None = None):
        self._bot_token = bot_token
        self._environment = environment
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="slack_client")
        self._msg_registry_lock = threading.Lock()
        # Maps your arbitrary "key" -> (channel, ts)
        self._message_registry: dict[str, tuple[str, str]] = {}

    # ---------------------------
    # Public async helpers
    # ---------------------------

    def notify_message_async(
        self,
        message: str,
        channel: str,
        emoji: str | None = None,
        blocks: Optional[list[dict]] = None,
        metadata: Optional[dict[str, Any]] = None,
        key: Optional[str] = None,
        thread_ts: Optional[str] = None,
    ) -> None:
        """
        Post or update a message.
        - If `key` is provided, upsert (create first time, then update by key).
        - If `blocks` is provided, use it. Otherwise build blocks from message/metadata.
        - If `thread_ts` is provided, post as a thread reply (no upsert in threads).

        Args:
            channel: Slack channel (e.g., "#bots")
            message: summary text (also used as fallback text)
            emoji: optional emoji prefix for the summary
            blocks: explicit Block Kit blocks to send
            metadata: simple key/value dict -> auto-rendered blocks if no blocks given
            key: idempotency key; when set, message is updated in place
            thread_ts: if set, this will post a thread reply under that parent ts (no upsert)
        """
        try:
            self._executor.submit(self._notify_impl, message, channel, emoji, blocks, metadata, key, thread_ts)
        except Exception as e:
            logger.error(f"[SlackClient] Failed to queue message: {e}")

    def post_thread_reply_async(self, message: str, parent_key: str, blocks: Optional[list[dict]] = None) -> None:
        """
        Reply under the parent message identified by `parent_key`.
        """
        try:
            with self._msg_registry_lock:
                entry = self._message_registry.get(parent_key)
            if not entry:
                logger.debug(f"[SlackClient] No parent for key={parent_key}; skipping thread reply")
                return
            channel, ts = entry
            self.notify_message_async(message=message, channel=channel, blocks=blocks, thread_ts=ts)
        except Exception as e:
            logger.error(f"[SlackClient] Failed to queue thread reply: {e}")

    # ---------------------------
    # Internal impl
    # ---------------------------

    def _notify_impl(
        self,
        message: str,
        channel: str,
        emoji: str | None,
        blocks: Optional[list[dict]],
        metadata: Optional[dict[str, Any]],
        key: Optional[str],
        thread_ts: Optional[str],
    ) -> bool:
        try:
            # Build blocks if not explicitly provided
            if blocks is None:
                blocks = self._build_blocks(message=message, emoji=emoji, metadata=metadata)

            # If thread reply requested -> always post (no upsert)
            if thread_ts:
                payload = {
                    "channel": channel,
                    "text": self._fallback_text(message, metadata),
                    "blocks": blocks,
                    "thread_ts": thread_ts,
                }
                return self._post(payload)

            # Upsert if key present; otherwise post a new message
            if key:
                with self._msg_registry_lock:
                    entry = self._message_registry.get(key)

                if entry is None:
                    payload = {"channel": channel, "text": self._fallback_text(message, metadata), "blocks": blocks}
                    ok, ts, channel_id = self._post_get_ts(payload)
                    if ok and ts and channel_id:
                        with self._msg_registry_lock:
                            self._message_registry[key] = (channel_id, ts)
                    return ok
                else:
                    channel_id, ts = entry
                    payload = {
                        "channel": channel_id,
                        "ts": ts,
                        "text": self._fallback_text(message, metadata),
                        "blocks": blocks,
                    }
                    return self._update(payload)
            else:
                payload = {"channel": channel, "text": self._fallback_text(message, metadata), "blocks": blocks}
                return self._post(payload)

        except Exception as e:
            logger.error(f"[SlackClient] notify_impl error: {e}")
            return False

    def _post(self, payload: dict) -> bool:
        ok, _, _ = self._post_get_ts(payload)
        return ok

    def _post_get_ts(self, payload: dict) -> tuple[bool, Optional[str], Optional[str]]:
        try:
            resp = requests.post(
                self.SLACK_POST_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._bot_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok", False):
                # Slack may return "error" or error details; log safely
                logger.warning(f"[SlackClient] chat.postMessage failed: {json.dumps(data, ensure_ascii=False)}")
                return False, None, None
            ts = data.get("ts")
            channel_id = data.get("channel")  # this is the channel ID, not the channel name
            return True, ts, channel_id
        except requests.RequestException as e:
            logger.error(f"[SlackClient] POST failed: {e}")
            return False, None, None

    def _update(self, payload: dict) -> bool:
        try:
            resp = requests.post(
                self.SLACK_UPDATE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._bot_token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok", False):
                logger.warning(f"[SlackClient] chat.update failed: {json.dumps(data, ensure_ascii=False)}")
                return False
            return True
        except requests.RequestException as e:
            logger.error(f"[SlackClient] UPDATE failed: {e}")
            return False

    # ---------------------------
    # Blocks builders
    # ---------------------------

    def _build_blocks(self, *, message: str, emoji: Optional[str], metadata: Optional[dict[str, Any]]) -> list[dict]:
        """
        If you provide blocks yourself, we use them.
        Otherwise, build a compact, nice default:
        - Header (message, optional emoji)
        - Key/Value pairs (from metadata) as a two-column list
        - Context with env + timestamp
        """
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = f"{emoji} {message}" if emoji else message

        blocks: list[dict] = []

        # Header (if present)
        if title:
            blocks.append({"type": "header", "text": {"type": "plain_text", "text": title[:150]}})

        # Metadata -> bulleted lines (two columns per row using fields-style)
        if metadata:
            lines = []
            # Make `key: value` pairs in a compact markdown list
            for k, v in metadata.items():
                v_str = "`" + str(v) + "`" if not isinstance(v, str) or (" " in v) else f"`{v}`"
                lines.append(f"*{k}*: {v_str}")

            # Chunk into sections if long (Slack section text limit is large; keep a bit conservative)
            chunk = []
            current_len = 0
            for line in lines:
                if current_len + len(line) > 2500:  # soft wrap to avoid huge sections
                    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(chunk)}})
                    blocks.append({"type": "divider"})
                    chunk = [line]
                    current_len = len(line)
                else:
                    chunk.append(line)
                    current_len += len(line) + 1
            if chunk:
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(chunk)}})

        # Footer context
        context_bits = []
        if self._environment:
            context_bits.append({"type": "mrkdwn", "text": f"*Environment:* {self._environment}"})
        context_bits.append({"type": "mrkdwn", "text": f"*Time:* {ts}"})
        blocks.append({"type": "context", "elements": context_bits})

        return blocks

    def _fallback_text(self, message: str, metadata: Optional[dict[str, Any]]) -> str:
        """Text that shows in notifications / fallback clients."""
        if not metadata:
            return message
        try:
            kv = ", ".join(f"{k}={v}" for k, v in list(metadata.items())[:10])
            if len(kv) > 300:
                kv = kv[:297] + "..."
            return f"{message} — {kv}"
        except Exception:
            return message

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


def _cell_text(text: str, *, bold: bool = False, code: bool = False) -> dict[str, Any]:
    style = {}
    if bold:
        style["bold"] = True
    if code:
        style["code"] = True
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": [{"type": "text", "text": text, **({"style": style} if style else {})}],
            }
        ],
    }


def _header_cell(text: str) -> dict[str, Any]:
    return _cell_text(text, bold=True)


def _code_cell(text: str) -> dict[str, Any]:
    return _cell_text(text, code=True)


def _plain_cell(text: str) -> dict[str, Any]:
    return _cell_text(text)


def blocks_make_table_section(data: pd.DataFrame, code_cells: set[str] | None = None) -> dict[str, Any]:
    """
    Build a Slack 'table' block from a pandas DataFrame.

    Args:
        data: Pandas DataFrame to build the table from.
        code_cells: Set of columns to render as code cells.

    Returns:
        A Slack 'table' block.
    """
    if code_cells is None:
        code_cells = set()

    header = [_header_cell(col) for col in data.columns]

    rows: list[list[dict]] = [header]

    for _, row in data.iterrows():
        cells = []
        for col, cell in row.items():
            if col in code_cells:
                cells.append(_code_cell(str(cell)))
            else:
                cells.append(_plain_cell(str(cell)))
        rows.append(cells)

    return {
        "type": "table",
        "rows": rows,
    }


def blocks_make_divider() -> dict[str, Any]:
    return {
        "type": "divider",
    }


def blocks_make_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"*{key}*: {value}"} for key, value in context.items()],
    }


def blocks_make_header(text: str) -> dict[str, Any]:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text},
    }
