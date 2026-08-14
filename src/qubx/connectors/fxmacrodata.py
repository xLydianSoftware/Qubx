from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class FXMacroDataCalendar:
    """FXMacroData release-calendar connector for Qubx strategies."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://fxmacrodata.com/api/v1",
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("FXMD_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def calendar(
        self,
        currency: str = "usd",
        start_date: str | None = None,
        end_date: str | None = None,
        top_tier_only: bool = False,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if self.api_key:
            params["api_key"] = self.api_key
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self.base_url}/calendar/{currency.lower()}{query}",
            headers={"Accept": "application/json", "User-Agent": "qubx-fxmacrodata"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        rows = list(payload.get("data") or [])
        if top_tier_only:
            rows = [row for row in rows if row.get("top_tier_for_currency") or row.get("market_tier") == 1]
        return rows

    def next_events(
        self,
        currency: str = "usd",
        now: datetime | None = None,
        lookahead: timedelta = timedelta(days=7),
        top_tier_only: bool = True,
    ) -> list[dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        end = now + lookahead
        rows = self.calendar(
            currency=currency,
            start_date=now.date().isoformat(),
            end_date=end.date().isoformat(),
            top_tier_only=top_tier_only,
        )
        return [row for row in rows if (event_time := _event_time(row)) is not None and now <= event_time <= end]


def _event_time(row: dict[str, Any]) -> datetime | None:
    value = row.get("announcement_datetime_utc") or row.get("announcement_datetime_local")
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

