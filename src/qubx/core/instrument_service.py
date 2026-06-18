"""Instrument blacklist service: pluggable HTTP-backed blacklist with Null default."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from qubx import logger
from qubx.core.basics import Instrument


@dataclass(frozen=True)
class BlacklistEntry:
    """A single blacklist rule. An empty asset+symbol entry blacklists the whole exchange/market_type."""

    exchange: str
    market_type: str | None
    asset: str | None
    symbol: str | None

    def matches(self, instrument: Instrument) -> bool:
        if instrument.exchange != self.exchange:
            return False
        if self.market_type is not None and instrument.market_type != self.market_type:
            return False
        return self._asset_symbol_match(instrument)

    def _asset_symbol_match(self, instrument: Instrument) -> bool:
        asset_set = self.asset is not None and self.asset != ""
        symbol_set = self.symbol is not None and self.symbol != ""
        if not asset_set and not symbol_set:
            return True
        asset_ok = (not asset_set) or (instrument.asset.upper() == self.asset.upper())  # type: ignore[union-attr]
        symbol_ok = (not symbol_set) or (instrument.symbol.upper() == self.symbol.upper())  # type: ignore[union-attr]
        return asset_ok and symbol_ok


@dataclass
class InstrumentServiceDiff:
    """Difference between two blacklist evaluations over a known instrument set.

    `blacklisted_added`/`blacklisted_removed` are scoped to the known universe (used to
    force-close held positions). `entries_changed` reflects whether the raw blacklist
    *entry set* changed at all — it flips even for edits to instruments outside the current
    universe (e.g. un-blacklisting an instrument that was already evicted), which the
    universe-scoped lists cannot detect. The manager uses it to decide whether to fire the
    re-fit callbacks."""

    blacklisted_added: list[Instrument] = field(default_factory=list)
    blacklisted_removed: list[Instrument] = field(default_factory=list)
    entries_changed: bool = False


class IInstrumentService:
    """Interface for the instrument blacklist service."""

    def get_blacklist_entries(self) -> list[BlacklistEntry]:
        raise NotImplementedError

    def refresh(self, known_instruments: list[Instrument]) -> InstrumentServiceDiff:
        raise NotImplementedError

    def is_blacklisted(self, instrument: Instrument) -> bool:
        raise NotImplementedError

    def matching_instruments(self, instruments: list[Instrument]) -> list[Instrument]:
        raise NotImplementedError


class NullInstrumentService(IInstrumentService):
    """No-op instrument service. The default in backtests and local runs."""

    def get_blacklist_entries(self) -> list[BlacklistEntry]:
        return []

    def refresh(self, known_instruments: list[Instrument]) -> InstrumentServiceDiff:
        return InstrumentServiceDiff()

    def is_blacklisted(self, instrument: Instrument) -> bool:
        return False

    def matching_instruments(self, instruments: list[Instrument]) -> list[Instrument]:
        return []


class HttpInstrumentService(IInstrumentService):
    """Fetches the blacklist over HTTP and caches it between refreshes."""

    _BLACKLIST_PATH = "/internal/instrument-service/blacklist"

    def __init__(
        self,
        base_url: str,
        exchanges: list[str],
        poll_interval_s: float = 60.0,
        timeout_s: float = 5.0,
        client: "httpx.Client | None" = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._exchanges = list(exchanges)
        self.poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)
        self._entries: list[BlacklistEntry] = []

    def get_blacklist_entries(self) -> list[BlacklistEntry]:
        return list(self._entries)

    def _fetch(self) -> list[BlacklistEntry] | None:
        url = f"{self._base_url}{self._BLACKLIST_PATH}"
        params = [("exchange", e) for e in self._exchanges]
        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:  # network / parse error: keep cache
            logger.warning(f"[InstrumentService] :: blacklist fetch failed, keeping previous cache: {e}")
            return None
        entries = []
        for item in payload.get("entries", []):
            entries.append(
                BlacklistEntry(
                    exchange=item["exchange"],
                    market_type=item.get("market_type"),
                    asset=item.get("asset"),
                    symbol=item.get("symbol"),
                )
            )
        return entries

    def refresh(self, known_instruments: list[Instrument]) -> InstrumentServiceDiff:
        prev_entries = self._entries
        new_entries = self._fetch()
        if new_entries is None:
            # network error: cache preserved, no diff
            return InstrumentServiceDiff()
        prev_matched = {i for i in known_instruments if self._any_match(prev_entries, i)}
        self._entries = new_entries
        now_matched = {i for i in known_instruments if self._any_match(new_entries, i)}
        added = [i for i in known_instruments if i in now_matched and i not in prev_matched]
        removed = [i for i in known_instruments if i in prev_matched and i not in now_matched]
        # entries_changed flips for ANY blacklist edit, including instruments outside the
        # known universe (BlacklistEntry is a frozen dataclass, so it is set-comparable).
        entries_changed = set(prev_entries) != set(new_entries)
        return InstrumentServiceDiff(
            blacklisted_added=added, blacklisted_removed=removed, entries_changed=entries_changed
        )

    @staticmethod
    def _any_match(entries: list[BlacklistEntry], instrument: Instrument) -> bool:
        return any(e.matches(instrument) for e in entries)

    def is_blacklisted(self, instrument: Instrument) -> bool:
        return self._any_match(self._entries, instrument)

    def matching_instruments(self, instruments: list[Instrument]) -> list[Instrument]:
        return [i for i in instruments if self._any_match(self._entries, i)]


def create_instrument_service(exchanges: list[str]) -> IInstrumentService:
    """Build the instrument service from the QUBX_INSTRUMENT_SERVICE_URL env var.

    Returns NullInstrumentService when the env var is unset/empty (the default for
    backtests and local runs), otherwise an HttpInstrumentService.
    """
    base_url = os.environ.get("QUBX_INSTRUMENT_SERVICE_URL", "").strip()
    if not base_url:
        return NullInstrumentService()
    return HttpInstrumentService(base_url=base_url, exchanges=exchanges)
