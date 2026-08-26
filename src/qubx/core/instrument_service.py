"""What we are allowed to trade: the operator blacklist, and what the venue actually lists."""

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class VenueInstruments:
    """The result of reconciling the lookup against a venue's market list."""

    # - venue key -> the one instrument both sides agree on. Safe to resolve inbound events with.
    by_key: dict[Any, Instrument] = field(default_factory=dict)
    # - (instrument, why) for every candidate the venue did not confirm
    dropped: list[tuple[Instrument, str]] = field(default_factory=list)
    # - venue keys with no instrument in the lookup at all: tradeable at the venue, invisible to us
    unmatched_venue_keys: list[Any] = field(default_factory=list)

    @property
    def instruments(self) -> list[Instrument]:
        return list(self.by_key.values())


def reconcile_instruments(
    exchange: str,
    venue_markets: Mapping[Any, str],
    candidates: Iterable[Instrument],
    key_of: Callable[[Instrument], Any],
    name_of: Callable[[Instrument], str] = lambda i: i.base,
) -> VenueInstruments:
    """
    Keep the instruments the venue confirms, drop the rest, and say why.

    The lookup is our database: it can hold delisted markets, and two entries can claim the same
    venue id after a rename — market 12 on Lighter was GRAM and TON at once for weeks. That is
    tolerable in the database and not tolerable in a live connector, which resolves inbound events
    by venue id: whichever entry a dict happened to keep won, so fills for one instrument were
    booked against another.

    Venues key their markets differently — ccxt by symbol string, Lighter by integer market id,
    Hyperliquid by coin — so the caller supplies the key. Identity is compared on
    ``Instrument.base`` by default, which is what all three name a market by.

    Args:
        exchange: for the log lines only.
        venue_markets: the venue's own list, venue key -> the venue's name for it.
        candidates: what the lookup offers for this exchange.
        key_of: the venue key an instrument claims. Return None when it claims none.
        name_of: the instrument's name in the venue's terms. Defaults to ``base``.
    """
    result = VenueInstruments()
    claimed: dict[Any, list[Instrument]] = {}

    def drop(instrument: Instrument, why: str) -> None:
        result.dropped.append((instrument, why))
        logger.error(f"[{exchange}] {instrument.symbol} dropped from trading: {why}")

    for instrument in candidates:
        try:
            key = key_of(instrument)
        except Exception as exc:  # noqa: BLE001 — a malformed entry must not stop the rest
            drop(instrument, f"venue key unreadable: {exc}")
            continue
        if key is None:
            drop(instrument, "carries no venue key")
        elif key not in venue_markets:
            drop(instrument, f"venue does not list {key}")
        else:
            claimed.setdefault(key, []).append(instrument)

    for key, instruments in claimed.items():
        venue_name = venue_markets[key]
        agreed = [i for i in instruments if name_of(i) == venue_name]
        if len(agreed) == 1:
            result.by_key[key] = agreed[0]
        for i in instruments:
            if len(agreed) == 1 and i is agreed[0]:
                continue
            if len(agreed) > 1 and i in agreed:
                # - same key AND same name twice: nothing tells them apart, so neither can be
                #   trusted to resolve an inbound event
                drop(i, f"{len(agreed)} lookup entries claim {key} as {venue_name!r}")
            else:
                drop(i, f"{key} is {venue_name!r} at the venue, not {name_of(i)!r}")

    result.unmatched_venue_keys = [k for k in venue_markets if k not in result.by_key]
    logger.info(
        f"[{exchange}] instruments: {len(result.by_key)} confirmed against {len(venue_markets)} venue markets, "
        f"{len(result.dropped)} dropped, {len(result.unmatched_venue_keys)} venue markets unmatched"
    )
    if result.unmatched_venue_keys:
        logger.warning(
            f"[{exchange}] no lookup entry for venue markets: "
            f"{', '.join(f'{k}={venue_markets[k]}' for k in result.unmatched_venue_keys[:20])}"
            + (" …" if len(result.unmatched_venue_keys) > 20 else "")
        )
    return result
