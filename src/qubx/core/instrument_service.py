"""Instrument blacklist service: pluggable HTTP-backed blacklist with Null default."""

from __future__ import annotations

from dataclasses import dataclass, field

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
        if asset_set and symbol_set:
            return asset_ok and symbol_ok
        return asset_ok and symbol_ok


@dataclass
class InstrumentServiceDiff:
    """Difference between two blacklist evaluations over a known instrument set."""

    blacklisted_added: list[Instrument] = field(default_factory=list)
    blacklisted_removed: list[Instrument] = field(default_factory=list)


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
