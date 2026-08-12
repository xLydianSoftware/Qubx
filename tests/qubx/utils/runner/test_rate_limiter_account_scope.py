"""The runner must scope account pools by the venue's own API key.

Dropping the ``api_key=`` argument in ``create_strategy_context`` is invisible everywhere else:
limiters are still built, they just fall back to a per-process account scope — so every bot on
the box shares one ``orders`` bucket. Data-only venues have no credentials entry and must keep
that fallback rather than inherit some other account's scope.
"""

from unittest.mock import MagicMock, patch

import pytest

from qubx.core.interfaces import IStrategy
from qubx.rate_limiting.manager import RateLimitManager
from qubx.utils.runner.accounts import ExchangeCredentials, ExchangeSettings
from qubx.utils.runner.configs import ExchangeConfig, LiveConfig, LoggingConfig, StrategyConfig
from qubx.utils.runner.runner import create_strategy_context


class _StopAfterVenueLoop(Exception):
    """Stands in for aux-config resolution — the first statement after the venue loop."""


class _Accounts:
    """AccountConfigurationManager stand-in: only the two per-venue lookups are used."""

    def __init__(self, api_keys: dict[str, str]) -> None:
        self._api_keys = api_keys

    def get_exchange_credentials(self, exchange: str) -> ExchangeCredentials:
        # KeyError for venues without an entry, exactly like the real manager
        return ExchangeCredentials(
            exchange=exchange, name="acct", api_key=self._api_keys[exchange.upper()], secret="secret"
        )

    def get_exchange_settings(self, exchange: str) -> ExchangeSettings:
        return ExchangeSettings(exchange=exchange)


class _Strategy(IStrategy):
    pass


def _config(venues: list[str]) -> StrategyConfig:
    return StrategyConfig(
        strategy=_Strategy,
        live=LiveConfig(
            exchanges={v: ExchangeConfig(connector="ccxt", universe=[]) for v in venues},
            logging=LoggingConfig(logger="InMemoryLogsWriter"),
        ),
    )


def _get_or_create_calls(venues: list[str], api_keys: dict[str, str]) -> list[tuple[str, str, str | None]]:
    """Run create_strategy_context up to the end of the venue loop, recording limiter creation."""
    calls: list[tuple[str, str, str | None]] = []

    def _record(self, exchange_name: str, connector_name: str, api_key: str | None = None):
        calls.append((exchange_name, connector_name, api_key))
        return None

    with (
        patch.object(RateLimitManager, "get_or_create", _record),
        patch("qubx.utils.runner.runner._setup_strategy_logging", return_value=MagicMock()),
        patch("qubx.utils.runner.runner._create_tcc", return_value=MagicMock()),
        patch("qubx.utils.runner.runner.ConnectorRegistry.get_data_provider", return_value=MagicMock()),
        patch("qubx.utils.runner.runner.ConnectorRegistry.get_connector", return_value=MagicMock()),
        patch("qubx.utils.runner.runner.resolve_aux_config", side_effect=_StopAfterVenueLoop),
        pytest.raises(_StopAfterVenueLoop),
    ):
        create_strategy_context(
            config=_config(venues),
            account_manager=_Accounts(api_keys),
            paper=False,
            restored_state=None,
            stg_name="test",
        )
    return calls


class TestRateLimiterAccountScope:
    def test_configured_account_key_reaches_the_limiter(self):
        assert _get_or_create_calls(["BINANCE.UM"], {"BINANCE.UM": "live-key"}) == [("BINANCE.UM", "ccxt", "live-key")]

    def test_each_venue_is_scoped_by_its_own_key(self):
        calls = _get_or_create_calls(["BINANCE.UM", "KRAKEN.F"], {"BINANCE.UM": "a", "KRAKEN.F": "b"})

        assert calls == [("BINANCE.UM", "ccxt", "a"), ("KRAKEN.F", "ccxt", "b")]

    def test_venue_without_credentials_keeps_the_fallback_scope(self):
        calls = _get_or_create_calls(["BINANCE.UM", "KRAKEN.F"], {"BINANCE.UM": "live-key"})

        assert calls == [("BINANCE.UM", "ccxt", "live-key"), ("KRAKEN.F", "ccxt", None)]
