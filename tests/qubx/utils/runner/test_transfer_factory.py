from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from qubx.backtester.transfers import SimulationTransferManager
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import ITimeProvider
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import ITransferManager
from qubx.transfers import XChangesTransferService
from qubx.utils.runner.configs import LiveConfig, TransfersConfig
from qubx.utils.runner.factory import create_transfer_manager


class _T(ITimeProvider):
    def time(self) -> np.datetime64:
        return np.datetime64("2026-07-14T00:00:00", "ns")


def _am() -> SimulatedAccountManager:
    return SimulatedAccountManager(
        connectors={"BINANCE.UM": MagicMock(), "HYPERLIQUID": MagicMock()},
        base_currencies={"BINANCE.UM": "USDT", "HYPERLIQUID": "USDT"},
        time=_T(),
    )


def _cfg(type: str = "xchanges", **parameters) -> TransfersConfig:
    params: dict = {"base_url": "http://svc/api", "provider": "XLYDIAN", "user": "bohdan"}
    params.update(parameters)
    return TransfersConfig(type=type, parameters=params)


def _create(
    cfg: TransfersConfig,
    paper: bool = False,
    read_only: bool = False,
    initializer: BasicStrategyInitializer | None = None,
) -> ITransferManager | None:
    return create_transfer_manager(
        cfg,
        paper=paper,
        read_only=read_only,
        account=_am(),
        time_provider=_T(),
        initializer=initializer if initializer is not None else BasicStrategyInitializer(simulation=False),
    )


class TestCreateTransferManager:
    def test_paper_returns_simulation_manager(self):
        assert isinstance(_create(_cfg(), paper=True), SimulationTransferManager)

    def test_read_only_returns_none(self):
        assert _create(_cfg(), read_only=True) is None

    def test_read_only_wins_over_paper(self):
        assert _create(_cfg(), paper=True, read_only=True) is None

    def test_live_returns_xchanges_service_with_simulation_guard(self):
        initializer = BasicStrategyInitializer(simulation=False)
        tm = _create(_cfg(), initializer=initializer)
        assert isinstance(tm, XChangesTransferService)

        # guard is wired to the live initializer: flipping simulation makes transfers raise
        initializer.simulation = True
        with pytest.raises(RuntimeError, match="simulation"):
            tm.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDC", 10.0)

    def test_unknown_type_raises_in_live(self):
        with pytest.raises(ValueError, match="Unknown transfer manager type: bogus.*Available types.*xchanges"):
            _create(_cfg(type="bogus"))

    def test_unknown_type_raises_in_paper_and_read_only(self):
        # type validated before the mode early-returns: a typo fails startup in every mode
        with pytest.raises(ValueError, match="Available types"):
            _create(_cfg(type="bogus"), paper=True)
        with pytest.raises(ValueError, match="Available types"):
            _create(_cfg(type="bogus"), read_only=True)


class TestTransfersConfig:
    def test_parses_from_live_config_fragment(self):
        live = LiveConfig(
            exchanges={"BINANCE.UM": {"connector": "ccxt", "universe": ["BTCUSDT"]}},
            logging={"logger": "InMemoryLogsWriter"},
            transfers={
                "type": "xchanges",
                "parameters": {"base_url": "http://transfer-service.platform.svc/api", "user": "bohdan"},
            },
        )
        assert live.transfers is not None
        assert live.transfers.type == "xchanges"
        assert live.transfers.parameters["base_url"] == "http://transfer-service.platform.svc/api"
        assert live.transfers.parameters["user"] == "bohdan"

    def test_parameters_default_empty(self):
        assert TransfersConfig(type="xchanges").parameters == {}

    def test_unknown_key_forbidden(self):
        with pytest.raises(ValidationError):
            TransfersConfig(type="xchanges", base_url="http://svc/api")

    def test_missing_type_forbidden(self):
        with pytest.raises(ValidationError):
            TransfersConfig(parameters={"base_url": "http://svc/api"})

    def test_transfers_default_none_on_live_config(self):
        live = LiveConfig(
            exchanges={"BINANCE.UM": {"connector": "ccxt", "universe": ["BTCUSDT"]}},
            logging={"logger": "InMemoryLogsWriter"},
        )
        assert live.transfers is None
