import pandas as pd

from qubx.backtester import simulator
from qubx.backtester.runner import SimulationRunner
from qubx.backtester.utils import SetupTypes, SimulationSetup, recognize_simulation_data_config
from qubx.core.basics import DEFAULT_MAINTENANCE_MARGIN, Instrument, MarketType
from qubx.core.interfaces import IStrategy, IStrategyInitializer
from qubx.data.registry import StorageRegistry
from qubx.utils.runner.configs import StrategyConfig
from qubx.utils.runner.runner import _build_sim_params


class _ProbeStrategy(IStrategy):
    def on_init(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription("ohlc(1h)")


def _instr(symbol: str = "BTCUSDT", listed_at: str | None = None) -> Instrument:
    return Instrument(
        symbol, MarketType.SWAP, "BINANCE.UM", "BTC", "USDT", "USDT", symbol, 0.1, 0.001, 0.001,
        listed_at=pd.Timestamp(listed_at) if listed_at else None,
    )  # fmt: skip


def _setup(setup_type: SetupTypes = SetupTypes.STRATEGY, **kwargs) -> SimulationSetup:
    return SimulationSetup(
        setup_type=setup_type,
        name="probe",
        generator=_ProbeStrategy(),
        tracker=None,
        instruments=kwargs.pop("instruments", [_instr()]),
        exchanges=["BINANCE.UM"],
        capital=10_000.0,
        base_currency="USDT",
        **kwargs,
    )


def _config(simulation: dict, live: dict | None = None) -> StrategyConfig:
    return StrategyConfig.model_validate(
        {
            "strategy": "qubx.core.interfaces.IStrategy",
            "simulation": {
                "capital": 1000,
                "instruments": ["BINANCE.UM:BTCUSDT"],
                "start": "2026-01-01",
                "stop": "2026-01-02",
                "data": {"storage": "csv::tests/data/storages/multi/"},
                **simulation,
            },
            **({"live": live} if live else {}),
        }
    )


_LIVE = {
    "exchanges": {"BINANCE.UM": {"connector": "ccxt", "universe": ["BTCUSDT"]}},
    "logging": {"logger": "InMemoryLogsWriter"},
    "account_manager": {"maint_margin_rate": 0.03},
}


class TestMaintMarginRate:
    def test_the_simulated_account_takes_the_setup_rate(self):
        runner = SimulationRunner(
            setup=_setup(maint_margin_rate=0.02),
            data_config=recognize_simulation_data_config(StorageRegistry.get("csv::tests/data/storages/multi/"), None),
            start="2026-01-01 00:00",
            stop="2026-01-01 05:00",
        )
        assert runner.account_manager.get_state("BINANCE.UM").maint_margin_rate == 0.02

    def test_the_setup_defaults_to_the_framework_rate(self):
        assert _setup().maint_margin_rate == DEFAULT_MAINTENANCE_MARGIN

    def test_sim_params_take_the_live_account_rate(self):
        _, params = _build_sim_params(_config({}, live=_LIVE))
        assert params["maint_margin_rate"] == 0.03

    def test_the_simulation_rate_wins_over_the_live_one(self):
        _, params = _build_sim_params(_config({"maint_margin_rate": 0.01}, live=_LIVE))
        assert params["maint_margin_rate"] == 0.01

    def test_no_rate_without_either(self):
        _, params = _build_sim_params(_config({}))
        assert "maint_margin_rate" not in params


class TestSignalStartAdjustment:
    def _start(self, monkeypatch, setup: SimulationSetup, listings: dict[str, list[Instrument]]) -> pd.Timestamp:
        captured = {}

        class _Runner:
            def __init__(self, **kwargs):
                captured["start"] = kwargs["start"]
                raise RuntimeError("stop here")

        monkeypatch.setattr(simulator, "SimulationRunner", _Runner)
        monkeypatch.setattr(simulator.lookup, "find_listings", lambda e, s, mt=None: listings.get(s, []))
        simulator._run_setup(
            0, "acc", setup, None, pd.Timestamp("2010-01-01"), pd.Timestamp("2027-01-01"), True, False, "5Min"
        )  # type: ignore[arg-type]
        return captured["start"]

    def test_a_relisted_symbol_starts_at_its_first_listing(self, monkeypatch):
        current = _instr(listed_at="2024-01-01")
        setup = _setup(SetupTypes.SIGNAL, instruments=[current])
        start = self._start(monkeypatch, setup, {"BTCUSDT": [current, _instr(listed_at="2019-01-01")]})
        assert start == pd.Timestamp("2019-01-01")

    def test_an_instrument_unknown_to_the_lookup_uses_its_own_date(self, monkeypatch):
        setup = _setup(SetupTypes.SIGNAL, instruments=[_instr(listed_at="2021-01-01")])
        assert self._start(monkeypatch, setup, {}) == pd.Timestamp("2021-01-01")
