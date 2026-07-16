from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from qubx.backtester.runner import SimulationRunner
from qubx.backtester.transfers import SimulationTransferManager
from qubx.backtester.utils import (
    SetupTypes,
    SimulationSetup,
    collect_transfers_log,
    recognize_simulation_data_config,
)
from qubx.core.account_manager import SimulatedAccountManager
from qubx.core.basics import Balance, Instrument, ITimeProvider, MarketType
from qubx.core.initializer import BasicStrategyInitializer
from qubx.core.interfaces import IStrategy, IStrategyInitializer, ITransferManager
from qubx.data.registry import StorageRegistry


class _T(ITimeProvider):
    def time(self) -> np.datetime64:
        return np.datetime64("2026-05-28T00:00:00", "ns")


def _am():
    am = SimulatedAccountManager(
        connectors={"E1": MagicMock(), "E2": MagicMock()},
        base_currencies={"E1": "USDT", "E2": "USDT"},
        time=_T(),
    )
    am.get_state("E1").update_balance("USDT", Balance(exchange="E1", currency="USDT", total=1000.0, free=1000.0))
    return am


def test_transfer_moves_balance_between_exchanges():
    am = _am()
    tm = SimulationTransferManager(am, _T())

    txid = tm.transfer_funds("E1", "E2", "USDT", 300.0)

    assert am.get_balance("USDT", exchange="E1").total == 700.0
    assert am.get_balance("USDT", exchange="E1").free == 700.0
    assert am.get_balance("USDT", exchange="E2").total == 300.0
    assert am.get_balance("USDT", exchange="E2").free == 300.0
    status = tm.get_transfer_status(txid)
    assert status["amount"] == 300.0
    assert status["from_exchange"] == "E1"
    assert status["to_exchange"] == "E2"
    assert status["status"] == "completed"
    assert txid in tm.get_transfers()


def test_transfer_preserves_destination_balance_identity():
    am = _am()
    am.get_state("E2").update_balance("USDT", Balance(exchange="E2", currency="USDT", total=50.0, free=50.0))
    dest_ref = am.get_balance("USDT", exchange="E2")
    tm = SimulationTransferManager(am, _T())

    tm.transfer_funds("E1", "E2", "USDT", 100.0)

    assert am.get_balance("USDT", exchange="E2") is dest_ref  # same object, updated in place
    assert dest_ref.total == 150.0


def _am_cross_stable():
    am = SimulatedAccountManager(
        connectors={"E1": MagicMock(), "E2": MagicMock()},
        base_currencies={"E1": "USDT", "E2": "USDC"},
        time=_T(),
    )
    am.get_state("E1").update_balance("USDT", Balance(exchange="E1", currency="USDT", total=1000.0, free=1000.0))
    return am


def test_cross_stable_transfer_credits_destination_base_currency():
    # a USDT credit on a USDC-based exchange would be invisible to the destination's total_capital
    am = _am_cross_stable()
    tm = SimulationTransferManager(am, _T())

    txid = tm.transfer_funds("E1", "E2", "USDT", 400.0)

    assert am.get_balance("USDT", exchange="E1").total == 600.0
    assert am.get_balance("USDC", exchange="E2").total == 400.0
    assert am.get_balance("USDT", exchange="E2").total == 0.0  # no phantom non-base stable
    status = tm.get_transfer_status(txid)
    assert status["currency"] == "USDT"
    assert status["credited_currency"] == "USDC"


def test_same_currency_transfer_credits_unchanged():
    am = _am()
    tm = SimulationTransferManager(am, _T())

    txid = tm.transfer_funds("E1", "E2", "USDT", 100.0)

    assert am.get_balance("USDT", exchange="E2").total == 100.0
    status = tm.get_transfer_status(txid)
    assert status["currency"] == "USDT"
    assert status["credited_currency"] == "USDT"


def test_non_stable_transfer_passes_through_unconverted():
    am = _am_cross_stable()
    am.get_state("E1").update_balance("BTC", Balance(exchange="E1", currency="BTC", total=2.0, free=2.0))
    tm = SimulationTransferManager(am, _T())

    txid = tm.transfer_funds("E1", "E2", "BTC", 1.0)

    assert am.get_balance("BTC", exchange="E1").total == 1.0
    assert am.get_balance("BTC", exchange="E2").total == 1.0
    assert am.get_balance("USDC", exchange="E2").total == 0.0
    status = tm.get_transfer_status(txid)
    assert status["currency"] == "BTC"
    assert status["credited_currency"] == "BTC"


def test_insufficient_funds_raises_and_leaves_balances_untouched():
    am = _am()
    tm = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="Insufficient funds"):
        tm.transfer_funds("E1", "E2", "USDT", 5000.0)
    assert am.get_balance("USDT", exchange="E1").total == 1000.0


def test_unknown_currency_raises():
    am = _am()
    tm = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="Insufficient funds"):
        tm.transfer_funds("E1", "E2", "DOGE", 1.0)


def test_unknown_transfer_status_raises():
    am = _am()
    tm = SimulationTransferManager(am, _T())
    with pytest.raises(ValueError, match="Transfer not found"):
        tm.get_transfer_status("nope")


def test_collect_transfers_log_populated_after_transfer():
    # The simulator must collect a non-None, populated transfers log from a
    # SimulationTransferManager (which only exposes get_transfers(), not the old
    # get_transfers_dataframe()). Today the simulator gates on get_transfers_dataframe
    # and always yields None — this asserts the corrected collection.
    am = _am()
    tm = SimulationTransferManager(am, _T())
    txid = tm.transfer_funds("E1", "E2", "USDT", 250.0)

    log = collect_transfers_log(tm)

    assert log is not None
    assert isinstance(log, pd.DataFrame)
    assert len(log) == 1
    assert txid in log["transaction_id"].values
    row = log[log["transaction_id"] == txid].iloc[0]
    assert row["from_exchange"] == "E1"
    assert row["to_exchange"] == "E2"
    assert row["currency"] == "USDT"
    assert row["amount"] == 250.0
    assert row["status"] == "completed"


def test_collect_transfers_log_empty_when_no_transfers():
    am = _am()
    tm = SimulationTransferManager(am, _T())

    log = collect_transfers_log(tm)

    # No transfers recorded -> empty frame (mirrors the legacy empty-schema behavior),
    # never the stale always-None result.
    assert log is not None
    assert isinstance(log, pd.DataFrame)
    assert len(log) == 0


def test_collect_transfers_log_none_for_missing_manager():
    # No manager at all -> None (nothing to collect).
    assert collect_transfers_log(None) is None


def test_collect_transfers_log_empty_for_base_manager():
    # get_transfers is on ITransferManager with a no-op default (empty), so a manager that
    # doesn't record a log yields an empty frame — no hasattr probing, no raise, never None.
    log = collect_transfers_log(ITransferManager())
    assert isinstance(log, pd.DataFrame)
    assert len(log) == 0


class _ProbeStrategy(IStrategy):
    def on_init(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription("ohlc(1h)")


def _instr(exchange: str, symbol: str, quote: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange=exchange,
        base="BTC",
        quote=quote,
        settle=quote,
        exchange_symbol=symbol,
        tick_size=0.1,
        lot_size=0.001,
        min_size=0.001,
    )


def _make_runner(
    warmup_mode: bool = False,
    initializer: BasicStrategyInitializer | None = None,
    strategy: IStrategy | None = None,
) -> SimulationRunner:
    # instruments built directly: lookup-based resolution silently drops exchanges absent
    # from the environment's instrument DB (CI), losing the HYPERLIQUID account state
    instruments = [_instr("BINANCE.UM", "BTCUSDT", "USDT"), _instr("HYPERLIQUID", "BTCUSDC", "USDC")]
    exchanges = ["BINANCE.UM", "HYPERLIQUID"]
    return SimulationRunner(
        setup=SimulationSetup(
            setup_type=SetupTypes.STRATEGY,
            name="transfers-probe",
            generator=strategy if strategy is not None else _ProbeStrategy(),
            tracker=None,
            instruments=instruments,
            exchanges=exchanges,
            capital=10_000.0,
            base_currency="USDT",
        ),
        data_config=recognize_simulation_data_config(StorageRegistry.get("csv::tests/data/storages/multi/"), None),
        start="2026-01-01 00:00",
        stop="2026-01-01 05:00",
        initializer=initializer,
        warmup_mode=warmup_mode,
    )


def test_simulation_context_has_transfer_manager():
    # regression: the pickup is StrategyContext.__post_init__, so the auto-assign must
    # precede ctx construction — otherwise plain sims run with _transfer_manager = None
    runner = _make_runner()

    assert isinstance(runner.ctx._transfer_manager, SimulationTransferManager)

    runner.ctx.transfer_funds("BINANCE.UM", "HYPERLIQUID", "USDT", 1000.0)
    assert runner.account_manager.get_balance("USDT", exchange="BINANCE.UM").total == 4000.0
    assert runner.account_manager.get_balance("USDT", exchange="HYPERLIQUID").total == 6000.0


def test_warmup_mode_force_assigns_sim_manager():
    stub = ITransferManager()
    initializer = BasicStrategyInitializer(simulation=True)
    initializer.set_transfer_manager(stub)

    runner = _make_runner(warmup_mode=True, initializer=initializer)

    assert isinstance(runner.ctx._transfer_manager, SimulationTransferManager)
    assert runner.ctx._transfer_manager is not stub


def test_on_init_injection_wins_in_plain_sim():
    stub = ITransferManager()

    class _InjectingStrategy(IStrategy):
        def on_init(self, initializer: IStrategyInitializer):
            initializer.set_base_subscription("ohlc(1h)")
            initializer.set_transfer_manager(stub)

    runner = _make_runner(strategy=_InjectingStrategy())

    assert runner.ctx._transfer_manager is stub
