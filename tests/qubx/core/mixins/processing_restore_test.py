"""Tests for restoring gatherer/tracker state re-persisting targets on the current run's log.

Regression coverage for the bug where `qubx run --restore` handed the restored
`TargetPosition`s to the gatherer purely in-memory without re-persisting them
through the current run's target logger. That meant the new run's
`*_targets.csv` stayed empty, so a subsequent `--restore` had nothing to read
and the restore chain broke after a single hop.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from qubx.core.basics import RestoredState, TargetPosition
from qubx.core.loggers import LogsWriter, StrategyLogging
from qubx.core.lookups import lookup
from qubx.core.mixins.processing import ProcessingManager


def _dt(seconds: int) -> np.datetime64:
    return np.datetime64("2024-01-01T00:00:00") + np.timedelta64(seconds, "s")


class _CapturingWriter(LogsWriter):
    """Records every write_data call so tests can assert on what got logged."""

    def __init__(self) -> None:
        super().__init__("acc", "strat", "run")
        self.written: list[tuple[str, list[dict]]] = []

    def write_data(self, log_type: str, data: list[dict]) -> None:
        self.written.append((log_type, data))


@pytest.fixture
def btc():
    instrument = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert instrument is not None
    return instrument


@pytest.fixture
def eth():
    instrument = lookup.find_symbol("BINANCE.UM", "ETHUSDT")
    assert instrument is not None
    return instrument


@pytest.fixture
def capturing_writer():
    return _CapturingWriter()


@pytest.fixture
def processing_manager(capturing_writer):
    """Build a ProcessingManager with mocked collaborators except for a real
    StrategyLogging (backed by a capturing writer) so we can observe whether
    the restore path actually writes through the same channel used to log
    targets on the normal (tracker-emits-a-target) path.
    """
    context = MagicMock()
    context.is_simulation = True
    # - no instruments so `_is_data_ready()` short-circuits to True (edge case: total == 0)
    context.instruments = []
    context._strategy_state = MagicMock()

    strategy = MagicMock()
    strategy.__class__.__name__ = "TestStrategy"

    # - real logging component: this is the same "current run's target logger" channel
    #   that `__preprocess_and_log_target_positions` uses via `self._logging.save_targets(...)`
    logging = StrategyLogging(logs_writer=capturing_writer, num_signals_records_to_write=1)

    market_data = MagicMock()
    market_data.get_market_data_cache.return_value = MagicMock()

    subscription_manager = MagicMock()
    time_provider = MagicMock()
    time_provider.time.return_value = _dt(0)

    account_manager = MagicMock()
    connectors = {}
    position_tracker = MagicMock()
    position_gathering = MagicMock()
    universe_manager = MagicMock()
    scheduler = MagicMock()

    health_monitor = MagicMock()
    health_monitor.return_value.__enter__ = MagicMock(return_value=None)
    health_monitor.return_value.__exit__ = MagicMock(return_value=False)

    delisting_detector = MagicMock()

    pm = ProcessingManager(
        context=context,
        strategy=strategy,
        logging=logging,
        market_data=market_data,
        subscription_manager=subscription_manager,
        time_provider=time_provider,
        account_manager=account_manager,
        connectors=connectors,
        position_tracker=position_tracker,
        position_gathering=position_gathering,
        universe_manager=universe_manager,
        scheduler=scheduler,
        is_simulation=True,
        health_monitor=health_monitor,
        delisting_detector=delisting_detector,
    )

    # - keep references around for assertions
    pm._test_position_tracker = position_tracker
    pm._test_position_gathering = position_gathering

    return pm


class TestRestoreTargetWriteThrough:
    def test_restore_seeds_new_runs_target_log(self, processing_manager, capturing_writer, btc, eth):
        """After a successful restore, the latest restored target per instrument must be
        re-persisted through the current run's target logger (preserving `options`), so the
        new run's targets.csv is seeded and a later `--restore` can chain off of it.
        """
        older_btc = TargetPosition(
            time=_dt(0), instrument=btc, target_position_size=0.5, options={"group": "hedge_paired"}
        )
        latest_btc = TargetPosition(
            time=_dt(10), instrument=btc, target_position_size=1.0, options={"group": "hedge_paired"}
        )
        latest_eth = TargetPosition(time=_dt(5), instrument=eth, target_position_size=-2.0, options={"group": "solo"})

        restored_state = RestoredState(
            time=_dt(10),
            balances=[],
            instrument_to_signal_positions={},
            instrument_to_target_positions={
                btc: [older_btc, latest_btc],
                eth: [latest_eth],
            },
            positions={},
        )

        processing_manager._restore_tracker_and_gatherer_state(restored_state)

        # - gatherer state is still restored the same way as before (only the latest target
        #   per instrument, not the full history)
        processing_manager._test_position_gathering.restore_from_target_positions.assert_called_once()
        _, gatherer_targets = processing_manager._test_position_gathering.restore_from_target_positions.call_args[0]
        assert {t.instrument for t in gatherer_targets} == {btc, eth}
        assert {t.target_position_size for t in gatherer_targets} == {1.0, -2.0}

        # - the same restored targets must now show up in the *current* run's target log
        target_writes = [data for log_type, data in capturing_writer.written if log_type == "targets"]
        assert len(target_writes) == 1, "restored targets were not re-persisted through the target logger"

        written_by_symbol = {row["symbol"]: row for row in target_writes[0]}
        assert set(written_by_symbol) == {"BTCUSDT", "ETHUSDT"}

        # - only the latest target position per instrument is re-persisted (matches gatherer restore)
        assert written_by_symbol["BTCUSDT"]["target_position"] == 1.0
        assert written_by_symbol["ETHUSDT"]["target_position"] == -2.0

        # - the options (carrying e.g. the maker-taker pairing "group") must round-trip untouched
        assert written_by_symbol["BTCUSDT"]["options"] == {"group": "hedge_paired"}
        assert written_by_symbol["ETHUSDT"]["options"] == {"group": "solo"}

    def test_no_write_through_when_nothing_to_restore(self, processing_manager, capturing_writer, btc):
        """When there are no target positions to restore, nothing should be logged and the
        gatherer should not be touched.
        """
        restored_state = RestoredState(
            time=_dt(0),
            balances=[],
            instrument_to_signal_positions={},
            instrument_to_target_positions={btc: []},
            positions={},
        )

        processing_manager._restore_tracker_and_gatherer_state(restored_state)

        processing_manager._test_position_gathering.restore_from_target_positions.assert_not_called()
        assert not any(log_type == "targets" for log_type, _ in capturing_writer.written)
