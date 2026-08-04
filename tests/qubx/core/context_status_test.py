import threading

import numpy as np

from qubx.core.fit_context import FitContext
from qubx.core.fit_executor import FitCycleState
from qubx.core.status import ContextStatus, DegradeReason, QubxStatus

T0 = np.datetime64("2026-08-04T10:00:00", "ns")


class TestContextStatus:
    def test_status_tracks_the_set_of_held_degradations(self):
        status = ContextStatus()
        status.add(DegradeReason.INTERNAL_QUEUE_OVERFLOW, T0)
        status.add(DegradeReason.EXCHANGE_MAINTENANCE, T0, scope="BINANCE.UM")

        status.clear(DegradeReason.INTERNAL_QUEUE_OVERFLOW)

        assert status.info.status is QubxStatus.DEGRADED
        assert [d.reason for d in status.info.degradations] == [DegradeReason.EXCHANGE_MAINTENANCE]

        status.clear(DegradeReason.EXCHANGE_MAINTENANCE, scope="BINANCE.UM")

        assert status.info.status is QubxStatus.NORMAL
        assert status.info.degradations == ()

    def test_same_reason_on_two_exchanges_stays_two_degradations(self):
        status = ContextStatus()
        status.add(DegradeReason.EXCHANGE_MAINTENANCE, T0, scope="BINANCE.UM")
        status.add(DegradeReason.EXCHANGE_MAINTENANCE, T0, scope="OKX")

        status.clear(DegradeReason.EXCHANGE_MAINTENANCE, scope="BINANCE.UM")

        assert [d.scope for d in status.info.degradations] == ["OKX"]

    def test_re_adding_a_live_degradation_keeps_its_original_since(self):
        status = ContextStatus()
        status.add(DegradeReason.INTERNAL_QUEUE_OVERFLOW, T0, message="queue size 5000")

        status.add(DegradeReason.INTERNAL_QUEUE_OVERFLOW, T0 + np.timedelta64(30, "s"), message="queue size 9000")

        info = status.info
        assert len(info.degradations) == 1
        assert info.degradations[0].since == T0
        assert info.degradations[0].message == "queue size 9000"

    def test_writes_from_many_threads_all_land(self):
        status = ContextStatus()
        scopes = [f"EXCHANGE{i}" for i in range(50)]
        start = threading.Barrier(len(scopes))

        def _write(scope: str) -> None:
            start.wait()
            status.add(DegradeReason.EXCHANGE_MAINTENANCE, T0, scope=scope)

        threads = [threading.Thread(target=_write, args=(s,)) for s in scopes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(d.scope for d in status.info.degradations) == sorted(scopes)  # type: ignore[type-var]


def test_fit_context_passes_status_through(mocker):
    context = mocker.MagicMock()
    context.status = ContextStatus().info
    fit_context = FitContext(context, FitCycleState())

    assert fit_context.status is context.status
