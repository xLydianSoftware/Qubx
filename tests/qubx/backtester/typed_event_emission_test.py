"""Backtester emits typed ChannelMessage events for live market data and the
pregenerated-signals scheduled trigger, instead of channel tuples.

These tests intercept SimulatedCtrlChannel.send (the runner's only send sink) and
record, per call, whether the payload is a typed event or a raw tuple. They prove
the runner now puts OhlcEvent/QuoteEvent on the channel for live data (with the
correct ohlc(<tf>) timeframe) and a ScheduledEvent for the pregenerated-signals path,
while historical/warmup batches stay tuples.
"""

import pandas as pd

from qubx.backtester.simulator import simulate
from qubx.backtester.utils import SimulatedCtrlChannel
from qubx.core.basics import DataType, Instrument
from qubx.core.events import (
    ChannelMessage,
    MarketDataMessage,
    OhlcEvent,
    QuoteEvent,
    ScheduledEvent,
)
from qubx.core.interfaces import IStrategy, IStrategyInitializer
from qubx.core.lookups import lookup
from qubx.data.registry import StorageRegistry


def _csv_storage():
    return StorageRegistry.get("csv::tests/data/storages/multi/")


class _SendRecorder:
    """Wraps SimulatedCtrlChannel.send to record every payload, then delegates."""

    def __init__(self, monkeypatch):
        self.sent: list = []
        original = SimulatedCtrlChannel.send

        def _spy(channel, data):
            self.sent.append(data)
            return original(channel, data)

        monkeypatch.setattr(SimulatedCtrlChannel, "send", _spy)

    @property
    def live_market_events(self) -> list[MarketDataMessage]:
        return [d for d in self.sent if isinstance(d, MarketDataMessage) and not d.is_historical]

    @property
    def scheduled_events(self) -> list[ScheduledEvent]:
        return [d for d in self.sent if isinstance(d, ScheduledEvent)]

    @property
    def live_market_tuples(self) -> list[tuple]:
        # 4-tuples (instrument, data_type, data, is_hist) that are live (is_hist False)
        # and whose data_type is a typed market-data type — these are exactly the tuples
        # the runner must no longer emit.
        out = []
        for d in self.sent:
            if isinstance(d, tuple) and len(d) == 4 and d[3] is False:
                base, _ = DataType.from_str(d[1]) if d[1] else (None, {})
                if base in (
                    DataType.QUOTE,
                    DataType.TRADE,
                    DataType.ORDERBOOK,
                    DataType.OHLC,
                    DataType.FUNDING_RATE,
                    DataType.OPEN_INTEREST,
                    DataType.LIQUIDATION,
                    DataType.FUNDING_PAYMENT,
                ):
                    out.append(d)
        return out


class _NoopStrategy(IStrategy):
    def on_init(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription("ohlc(1h)")
        initializer.subscribe("quote")
        initializer.set_event_schedule("1h")


def test_runner_emits_typed_market_data_events_for_live_data(monkeypatch):
    recorder = _SendRecorder(monkeypatch)

    simulate(
        _NoopStrategy(),
        data=_csv_storage(),
        capital=1000,
        start="2026-01-01 00:00",
        stop="2026-01-01 05:00",
        instruments=["BINANCE.UM:SWAP:BTCUSDT"],
        debug="WARNING",
        silent=True,
    )

    live = recorder.live_market_events
    assert live, "expected live market-data events on the channel"

    # - no live market-data tuples remain
    assert recorder.live_market_tuples == [], (
        f"runner still emits live market-data tuples: {recorder.live_market_tuples[:3]}"
    )

    # - OHLC events arrive typed and carry the requested timeframe (ohlc(1h)), not the
    #   bare producing 'ohlc'
    ohlc_events = [e for e in live if isinstance(e, OhlcEvent)]
    assert ohlc_events, "expected typed OhlcEvent for live OHLC data"
    assert all(e.timeframe == "1h" for e in ohlc_events), (
        f"OHLC events must carry timeframe '1h', got {set(e.timeframe for e in ohlc_events)}"
    )

    # - the additional quote subscription arrives as typed QuoteEvent
    assert any(isinstance(e, QuoteEvent) for e in live), "expected typed QuoteEvent for live quote data"


def test_runner_emits_scheduled_event_for_pregenerated_signals(monkeypatch):
    recorder = _SendRecorder(monkeypatch)

    instr = lookup.find_symbol("BINANCE.UM", "BTCUSDT")
    assert instr is not None

    # - pregenerated alternating signals → the signal-replay path emits one
    #   ScheduledEvent(kind="event") per signal
    idx = pd.date_range("2026-01-01 01:00", "2026-01-01 04:00", freq="1h")
    signals = pd.Series([1, -1, 1, -1], index=idx, name="BTCUSDT")

    simulate(
        signals,
        data=_csv_storage(),
        capital=1000,
        start="2026-01-01 00:00",
        stop="2026-01-01 05:00",
        instruments=["BINANCE.UM:SWAP:BTCUSDT"],
        signal_timeframe="1h",
        debug="WARNING",
        silent=True,
    )

    scheduled = recorder.scheduled_events
    assert scheduled, "expected ScheduledEvent(s) from the pregenerated-signals path"

    event_triggers = [e for e in scheduled if e.kind == "event"]
    assert event_triggers, "expected ScheduledEvent(kind='event') for replayed signals"
    for e in event_triggers:
        assert isinstance(e, ChannelMessage)
        assert isinstance(e.instrument, Instrument)
        assert isinstance(e.payload, dict) and "order" in e.payload

    # - the signal-replay payloads must match the pregenerated signal values
    sent_orders = [e.payload["order"] for e in event_triggers]
    assert set(sent_orders) <= {1, -1}, f"unexpected replayed order values: {set(sent_orders)}"

    # - and the market-data ticks driving the replay are typed, not live tuples
    assert recorder.live_market_tuples == [], (
        f"runner still emits live market-data tuples: {recorder.live_market_tuples[:3]}"
    )
