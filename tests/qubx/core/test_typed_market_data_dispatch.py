"""Typed isinstance-based market-data dispatch (commit 6.5.2).

Two layers of coverage:

* focused unit tests drive ProcessingManager._dispatch_market_data / _fire_md_reaction /
  _md_payload / process_data directly with a mocked manager, asserting the dispatch
  wiring (base-data update, MarketEvent into the pipeline, reaction-callback routing,
  the throttle gate, and the tuple->typed-event adapter).
* an end-to-end simulation drives the full ProcessingManager so we can assert that
  on_market_data and the on_quote/on_trade reaction callbacks fire together while the
  OHLC cache (and therefore AM mark-to-market off the quote mid) is exercised.
"""

from collections import defaultdict
from unittest.mock import Mock

import numpy as np
import pandas as pd

from qubx.backtester.simulator import simulate
from qubx.core.basics import DataType, Instrument, MarketEvent, MarketType, Signal
from qubx.core.events import (
    OhlcEvent,
    OrderBookEvent,
    QuoteEvent,
    TradeEvent,
)
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer
from qubx.core.mixins.processing import ProcessingManager
from qubx.core.series import Bar, OrderBook, Quote, Trade
from qubx.data.registry import StorageRegistry


def _instrument(symbol: str = "BTCUSDT") -> Instrument:
    return Instrument(
        symbol=symbol,
        market_type=MarketType.SWAP,
        exchange="binance",
        base="BTC",
        quote="USDT",
        settle="USDT",
        exchange_symbol=symbol,
        tick_size=0.01,
        lot_size=0.001,
        min_size=0.001,
    )


# ---------------------------------------------------------------------------
# Focused unit tests on the dispatch internals
# ---------------------------------------------------------------------------


def _mock_manager():
    pm = Mock()
    pm._time_provider = Mock()
    pm._time_provider.time.return_value = pd.Timestamp("2025-01-08 00:00:00").asm8
    pm._data_throttler = None
    pm._md_payload = ProcessingManager._md_payload
    pm._ProcessingManager__update_base_data = Mock(return_value=True)
    return pm


def test_dispatch_market_data_quote_updates_cache_and_runs_pipeline():
    instr = _instrument()
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)
    pm = _mock_manager()

    ProcessingManager._dispatch_market_data(pm, QuoteEvent(instrument=instr, quote=quote))

    # (a) base-data cache update with the right data type and payload
    pm._ProcessingManager__update_base_data.assert_called_once_with(instr, DataType.QUOTE, quote)

    # (b) a MarketEvent flows into the strategy pipeline (on_market_data -> signals)
    pm._run_strategy_pipeline.assert_called_once()
    mkt = pm._run_strategy_pipeline.call_args[0][0]
    assert isinstance(mkt, MarketEvent)
    assert mkt.type == DataType.QUOTE
    assert mkt.instrument is instr
    assert mkt.data is quote
    assert mkt.is_trigger is True

    # (c) the typed event rides along as the reaction-callback source
    assert isinstance(pm._run_strategy_pipeline.call_args.kwargs["md_reaction"], QuoteEvent)


def test_dispatch_market_data_ohlc_uses_parameterized_type():
    instr = _instrument()
    bar = Bar(0, 1.0, 2.0, 0.5, 1.5, 10.0)
    pm = _mock_manager()

    ProcessingManager._dispatch_market_data(pm, OhlcEvent(instrument=instr, bar=bar, timeframe="1h"))

    # the cache keys OHLC by the parameterized data-type string
    pm._ProcessingManager__update_base_data.assert_called_once_with(instr, DataType.OHLC["1h"], bar)
    mkt = pm._run_strategy_pipeline.call_args[0][0]
    assert mkt.type == DataType.OHLC["1h"]


def test_dispatch_market_data_throttle_gate_skips():
    instr = _instrument()
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)
    pm = _mock_manager()
    pm._data_throttler = Mock()
    pm._data_throttler.should_send.return_value = False

    ProcessingManager._dispatch_market_data(pm, QuoteEvent(instrument=instr, quote=quote))

    pm._data_throttler.should_send.assert_called_once_with(DataType.QUOTE, instr)
    pm._ProcessingManager__update_base_data.assert_not_called()
    pm._run_strategy_pipeline.assert_not_called()


def test_fire_md_reaction_routes_to_typed_callbacks():
    instr = _instrument()
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)
    trade = Trade(0, 100.0, 1.0)
    ob = OrderBook(0, 100.0, 0.01, 1, np.array([1.0]), np.array([1.0]))

    pm = Mock()
    pm._strategy = Mock()

    ProcessingManager._fire_md_reaction(pm, QuoteEvent(instrument=instr, quote=quote))
    pm._safe_call.assert_called_once_with(pm._strategy.on_quote, quote)

    pm._safe_call.reset_mock()
    ProcessingManager._fire_md_reaction(pm, TradeEvent(instrument=instr, trade=trade))
    pm._safe_call.assert_called_once_with(pm._strategy.on_trade, trade)

    pm._safe_call.reset_mock()
    ProcessingManager._fire_md_reaction(pm, OrderBookEvent(instrument=instr, orderbook=ob))
    pm._safe_call.assert_called_once_with(pm._strategy.on_orderbook, ob)


def test_quote_marks_account_to_market():
    """A quote on the market-data path marks the AM position off the quote mid
    (the base-data update calls _mark_to_market, which forwards Quotes to AM)."""
    instr = _instrument()
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)
    pm = Mock()
    pm._account_manager = Mock()

    ProcessingManager._mark_to_market(pm, instr, quote)

    pm._account_manager.on_market_quote.assert_called_once_with(instr, quote)


def test_md_payload_extracts_each_event():
    instr = _instrument()
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)
    trade = Trade(0, 100.0, 1.0)
    ob = OrderBook(0, 100.0, 0.01, 1, np.array([1.0]), np.array([1.0]))
    bar = Bar(0, 1.0, 2.0, 0.5, 1.5, 10.0)

    assert ProcessingManager._md_payload(QuoteEvent(instrument=instr, quote=quote)) is quote
    assert ProcessingManager._md_payload(TradeEvent(instrument=instr, trade=trade)) is trade
    assert ProcessingManager._md_payload(OrderBookEvent(instrument=instr, orderbook=ob)) is ob
    assert ProcessingManager._md_payload(OhlcEvent(instrument=instr, bar=bar, timeframe="1h")) is bar


def test_process_data_routes_live_market_data_through_process_event():
    instr = _instrument()
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)
    pm = Mock()
    pm._LIVE_MARKET_DATA_TYPES = ProcessingManager._LIVE_MARKET_DATA_TYPES
    pm._context = Mock()
    pm._context.emitter = None

    ProcessingManager.process_data(pm, instr, DataType.QUOTE, quote, is_historical=False)

    # live market data is wrapped in its typed event and routed through process_event
    pm.process_event.assert_called_once()
    ev = pm.process_event.call_args[0][0]
    assert isinstance(ev, QuoteEvent)
    assert ev.quote is quote
    # the legacy tuple path is bypassed for live market data
    pm._ProcessingManager__process_data.assert_not_called()


def test_process_data_historical_falls_to_legacy_path():
    instr = _instrument()
    quote = Quote(0, 100.0, 101.0, 1.0, 1.0)
    pm = Mock()
    pm._LIVE_MARKET_DATA_TYPES = ProcessingManager._LIVE_MARKET_DATA_TYPES
    pm._context = Mock()
    pm._context.emitter = None

    ProcessingManager.process_data(pm, instr, DataType.QUOTE, quote, is_historical=True)

    pm.process_event.assert_not_called()
    pm._ProcessingManager__process_data.assert_called_once_with(instr, DataType.QUOTE, quote, True)


# ---------------------------------------------------------------------------
# End-to-end: full ProcessingManager via the simulator
# ---------------------------------------------------------------------------


class _ReactionStrategy(IStrategy):
    """Records market-data delivery on every path so the test can assert that the
    typed dispatch feeds on_market_data, the reaction callbacks, and the cache."""

    def on_init(self, initializer: IStrategyInitializer) -> None:
        initializer.set_base_subscription("ohlc_quotes(1h)")
        initializer.subscribe("ohlc_trades(1h)")
        self._md_hits: dict[str, int] = defaultdict(int)
        self._quotes = 0
        self._trades = 0
        self._ohlc_len = 0

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent) -> list[Signal] | Signal | None:
        self._md_hits[data.type] += 1
        # cache must be populated by the time market data is delivered
        series = ctx.ohlc(ctx.instruments[0], "1h")
        self._ohlc_len = max(self._ohlc_len, len(series))

    def on_quote(self, ctx: IStrategyContext, quote: Quote) -> None:
        self._quotes += 1

    def on_trade(self, ctx: IStrategyContext, trade: Trade) -> None:
        self._trades += 1


def test_simulated_typed_dispatch_fires_callbacks_and_populates_cache():
    storage = StorageRegistry.get("csv::tests/data/storages/multi/")
    simulate(
        (s := _ReactionStrategy()),
        data=storage,
        capital=1000,
        start="2026-01-01 00:00",
        stop="2026-01-01 05:00",
        instruments=["BINANCE.UM:SWAP:BTCUSDT"],
        debug="INFO",
        silent=True,
    )

    # on_market_data is reached on both the quote and trade paths
    assert s._md_hits["quote"] > 0, f"no quote market events, got {dict(s._md_hits)}"
    assert s._md_hits["trade"] > 0, f"no trade market events, got {dict(s._md_hits)}"

    # the reaction callbacks fire alongside on_market_data, once per matching event
    assert s._quotes == s._md_hits["quote"], (s._quotes, s._md_hits["quote"])
    assert s._trades == s._md_hits["trade"], (s._trades, s._md_hits["trade"])

    # the OHLC cache was populated through the typed dispatch
    assert s._ohlc_len > 0
