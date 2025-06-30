import asyncio
import concurrent.futures
import re
from asyncio.exceptions import CancelledError
from collections import defaultdict
from threading import Thread
from types import FunctionType
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

import ccxt.pro as cxp
from ccxt import (
    ExchangeClosedByUser,
    ExchangeError,
    ExchangeNotAvailable,
    NetworkError,
)
from ccxt.pro import Exchange
from qubx import logger
from qubx.core.basics import CtrlChannel, DataType, Instrument, ITimeProvider, dt_64
from qubx.core.helpers import BasicScheduler
from qubx.core.interfaces import IDataProvider, IHealthMonitor
from qubx.core.series import Bar, Quote
from qubx.health import DummyHealthMonitor
from qubx.utils.misc import AsyncThreadLoop

from .exceptions import CcxtLiquidationParsingError, CcxtSymbolNotRecognized
from .utils import (
    ccxt_convert_funding_rate,
    ccxt_convert_liquidation,
    ccxt_convert_orderbook,
    ccxt_convert_ticker,
    ccxt_convert_trade,
    ccxt_find_instrument,
    instrument_to_ccxt_symbol,
)


class CcxtDataProvider(IDataProvider):
    time_provider: ITimeProvider
    _exchange: Exchange
    _scheduler: BasicScheduler | None = None

    # - subscriptions
    _subscriptions: Dict[str, Set[Instrument]]
    _pending_subscriptions: Dict[str, Set[Instrument]]  # Track subscriptions being established
    _sub_to_coro: Dict[str, concurrent.futures.Future]
    _sub_to_name: Dict[str, str]
    _sub_to_unsubscribe: Dict[str, Callable[[], Awaitable[None]]]
    _is_sub_name_enabled: Dict[str, bool]
    _sub_connection_ready: Dict[str, bool]  # Track if connection is actually ready

    _sub_instr_to_time: Dict[Tuple[str, Instrument], dt_64]
    _last_quotes: Dict[Instrument, Optional[Quote]]
    _loop: AsyncThreadLoop
    _thread_event_loop: Thread
    _warmup_timeout: int

    _subscribers: Dict[str, Callable]
    _warmupers: Dict[str, Callable]

    def __init__(
        self,
        exchange: cxp.Exchange,
        time_provider: ITimeProvider,
        channel: CtrlChannel,
        max_ws_retries: int = 10,
        warmup_timeout: int = 120,
        health_monitor: IHealthMonitor | None = None,
    ):
        self._exchange_id = str(exchange.name)
        self.time_provider = time_provider
        self.channel = channel
        self.max_ws_retries = max_ws_retries
        self._warmup_timeout = warmup_timeout
        self._health_monitor = health_monitor or DummyHealthMonitor()

        # - create new even loop
        self._exchange = exchange
        self._loop = AsyncThreadLoop(self._exchange.asyncio_loop)

        self._last_quotes = defaultdict(lambda: None)
        self._subscriptions = defaultdict(set)
        self._pending_subscriptions = defaultdict(set)
        self._sub_to_coro = {}
        self._sub_to_name = {}
        self._sub_to_unsubscribe = {}
        self._is_sub_name_enabled = defaultdict(lambda: False)
        self._sub_connection_ready = defaultdict(lambda: False)
        self._symbol_to_instrument = {}
        self._subscribers = {
            n.split("_subscribe_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) is FunctionType and n.startswith("_subscribe_")
        }
        self._warmupers = {
            n.split("_warmup_")[1]: f
            for n, f in self.__class__.__dict__.items()
            if type(f) is FunctionType and n.startswith("_warmup_")
        }
        logger.info(f"<yellow>{self._exchange_id}</yellow> Initialized")

    @property
    def is_simulation(self) -> bool:
        return False

    def subscribe(
        self,
        subscription_type: str,
        instruments: List[Instrument],
        reset: bool = False,
    ) -> None:
        _updated_instruments = set(instruments)
        # - update symbol to instrument mapping
        self._symbol_to_instrument.update({i.symbol: i for i in instruments})

        # - add existing subscription instruments if reset is False
        if not reset:
            _current_instruments = self.get_subscribed_instruments(subscription_type)
            _updated_instruments = _updated_instruments.union(_current_instruments)

        # - update subscriptions
        self._subscribe(_updated_instruments, subscription_type)

    def unsubscribe(self, subscription_type: str, instruments: List[Instrument]) -> None:
        # _current_instruments = self.get_subscribed_instruments(subscription_type)
        # _updated_instruments = set(_current_instruments).difference(instruments)
        # self._subscribe(_updated_instruments, subscription_type)
        # unsubscribe functionality is handled for ccxt via subscribe with reset=True
        if subscription_type in self._subscriptions:
            self._subscriptions[subscription_type] = self._subscriptions[subscription_type].difference(instruments)

    def get_subscriptions(self, instrument: Instrument | None = None) -> List[str]:
        if instrument is not None:
            return [sub for sub, instrs in self._subscriptions.items() if instrument in instrs]
        return [sub for sub, instruments in self._subscriptions.items() if instruments]

    def get_subscribed_instruments(self, subscription_type: str | None = None) -> list[Instrument]:
        if not subscription_type:
            return list(self.subscribed_instruments)

        # Return active subscriptions, fallback to pending if no active ones
        _sub_type, _ = DataType.from_str(subscription_type)
        if _sub_type in self._subscriptions:
            return list(self._subscriptions[_sub_type])
        elif _sub_type in self._pending_subscriptions:
            return list(self._pending_subscriptions[_sub_type])
        else:
            return []

    def has_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        sub_type, _ = DataType.from_str(subscription_type)
        # Only return True if subscription is actually active (not just pending)
        return (
            sub_type in self._subscriptions
            and instrument in self._subscriptions[sub_type]
            and self._sub_connection_ready.get(sub_type, False)
        )

    def has_pending_subscription(self, instrument: Instrument, subscription_type: str) -> bool:
        """Check if a subscription is pending (connection being established)."""
        sub_type, _ = DataType.from_str(subscription_type)
        return (
            sub_type in self._pending_subscriptions
            and instrument in self._pending_subscriptions[sub_type]
            and not self._sub_connection_ready.get(subscription_type, False)
        )

    def warmup(self, warmups: Dict[Tuple[str, Instrument], str]) -> None:
        _coros = []

        for (sub_type, instrument), period in warmups.items():
            _sub_type, _params = DataType.from_str(sub_type)
            _warmuper = self._warmupers.get(_sub_type)
            if _warmuper is None:
                logger.warning(f"<yellow>{self._exchange_id}</yellow> Warmup for {sub_type} is not supported")
                continue
            _coros.append(
                _warmuper(
                    self,
                    channel=self.channel,
                    instrument=instrument,
                    warmup_period=period,
                    **_params,
                )
            )

        async def gather_coros():
            return await asyncio.gather(*_coros)

        if _coros:
            self._loop.submit(gather_coros()).result(self._warmup_timeout)

    def get_quote(self, instrument: Instrument) -> Quote | None:
        return self._last_quotes[instrument]

    def get_ohlc(self, instrument: Instrument, timeframe: str, nbarsback: int) -> List[Bar]:
        assert nbarsback >= 1
        symbol = instrument.symbol
        since = self._time_msec_nbars_back(timeframe, nbarsback)

        # - retrieve OHLC data
        # - TODO: check if nbarsback > max_limit (1000) we need to do more requests
        # - TODO: how to get quoted volumes ?
        async def _get():
            return await self._exchange.fetch_ohlcv(
                symbol, self._get_exch_timeframe(timeframe), since=since, limit=nbarsback + 1
            )  # type: ignore

        res = self._loop.submit(_get()).result(60)

        _arr = []
        for oh in res:  # type: ignore
            _arr.append(
                Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7])
                if len(oh) > 6
                else Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[5])
            )

        return _arr

    def close(self):
        try:
            if hasattr(self._exchange, "close"):
                future = self._loop.submit(self._exchange.close())  # type: ignore
                # - wait for 5 seconds for connection to close
                future.result(5)
            else:
                del self._exchange
        except Exception as e:
            logger.error(e)

    @property
    def subscribed_instruments(self) -> Set[Instrument]:
        active = set.union(*self._subscriptions.values()) if self._subscriptions else set()
        pending = set.union(*self._pending_subscriptions.values()) if self._pending_subscriptions else set()
        return active.union(pending)

    @property
    def is_read_only(self) -> bool:
        _key = self._exchange.apiKey
        return _key is None or _key == ""

    def _subscribe(
        self,
        instruments: Set[Instrument],
        sub_type: str,
    ) -> None:
        _sub_type, _params = DataType.from_str(sub_type)
        _subscriber = self._subscribers.get(_sub_type)
        if _subscriber is None:
            raise ValueError(f"{self._exchange_id}: Subscription type {sub_type} is not supported")

        # Save old subscription state before starting cleanup
        old_sub_info = None
        if sub_type in self._sub_to_coro:
            old_sub_info = {"name": self._sub_to_name[sub_type], "coro": self._sub_to_coro[sub_type]}
            logger.debug(
                f"<yellow>{self._exchange_id}</yellow> Canceling existing {sub_type} subscription for {self._subscriptions.get(_sub_type, set())}"
            )

            # Clear state immediately to prevent interference with new subscription
            del self._sub_to_coro[sub_type]
            del self._sub_to_name[sub_type]
            # Clean up both active and pending subscriptions
            self._subscriptions.pop(_sub_type, None)
            self._pending_subscriptions.pop(_sub_type, None)
            self._sub_connection_ready.pop(sub_type, None)

        if instruments is not None and len(instruments) == 0:
            return

        # Mark subscription as pending (not active yet)
        self._pending_subscriptions[_sub_type] = instruments
        self._sub_connection_ready[sub_type] = False

        kwargs = {"instruments": instruments, **_params}
        _subscriber = self._subscribers[_sub_type]
        _subscriber_params = set(_subscriber.__code__.co_varnames[: _subscriber.__code__.co_argcount])
        # - get only parameters that are needed for subscriber
        kwargs = {k: v for k, v in kwargs.items() if k in _subscriber_params}
        self._sub_to_name[sub_type] = (name := self._get_subscription_name(_sub_type, **kwargs))
        self._sub_to_coro[sub_type] = self._loop.submit(_subscriber(self, name, _sub_type, self.channel, **kwargs))

        # Now stop the old subscriber after new one is started (to avoid interference)
        if old_sub_info is not None:
            # Stop old subscriber in background to avoid blocking
            self._loop.submit(self._stop_old_subscriber(old_sub_info["name"], old_sub_info["coro"]))

        # Don't set _subscriptions here - it will be set when connection is established

    def _time_msec_nbars_back(self, timeframe: str, nbarsback: int = 1) -> int:
        return (self.time_provider.time() - nbarsback * pd.Timedelta(timeframe)).asm8.item() // 1000000

    def _get_exch_timeframe(self, timeframe: str) -> str:
        if timeframe is not None:
            _t = re.match(r"(\d+)(\w+)", timeframe)
            timeframe = f"{_t[1]}{_t[2][0].lower()}" if _t and len(_t.groups()) > 1 else timeframe

        tframe = self._exchange.find_timeframe(timeframe)
        if tframe is None:
            raise ValueError(f"timeframe {timeframe} is not supported by {self._exchange.name}")

        return tframe

    def _get_exch_symbol(self, instrument: Instrument) -> str:
        return f"{instrument.base}/{instrument.quote}:{instrument.settle}"

    def _get_subscription_name(
        self, subscription: str, instruments: List[Instrument] | Set[Instrument] | Instrument | None = None, **kwargs
    ) -> str:
        if isinstance(instruments, Instrument):
            instruments = [instruments]
        _symbols = [instrument_to_ccxt_symbol(i) for i in instruments] if instruments is not None else []
        _name = f"{','.join(_symbols)} {subscription}" if _symbols else subscription
        if kwargs:
            kwargs_str = ",".join(f"{k}={v}" for k, v in kwargs.items())
            _name += f" ({kwargs_str})"
        return _name

    def _mark_subscription_active(self, sub_type: str) -> None:
        """Mark a subscription as active once the WebSocket connection is established."""
        _sub_type, _ = DataType.from_str(sub_type)
        if _sub_type in self._pending_subscriptions:
            self._subscriptions[_sub_type] = self._pending_subscriptions[_sub_type]
            self._sub_connection_ready[sub_type] = True
            logger.debug(f"<yellow>{self._exchange_id}</yellow> Subscription {sub_type} is now active")

    async def _stop_subscriber(self, sub_type: str, sub_name: str) -> None:
        try:
            self._is_sub_name_enabled[sub_name] = False  # stop the subscriber
            future = self._sub_to_coro[sub_type]
            total_sleep_time = 0.0
            while future.running():
                await asyncio.sleep(1.0)
                total_sleep_time += 1.0
                if total_sleep_time >= 20.0:
                    break

            if future.running():
                logger.warning(
                    f"<yellow>{self._exchange_id}</yellow> Subscriber {sub_name} is still running. Cancelling it."
                )
                future.cancel()
            else:
                logger.debug(f"<yellow>{self._exchange_id}</yellow> Subscriber {sub_name} has been stopped")

            if sub_name in self._sub_to_unsubscribe:
                logger.debug(f"<yellow>{self._exchange_id}</yellow> Unsubscribing from {sub_name}")
                await self._sub_to_unsubscribe[sub_name]()
                del self._sub_to_unsubscribe[sub_name]

            del self._is_sub_name_enabled[sub_name]

            # Clean up connection state for this subscription
            for sub_type, stream_name in list(self._sub_to_name.items()):
                if stream_name == sub_name:
                    self._sub_connection_ready.pop(sub_type, None)
                    break

            logger.debug(f"<yellow>{self._exchange_id}</yellow> Unsubscribed from {sub_name}")
        except Exception as e:
            logger.error(f"<yellow>{self._exchange_id}</yellow> Error stopping {sub_name}")
            logger.exception(e)

    async def _stop_old_subscriber(self, old_name: str, old_coro: concurrent.futures.Future) -> None:
        """Stop an old subscriber safely without interfering with new subscriptions."""
        try:
            # Disable the old stream by name
            self._is_sub_name_enabled[old_name] = False

            # Wait for the old coroutine to finish
            total_sleep_time = 0.0
            while old_coro.running():
                await asyncio.sleep(1.0)
                total_sleep_time += 1.0
                if total_sleep_time >= 20.0:
                    break

            if old_coro.running():
                logger.warning(
                    f"<yellow>{self._exchange_id}</yellow> Old subscriber {old_name} is still running. Cancelling it."
                )
                old_coro.cancel()
            else:
                logger.debug(f"<yellow>{self._exchange_id}</yellow> Old subscriber {old_name} has been stopped")

            # Clean up old unsubscriber if it exists
            if old_name in self._sub_to_unsubscribe:
                logger.debug(f"<yellow>{self._exchange_id}</yellow> Calling old unsubscriber for {old_name}")
                await self._sub_to_unsubscribe[old_name]()
                # Use pop to safely remove, in case it was already removed
                self._sub_to_unsubscribe.pop(old_name, None)

            # Clean up old stream state
            if old_name in self._is_sub_name_enabled:
                del self._is_sub_name_enabled[old_name]

            logger.debug(f"<yellow>{self._exchange_id}</yellow> Old subscription {old_name} cleaned up")

        except Exception as e:
            logger.error(f"<yellow>{self._exchange_id}</yellow> Error stopping old subscriber {old_name}")
            logger.exception(e)

    async def _listen_to_stream(
        self,
        subscriber: Callable[[], Awaitable[None]],
        exchange: Exchange,
        channel: CtrlChannel,
        name: str,
        unsubscriber: Callable[[], Awaitable[None]] | None = None,
    ):
        logger.info(f"<yellow>{self._exchange_id}</yellow> Listening to {name}")
        if unsubscriber is not None:
            self._sub_to_unsubscribe[name] = unsubscriber

        self._is_sub_name_enabled[name] = True
        n_retry = 0
        connection_established = False

        while channel.control.is_set() and self._is_sub_name_enabled[name]:
            try:
                await subscriber()
                n_retry = 0

                # Mark subscription as active on first successful data reception
                if not connection_established:
                    # Find the subscription type for this stream name
                    for sub_type, stream_name in self._sub_to_name.items():
                        if stream_name == name:
                            self._mark_subscription_active(sub_type)
                            connection_established = True
                            break

                if not self._is_sub_name_enabled[name]:
                    break
            except CcxtSymbolNotRecognized:
                continue
            except CancelledError:
                break
            except ExchangeClosedByUser:
                # - we closed connection so just stop it
                logger.info(f"<yellow>{self._exchange_id}</yellow> {name} listening has been stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                logger.error(f"<yellow>{self._exchange_id}</yellow> Error in {name} : {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                if not channel.control.is_set() or not self._is_sub_name_enabled[name]:
                    # If the channel is closed, then ignore all exceptions and exit
                    break
                logger.error(f"<yellow>{self._exchange_id}</yellow> Exception in {name}: {e}")
                n_retry += 1
                if n_retry >= self.max_ws_retries:
                    logger.error(
                        f"<yellow>{self._exchange_id}</yellow> Max retries reached for {name}. Closing connection."
                    )
                    del exchange
                    break
                await asyncio.sleep(min(2**n_retry, 60))  # Exponential backoff with a cap at 60 seconds

    #############################
    # - Warmup methods
    #############################
    async def _warmup_ohlc(
        self, channel: CtrlChannel, instrument: Instrument, warmup_period: str, timeframe: str
    ) -> None:
        nbarsback = pd.Timedelta(warmup_period) // pd.Timedelta(timeframe)
        exch_timeframe = self._get_exch_timeframe(timeframe)
        start = self._time_msec_nbars_back(timeframe, nbarsback)
        ohlcv = await self._exchange.fetch_ohlcv(instrument.symbol, exch_timeframe, since=start, limit=nbarsback + 1)
        logger.debug(f"<yellow>{self._exchange_id}</yellow> {instrument}: loaded {len(ohlcv)} {timeframe} bars")
        channel.send(
            (
                instrument,
                DataType.OHLC[timeframe],
                [Bar(oh[0] * 1_000_000, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]) for oh in ohlcv],
                True,
            )
        )

    async def _warmup_trade(self, channel: CtrlChannel, instrument: Instrument, warmup_period: str):
        trades = await self._exchange.fetch_trades(instrument.symbol, since=self._time_msec_nbars_back(warmup_period))
        logger.debug(f"<yellow>{self._exchange_id}</yellow> Loaded {len(trades)} trades for {instrument}")
        channel.send(
            (
                instrument,
                DataType.TRADE,
                [ccxt_convert_trade(trade) for trade in trades],
                True,
            )
        )

    def _call_by_market_type(
        self, subscriber: Callable[[list[Instrument]], Awaitable[None]], instruments: set[Instrument]
    ) -> Any:
        """Call subscriber for each market type"""
        _instr_by_type: dict[str, list[Instrument]] = defaultdict(list)
        for instr in instruments:
            _instr_by_type[instr.market_type].append(instr)

        # sort instruments by symbol
        for instrs in _instr_by_type.values():
            instrs.sort(key=lambda i: i.symbol)

        async def _call_subscriber():
            await asyncio.gather(*[subscriber(instrs) for instrs in _instr_by_type.values()])

        return _call_subscriber

    #############################
    # - Subscription methods
    #############################
    async def _subscribe_ohlc(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        timeframe: str = "1m",
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _exchange_timeframe = self._get_exch_timeframe(timeframe)
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_ohlcv(instruments: list[Instrument]):
            _symbol_timeframe_pairs = [[_instr_to_ccxt_symbol[i], _exchange_timeframe] for i in instruments]
            ohlcv = await self._exchange.watch_ohlcv_for_symbols(_symbol_timeframe_pairs)
            # - ohlcv is symbol -> timeframe -> list[timestamp, open, high, low, close, volume]
            for exch_symbol, _data in ohlcv.items():
                instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                for _, ohlcvs in _data.items():
                    for oh in ohlcvs:
                        timestamp_ns = oh[0] * 1_000_000
                        self._health_monitor.record_data_arrival(sub_type, dt_64(timestamp_ns, "ns"))
                        channel.send(
                            (
                                instrument,
                                sub_type,
                                Bar(timestamp_ns, oh[1], oh[2], oh[3], oh[4], oh[6], oh[7]),
                                False,  # not historical bar
                            )
                        )
                    if not (
                        self.has_subscription(instrument, DataType.ORDERBOOK)
                        or self.has_subscription(instrument, DataType.QUOTE)
                    ):
                        _price = ohlcvs[-1][4]
                        _s2 = instrument.tick_size / 2.0
                        _bid, _ask = _price - _s2, _price + _s2
                        self._last_quotes[instrument] = Quote(oh[0] * 1_000_000, _bid, _ask, 0.0, 0.0)

        # ohlc subscription reuses the same connection always, unsubscriptions don't work properly
        # but it's likely not very needed
        # async def un_watch_ohlcv(instruments: list[Instrument]):
        #     _symbol_timeframe_pairs = [[_instr_to_ccxt_symbol[i], _exchange_timeframe] for i in instruments]
        #     await self._exchange.un_watch_ohlcv_for_symbols(_symbol_timeframe_pairs)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_ohlcv, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            # unsubscriber=self._call_by_market_type(un_watch_ohlcv, instruments),
        )

    async def _subscribe_trade(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_trades(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            trades = await self._exchange.watch_trades_for_symbols(symbols)
            exch_symbol = trades[0]["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
            for trade in trades:
                converted_trade = ccxt_convert_trade(trade)
                self._health_monitor.record_data_arrival(sub_type, dt_64(converted_trade.time, "ns"))
                channel.send((instrument, sub_type, converted_trade, False))

        async def un_watch_trades(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            await self._exchange.un_watch_trades_for_symbols(symbols)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_trades, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_trades, instruments),
        )

    async def _subscribe_orderbook(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        tick_size_pct: float = 0.01,
        depth: int = 200,
    ):
        if self._exchange.has.get("watchOrderBookForSymbols", False):
            await self._subscribe_orderbook_for_instruments(name, sub_type, channel, instruments, tick_size_pct, depth)
        else:
            subs = []
            for instrument in instruments:
                subs.append(
                    self._subscribe_orderbook_for_instrument(name, sub_type, channel, instrument, tick_size_pct, depth)
                )
            await asyncio.gather(*subs)

    async def _subscribe_orderbook_for_instrument(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instrument: Instrument,
        tick_size_pct: float,
        depth: int,
    ):
        ccxt_symbol = instrument_to_ccxt_symbol(instrument)

        async def watch_orderbook():
            ccxt_ob = await self._exchange.watch_order_book(ccxt_symbol)
            ob = ccxt_convert_orderbook(
                ccxt_ob,
                instrument,
                levels=depth,
                tick_size_pct=tick_size_pct,
                current_timestamp=self.time_provider.time(),
            )

            if ob is None:
                return

            self._health_monitor.record_data_arrival(sub_type, dt_64(ob.time, "ns"))

            if not self.has_subscription(instrument, DataType.QUOTE):
                quote = ob.to_quote()
                self._last_quotes[instrument] = quote

            channel.send((instrument, sub_type, ob, False))

        async def un_watch_orderbook():
            if hasattr(self._exchange, "un_watch_order_book"):
                await self._exchange.un_watch_order_book(ccxt_symbol)

        await self._listen_to_stream(
            subscriber=watch_orderbook,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_orderbook,
        )

    async def _subscribe_orderbook_for_instruments(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
        tick_size_pct: float,
        depth: int,
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_orderbook(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            ccxt_ob = await self._exchange.watch_order_book_for_symbols(symbols)
            exch_symbol = ccxt_ob["symbol"]
            instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
            ob = ccxt_convert_orderbook(ccxt_ob, instrument, levels=depth, tick_size_pct=tick_size_pct)
            if ob is None:
                return

            self._health_monitor.record_data_arrival(sub_type, dt_64(ob.time, "ns"))

            if not self.has_subscription(instrument, DataType.QUOTE):
                quote = ob.to_quote()
                self._last_quotes[instrument] = quote

            channel.send((instrument, sub_type, ob, False))

        async def un_watch_orderbook(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            await self._exchange.un_watch_order_book_for_symbols(symbols)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_orderbook, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_orderbook, instruments),
        )

    async def _subscribe_quote(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        if not self._exchange.has.get("watchBidsAsks", False):
            logger.warning(f"<yellow>{self._exchange_id}</yellow> watchBidsAsks is not supported for {name}")
            self.unsubscribe(sub_type, list(instruments))
            return

        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_quote(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            ccxt_tickers: dict[str, dict] = await self._exchange.watch_bids_asks(symbols)
            for exch_symbol, ccxt_ticker in ccxt_tickers.items():  # type: ignore
                instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                quote = ccxt_convert_ticker(ccxt_ticker)
                last_quote = self._last_quotes[instrument]
                if last_quote is None or quote.time > last_quote.time:
                    self._health_monitor.record_data_arrival(sub_type, dt_64(quote.time, "ns"))
                    self._last_quotes[instrument] = quote
                    channel.send((instrument, sub_type, quote, False))

        async def un_watch_quote(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            if hasattr(self._exchange, "un_watch_bids_asks"):
                await getattr(self._exchange, "un_watch_bids_asks")(symbols)
            else:
                await self._exchange.un_watch_tickers(symbols)

        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_quote, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_quote, instruments),
        )

    async def _subscribe_liquidation(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
        instruments: Set[Instrument],
    ):
        _instr_to_ccxt_symbol = {i: instrument_to_ccxt_symbol(i) for i in instruments}
        _symbol_to_instrument = {_instr_to_ccxt_symbol[i]: i for i in instruments}

        async def watch_liquidation(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            liquidations = await self._exchange.watch_liquidations_for_symbols(symbols)
            for liquidation in liquidations:
                try:
                    exch_symbol = liquidation["symbol"]
                    instrument = ccxt_find_instrument(exch_symbol, self._exchange, _symbol_to_instrument)
                    liquidation_event = ccxt_convert_liquidation(liquidation)
                    self._health_monitor.record_data_arrival(sub_type, dt_64(liquidation_event.time, "ns"))
                    channel.send((instrument, sub_type, liquidation_event, False))
                except CcxtLiquidationParsingError:
                    logger.debug(f"<yellow>{self._exchange_id}</yellow> Could not parse liquidation {liquidation}")
                    continue

        async def un_watch_liquidation(instruments: list[Instrument]):
            symbols = [_instr_to_ccxt_symbol[i] for i in instruments]
            unwatch = getattr(self._exchange, "un_watch_liquidations_for_symbols", lambda _: None)(symbols)
            if unwatch is not None:
                await unwatch

        # - fetching of liquidations for warmup is not supported by ccxt
        await self._listen_to_stream(
            subscriber=self._call_by_market_type(watch_liquidation, instruments),
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=self._call_by_market_type(un_watch_liquidation, instruments),
        )

    async def _subscribe_funding_rate(
        self,
        name: str,
        sub_type: str,
        channel: CtrlChannel,
    ):
        # it is expected that we can retrieve funding rates for all instruments
        async def watch_funding_rates():
            funding_rates = await self._exchange.watch_funding_rates()  # type: ignore
            instrument_to_funding_rate = {}
            current_time = self.time_provider.time()

            for symbol, info in funding_rates.items():
                try:
                    instrument = ccxt_find_instrument(symbol, self._exchange)
                    funding_rate = ccxt_convert_funding_rate(info)
                    instrument_to_funding_rate[instrument] = funding_rate
                    self._health_monitor.record_data_arrival(sub_type, dt_64(current_time, "s"))
                except CcxtSymbolNotRecognized:
                    continue

            channel.send((None, sub_type, instrument_to_funding_rate, False))

        async def un_watch_funding_rates():
            unwatch = getattr(self._exchange, "un_watch_funding_rates", lambda: None)()
            if unwatch is not None:
                await unwatch

        await self._listen_to_stream(
            subscriber=watch_funding_rates,
            exchange=self._exchange,
            channel=channel,
            name=name,
            unsubscriber=un_watch_funding_rates,
        )

    def exchange(self) -> str:
        return self._exchange_id.upper()
