import asyncio
import concurrent.futures
from asyncio.exceptions import CancelledError
from collections import defaultdict
from typing import Awaitable, Callable

import numpy as np
import pandas as pd

import ccxt.pro as cxp
from ccxt import ExchangeClosedByUser, ExchangeError, ExchangeNotAvailable, NetworkError
from qubx import logger
from qubx.core.account import BasicAccountProcessor
from qubx.core.basics import (
    CtrlChannel,
    Deal,
    Instrument,
    ITimeProvider,
    Order,
    Position,
    Timestamped,
    TransactionCostsCalculator,
    dt_64,
)
from qubx.core.helpers import extract_price
from qubx.core.interfaces import ISubscriptionManager
from qubx.utils.marketdata.ccxt import ccxt_symbol_to_instrument
from qubx.utils.misc import AsyncThreadLoop

from .exceptions import CcxtSymbolNotRecognized
from .utils import (
    ccxt_convert_balance,
    ccxt_convert_deal_info,
    ccxt_convert_order_info,
    ccxt_convert_positions,
    ccxt_convert_ticker,
    ccxt_extract_deals_from_exec,
    ccxt_find_instrument,
    ccxt_restore_position_from_deals,
    instrument_to_ccxt_symbol,
)


class CcxtAccountProcessor(BasicAccountProcessor):
    """
    Subscribes to account information from the exchange.
    """

    exchange: cxp.Exchange
    channel: CtrlChannel
    base_currency: str
    balance_interval: str
    position_interval: str
    subscription_interval: str
    max_position_restore_days: int
    max_retries: int

    _loop: AsyncThreadLoop
    _polling_tasks: dict[str, concurrent.futures.Future]
    _subscription_manager: ISubscriptionManager | None
    _polling_to_init: dict[str, bool]
    _required_instruments: set[Instrument]
    _latest_instruments: set[Instrument]

    _free_capital: float = np.nan
    _total_capital: float = np.nan
    _instrument_to_last_price: dict[Instrument, tuple[dt_64, float]]

    def __init__(
        self,
        account_id: str,
        exchange: cxp.Exchange,
        channel: CtrlChannel,
        time_provider: ITimeProvider,
        base_currency: str,
        tcc: TransactionCostsCalculator,
        balance_interval: str = "30Sec",
        position_interval: str = "30Sec",
        subscription_interval: str = "10Sec",
        open_order_interval: str = "1Min",
        open_order_backoff: str = "1Min",
        max_position_restore_days: int = 5,
        max_retries: int = 10,
        read_only: bool = False,
    ):
        super().__init__(
            account_id=account_id,
            time_provider=time_provider,
            base_currency=base_currency,
            tcc=tcc,
            initial_capital=0,
        )
        self.exchange = exchange
        self.channel = channel
        self.max_retries = max_retries
        self.balance_interval = balance_interval
        self.position_interval = position_interval
        self.subscription_interval = subscription_interval
        self.open_order_interval = open_order_interval
        self.open_order_backoff = open_order_backoff
        self.max_position_restore_days = max_position_restore_days
        self._loop = AsyncThreadLoop(exchange.asyncio_loop)
        self._is_running = False
        self._polling_tasks = {}
        self._polling_to_init = defaultdict(bool)
        self._instrument_to_last_price = {}
        self._required_instruments = set()
        self._latest_instruments = set()
        self._subscription_manager = None
        self._read_only = read_only

    def set_subscription_manager(self, manager: ISubscriptionManager) -> None:
        self._subscription_manager = manager

    def start(self):
        """Start the balance and position polling tasks"""
        channel = self.channel
        if channel is None or not channel.control.is_set():
            return
        if self._subscription_manager is None:
            return
        if self._is_running:
            logger.debug("Account polling is already running")
            return

        self._is_running = True

        if not self.exchange.isSandboxModeEnabled:
            # - start polling tasks
            self._polling_tasks["balance"] = self._loop.submit(
                self._poller("balance", self._update_balance, self.balance_interval)
            )
            self._polling_tasks["position"] = self._loop.submit(
                self._poller("position", self._update_positions, self.position_interval)
            )

            # - start initialization tasks
            _init_tasks = [
                self._loop.submit(self._init_spot_positions()),  # restore spot positions
                self._loop.submit(self._init_open_orders()),  # fetch open orders
            ]

            logger.info("Waiting for account polling tasks to be initialized")
            _waiter = self._loop.submit(self._wait_for_init(*_init_tasks))
            _waiter.result()
            logger.info("Account polling tasks have been initialized")

        # - start subscription polling task
        self._polling_tasks["subscription"] = self._loop.submit(
            self._poller("subscription", self._update_subscriptions, self.subscription_interval)
        )
        # - subscribe to order executions
        self._polling_tasks["executions"] = self._loop.submit(self._subscribe_executions("executions", channel))
        # - sync open orders
        self._polling_tasks["open_orders"] = self._loop.submit(
            self._poller(
                "open_orders", self._sync_open_orders, self.open_order_interval, backoff=self.open_order_backoff
            )
        )

    def stop(self):
        """Stop all polling tasks"""
        for task in self._polling_tasks.values():
            if not task.done():
                task.cancel()
        self._polling_tasks.clear()
        self._is_running = False

    def update_position_price(self, time: dt_64, instrument: Instrument, update: float | Timestamped) -> None:
        self._instrument_to_last_price[instrument] = (time, extract_price(update))
        super().update_position_price(time, instrument, update)

    def get_total_capital(self, exchange: str | None = None) -> float:
        # sum of balances + market value of all positions on non spot/margin
        _currency_to_value = {c: self._get_currency_value(b.total, c) for c, b in self._balances.items()}
        _positions_value = sum([p.market_value_funds for p in self._positions.values() if p.instrument.is_futures()])
        return sum(_currency_to_value.values()) + _positions_value

    def _get_instrument_for_currency(self, currency: str) -> Instrument:
        symbol = f"{currency}/{self.base_currency}"
        market = self.exchange.market(symbol)
        exchange_name = self.exchange.name
        assert exchange_name is not None
        return ccxt_symbol_to_instrument(exchange_name, market)

    def _get_currency_value(self, amount: float, currency: str) -> float:
        if not amount:
            return 0.0
        if currency == self.base_currency:
            return amount
        instr = self._get_instrument_for_currency(currency)
        _dt, _price = self._instrument_to_last_price.get(instr, (None, None))
        if not _dt or not _price:
            logger.warning(f"Price for {instr} not available. Using 0.")
            return 0.0
        return amount * _price

    async def _poller(
        self,
        name: str,
        coroutine: Callable[[], Awaitable],
        interval: str,
        backoff: str | None = None,
    ):
        sleep_time = pd.Timedelta(interval).total_seconds()
        retries = 0

        if backoff is not None:
            sleep_time = pd.Timedelta(backoff).total_seconds()
            await asyncio.sleep(sleep_time)

        while self.channel.control.is_set():
            try:
                await coroutine()

                if not self._polling_to_init[name]:
                    logger.info(f"{name} polling task has been initialized")
                    self._polling_to_init[name] = True

                retries = 0  # Reset retry counter on success
            except CancelledError:
                logger.info(f"{name} listening has been cancelled")
                break
            except ExchangeClosedByUser:
                logger.info(f"{name} listening has been stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                logger.error(f"Error polling account data: {e}")
                retries += 1
                if retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached. Stopping poller.")
                    break
            except Exception as e:
                if not self.channel.control.is_set():
                    # If the channel is closed, then ignore all exceptions and exit
                    break
                logger.error(f"Unexpected error during account polling: {e}")
                logger.exception(e)
                retries += 1
                if retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached. Stopping poller.")
                    break
            finally:
                if not self.channel.control.is_set():
                    break
                await asyncio.sleep(min(sleep_time * (2 ** (retries)), 60))  # Exponential backoff capped at 60s

        logger.debug(f"{name} polling task has been stopped")

    async def _wait(self, condition: Callable[[], bool], sleep: float = 0.1) -> None:
        while not condition():
            await asyncio.sleep(sleep)

    async def _wait_for_init(self, *futures: concurrent.futures.Future) -> None:
        await self._wait(lambda: all(self._polling_to_init.values()))
        await self._wait(lambda: all([f.done() for f in futures]))

    async def _update_subscriptions(self) -> None:
        """Subscribe to required instruments"""
        assert self._subscription_manager is not None
        await asyncio.sleep(pd.Timedelta(self.subscription_interval).total_seconds())

        # if required instruments have changed, subscribe to them
        if not self._latest_instruments.issuperset(self._required_instruments):
            await self._subscribe_instruments(list(self._required_instruments))
            self._latest_instruments.update(self._required_instruments)

    async def _update_balance(self) -> None:
        """Fetch and update balances from exchange"""
        balances_raw = await self.exchange.fetch_balance()
        balances = ccxt_convert_balance(balances_raw)
        current_balances = self.get_balances()

        # remove balances that are not there anymore
        _removed_currencies = set(current_balances.keys()) - set(balances.keys())
        for currency in _removed_currencies:
            self.update_balance(currency, 0, 0)

        # update current balances
        for currency, data in balances.items():
            self.update_balance(currency=currency, total=data.total, locked=data.locked)

        # update required instruments that we need to subscribe to
        currencies = list(self.get_balances().keys())
        instruments = [
            self._get_instrument_for_currency(c) for c in currencies if c.upper() != self.base_currency.upper()
        ]
        self._required_instruments.update(instruments)

        # fetch tickers for instruments that don't have recent price updates
        await self._fetch_missing_tickers(instruments)

    async def _update_positions(self) -> None:
        # fetch and update positions from exchange
        ccxt_positions = await self.exchange.fetch_positions()
        positions = ccxt_convert_positions(ccxt_positions, self.exchange.name, self.exchange.markets)  # type: ignore
        # update required instruments that we need to subscribe to
        self._required_instruments.update([p.instrument for p in positions])
        # update positions
        _instrument_to_position = {p.instrument: p for p in positions}
        _current_instruments = set(self._positions.keys())
        _new_instruments = set([p.instrument for p in positions])
        # - spot positions should not be updated here, because exchanges don't provide spot positions
        # - so we have to trust deal updates to update spot positions
        _to_remove = {instr for instr in _current_instruments - _new_instruments if instr.is_futures()}
        _to_add = _new_instruments - _current_instruments
        _to_modify = _current_instruments.intersection(_new_instruments)
        _update_positions = [Position(i) for i in _to_remove] + [_instrument_to_position[i] for i in _to_modify]
        # - add new positions
        for i in _to_add:
            self._positions[i] = _instrument_to_position[i]
        # - modify existing positions
        _time = self.time_provider.time()
        for pos in _update_positions:
            self._update_instrument_position(_time, self._positions[pos.instrument], pos)

    def _update_instrument_position(self, timestamp: dt_64, current_pos: Position, new_pos: Position) -> None:
        instrument = current_pos.instrument
        quantity_diff = new_pos.quantity - current_pos.quantity
        if abs(quantity_diff) < instrument.lot_size:
            return
        _current_price = current_pos.last_update_price
        current_pos.change_position_by(timestamp, quantity_diff, _current_price)

    def _get_start_time_in_ms(self, days_before: int) -> int:
        return (self.time_provider.time() - days_before * pd.Timedelta("1d")).asm8.item() // 1000000

    def _is_our_order(self, order: Order) -> bool:
        if order.client_id is None:
            return False
        return order.client_id.startswith("qubx_")

    def _is_base_currency(self, currency: str) -> bool:
        return currency.upper() == self.base_currency

    async def _subscribe_instruments(self, instruments: list[Instrument]) -> None:
        assert self._subscription_manager is not None

        # find missing subscriptions
        _base_sub = self._subscription_manager.get_base_subscription()
        _subscribed_instruments = self._subscription_manager.get_subscribed_instruments(_base_sub)
        _add_instruments = list(set(instruments) - set(_subscribed_instruments))

        if _add_instruments:
            # subscribe to instruments
            self._subscription_manager.subscribe(_base_sub, _add_instruments)
            self._subscription_manager.commit()

    async def _fetch_missing_tickers(self, instruments: list[Instrument]) -> None:
        _current_time = self.time_provider.time()
        _fetch_instruments: list[Instrument] = []
        for instr in instruments:
            _dt, _ = self._instrument_to_last_price.get(instr, (None, None))
            if _dt is None or pd.Timedelta(_current_time - _dt) > pd.Timedelta(self.balance_interval):
                _fetch_instruments.append(instr)

        _symbol_to_instrument = {instr.symbol: instr for instr in instruments}
        if _fetch_instruments:
            logger.debug(f"Fetching missing tickers for {_fetch_instruments}")
            _fetch_symbols = [instrument_to_ccxt_symbol(instr) for instr in _fetch_instruments]
            tickers: dict[str, dict] = await self.exchange.fetch_tickers(_fetch_symbols)
            for symbol, ticker in tickers.items():
                instr = _symbol_to_instrument.get(symbol)
                if instr is not None:
                    quote = ccxt_convert_ticker(ticker)
                    self.update_position_price(_current_time, instr, quote)

    async def _init_spot_positions(self) -> None:
        # - wait for balance to be initialized
        await self._wait(lambda: self._polling_to_init["balance"])
        logger.debug("Restoring spot positions ...")

        # - get nonzero balances
        _nonzero_balances = {
            c: b.total for c, b in self._balances.items() if b.total > 0 and not self._is_base_currency(c)
        }
        _positions = []

        async def _restore_pos(currency: str, balance: float) -> None:
            try:
                _instrument = self._get_instrument_for_currency(currency)
                # - get latest order for instrument and check client id
                _latest_orders = await self._fetch_orders(_instrument, limit=1)
                if not _latest_orders:
                    return
                _latest_order = list(_latest_orders.values())[-1]
                if self._is_our_order(_latest_order):
                    # - if it's our order, then we fetch the deals and restore position
                    _deals = await self._fetch_deals(_instrument, self.max_position_restore_days)
                    _position = ccxt_restore_position_from_deals(Position(_instrument), balance, _deals)
                    _positions.append(_position)
            except Exception as e:
                logger.warning(f"Error restoring position for {currency}: {e}")

        # - restore positions
        await asyncio.gather(*[_restore_pos(c, b) for c, b in _nonzero_balances.items()])

        # - attach positions
        if _positions:
            self.attach_positions(*_positions)
            logger.debug("Restored positions ->")
            for p in _positions:
                logger.debug(f"  ::  {p}")

    async def _init_open_orders(self) -> None:
        # wait for balances and positions to be initialized
        await self._wait(lambda: all([self._polling_to_init[task] for task in ["balance", "position"]]))
        await self._sync_open_orders(initial_call=True)

    async def _sync_open_orders(self, initial_call: bool = False) -> None:
        logger.debug("[SYNC] Fetching open orders ...")

        # in order to minimize order requests we only fetch open orders for instruments that we have positions in
        _nonzero_balances = {
            c: b.total for c, b in self._balances.items() if b.total > 0 and not self._is_base_currency(c)
        }
        _balance_instruments = [self._get_instrument_for_currency(c) for c in _nonzero_balances.keys()]
        _position_instruments = list(self._positions.keys())
        _instruments = list(set(_balance_instruments + _position_instruments))

        _open_orders: dict[str, Order] = {}

        async def _add_open_orders(instrument: Instrument) -> None:
            try:
                _orders = await self._fetch_orders(instrument, is_open=True)
                _open_orders.update(_orders)
            except Exception as e:
                logger.warning(f"[SYNC] Error fetching open orders for {instrument}: {e}")

        await asyncio.gather(*[_add_open_orders(i) for i in _instruments])

        if initial_call:
            # - when it's the initial call, we add the open orders to the account
            self.add_active_orders(_open_orders)
            logger.debug(f"[SYNC] Found {len(_open_orders)} open orders ->")
            _instr_to_open_orders: dict[Instrument, list[Order]] = defaultdict(list)
            for od in _open_orders.values():
                _instr_to_open_orders[od.instrument].append(od)
            for instr, orders in _instr_to_open_orders.items():
                logger.debug(f"  :: [SYNC] {instr} ->")
                for order in orders:
                    logger.debug(f"    :: [SYNC] {order.side} {order.quantity} @ {order.price} ({order.status})")
        else:
            # - we need to cancel the unexpected orders
            if not self._read_only:
                await self._cancel_unexpected_orders(_open_orders)

    async def _cancel_unexpected_orders(self, open_orders: dict[str, Order]) -> None:
        _expected_orders = set(self._active_orders.keys())
        _unexpected_orders = set(open_orders.keys()) - _expected_orders
        if _unexpected_orders:
            logger.info(f"[SYNC] Canceling {len(_unexpected_orders)} unexpected open orders ...")
            _instr_to_orders = defaultdict(list)
            for _id in _unexpected_orders:
                _order = open_orders[_id]
                _instr_to_orders[_order.instrument].append(_order)

            async def _cancel_order(order: Order) -> None:
                try:
                    await self.exchange.cancel_order(order.id, symbol=instrument_to_ccxt_symbol(order.instrument))
                    logger.debug(
                        f"  :: [SYNC] Canceled {order.id} {order.instrument.symbol} {order.side} {order.quantity} @ {order.price} ({order.status})"
                    )
                except Exception as e:
                    logger.warning(f"[SYNC] Error canceling order {order.id}: {e}")

            for instr, orders in _instr_to_orders.items():
                logger.debug(
                    f"[SYNC] Canceling {len(orders)} (out of {len(open_orders)}) unexpected open orders for {instr}"
                )
                await asyncio.gather(*[_cancel_order(order) for order in orders])

    async def _fetch_orders(
        self, instrument: Instrument, days_before: int = 30, limit: int | None = None, is_open: bool = False
    ) -> dict[str, Order]:
        _start_ms = self._get_start_time_in_ms(days_before) if limit is None else None
        _ccxt_symbol = instrument_to_ccxt_symbol(instrument)
        _fetcher = self.exchange.fetch_open_orders if is_open else self.exchange.fetch_orders
        _raw_orders = await _fetcher(_ccxt_symbol, since=_start_ms, limit=limit)
        _orders = [ccxt_convert_order_info(instrument, o) for o in _raw_orders]
        _id_to_order = {o.id: o for o in _orders}
        return dict(sorted(_id_to_order.items(), key=lambda x: x[1].time, reverse=False))

    async def _fetch_deals(self, instrument: Instrument, days_before: int = 30) -> list[Deal]:
        _start_ms = self._get_start_time_in_ms(days_before)
        _ccxt_symbol = instrument_to_ccxt_symbol(instrument)
        deals_data = await self.exchange.fetch_my_trades(_ccxt_symbol, since=_start_ms)
        deals: list[Deal] = [ccxt_convert_deal_info(o) for o in deals_data]
        return sorted(deals, key=lambda x: x.time) if deals else []

    async def _listen_to_stream(
        self,
        subscriber: Callable[[], Awaitable[None]],
        exchange: cxp.Exchange,
        channel: CtrlChannel,
        name: str,
    ):
        logger.info(f"Listening to {name}")
        n_retry = 0
        while channel.control.is_set():
            try:
                await subscriber()
                n_retry = 0
            except CcxtSymbolNotRecognized:
                continue
            except CancelledError:
                break
            except ExchangeClosedByUser:
                # - we closed connection so just stop it
                logger.info(f"{name} listening has been stopped")
                break
            except (NetworkError, ExchangeError, ExchangeNotAvailable) as e:
                logger.error(f"Error in {name} : {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                if not channel.control.is_set():
                    # If the channel is closed, then ignore all exceptions and exit
                    break
                logger.error(f"exception in {name} : {e}")
                logger.exception(e)
                n_retry += 1
                if n_retry >= self.max_retries:
                    logger.error(f"Max retries reached for {name}. Closing connection.")
                    del exchange
                    break
                await asyncio.sleep(min(2**n_retry, 60))  # Exponential backoff with a cap at 60 seconds

    async def _subscribe_executions(self, name: str, channel: CtrlChannel):
        _symbol_to_instrument = {}

        async def _watch_executions():
            exec = await self.exchange.watch_orders()
            for report in exec:
                instrument = ccxt_find_instrument(report["symbol"], self.exchange, _symbol_to_instrument)
                order = ccxt_convert_order_info(instrument, report)
                deals = ccxt_extract_deals_from_exec(report)
                channel.send((instrument, "order", order, False))
                if deals:
                    channel.send((instrument, "deals", deals, False))

        await self._listen_to_stream(
            subscriber=_watch_executions,
            exchange=self.exchange,
            channel=channel,
            name=name,
        )
