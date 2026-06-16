from qubx import logger
from qubx.core.basics import DataType, Instrument
from qubx.core.detectors import DelistingDetector
from qubx.core.instrument_service import IInstrumentService, NullInstrumentService
from qubx.core.interfaces import (
    IAccountProcessor,
    IMarketManager,
    IPositionGathering,
    IStrategy,
    IStrategyContext,
    ISubscriptionManager,
    ITimeProvider,
    ITradingManager,
    IUniverseManager,
    RemovalPolicy,
)
from qubx.core.loggers import StrategyLogging
from qubx.utils.time import to_timestamp


class UniverseManager(IUniverseManager):
    _context: IStrategyContext
    _strategy: IStrategy
    _mkt_manager: IMarketManager
    _logging: StrategyLogging
    _subscription_manager: ISubscriptionManager
    _trading_manager: ITradingManager
    _time_provider: ITimeProvider
    _account: IAccountProcessor
    _position_gathering: IPositionGathering
    _warmup_position_gathering: IPositionGathering
    _removal_queue: dict[Instrument, tuple[RemovalPolicy, bool]]
    _delisting_detector: DelistingDetector
    _instrument_service: IInstrumentService

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        market_data_manager: IMarketManager,
        logging: StrategyLogging,
        subscription_manager: ISubscriptionManager,
        trading_manager: ITradingManager,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        position_gathering: IPositionGathering,
        delisting_detector: DelistingDetector,
        instrument_service: IInstrumentService | None = None,
    ):
        self._context = context
        self._strategy = strategy
        self._mkt_manager = market_data_manager
        self._logging = logging
        self._subscription_manager = subscription_manager
        self._trading_manager = trading_manager
        self._time_provider = time_provider
        self._account = account
        self._position_gathering = position_gathering
        self._instruments = set()
        self._removal_queue = {}
        self._removal_in_progress = set()
        self._delisting_detector = delisting_detector
        self._instrument_service = instrument_service if instrument_service is not None else NullInstrumentService()

    def _has_position(self, instrument: Instrument) -> bool:
        return (
            instrument in self._account.positions
            and abs(self._account.positions[instrument].quantity) > instrument.min_size
        )

    def _is_market_gone(self, instrument: Instrument) -> bool:
        """State B only: the market no longer exists / is untradeable.
        A future delist_date (state A) is NOT gone."""
        if not self._mkt_manager.is_instrument_listed(instrument):
            return True  # authoritative live signal
        d = instrument.delist_date
        if d is None:
            return False
        try:
            delist_ts = to_timestamp(d).replace(tzinfo=None)
            now_ts = to_timestamp(self._time_provider.time())
        except (TypeError, ValueError):
            # - unparseable delist_date: fail-open, treat as not gone
            return False
        return delist_ts <= now_ts

    def _settle_if_held(self, instrument: Instrument) -> None:
        if not self._has_position(instrument):
            return
        if not self._mkt_manager.is_instrument_listed(instrument):
            # live-confirmed gone: cannot trade out, exchange already settled
            self._trading_manager.cancel_orders(instrument)
            self._account.settle_position(instrument)
            logger.warning(f"[UniverseManager] Settled delisted position {instrument.symbol} (market gone)")
        else:
            # flagged by past delist_date metadata only, but still listed (settlement
            # overlap) -- leave it for the close-via-trade path / manual review
            logger.warning(
                f"[UniverseManager] {instrument.symbol} flagged delisted by metadata but still listed; "
                "leaving position for close-via-trade / manual review"
            )

    def _notify_gone(self, instruments: list[Instrument]) -> None:
        symbols = ", ".join(sorted(i.symbol for i in instruments))
        logger.warning(f"[UniverseManager] Dropping delisted (gone) instruments: {symbols}")
        notifier = getattr(self._context, "notifier", None)
        if notifier is not None and not self._context.is_simulation:
            notifier.notify_message(
                f"[{self._context.strategy_name}] Dropped delisted (gone) instruments: {symbols}",
                metadata={"event": "delisted_gone", "instruments": symbols},
            )

    def _drop_gone(self, instruments: list[Instrument]) -> list[Instrument]:
        gone = [i for i in instruments if self._is_market_gone(i)]
        if not gone:
            return instruments
        for i in gone:
            self._settle_if_held(i)
        self._notify_gone(gone)
        gone_set = set(gone)
        return [i for i in instruments if i not in gone_set]

    def _filter_blacklisted(self, instruments: list[Instrument]) -> list[Instrument]:
        kept = [i for i in instruments if not self._instrument_service.is_blacklisted(i)]
        dropped = [i for i in instruments if i not in set(kept)]
        if dropped:
            symbols = ", ".join(sorted(i.symbol for i in dropped))
            logger.info(f"[UniverseManager] Dropping blacklisted instruments: {symbols}")
        return kept

    def set_universe(
        self,
        instruments: list[Instrument],
        skip_callback: bool = False,
        if_has_position_then: RemovalPolicy = "close",
    ) -> None:
        assert if_has_position_then in (
            "close",
            "wait_for_close",
            "wait_for_change",
        ), "Invalid if_has_position_then policy"

        # Settle & exclude instruments whose market is already gone (state B) FIRST,
        # so a gone instrument that also carries a delist_date is settled in place
        # before the delisting filter (state A) would otherwise strip it from the list.
        instruments = self._drop_gone(instruments)

        # Then filter out instruments with upcoming/scheduled delist dates (state A:
        # still listed -> closed via trade through the normal removal path).
        instruments = self._delisting_detector.filter_delistings(instruments)

        instruments = self._filter_blacklisted(instruments)

        new_set = set(instruments)
        prev_set = self._instruments.copy()

        # - determine instruments to remove depending on if_has_position_then policy
        may_be_removed = list(prev_set - new_set)

        # - split instruments into removable and keepable
        to_remove, to_keep = self._get_what_can_be_removed_or_kept(may_be_removed, skip_callback, if_has_position_then)

        to_add = list(new_set - prev_set)
        self.__do_add_instruments(to_add)
        self.__do_remove_instruments(to_remove)

        # - cleanup removal queue
        self.__cleanup_removal_queue(instruments)

        self._subscription_manager.commit()  # apply pending changes

        if not skip_callback and (to_add or to_remove):
            self._strategy.on_universe_change(self._context, to_add, to_remove)

        # set new instruments
        self._instruments = new_set | set(to_keep)

    def _get_what_can_be_removed_or_kept(
        self, may_be_removed: list[Instrument], skip_callback: bool, if_has_position_then: RemovalPolicy
    ) -> tuple[list[Instrument], list[Instrument]]:
        immediately_close = if_has_position_then == "close"
        to_remove, to_keep = [], []
        for instr in may_be_removed:
            if immediately_close:
                to_remove.append(instr)
            elif self._has_position(instr):
                self._removal_queue[instr] = (if_has_position_then, skip_callback)
                to_keep.append(instr)
            else:
                to_remove.append(instr)
        return to_remove, to_keep

    def __cleanup_removal_queue(self, instruments: list[Instrument]):
        for instr in instruments:
            # - if it's still in the removal queue, remove it
            if instr in self._removal_queue:
                self._removal_queue.pop(instr)

    def add_instruments(self, instruments: list[Instrument]):
        instruments = self._filter_blacklisted(instruments)
        to_add = list(set([instr for instr in instruments if instr not in self._instruments]))
        self.__do_add_instruments(to_add)
        self.__cleanup_removal_queue(instruments)
        self._strategy.on_universe_change(self._context, to_add, [])
        self._subscription_manager.commit()
        self._instruments.update(to_add)

    def remove_instruments(
        self,
        instruments: list[Instrument],
        if_has_position_then: RemovalPolicy = "close",
    ):
        assert if_has_position_then in (
            "close",
            "wait_for_close",
            "wait_for_change",
        ), "Invalid if_has_position_then policy"

        # - split instruments into removable and keepable
        to_remove, to_keep = self._get_what_can_be_removed_or_kept(instruments, False, if_has_position_then)

        # - remove ones that can be removed immediately
        self.__do_remove_instruments(to_remove)
        self._strategy.on_universe_change(self._context, [], to_remove)
        self._subscription_manager.commit()

        # - update instruments list
        self._instruments = (self._instruments - set(to_remove)) | set(to_keep)

    @property
    def instruments(self) -> list[Instrument]:
        return list(self._instruments)

    def __do_remove_instruments(self, instruments: list[Instrument]):
        """
        Remove symbols from universe. Steps:
        - [v] cancel all open orders
        - [v] close all open positions
        - [v] unsubscribe from market data
        - [v] remove from data cache

        We are still keeping the symbols in the positions dictionary.
        """
        if not instruments:
            return

        # - preprocess instruments and cancel all open orders
        for instr in instruments:
            # - remove instrument from the removal queue if it's there
            self._removal_queue.pop(instr, None)

            # - cancel all open orders
            self._trading_manager.cancel_orders(instr)

        # - close all open positions
        exit_targets = []
        for instr in instruments:
            if self._has_position(instr):
                if not self._mkt_manager.is_instrument_listed(instr):
                    # market gone -- cannot trade; settle in place
                    self._account.settle_position(instr)
                    logger.warning(f"[UniverseManager] Settled delisted position {instr.symbol} on removal")
                    continue

                # - create exit target
                exit_targets.append(instr.target(self._context, 0))

                self._removal_in_progress.add(instr)

                # - emit service signals for instruments that are being removed
                self._context.emit_signal(
                    instr.service_signal(self._context, 0, group="Universe", comment="Universe change")
                )

        # - alter positions
        self._position_gathering.alter_positions(self._context, exit_targets)

        # - unsubscribe from market data
        for instr in instruments:
            if instr not in self._removal_in_progress:
                self._subscription_manager.unsubscribe(DataType.ALL, instr)
                self._mkt_manager.get_market_data_cache().remove(instr)

    def __do_add_instruments(self, instruments: list[Instrument]) -> None:
        # - create positions for instruments
        self._create_and_update_positions(instruments)

        # - initialize ohlcv for new instruments
        for instr in instruments:
            self._mkt_manager.get_market_data_cache().init_ohlcv(instr)

            if instr in self._removal_in_progress:
                self._removal_in_progress.discard(instr)

        # - subscribe to market data
        self._subscription_manager.subscribe(
            (
                DataType.ALL
                if self._subscription_manager.auto_subscribe
                else self._subscription_manager.get_base_subscription()
            ),
            instruments,
        )

        # - reinitialize strategy loggers
        self._logging.initialize(self._time_provider.time(), self._account.positions, self._account.get_balances(), self._account)

    def _create_and_update_positions(self, instruments: list[Instrument]):
        for instrument in instruments:
            _ = self._account.get_position(instrument)

            # - check if we need any aux instrument for calculating pnl ?
            # TODO: test edge cases for aux symbols (UniverseManager)
            # aux = lookup.find_aux_instrument_for(instrument, self._account.get_base_currency())
            # if aux is not None:
            #     instrument._aux_instrument = aux
            #     instruments.append(aux)
            #     _ = self._account.get_position(aux)

    def on_alter_position(self, instrument: Instrument) -> None:
        """
        Called when the position of an instrument changes.
        It can be used for postponed unsubscribed events
        """
        # - check if need to remove instrument from the universe
        if instrument in self._removal_queue:
            _, skip_callback = self._removal_queue[instrument]

            # - if no position, remove instrument from the universe
            if not self._has_position(instrument):
                self.__do_remove_instruments([instrument])

                if not skip_callback:
                    self._strategy.on_universe_change(self._context, [], [instrument])

                # - commit changes and remove instrument from the universe
                self._subscription_manager.commit()
                self._instruments.discard(instrument)

        if instrument in self._removal_in_progress:
            if not self._has_position(instrument):
                self._removal_in_progress.discard(instrument)
                self._subscription_manager.unsubscribe(DataType.ALL, instrument)
                self._mkt_manager.get_market_data_cache().remove(instrument)
                self._subscription_manager.commit()
                self._instruments.discard(instrument)

    def is_trading_allowed(self, instrument: Instrument) -> bool:
        if instrument in self._removal_queue:
            policy, skip_callback = self._removal_queue[instrument]

            if policy == "wait_for_change":
                self.__do_remove_instruments([instrument])

                if not skip_callback:
                    self._strategy.on_universe_change(self._context, [], [instrument])

                # - commit changes and remove instrument from the universe
                self._subscription_manager.commit()
                self._instruments.discard(instrument)
                return False

        return True
