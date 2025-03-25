from qubx.core.basics import DataType, Instrument, TargetPosition
from qubx.core.exceptions import SymbolNotFound
from qubx.core.helpers import CachedMarketDataHolder
from qubx.core.interfaces import (
    IAccountProcessor,
    IBroker,
    IDataProvider,
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
from qubx.core.lookups import lookup


class UniverseManager(IUniverseManager):
    _context: IStrategyContext
    _strategy: IStrategy
    _broker: IDataProvider
    _trading_service: IBroker
    _cache: CachedMarketDataHolder
    _logging: StrategyLogging
    _subscription_manager: ISubscriptionManager
    _trading_manager: ITradingManager
    _time_provider: ITimeProvider
    _account: IAccountProcessor
    _position_gathering: IPositionGathering
    _removal_queue: dict[Instrument, tuple[RemovalPolicy, bool]]

    def __init__(
        self,
        context: IStrategyContext,
        strategy: IStrategy,
        broker: IDataProvider,
        trading_service: IBroker,
        cache: CachedMarketDataHolder,
        logging: StrategyLogging,
        subscription_manager: ISubscriptionManager,
        trading_manager: ITradingManager,
        time_provider: ITimeProvider,
        account: IAccountProcessor,
        position_gathering: IPositionGathering,
    ):
        self._context = context
        self._strategy = strategy
        self._broker = broker
        self._trading_service = trading_service
        self._cache = cache
        self._logging = logging
        self._subscription_manager = subscription_manager
        self._trading_manager = trading_manager
        self._time_provider = time_provider
        self._account = account
        self._position_gathering = position_gathering
        self._instruments = []
        self._removal_queue = {}

    def _has_position(self, instrument: Instrument) -> bool:
        return (
            instrument in self._account.positions
            and abs(self._account.positions[instrument].quantity) > instrument.min_size
        )

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

        new_set = set(instruments)
        prev_set = set(self._instruments)

        # - determine instruments to remove depending on if_has_position_then policy
        may_be_removed = list(prev_set - new_set)

        # - split instruments into removable and keepable
        to_remove, to_keep = self._get_what_can_be_removed_or_kept(may_be_removed, skip_callback, if_has_position_then)

        to_add = list(new_set - prev_set)
        self.__do_add_instruments(to_add)
        self.__do_remove_instruments(to_remove)

        # - cleanup removal queue
        self.__cleanup_removal_queue(instruments)

        if not skip_callback and (to_add or to_remove):
            self._strategy.on_universe_change(self._context, to_add, to_remove)

        self._subscription_manager.commit()  # apply pending changes

        # set new instruments
        self._instruments.clear()
        self._instruments.extend(instruments)
        self._instruments.extend(to_keep)

    def _get_what_can_be_removed_or_kept(
        self, may_be_removed: list[Instrument], skip_callback: bool, if_has_position_then: RemovalPolicy
    ) -> tuple[list[Instrument], list[Instrument]]:
        immediately_close = if_has_position_then == "close"
        to_remove, to_keep = [], []
        for instr in may_be_removed:
            if immediately_close:
                to_remove.append(instr)
            else:
                if self._has_position(instr):
                    self._removal_queue[instr] = (if_has_position_then, skip_callback)
                    to_keep.append(instr)
        return to_remove, to_keep

    def __cleanup_removal_queue(self, instruments: list[Instrument]):
        for instr in instruments:
            # - if it's still in the removal queue, remove it
            if instr in self._removal_queue:
                self._removal_queue.pop(instr)

    def add_instruments(self, instruments: list[Instrument]):
        self.__do_add_instruments(instruments)
        self.__cleanup_removal_queue(instruments)
        self._strategy.on_universe_change(self._context, instruments, [])
        self._subscription_manager.commit()
        self._instruments.extend(instruments)

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
        self._instruments = list(set(self._instruments) - set(to_remove))
        self._instruments.extend(to_keep)

    def find_instrument(self, symbol: str, exchange: str | None = None) -> Instrument:
        if exchange is None:
            parts = symbol.split(":")
            if len(parts) == 2:
                exchange, symbol = parts
            else:
                raise SymbolNotFound(f"Invalid symbol: {symbol}")

        instrument = lookup.find_symbol(exchange, symbol)
        if instrument is None:
            raise SymbolNotFound(f"Symbol not found: {symbol} on {exchange}")
        return instrument

    @property
    def instruments(self) -> list[Instrument]:
        return self._instruments

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
        exit_targets = [
            TargetPosition.zero(self._context, instr.signal(0, group="Universe", comment="Universe change"))
            for instr in instruments
            if self._has_position(instr)
        ]
        self._position_gathering.alter_positions(self._context, exit_targets)

        # - if still open positions close them manually
        for instr in instruments:
            pos = self._account.positions.get(instr)
            if pos and abs(pos.quantity) > instr.min_size:
                self._trading_manager.trade(instr, -pos.quantity)

        # - unsubscribe from market data
        for instr in instruments:
            self._subscription_manager.unsubscribe(DataType.ALL, instr)

        # - remove from data cache
        for instr in instruments:
            self._cache.remove(instr)

    def __do_add_instruments(self, instruments: list[Instrument]) -> None:
        # - create positions for instruments
        self._create_and_update_positions(instruments)

        # - initialize ohlcv for new instruments
        for instr in instruments:
            self._cache.init_ohlcv(instr)

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
        self._logging.initialize(self._time_provider.time(), self._account.positions, self._account.get_balances())

    def _create_and_update_positions(self, instruments: list[Instrument]):
        for instrument in instruments:
            _ = self._account.get_position(instrument)

            # - check if we need any aux instrument for calculating pnl ?
            # TODO: test edge cases for aux symbols (UniverseManager)
            # aux = lookup.find_aux_instrument_for(instrument, self._account.get_base_currency())
            # if aux is not None:
            #     instrument._aux_instrument = aux
            #     instruments.append(aux)
            #     _ = self._trading_service.get_position(aux)

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
                self._instruments.remove(instrument)

    def is_trading_allowed(self, instrument: Instrument) -> bool:
        if instrument in self._removal_queue:
            policy, skip_callback = self._removal_queue[instrument]

            if policy == "wait_for_change":
                self.__do_remove_instruments([instrument])

                if not skip_callback:
                    self._strategy.on_universe_change(self._context, [], [instrument])

                # - commit changes and remove instrument from the universe
                self._subscription_manager.commit()
                self._instruments.remove(instrument)
                return False

        return True
