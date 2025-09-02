from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from qubx import logger
from qubx.core.basics import (
    OPTION_FILL_AT_SIGNAL_PRICE,
    OPTION_SIGNAL_PRICE,
    OPTION_SKIP_PRICE_CROSS_CONTROL,
    Deal,
    InitializingSignal,
    Instrument,
    Signal,
    TargetPosition,
)
from qubx.core.exceptions import OrderNotFound
from qubx.core.interfaces import IPositionSizer, IStrategyContext, PositionsTracker
from qubx.core.series import Bar, OrderBook, Quote, Trade
from qubx.emitters import indicator_emitter
from qubx.ta.indicators import atr, psar, swings
from qubx.trackers.sizers import FixedSizer

RiskControllingSide: TypeAlias = Literal["broker", "client"]


class State(Enum):
    NEW = 0
    OPEN = 1
    RISK_TRIGGERED = 2
    DONE = 3


@dataclass
class SgnCtrl:
    signal: Signal
    target: TargetPosition
    status: State = State.NEW
    take_order_id: str | None = None
    stop_order_id: str | None = None
    take_executed_price: float | None = None
    stop_executed_price: float | None = None


class RiskCalculator:
    def calculate_risks(self, ctx: IStrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        return signal


class RiskController(PositionsTracker):
    _trackings: dict[Instrument, SgnCtrl]
    _waiting: dict[Instrument, SgnCtrl]
    _risk_calculator: RiskCalculator

    def __init__(self, name: str, risk_calculator: RiskCalculator, sizer: IPositionSizer) -> None:
        self._name = f"{name}.{self.__class__.__name__}"
        self._risk_calculator = risk_calculator
        self._trackings = {}
        self._waiting = {}
        super().__init__(sizer)

    @staticmethod
    def _get_price(update: float | Quote | Trade | Bar | OrderBook, direction: int) -> float:
        if isinstance(update, float):
            return update
        elif isinstance(update, Quote):
            return update.ask if direction > 0 else update.bid
        elif isinstance(update, OrderBook):
            return update.top_ask if direction > 0 else update.top_bid
        elif isinstance(update, Trade):
            return update.price
        elif isinstance(update, Bar):
            return update.close
        else:
            raise ValueError(f"Unknown update type: {type(update)}")

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        targets = []
        for s in signals:
            quote = ctx.quote(s.instrument)
            if quote is None:
                logger.warning(
                    f"[<y>{self._name}</y>] :: Quote is not available for <g>{s.instrument}</g>. Skipping signal {s}"
                )
                continue

            # - calculate risk, we allow modifications of the original signal
            signal_with_risk = self._risk_calculator.calculate_risks(ctx, quote, s)
            if signal_with_risk is None:
                continue

            # - final step - calculate actual target position and check if tracker can approve it
            target = self.get_position_sizer().calculate_target_positions(ctx, [signal_with_risk])[0]
            if self.handle_new_target(ctx, s, target):
                targets.append(target)

        return targets

    def handle_new_target(self, ctx: IStrategyContext, signal: Signal, target: TargetPosition) -> bool:
        """
        As it doesn't use any referenced orders for position - new target is always approved
        """
        # - add first in waiting list
        self._waiting[signal.instrument] = SgnCtrl(signal, target, State.NEW)
        logger.debug(
            f"[<y>{self._name}</y>(<g>{signal.instrument}</g>)] :: Processing signal ({signal.signal}) to target: <c><b>{target}</b></c>"
        )

        return True

    def get_stop_level(self, ctx: IStrategyContext, instrument: Instrument) -> float | None:
        """
        Get current stop level for the given instrument. If there is no stop level or positions is still not open - return None
        This method must be overriden in class that inherits from RiskController
        """
        # - no tracking for this instrument
        if (c := self._trackings.get(instrument)) is None:
            return None
        return c.signal.stop

    def get_take_level(self, ctx: IStrategyContext, instrument: Instrument) -> float | None:
        """
        Get current take level for the given instrument. If there is no take level or positions is still not open - return None
        This method can be overriden in class that inherits from RiskController
        """
        # - no tracking for this instrument
        if (c := self._trackings.get(instrument)) is None:
            return None
        return c.signal.take

    def update_stop_level(self, ctx: IStrategyContext, instrument: Instrument, new_stop_level: float) -> bool:
        """
        Update stop level for the given instrument.
        This method must be implemented in class that inherits from RiskController

        Args:
            ctx: IStrategyContext
            instrument: Instrument
            new_stop_level: float

        Returns:
            bool: True if stop level was updated successfully, False otherwise
        """
        ...

    def update_take_level(self, ctx: IStrategyContext, instrument: Instrument, new_take_level: float) -> bool:
        """
        Update take level for the given instrument
        This method must be implemented in class that inherits from RiskController

        Args:
            ctx: IStrategyContext
            instrument: Instrument
            new_take_level: float

        Returns:
            bool: True if take level was updated successfully, False otherwise
        """
        return False

    def is_active(self, instrument: Instrument) -> bool:
        return instrument in self._trackings


class ClientSideRiskController(RiskController):
    """
    Risk is controlled on client (Qubx) side without using limit order for take and stop order for loss.
    So when risk is triggered, it uses market orders to close position immediatelly.
    As result it may lead to significant slippage.
    """

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar | OrderBook
    ) -> list[TargetPosition] | TargetPosition:
        c = self._trackings.get(instrument)
        if c is None:
            return []

        match c.status:
            case State.NEW:
                # - nothing to do just waiting for position to be open
                pass

            case State.RISK_TRIGGERED:
                # - nothing to do just waiting for position to be closed
                pass

            case State.OPEN:
                pos = c.target.target_position_size
                if c.signal.stop:
                    if (
                        pos > 0
                        and self._get_price(update, +1) <= c.signal.stop
                        or (pos < 0 and self._get_price(update, -1) >= c.signal.stop)
                    ):
                        c.status = State.RISK_TRIGGERED
                        logger.debug(
                            f"[<y>{self._name}</y>(<g>{c.signal.instrument}</g>)] :: triggered <red>STOP LOSS</red> at {c.signal.stop}"
                        )

                        # - emit service signal that risk triggeres (it won't be processed by StrategyContext)
                        ctx.emit_signal(
                            instrument.service_signal(
                                ctx, 0, price=c.signal.stop, group="Risk Manager", comment="Stop triggered"
                            )
                        )

                        return instrument.target(
                            ctx,
                            0.0,
                            options={
                                OPTION_FILL_AT_SIGNAL_PRICE: True,
                                OPTION_SIGNAL_PRICE: c.signal.stop,
                                OPTION_SKIP_PRICE_CROSS_CONTROL: True,
                            },
                        )

                if c.signal.take:
                    if (
                        pos > 0
                        and self._get_price(update, -1) >= c.signal.take
                        or (pos < 0 and self._get_price(update, +1) <= c.signal.take)
                    ):
                        c.status = State.RISK_TRIGGERED
                        logger.debug(
                            f"[<y>{self._name}</y>(<g>{c.signal.instrument}</g>)] :: triggered <g>TAKE PROFIT</g> at {c.signal.take}"
                        )

                        # - emit service signal that risk triggeres (it won't be processed by StrategyContext)
                        ctx.emit_signal(
                            instrument.service_signal(
                                ctx, 0, price=c.signal.take, group="Risk Manager", comment="Take triggered"
                            )
                        )
                        return instrument.target(
                            ctx,
                            0.0,
                            options={
                                OPTION_FILL_AT_SIGNAL_PRICE: True,
                                OPTION_SIGNAL_PRICE: c.signal.take,
                                OPTION_SKIP_PRICE_CROSS_CONTROL: True,
                            },
                        )

            case State.DONE:
                logger.debug(f"[<y>{self._name}</y>(<g>{c.signal.instrument}</g>)] :: <m>Stop tracking</m>")
                self._trackings.pop(instrument)

        return []

    def cancel_tracking(self, ctx: IStrategyContext, instrument: Instrument):
        # - remove from waiting list if exists
        if instrument in self._waiting:
            logger.debug(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: <m>Tracking cancelled and removed from waiting list</m>"
            )
            self._waiting.pop(instrument)

        c = self._trackings.get(instrument)

        # - if there is no tracking or it's already done - do nothing
        if c is None or c.status == State.DONE:
            return

        if c.status != State.OPEN:
            logger.warning(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: <m>Canceling tracking when tracker in active state (position: {ctx.positions[instrument].quantity})</m> : <red>({c.status})</red>"
            )
        # - cancel everything in any case ("")
        logger.debug(
            f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: <m>Tracking cancelled and removed from active list</m>"
        )
        self._trackings.pop(instrument)

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal):
        pos = ctx.positions[instrument].quantity

        # - check what is in the waiting list
        if (c_w := self._waiting.get(instrument)) is not None:
            if abs(pos - c_w.target.target_position_size) <= instrument.min_size:
                c_w.status = State.OPEN
                self._trackings[instrument] = c_w  # add to tracking
                self._waiting.pop(instrument)  # remove from waiting
                logger.debug(
                    f"[<y>{self._name}</y>(<g>{c_w.signal.instrument.symbol}</g>)] :: Start tracking <cyan><b>{c_w.target}</b></cyan>"
                )
                return

        # - check what is in the tracking list
        if (c_t := self._trackings.get(instrument)) is not None:
            if c_t.status == State.RISK_TRIGGERED and abs(pos) <= instrument.min_size:
                c_t.status = State.DONE

    def update_stop_level(self, ctx: IStrategyContext, instrument: Instrument, new_stop_level: float) -> bool:
        # - find the tracking and update the stop level
        if (c := self._trackings.get(instrument)) is None:
            return False

        # - update the stop level only if position is open
        if c.status == State.OPEN:
            logger.debug(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Move stop level from {c.signal.stop} to {new_stop_level}"
            )
            c.signal.stop = new_stop_level
            return True
        else:
            logger.warning(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Can't update stop level for non active position: {c.status}"
            )
        return False

    def update_take_level(self, ctx: IStrategyContext, instrument: Instrument, new_take_level: float) -> bool:
        # - find the tracking and update the take level
        if (c := self._trackings.get(instrument)) is None:
            return False

        # - update the stop level only if position is open
        if c.status == State.OPEN:
            logger.debug(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Move take level from {c.signal.stop} to {new_take_level}"
            )
            c.signal.take = new_take_level
            return True
        else:
            logger.warning(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Can't update take level for non active position: {c.status}"
            )
        return False


class BrokerSideRiskController(RiskController):
    """
    Risk is managed on the broker's side by using limit orders for take and stop order for loss.
    For backtesting we assume that stop orders are executed by it's price.
    """

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar | OrderBook
    ) -> list[TargetPosition]:
        # fmt: off
        c = self._trackings.get(instrument)
        if c is None:
            return []

        match c.status:
            case State.NEW:
                # - nothing to do just waiting for position to be open
                pass

            case State.RISK_TRIGGERED:
                c.status = State.DONE

                # - remove from the tracking list
                logger.debug(
                    f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: risk triggered - <m>Stop tracking</m>"
                )
                self._trackings.pop(instrument)

                if c.stop_executed_price: 
                    # - emit service signal that risk triggeres (it won't be processed by StrategyContext)
                    ctx.emit_signal(
                        instrument.service_signal(ctx, 0, price=c.stop_executed_price, group="Risk Manager", comment="Stop triggered")
                    )

                elif c.take_executed_price: 
                    # - emit service signal that risk triggeres (it won't be processed by StrategyContext)
                    ctx.emit_signal(
                        instrument.service_signal(ctx, 0, price=c.take_executed_price, group="Risk Manager", comment="Take triggered")
                    )

            case State.DONE:
                logger.debug(
                    f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] state is done - <m>Stop tracking</m>"
                )
                self._trackings.pop(instrument)

        # fmt: on
        return []

    def __cncl_stop(self, ctx: IStrategyContext, ctrl: SgnCtrl):
        if ctrl.stop_order_id is not None:
            logger.debug(
                f"[<y>{self._name}</y>(<g>{ctrl.signal.instrument}</g>)] :: <m>Canceling stop order</m> <red>{ctrl.stop_order_id}</red>"
            )
            try:
                ctx.cancel_order(ctrl.stop_order_id)
            except OrderNotFound:
                # - order was already cancelled (expected during universe changes)
                logger.debug(
                    f"[<y>{self._name}</y>(<g>{ctrl.signal.instrument}</g>)] :: <m>Stop order</m> <red>{ctrl.stop_order_id}</red> already cancelled"
                )
            except Exception as e:
                # - unexpected cancellation error
                logger.error(
                    f"[<y>{self._name}</y>(<g>{ctrl.signal.instrument}</g>)] :: <m>Canceling stop order</m> <red>{ctrl.stop_order_id}</red> failed: {str(e)}"
                )
            ctrl.stop_order_id = None

    def __cncl_take(self, ctx: IStrategyContext, ctrl: SgnCtrl):
        if ctrl.take_order_id is not None:
            logger.debug(
                f"[<y>{self._name}(<g>{ctrl.signal.instrument}</g>)</y>] :: <m>Canceling take order</m> <r>{ctrl.take_order_id}</r>"
            )
            try:
                ctx.cancel_order(ctrl.take_order_id)
            except OrderNotFound:
                # - order was already cancelled (expected during universe changes)
                logger.debug(
                    f"[<y>{self._name}(<g>{ctrl.signal.instrument}</g>)</y>] :: <m>Take order</m> <r>{ctrl.take_order_id}</r> already cancelled"
                )
            except Exception as e:
                # - unexpected cancellation error
                logger.error(
                    f"[<y>{self._name}(<g>{ctrl.signal.instrument}</g>)</y>] :: <m>Canceling take order</m> <r>{ctrl.take_order_id}</r> failed: {str(e)}"
                )
            ctrl.take_order_id = None

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal):
        pos = ctx.positions[instrument].quantity
        _waiting = self._waiting.get(instrument)

        # - check if there is any waiting signals
        if _waiting is not None:
            _tracking = self._trackings.get(instrument)
            # - when asked to process 0 signal and got new execution - we need to remove all previous orders
            if _waiting.target.target_position_size == 0:
                self._waiting.pop(instrument)  # remove from waiting

                if _tracking is not None:
                    logger.debug(
                        f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: got execution from <r>{deal.order_id}</r> and before was asked to process 0 by <c>{_waiting.signal}</c>"
                    )
                    self.__cncl_stop(ctx, _tracking)
                    self.__cncl_take(ctx, _tracking)

                return

            # - when gathered asked position
            if abs(pos - _waiting.target.target_position_size) <= instrument.min_size:
                _waiting.status = State.OPEN
                logger.debug(
                    f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: <r>{deal.order_id}</r> opened position for <c>{_waiting.signal}</c>"
                )

                # - check if we need to cancel previous stop / take orders
                if _tracking is not None:
                    self.__cncl_stop(ctx, _tracking)
                    self.__cncl_take(ctx, _tracking)

                self._trackings[instrument] = _waiting  # add to tracking
                self._waiting.pop(instrument)  # remove from waiting

                if _waiting.target.take:
                    try:
                        logger.debug(
                            f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: sending <g>take limit</g> order at {_waiting.target.take}"
                        )
                        order = ctx.trade(instrument, -pos, _waiting.target.take)
                        _waiting.take_order_id = order.id

                        # - if order was executed immediately we don't need to send stop order
                        if order.status == "CLOSED":
                            _waiting.status = State.RISK_TRIGGERED
                            logger.debug(
                                f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: <g>TAKE PROFIT</g> was exected immediately at {_waiting.target.take}"
                            )
                            return

                    except Exception as e:
                        logger.error(
                            f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: couldn't send take limit order: {str(e)}"
                        )

                if _waiting.target.stop:
                    try:
                        logger.debug(
                            f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: sending <g>stop</g> order at {_waiting.target.stop}"
                        )
                        # - for simulation purposes we assume that stop order will be executed at stop price
                        order = ctx.trade(
                            instrument,
                            -pos,
                            _waiting.target.stop,
                            stop_type="market",
                            fill_at_signal_price=True,
                            avoid_stop_order_price_validation=True,
                        )
                        _waiting.stop_order_id = order.id
                    except Exception as e:
                        logger.error(
                            f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: couldn't send stop order: {str(e)}"
                        )

        # - check tracked signal
        if (_tracking := self._trackings.get(instrument)) is not None:
            if _tracking.status == State.OPEN and abs(pos) <= instrument.min_size:
                if deal.order_id == _tracking.take_order_id:
                    _tracking.status = State.RISK_TRIGGERED
                    _tracking.take_executed_price = deal.price
                    logger.debug(
                        f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: triggered <green>TAKE PROFIT</green> (<red>{_tracking.take_order_id}</red>) at {_tracking.take_executed_price}"
                    )
                    # - cancel stop if need
                    self.__cncl_stop(ctx, _tracking)

                elif deal.order_id == _tracking.stop_order_id:
                    _tracking.status = State.RISK_TRIGGERED
                    _tracking.stop_executed_price = deal.price
                    logger.debug(
                        f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: triggered <magenta>STOP LOSS</magenta> (<red>{_tracking.take_order_id}</red>) at {_tracking.stop_executed_price}"
                    )
                    # - cancel take if need
                    self.__cncl_take(ctx, _tracking)

                else:
                    # - closed by opposite signal or externally
                    _tracking.status = State.DONE
                    self.__cncl_stop(ctx, _tracking)
                    self.__cncl_take(ctx, _tracking)

    def cancel_tracking(self, ctx: IStrategyContext, instrument: Instrument):
        _waiting = self._waiting.get(instrument)
        _tracking = self._trackings.get(instrument)

        # - check if there is any waiting signals
        if _waiting is not None:
            self._waiting.pop(instrument)

        if _tracking is not None:
            logger.warning(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: <m>Canceling tracking when tracker in active state (position: {ctx.positions[instrument].quantity})</m> : <red>({_tracking.status})</red>"
            )
            _tracking.status = State.DONE
            self._trackings.pop(instrument)
            self.__cncl_stop(ctx, _tracking)
            self.__cncl_take(ctx, _tracking)

    def update_stop_level(self, ctx: IStrategyContext, instrument: Instrument, new_stop_level: float) -> bool:
        # - find the tracking and update the stop level
        if (_tracked := self._trackings.get(instrument)) is None:
            return False

        # - update the stop level only if position is open
        if _tracked.status != State.OPEN:
            logger.warning(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Can't update stop level for non active position: {_tracked.status}"
            )
            return False

        logger.debug(
            f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Move stop level from {_tracked.signal.stop} to {new_stop_level}"
        )

        # - set new level for reference
        _tracked.signal.stop = new_stop_level

        # - cancel previos stop order
        self.__cncl_stop(ctx, _tracked)

        # - try to send new stop order
        try:
            pos = ctx.positions[instrument].quantity
            logger.debug(
                f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: sending updated <g>stop</g> order at {new_stop_level}"
            )
            # - for simulation purposes we assume that stop order will be executed at stop price
            order = ctx.trade(
                instrument,
                -pos,
                new_stop_level,
                stop_type="market",
                fill_at_signal_price=True,
                avoid_stop_order_price_validation=True,
            )
            _tracked.stop_order_id = order.id
        except Exception as e:
            logger.error(f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: couldn't send stop order: {str(e)}")

        return True

    def update_take_level(self, ctx: IStrategyContext, instrument: Instrument, new_take_level: float) -> bool:
        if (_tracked := self._trackings.get(instrument)) is None:
            return False

        # - update the stop level only if position is open
        if _tracked.status != State.OPEN:
            logger.warning(
                f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Can't update take level for non active position: {_tracked.status}"
            )
            return False

        logger.debug(
            f"[<y>{self._name}</y>(<g>{instrument.symbol}</g>)] :: Move stop level from {_tracked.signal.take} to {new_take_level}"
        )

        # - set new level for reference
        _tracked.signal.take = new_take_level

        # - try to send new stop order
        try:
            pos = ctx.positions[instrument].quantity
            logger.debug(
                f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: sending updated <g>take</g> order at {new_take_level}"
            )
            # - for simulation purposes we assume that stop order will be executed at stop price
            order = ctx.trade(instrument, -pos, new_take_level)
            _tracked.take_order_id = order.id
        except Exception as e:
            logger.error(f"[<y>{self._name}</y>(<g>{instrument}</g>)] :: couldn't send take order: {str(e)}")

        return True


class GenericRiskControllerDecorator(PositionsTracker, RiskCalculator):
    riskctrl: RiskController

    def __init__(
        self,
        sizer: IPositionSizer,
        riskctrl: RiskController,
    ) -> None:
        super().__init__(sizer)
        self.riskctrl = riskctrl

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        return self.riskctrl.process_signals(ctx, signals)

    def is_active(self, instrument: Instrument) -> bool:
        return self.riskctrl.is_active(instrument)

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar | OrderBook
    ) -> list[TargetPosition] | TargetPosition:
        return self.riskctrl.update(ctx, instrument, update)

    def on_execution_report(self, ctx: IStrategyContext, instrument: Instrument, deal: Deal):
        return self.riskctrl.on_execution_report(ctx, instrument, deal)

    def calculate_risks(self, ctx: IStrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        raise NotImplementedError("calculate_risks should be implemented by subclasses")

    @staticmethod
    def create_risk_controller_for_side(
        name: str, risk_controlling_side: RiskControllingSide, risk_calculator: RiskCalculator, sizer: IPositionSizer
    ) -> RiskController:
        match risk_controlling_side:
            case "broker":
                return BrokerSideRiskController(name, risk_calculator, sizer)
            case "client":
                return ClientSideRiskController(name, risk_calculator, sizer)
            case _:
                raise ValueError(
                    f"Invalid risk controlling side: {risk_controlling_side} for {name} only 'broker' or 'client' are supported"
                )

    def cancel_tracking(self, ctx: IStrategyContext, instrument: Instrument):
        self.riskctrl.cancel_tracking(ctx, instrument)


class StopTakePositionTracker(GenericRiskControllerDecorator):
    """
    Basic fixed stop-take position tracker. It observes position opening or closing and controls stop-take logic.
    It may use either limit and stop orders for managing risk or market orders depending on 'risk_controlling_side' parameter.
    """

    def __init__(
        self,
        take_target: float | None = None,
        stop_risk: float | None = None,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
        risk_controlling_side: RiskControllingSide = "broker",
        purpose: str = "",  # if we need to distinguish different instances of the same tracker, i.e. for shorts or longs etc
    ) -> None:
        self.take_target = take_target
        self.stop_risk = stop_risk
        self._take_target_fraction = take_target / 100 if take_target else None
        self._stop_risk_fraction = stop_risk / 100 if stop_risk else None

        super().__init__(
            sizer,
            GenericRiskControllerDecorator.create_risk_controller_for_side(
                f"{self.__class__.__name__}{purpose}", risk_controlling_side, self, sizer
            ),
        )

    def calculate_risks(self, ctx: IStrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        if signal.signal > 0:
            entry = signal.price if signal.price else quote.ask
            if self._take_target_fraction:
                signal.take = entry * (1 + self._take_target_fraction)
            if self._stop_risk_fraction:
                signal.stop = entry * (1 - self._stop_risk_fraction)

        elif signal.signal < 0:
            entry = signal.price if signal.price else quote.bid
            if self._take_target_fraction:
                signal.take = entry * (1 - self._take_target_fraction)
            if self._stop_risk_fraction:
                signal.stop = entry * (1 + self._stop_risk_fraction)

        if signal.stop is not None:
            signal.stop = signal.instrument.round_price_down(signal.stop)
        if signal.take is not None:
            signal.take = signal.instrument.round_price_down(signal.take)

        return signal


class SignalRiskPositionTracker(GenericRiskControllerDecorator):
    """
    Tracker just uses signal's take stop levels
    """

    def __init__(
        self,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
        risk_controlling_side: RiskControllingSide = "broker",
        purpose: str = "",  # if we need to distinguish different instances of the same tracker, i.e. for shorts or longs etc
    ) -> None:
        super().__init__(
            sizer,
            GenericRiskControllerDecorator.create_risk_controller_for_side(
                f"{self.__class__.__name__}{purpose}", risk_controlling_side, self, sizer
            ),
        )

    def calculate_risks(self, ctx: IStrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        if signal.stop is not None:
            signal.stop = signal.instrument.round_price_down(signal.stop)
        if signal.take is not None:
            signal.take = signal.instrument.round_price_down(signal.take)

        return signal


class AtrRiskTracker(GenericRiskControllerDecorator):
    """
    ATR based risk management
     - Take at entry +/- ATR[1] * take_target
     - Stop at entry -/+ ATR[1] * stop_risk
    It may use either limit and stop orders for managing risk or market orders depending on 'risk_controlling_side' parameter.
    """

    def __init__(
        self,
        take_target: float | None,
        stop_risk: float | None,
        atr_timeframe: str,
        atr_period: int,
        atr_smoother="sma",
        sizer: IPositionSizer = FixedSizer(1.0),
        risk_controlling_side: RiskControllingSide = "broker",
        purpose: str = "",  # if we need to distinguish different instances of the same tracker, i.e. for shorts or longs etc
    ) -> None:
        self.take_target = take_target
        self.stop_risk = stop_risk
        self.atr_timeframe = atr_timeframe
        self.atr_period = atr_period
        self.atr_smoother = atr_smoother
        self._full_name = f"{self.__class__.__name__}{purpose}"
        self._instrument_initialized: dict[Instrument, bool] = {}

        super().__init__(
            sizer,
            GenericRiskControllerDecorator.create_risk_controller_for_side(
                f"{self._full_name}", risk_controlling_side, self, sizer
            ),
        )

    def _get_volatility(self, ctx: IStrategyContext, instrument: Instrument) -> list[float]:
        return atr(
            ctx.ohlc(instrument, self.atr_timeframe, 2 * self.atr_period),
            self.atr_period,
            smoother=self.atr_smoother,
            percentage=False,
        )

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar | OrderBook
    ) -> list[TargetPosition] | TargetPosition:
        if ctx.is_live_or_warmup and not self._instrument_initialized.get(instrument, False):
            # - emit volatility indicator in live mode
            indicator_emitter(
                wrapped_indicator=self._get_volatility(ctx, instrument),
                metric_emitter=ctx.emitter,
                instrument=instrument,
            )
            self._instrument_initialized[instrument] = True

        return super().update(ctx, instrument, update)

    def calculate_risks(self, ctx: IStrategyContext, quote: Quote | None, signal: Signal) -> Signal | None:
        volatility = self._get_volatility(ctx, signal.instrument)

        if len(volatility) < 2 or ((last_volatility := volatility[1]) is None or not np.isfinite(last_volatility)):
            logger.debug(
                f"[<y>{self._full_name}</y>(<g>{signal.instrument}</g>)] :: not enough ATR data, skipping risk calculation"
            )
            return None

        if quote is None:
            logger.debug(
                f"[<y>{self._full_name}</y>(<g>{signal.instrument}</g>)] :: there is no actual market data, skipping risk calculation"
            )
            return None

        if signal.signal > 0:
            entry = signal.price if signal.price else quote.ask
            if self.stop_risk:
                signal.stop = entry - self.stop_risk * last_volatility
            if self.take_target:
                signal.take = entry + self.take_target * last_volatility

        elif signal.signal < 0:
            entry = signal.price if signal.price else quote.bid
            if self.stop_risk:
                signal.stop = entry + self.stop_risk * last_volatility
            if self.take_target:
                signal.take = entry - self.take_target * last_volatility

        if ctx.is_live_or_warmup:
            # - additional comments for live debugging
            mid_price = quote.mid_price()
            volatility_pct = last_volatility / mid_price
            signal.comment += f", ATR: {volatility_pct:.2%} ({last_volatility:.4f})"
            signal.comment += f", stop_risk: {self.stop_risk}"
            signal.comment += f", take_target: {self.take_target}"

        return signal


class MinAtrExitDistanceTracker(PositionsTracker):
    """
    Allow exit only if price has moved away from entry by the specified distance in ATR units.
    """

    _signals: dict[Instrument, Signal]

    def __init__(
        self,
        take_target: float | None,
        stop_target: float | None,
        atr_timeframe: str,
        atr_period: int,
        atr_smoother="sma",
        sizer: IPositionSizer = FixedSizer(1.0),
    ) -> None:
        super().__init__(sizer)
        self.atr_timeframe = atr_timeframe
        self.atr_period = atr_period
        self.atr_smoother = atr_smoother
        self.take_target = take_target
        self.stop_target = stop_target
        self._signals = dict()

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        targets = []
        for s in signals:
            volatility = atr(
                ctx.ohlc(s.instrument, self.atr_timeframe, 2 * self.atr_period),
                self.atr_period,
                smoother=self.atr_smoother,
                percentage=False,
            )
            if len(volatility) < 2:
                continue
            last_volatility = volatility[1]
            quote = ctx.quote(s.instrument)
            if last_volatility is None or not np.isfinite(last_volatility) or quote is None:
                continue

            self._signals[s.instrument] = s

            if s.signal != 0:
                # if signal is not 0, atr thresholds don't apply
                # set expected stop price in case sizer needs it
                if s.stop is None:
                    price = quote.ask if s.signal > 0 else quote.bid
                    s.stop = (
                        price - self.stop_target * last_volatility
                        if s.signal > 0
                        else price + self.stop_target * last_volatility
                    )

                target = self.get_position_sizer().calculate_target_positions(ctx, [s])[0]
                targets.append(target)
                continue

            if self.__check_exit(ctx, s.instrument):
                logger.debug(
                    f"[<y>{self.__class__.__name__}</y>(<g>{s.instrument.symbol}</g>)] :: <y>Min ATR distance reached</y>"
                )

                ctx.emit_signal(
                    s.instrument.service_signal(ctx, 0, group="Risk Manager", comment="Min ATR distance reached")
                )
                targets.append(s.instrument.target(ctx, 0.0))

        return targets

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar | OrderBook
    ) -> list[TargetPosition] | TargetPosition:
        signal = self._signals.get(instrument)
        if signal is None or signal.signal != 0:
            return []
        if not self.__check_exit(ctx, instrument):
            return []
        logger.debug(
            f"[<y>{self.__class__.__name__}</y>(<g>{instrument.symbol}</g>)] :: <y>Min ATR distance reached</y>"
        )

        # - emit service signal that risk triggeres (it won't be processed by StrategyContext)
        ctx.emit_signal(
            instrument.service_signal(
                ctx,
                0,
                group="Risk Manager",
                comment=f"Min ATR distance reached. Original signal price: {signal.reference_price}",
            )
        )

        # - return target position with 0 size
        return instrument.target(ctx, 0)

    def __check_exit(self, ctx: IStrategyContext, instrument: Instrument) -> bool:
        volatility = atr(
            ctx.ohlc(instrument, self.atr_timeframe, 2 * self.atr_period),
            self.atr_period,
            smoother=self.atr_smoother,
            percentage=False,
        )
        if len(volatility) < 2:
            return False

        last_volatility = volatility[1]
        quote = ctx.quote(instrument)
        if last_volatility is None or not np.isfinite(last_volatility) or quote is None:
            return False

        pos = ctx.positions.get(instrument)
        if pos is None:
            return False

        entry = pos.position_avg_price
        allow_exit = False
        if pos.quantity > 0:
            stop = entry - self.stop_target * last_volatility
            take = entry + self.take_target * last_volatility
            if quote.bid <= stop or quote.ask >= take:
                allow_exit = True
        else:
            stop = entry + self.stop_target * last_volatility
            take = entry - self.take_target * last_volatility
            if quote.ask >= stop or quote.bid <= take:
                allow_exit = True
        return allow_exit


class _InitializationStageTracker(GenericRiskControllerDecorator, IPositionSizer):
    """
    Tracker for initialization stage. It is used to manage risk during initialization stage.
    It is not used for standard signals.

    TODO: we need to think about better method to manage initialization stage.
    """

    def __init__(
        self,
        risk_controlling_side: RiskControllingSide = "broker",
    ) -> None:
        super().__init__(
            sizer=self,
            riskctrl=GenericRiskControllerDecorator.create_risk_controller_for_side(
                f"{self.__class__.__name__}", risk_controlling_side, self, self
            ),
        )

    def process_signals(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        _to_proceed = []

        for s in signals:
            if not isinstance(s, InitializingSignal):
                logger.warning(
                    f"[<y>{self.__class__.__name__}</y>] :: <r>Received standard signal - skip it</r> :: {s}"
                )
                continue

            _current_pos = ctx.get_position(s.instrument).quantity
            logger.info(
                f"[<y>{self.__class__.__name__}</y>] :: <y>Processing init signal</y> :: {s} :: Position is {_current_pos}"
            )
            _to_proceed.append(s)

        return super().process_signals(ctx, _to_proceed)

    def calculate_risks(self, ctx: IStrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        return signal

    def calculate_target_positions(self, ctx: IStrategyContext, signals: list[Signal]) -> list[TargetPosition]:
        return [s.target_for_amount(s.signal) for s in signals]

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar | OrderBook
    ) -> list[TargetPosition] | TargetPosition:
        return super().update(ctx, instrument, update)

    def cancel_tracking(self, ctx: IStrategyContext, instrument: Instrument):
        logger.info(f"[<y>{self.__class__.__name__}</y>] :: <y>Cancelling tracking</y> for {instrument}")
        super().cancel_tracking(ctx, instrument)


class AbstractTrailingRiskPositionTracker(GenericRiskControllerDecorator):
    """
    Abstract class for trailing stop position trackers.
    """

    def __init__(
        self,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
        risk_controlling_side: RiskControllingSide = "broker",
        purpose: str = "",
    ) -> None:
        super().__init__(
            sizer,
            GenericRiskControllerDecorator.create_risk_controller_for_side(
                f"{self.__class__.__name__}{purpose}", risk_controlling_side, self, sizer
            ),
        )

    def get_trailing_stop_level(
        self, ctx: IStrategyContext, instrument: Instrument, current_market_price: float
    ) -> float | None:
        """
        This method can be called in impl to get new stop level
        """
        return None

    def get_trailing_take_level(
        self, ctx: IStrategyContext, instrument: Instrument, current_market_price: float
    ) -> float | None:
        """
        This method can be called in impl to get new take level
        """
        return None

    def update(
        self, ctx: IStrategyContext, instrument: Instrument, update: Quote | Trade | Bar | OrderBook
    ) -> list[TargetPosition] | TargetPosition:
        # - if position is not flat, check if trailing levels needed to be updated
        if (p := ctx.get_position(instrument)).is_open():
            # - update stop and take levels if needed
            if _new_stop := self.get_trailing_stop_level(
                ctx, instrument, self.riskctrl._get_price(update, p.quantity > 0)
            ):
                self.riskctrl.update_stop_level(ctx, instrument, _new_stop)

            # - update take level if needed
            if _new_take := self.get_trailing_take_level(
                ctx, instrument, self.riskctrl._get_price(update, p.quantity > 0)
            ):
                self.riskctrl.update_take_level(ctx, instrument, _new_take)

        return self.riskctrl.update(ctx, instrument, update)


class TrailingStopPositionTracker(AbstractTrailingRiskPositionTracker):
    """
    Classical trailing stop implementation (simplest version).

    When price moves away from entry by the specified number of ticks (min_price_change_ticks), trailing stop is updated as following:
        new_long_stop = market_price * (1 - trailing_stop_percentage / 100)
    or
        new_short_stop = market_price * (1 + trailing_stop_percentage / 100)
    """

    def __init__(
        self,
        trailing_stop_percentage: float,
        min_price_change_ticks: int = 10,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
        risk_controlling_side: RiskControllingSide = "client",
        purpose: str = "",
    ) -> None:
        self.trailing_stop_ratio = trailing_stop_percentage / 100
        self.min_price_change_ticks = min_price_change_ticks
        super().__init__(sizer, risk_controlling_side, purpose)

    def get_trailing_stop_level(
        self, ctx: IStrategyContext, instrument: Instrument, current_market_price: float
    ) -> float | None:
        if (stop := self.riskctrl.get_stop_level(ctx, instrument)) is None:
            return None

        pos = ctx.get_position(instrument)
        entry = pos.position_avg_price
        price_move = int((current_market_price - entry) / (self.min_price_change_ticks * instrument.tick_size))
        _updated_stop = instrument.round_price_down(
            current_market_price * (1 - np.sign(pos.quantity) * self.trailing_stop_ratio)
        )

        if pos.quantity > 0 and price_move >= 1 and _updated_stop > stop:
            return _updated_stop

        if pos.quantity < 0 and price_move <= -1 and _updated_stop < stop:
            return _updated_stop

        return None

    def get_trailing_take_level(
        self, ctx: IStrategyContext, instrument: Instrument, current_market_price: float
    ) -> float | None:
        return None

    def calculate_risks(self, ctx: IStrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        # - if signal has predefined stop, just leave it as is
        if signal.stop is not None:
            signal.stop = signal.instrument.round_price_down(signal.stop)
        else:
            # - otherwise calculate stop level based on the entry price and trailing stop ratio
            entry = signal.price if signal.price else (quote.ask if signal.signal > 0 else quote.bid)
            signal.stop = signal.instrument.round_price_down(
                entry * (1 - np.sign(signal.signal) * self.trailing_stop_ratio)
            )

        # - keep take if presented in signal
        if signal.take is not None:
            signal.take = signal.instrument.round_price_down(signal.take)

        return signal


class SwingsStopLevels(AbstractTrailingRiskPositionTracker):
    """
    Trailing stop loss based on previous market pivots points identified by swing indicator.

    If signal doesn't have any predefined stop level, previous bottom | top is taken
    Then when market evolves and new tops and bottoms are spotted it moves stop loss using most recent pivot level.

    """

    def __init__(
        self,
        timeframe: str,
        iaf=0.02,
        maxaf=0.2,
        sizer: IPositionSizer = FixedSizer(1.0, amount_in_quote=False),
        risk_controlling_side: RiskControllingSide = "client",
        purpose: str = "",
        historical_bars=100,
    ) -> None:
        """
        Parameters:
            - timeframe: str - timeframe to use for swing indicator calculation
            - iaf: float - initial acceleration factor (default: 0.02)
            - maxaf: float - maximum acceleration factor (default: 0.2)
            - sizer: IPositionSizer - sizer to use for position sizing
            - risk_controlling_side: RiskControllingSide - side to control risk on ("client" or "broker")
            - purpose: str - purpose of the tracker
            - historical_bars: int - number of bars to use for swing indicator (default: 100)

        Example:
            ```python
            tracker = SwingsStopLevels(timeframe="1h", iaf=0.02, maxaf=0.2, sizer=FixedLeverageSizer(1.0), risk_controlling_side="client")
            ```
        """
        self.iaf = iaf
        self.maxaf = maxaf
        self.timeframe = timeframe
        self.historical_bars = historical_bars
        super().__init__(sizer, risk_controlling_side, purpose)

    def _get_lows_tops(
        self, ctx: IStrategyContext, instrument: Instrument, max_lookback=100
    ) -> tuple[list[tuple[pd.Timestamp, float]], list[tuple[pd.Timestamp, float]]]:
        xT = lambda t: pd.Timestamp(t, unit="ns")  # noqa: E731
        ohlc = ctx.ohlc(instrument, self.timeframe, self.historical_bars)
        swings_indicator = swings(ohlc, psar, iaf=self.iaf, maxaf=self.maxaf)
        _n_tops, _n_btm = len(swings_indicator.tops), len(swings_indicator.bottoms)  # type: ignore

        bottoms, tops = [], []
        for i in range(min(_n_tops, _n_btm, max_lookback)):
            if _n_tops > i and _n_btm > i:  # type: ignore
                bottoms.append((xT(swings_indicator.bottoms.times[i]), swings_indicator.bottoms[i]))  # type: ignore
                tops.append((xT(swings_indicator.tops.times[i]), swings_indicator.tops[i]))  # type: ignore
        return bottoms, tops

    def get_trailing_stop_level(
        self, ctx: IStrategyContext, instrument: Instrument, current_market_price: float
    ) -> float | None:
        if (stop := self.riskctrl.get_stop_level(ctx, instrument)) is None:
            return None

        pos = ctx.get_position(instrument)

        # - we just check last swing
        lows, tops = self._get_lows_tops(ctx, instrument, 1)

        # - look at last bottom - if it's higher than previous stop and lower than current market price, update stop
        last_low = lows[0][1]
        if pos.quantity > 0 and last_low > stop and last_low < current_market_price:
            return last_low

        # - look at last top - if it's lower than previous stop and higher than current market price, update stop
        last_top = tops[0][1]
        if pos.quantity < 0 and last_top < stop and last_top > current_market_price:
            return last_top

        return None

    def calculate_risks(self, ctx: IStrategyContext, quote: Quote, signal: Signal) -> Signal | None:
        # - if signal has predefined stop, just leave it as is
        if signal.stop is not None:
            signal.stop = signal.instrument.round_price_down(signal.stop)
        else:
            # - initial stop setup
            # - otherwise find first appropriate level as support or resistance
            entry = signal.price if signal.price else (quote.ask if signal.signal > 0 else quote.bid)
            lows, tops = self._get_lows_tops(ctx, signal.instrument)

            if signal.signal > 0:
                # - start looking for first appropriate bottom
                for t, p in lows:
                    if p < entry:
                        signal.stop = signal.instrument.round_price_down(p)
                        break
                else:
                    raise ValueError(
                        "Can't find any appropriate previous low pivot to be used as stop level. Try to increase 'historical_bars' parameter !"
                    )

            elif signal.signal < 0:
                # - start looking for first appropriate top
                for t, p in tops:
                    if p > entry:
                        signal.stop = signal.instrument.round_price_down(p)
                        break
                else:
                    raise ValueError(
                        "Can't find any appropriate previous top pivot to be used as stop level. Try to increase 'historical_bars' parameter !"
                    )

        # - keep take if presented in signal
        if signal.take is not None:
            signal.take = signal.instrument.round_price_down(signal.take)

        return signal
