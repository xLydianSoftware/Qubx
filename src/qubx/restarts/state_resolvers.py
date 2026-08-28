from qubx import logger
from qubx.core.basics import InitializingSignal, Instrument, Order, Position, TargetPosition
from qubx.core.exceptions import OrderNotFound
from qubx.core.interfaces import IStrategyContext


def _no_warmup_output(
    sim_positions: dict[Instrument, Position],
    sim_orders: dict[Instrument, list[Order]],
    sim_active_targets: dict[Instrument, TargetPosition],
) -> bool:
    """All-empty sim args ⇔ no warmup sim ran (a sim that ran captures a Position per
    instrument, flat included). Resolvers that steer toward sim state must hold then."""
    if sim_positions or sim_orders or sim_active_targets:
        return False
    logger.warning(
        "<yellow>State resolver received no warmup output — holding the live book as-is. "
        "Register a custom resolver (or StateResolver.HOLD) to silence this.</yellow>"
    )
    return True


def _cancel_all_orders(ctx: IStrategyContext) -> None:
    orders = ctx.get_orders()
    if not orders:
        return
    logger.info(f"Cancelling {len(orders)} live orders ...")
    for order in orders.values():
        try:
            # - route by the id we actually hold: venue id when acked, else the client id
            if order.venue_order_id:
                ctx.cancel_order(order_id=order.venue_order_id)
            else:
                ctx.cancel_order(client_order_id=order.client_order_id)
        except OrderNotFound:
            logger.debug(f"Order {order.venue_order_id or order.client_order_id} already cancelled or doesn't exist")


class StateResolver:
    """
    Collection of static methods for resolving position mismatches between
    warmup simulation and live trading.
    These methods can be used with IStrategyInitializer.set_state_resolver().

    sim_orders is unused by all stock resolvers; it remains in the signature for custom resolvers.
    """

    @staticmethod
    def NONE(
        ctx: IStrategyContext,
        sim_positions: dict[Instrument, Position],
        sim_orders: dict[Instrument, list[Order]],
        sim_active_targets: dict[Instrument, TargetPosition],
    ) -> None:
        """
        Do nothing.
        """
        pass

    @staticmethod
    def HOLD(
        ctx: IStrategyContext,
        sim_positions: dict[Instrument, Position],
        sim_orders: dict[Instrument, list[Order]],
        sim_active_targets: dict[Instrument, TargetPosition],
    ) -> None:
        """
        Cancel all open live orders, keep all live positions untouched, emit nothing.
        The recommended partner of initializer.set_fit_on_start(True): let the first
        live fit reconcile positions through the strategy's own tracker.
        """
        _cancel_all_orders(ctx)

    @staticmethod
    def REDUCE_ONLY(
        ctx: IStrategyContext,
        sim_positions: dict[Instrument, Position],
        sim_orders: dict[Instrument, list[Order]],
        sim_active_targets: dict[Instrument, TargetPosition],
    ) -> None:
        """
        Only allow reducing positions that exist in both simulation and live.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
        """
        if _no_warmup_output(sim_positions, sim_orders, sim_active_targets):
            return

        # Get current live positions
        live_positions = ctx.get_positions()

        # Process each live position
        for instrument, live_pos in live_positions.items():
            live_qty = live_pos.quantity

            # Skip positions with zero quantity
            if abs(live_qty) <= instrument.lot_size:
                continue

            # Check if the instrument exists in simulation positions
            if instrument in sim_positions:
                sim_qty = sim_positions[instrument].quantity

                # If signs are opposite, close the live position
                if live_qty * sim_qty < 0:
                    logger.info(f"Closing position for {instrument.symbol} due to opposite direction: {live_qty} -> 0")
                    ctx.emit_signal(InitializingSignal(time=ctx.time(), instrument=instrument, signal=0.0))

                # If live position is larger than sim position (same direction), reduce it
                elif abs(live_qty) > abs(sim_qty) and abs(live_qty) > instrument.lot_size:
                    qty_diff = sim_qty - live_qty
                    logger.info(
                        f"Reducing position for {instrument.symbol}: {live_qty} -> {sim_qty} (diff: {qty_diff:.4f})"
                    )
                    ctx.emit_signal(InitializingSignal(time=ctx.time(), instrument=instrument, signal=sim_qty))

                # If sim position is larger or equal (same direction), do nothing
                else:
                    logger.info(f"Keeping position for {instrument.symbol} as is: {live_qty}")

            # If the instrument doesn't exist in simulation, close the position
            else:
                logger.info(f"Closing position for {instrument.symbol} not in simulation: {live_qty} -> 0")
                ctx.emit_signal(InitializingSignal(time=ctx.time(), instrument=instrument, signal=0.0))

    @staticmethod
    def CLOSE_ALL(
        ctx: IStrategyContext,
        sim_positions: dict[Instrument, Position],
        sim_orders: dict[Instrument, list[Order]],
        sim_active_targets: dict[Instrument, TargetPosition],
    ) -> None:
        """
        Close all positions and start fresh.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
            sim_active_targets (dict[Instrument, list[TargetPosition]]): Active targets from the simulation
        """
        # TODO: optimize with batch requests
        _cancel_all_orders(ctx)

        # Get current live positions
        live_positions = ctx.get_positions()

        # Close all live positions
        for instrument, position in live_positions.items():
            if abs(position.quantity) > instrument.lot_size:
                logger.info(f"Closing position for {instrument.symbol}: {position.quantity} -> 0")
                ctx.emit_signal(InitializingSignal(time=ctx.time(), instrument=instrument, signal=0.0))

    @staticmethod
    def SYNC_STATE(
        ctx: IStrategyContext,
        sim_positions: dict[Instrument, Position],
        sim_orders: dict[Instrument, list[Order]],
        sim_active_targets: dict[Instrument, TargetPosition],
    ) -> None:
        """
        Synchronize the live state with the simulation state.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
            sim_active_targets (dict[Instrument, list[TargetPosition]]): Active targets from the simulation
        """
        if _no_warmup_output(sim_positions, sim_orders, sim_active_targets):
            return

        # Get current live positions
        live_positions = ctx.get_positions()

        # - process last active targets from simulation and send them as initializing signals
        for instrument, a_tgt in sim_active_targets.items():
            s = InitializingSignal(
                time=ctx.time(),
                instrument=instrument,
                signal=a_tgt.target_position_size,
                price=a_tgt.price,
                stop=a_tgt.stop,
                take=a_tgt.take,
            )
            ctx.emit_signal(s)

        # - now check which positions are open in live and we didn't update them by InitializingSignal
        for instrument, live_pos in live_positions.items():
            ctx.cancel_orders(live_pos.instrument)
            if live_pos.is_open() and instrument not in sim_active_targets:
                # - just close the position
                ctx.emit_signal(InitializingSignal(time=ctx.time(), instrument=instrument, signal=0.0))
