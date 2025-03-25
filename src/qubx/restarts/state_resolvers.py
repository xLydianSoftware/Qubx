from qubx import logger
from qubx.core.basics import Instrument, Order, Position
from qubx.core.interfaces import IStrategyContext


class StateResolver:
    """
    Collection of static methods for resolving position mismatches between
    warmup simulation and live trading.
    These methods can be used with IStrategyInitializer.set_state_resolver().
    """

    @staticmethod
    def NONE(
        ctx: IStrategyContext, sim_positions: dict[Instrument, Position], sim_orders: dict[Instrument, list[Order]]
    ) -> None:
        """
        Do nothing.
        """
        pass

    @staticmethod
    def REDUCE_ONLY(
        ctx: IStrategyContext, sim_positions: dict[Instrument, Position], sim_orders: dict[Instrument, list[Order]]
    ) -> None:
        """
        Only allow reducing positions that exist in both simulation and live.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
        """
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
                    ctx.trade(instrument, -live_qty)

                # If live position is larger than sim position (same direction), reduce it
                elif abs(live_qty) > abs(sim_qty) and abs(live_qty) > instrument.lot_size:
                    qty_diff = sim_qty - live_qty
                    logger.info(
                        f"Reducing position for {instrument.symbol}: {live_qty} -> {sim_qty} (diff: {qty_diff})"
                    )
                    ctx.trade(instrument, qty_diff)

                # If sim position is larger or equal (same direction), do nothing
                else:
                    logger.info(f"Keeping position for {instrument.symbol} as is: {live_qty}")

            # If the instrument doesn't exist in simulation, close the position
            else:
                logger.info(f"Closing position for {instrument.symbol} not in simulation: {live_qty} -> 0")
                ctx.trade(instrument, -live_qty)

    @staticmethod
    def CLOSE_ALL(
        ctx: IStrategyContext, sim_positions: dict[Instrument, Position], sim_orders: dict[Instrument, list[Order]]
    ) -> None:
        """
        Close all positions and start fresh.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
        """
        # TODO: optimize with batch requests
        orders = ctx.get_orders()
        if orders:
            logger.info(f"Cancelling {len(orders)} live orders ...")
            for order in orders.values():
                ctx.cancel_order(order.id)

        # Get current live positions
        live_positions = ctx.get_positions()

        # Close all live positions
        for instrument, position in live_positions.items():
            if abs(position.quantity) > instrument.lot_size:
                logger.info(f"Closing position for {instrument.symbol}: {position.quantity} -> 0")
                ctx.trade(instrument, -position.quantity)

    @staticmethod
    def SYNC_STATE(
        ctx: IStrategyContext, sim_positions: dict[Instrument, Position], sim_orders: dict[Instrument, list[Order]]
    ) -> None:
        """
        Synchronize the live state with the simulation state.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
        """
        # Get current live positions
        live_positions = ctx.get_positions()

        # Process instruments in simulation positions
        for instrument, sim_pos in sim_positions.items():
            live_qty = 0
            if instrument in live_positions:
                live_qty = live_positions[instrument].quantity

            # Calculate the difference needed to match simulation position
            qty_diff = sim_pos.quantity - live_qty

            # Only trade if there's a difference
            if abs(qty_diff) > instrument.lot_size:
                logger.info(
                    f"Syncing position for {instrument.symbol}: {live_qty} -> {sim_pos.quantity} (diff: {qty_diff})"
                )
                ctx.trade(instrument, qty_diff)

        # Close positions that exist in live but not in simulation
        for instrument, live_pos in live_positions.items():
            if instrument not in sim_positions and abs(live_pos.quantity) > instrument.lot_size:
                logger.info(f"Closing position for {instrument.symbol} not in simulation: {live_pos.quantity} -> 0")
                ctx.trade(instrument, -live_pos.quantity)
