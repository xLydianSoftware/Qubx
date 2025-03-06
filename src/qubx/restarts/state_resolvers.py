from qubx.core.basics import Instrument, Order, Position
from qubx.core.interfaces import IStrategyContext


class StateResolver:
    """
    Collection of static methods for resolving position mismatches between
    warmup simulation and live trading.
    These methods can be used with IStrategyInitializer.set_mismatch_resolver().
    """

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
        ...

    @staticmethod
    def CLOSE_ON_MISMATCH(
        ctx: IStrategyContext, sim_positions: dict[Instrument, Position], sim_orders: dict[Instrument, list[Order]]
    ) -> None:
        """
        Close positions that don't match between simulation and live.

        Args:
            ctx (IStrategyContext): The strategy context
            sim_positions (dict[Instrument, Position]): Positions from the simulation
            sim_orders (dict[Instrument, list[Order]]): Orders from the simulation
        """
        ...

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
        ...

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
        ...
