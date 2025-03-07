from qubx.core.basics import RestoredState, dt_64


class TimeFinder:
    """
    Collection of static methods for finding start times for warmup simulations.
    These methods can be used with IStrategyInitializer.set_warmup().
    """

    @staticmethod
    def NOW(time: dt_64, state: RestoredState) -> dt_64:
        """
        Use the current time as the start time.

        Args:
            state (RestoredState): The restored state from a previous run

        Returns:
            dt_64: The current time
        """
        return time

    @staticmethod
    def LAST_SIGNAL(time: dt_64, state: RestoredState) -> dt_64:
        """
        Use the time of the last signal as the start time.

        Args:
            state (RestoredState): The restored state from a previous run

        Returns:
            dt_64: The time of the last signal
        """
        ...
