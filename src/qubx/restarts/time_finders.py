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
        instrument_to_start_time = {}
        for instrument, signals in state.instrument_to_signal_positions.items():
            if not signals:
                continue

            # sort signals in descending order of time (newest first)
            sorted_signals = sorted(signals, key=lambda x: x.time, reverse=True)

            # Go back in time as long as positions are nonzero
            # When we find a zero position, take the timestamp of the target before it
            found_nonzero = False
            for i, signal in enumerate(sorted_signals):
                if abs(signal.signal) > 0:
                    found_nonzero = True
                else:  # Found a zero position
                    if found_nonzero and i > 0:
                        # Take the timestamp of the previous target (which was nonzero)
                        instrument_to_start_time[instrument] = sorted_signals[i - 1].time
                        break

            # If all positions were nonzero or we didn't find a transition from nonzero to zero,
            # use the oldest signal's time if it exists
            if instrument not in instrument_to_start_time and sorted_signals and found_nonzero:
                instrument_to_start_time[instrument] = sorted_signals[-1].time

        # If no suitable positions found, return the current time
        if not instrument_to_start_time:
            return time

        # Return the minimum time among all instruments' last signals
        return min(instrument_to_start_time.values())

    @staticmethod
    def LAST_TARGET(time: dt_64, state: RestoredState) -> dt_64:
        """
        Use the time of the last target as the start time.

        Args:
            state (RestoredState): The restored state from a previous run

        Returns:
            dt_64: The time of the last target
        """
        instrument_to_start_time = {}
        for instrument, target_positions in state.instrument_to_target_positions.items():
            if not target_positions:
                continue

            # sort signals in descending order of time (newest first)
            sorted_targets = sorted(target_positions, key=lambda x: x.time, reverse=True)

            # Go back in time as long as positions are nonzero
            # When we find a zero position, take the timestamp of the target before it
            found_nonzero = False
            for i, target in enumerate(sorted_targets):
                if abs(target.target_position_size) > 0:
                    found_nonzero = True
                else:  # Found a zero position
                    if found_nonzero and i > 0:
                        # Take the timestamp of the previous target (which was nonzero)
                        instrument_to_start_time[instrument] = sorted_targets[i - 1].time
                        break

            # If all positions were nonzero or we didn't find a transition from nonzero to zero,
            # use the oldest signal's time if it exists
            if instrument not in instrument_to_start_time and sorted_targets and found_nonzero:
                instrument_to_start_time[instrument] = sorted_targets[-1].time

        # If no suitable positions found, return the current time
        if not instrument_to_start_time:
            return time

        # Return the minimum time among all instruments' last signals
        return min(instrument_to_start_time.values())
