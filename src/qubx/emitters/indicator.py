"""
Indicator Emitter Module.

This module provides the IndicatorEmitter class that can wrap around any indicator
and automatically emit their values when there are updates.
"""

import numpy as np

from qubx import logger
from qubx.core.basics import Instrument
from qubx.core.interfaces import IMetricEmitter
from qubx.core.series import Indicator
from qubx.core.utils import recognize_time


class IndicatorEmitter(Indicator):
    """
    An indicator that wraps around another indicator and automatically emits its values.

    This class solves the problem of wanting to emit indicator values independently
    of when certain calculation methods are called. For example, you might want to
    emit ATR values on every update, not just when calculate_risks() is called.

    Example:
        # Create ATR indicator
        volatility = atr(ohlc_series, period=14, smoother="sma")

        # Wrap it with an emitter to automatically emit values
        volatility_emitter = IndicatorEmitter.wrap_with_emitter(
            volatility,
            metric_emitter=my_emitter,
            metric_name="atr_volatility",
            instrument=instrument,
            tags={"timeframe": "1m", "period": "14"}
        )
    """

    def __init__(
        self,
        name: str,
        wrapped_indicator: Indicator,
        metric_emitter: IMetricEmitter,
        metric_name: str | None = None,
        instrument: Instrument | None = None,
        tags: dict[str, str] | None = None,
        emit_on_new_item_only: bool = True,
    ):
        """
        Initialize the IndicatorEmitter.

        Args:
            name: Name of this emitter indicator
            wrapped_indicator: The indicator to wrap and emit values from
            metric_emitter: The metric emitter to use for sending values
            metric_name: Name to use for the emitted metric
            instrument: Optional instrument to include in tags
            tags: Optional additional tags to include with emitted metrics
            emit_on_new_item_only: If True, only emit when new_item_started=True;
                                  if False, emit on every update
        """
        # Store our configuration BEFORE calling super().__init__()
        # because super().__init__() triggers initial data recalculation
        self._wrapped_indicator = wrapped_indicator
        self._metric_emitter = metric_emitter
        self._metric_name = metric_name if metric_name is not None else wrapped_indicator.name
        self._instrument = instrument
        self._tags = tags or {}
        self._emit_on_new_item_only = emit_on_new_item_only
        self._has_emitted = False

        # Initialize as an indicator subscribed to the wrapped indicator's updates
        # Since Indicator extends TimeSeries, we can pass the wrapped_indicator directly
        super().__init__(name, wrapped_indicator)

        logger.debug(
            f"[IndicatorEmitter] Created emitter '{name}' wrapping '{wrapped_indicator.name}' -> metric '{self._metric_name}'"
        )

    def calculate(self, time: int, value: float, new_item_started: bool) -> float:
        """
        Calculate method that handles the emission logic.

        Args:
            time: Timestamp of the update
            value: The input value (from the wrapped indicator - should be numeric)
            new_item_started: Whether this is a new item or an update to existing

        Returns:
            The current value of the wrapped indicator
        """
        current_value = float(value)
        if not np.isfinite(current_value):
            current_value = np.nan

        # Decide whether to emit based on our configuration
        should_emit = False

        if not self._emit_on_new_item_only:
            # Emit on every update
            should_emit = not np.isnan(current_value)
        else:
            # Only emit when new item starts AND we have a previous value to emit
            if new_item_started and len(self._wrapped_indicator) >= 2:
                should_emit = True
                # Use the previous (completed) value, not the current one
                current_value = self._wrapped_indicator[1] if not self.is_initial_recalculate else value

        # Emit if we should and the value is valid
        if should_emit and not np.isnan(current_value):
            try:
                # Prepare tags for emission
                emission_tags = self._tags.copy()

                # Emit the metric with the proper timestamp
                self._metric_emitter.emit(
                    name=self._metric_name,
                    value=float(current_value),
                    tags=emission_tags,
                    timestamp=recognize_time(time),
                    instrument=self._instrument,
                )

                if not self._has_emitted:
                    logger.debug(
                        f"[IndicatorEmitter] '{self.name}' started emitting '{self._metric_name}' "
                        f"values from '{self._wrapped_indicator.name}'"
                    )
                    self._has_emitted = True

            except Exception as e:
                logger.error(
                    f"[IndicatorEmitter] Failed to emit metric '{self._metric_name}' "
                    f"from '{self._wrapped_indicator.name}': {e}"
                )

        # Return the current value
        return current_value

    @classmethod
    def wrap_with_emitter(
        cls,
        indicator: Indicator,
        metric_emitter: IMetricEmitter,
        metric_name: str | None = None,
        instrument: Instrument | None = None,
        tags: dict[str, str] | None = None,
        emit_on_new_item_only: bool = True,
    ) -> "Indicator":
        """
        Convenience method to wrap an existing indicator with an emitter.

        Args:
            indicator: The indicator to wrap
            metric_emitter: The metric emitter to use
            metric_name: Name for the emitted metric (defaults to indicator name)
            instrument: Optional instrument for tagging
            tags: Optional additional tags
            emit_on_new_item_only: Whether to emit only on new items or on every update

        Returns:
            IndicatorEmitter: The wrapped indicator with emission capability
        """
        return cls.wrap(
            indicator,
            metric_emitter=metric_emitter,
            metric_name=metric_name,
            instrument=instrument,
            tags=tags,
            emit_on_new_item_only=emit_on_new_item_only,
        )

    def __getitem__(self, idx):
        """Delegate indexing to the wrapped indicator."""
        return self._wrapped_indicator[idx]

    def __len__(self):
        """Delegate length to the wrapped indicator."""
        return len(self._wrapped_indicator)


def indicator_emitter(
    wrapped_indicator: Indicator,
    metric_emitter: IMetricEmitter,
    metric_name: str | None = None,
    instrument: Instrument | None = None,
    tags: dict[str, str] | None = None,
    emit_on_new_item_only: bool = True,
) -> IndicatorEmitter:
    """
    Helper function to create an IndicatorEmitter following the standard pattern.

    Args:
        wrapped_indicator: The indicator to wrap
        metric_emitter: The metric emitter to use
        metric_name: Name for the emitted metric (defaults to indicator name)
        instrument: Optional instrument for tagging
        tags: Optional additional tags
        emit_on_new_item_only: Whether to emit only on new items or on every update

    Returns:
        IndicatorEmitter: The wrapped indicator with emission capability
    """
    emitter = IndicatorEmitter.wrap_with_emitter(
        indicator=wrapped_indicator,
        metric_emitter=metric_emitter,
        metric_name=metric_name,
        instrument=instrument,
        tags=tags,
        emit_on_new_item_only=emit_on_new_item_only,
    )
    assert isinstance(emitter, IndicatorEmitter)
    return emitter
