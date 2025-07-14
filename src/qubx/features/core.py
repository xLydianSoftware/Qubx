from collections import defaultdict
from typing import Any

from qubx.core.basics import DataType, Instrument, MarketEvent
from qubx.core.helpers import set_parameters_to_object
from qubx.core.interfaces import IMarketManager, IStrategyContext
from qubx.core.series import TimeSeries, time_as_nsec
from qubx.data.readers import DataReader

FeatureMapping = dict[str, dict[str, TimeSeries]]  # feature_provider -> feature_name -> TimeSeries


class FeatureManager:
    reader: DataReader | None

    feature_providers: list["FeatureProvider"]
    subscription_to_providers: dict[str, list["FeatureProvider"]]
    instrument_features: dict[Instrument, FeatureMapping]
    _subscriptions: set[DataType]

    _max_series_length: int

    def __init__(
        self,
        reader: DataReader | None = None,
        max_series_length: int = 1000,
    ):
        self.reader = reader
        self.feature_providers = []
        self.subscription_to_providers = defaultdict(list)
        self.instrument_features = defaultdict(lambda: defaultdict(dict))
        self._max_series_length = max_series_length
        self._symbol_to_instrument = {}
        self._subscriptions = set()

    def __add__(self, feature_provider: "FeatureProvider") -> "FeatureManager":
        """
        Subscribes the given feature provider to the manager.
        """
        # - add the required subscriptions to the manager and update routing table if not already added
        if feature_provider not in self.feature_providers:
            self.feature_providers.append(feature_provider)
            for input_name in feature_provider.inputs():
                self._subscriptions.add(input_name)
                dtype, _ = DataType.from_str(input_name)
                self.subscription_to_providers[dtype].append(feature_provider)

            # - notify the feature provider that it has been subscribed
            feature_provider.on_subscribe(self)
        return self

    def __str__(self) -> str:
        provider_info = "\n".join(
            f"- {provider.name}: {', '.join(provider.outputs())}" for provider in self.feature_providers
        )
        return f"FeatureManager\n{provider_info}"

    def __repr__(self) -> str:
        return self.__str__()

    def on_start(self, ctx: IStrategyContext) -> None:
        for provider in self.feature_providers:
            provider.on_start(ctx)

        # - initialize subscriptions for existing instruments
        self.on_universe_change(ctx, ctx.get_instruments(), [])

    def on_universe_change(
        self,
        ctx: IStrategyContext,
        add_instruments: list[Instrument],
        rm_instruments: list[Instrument],
    ) -> None:
        self._symbol_to_instrument.update({instrument.symbol: instrument for instrument in add_instruments})

        # - subscribe to the required data sources
        for instrument in add_instruments:
            self._subscribe_instrument(ctx, instrument)

        # - notify all feature providers about the universe change and initialize the features
        for provider in self.feature_providers:
            output_features = provider.outputs()
            instr_to_features = provider.on_universe_change(ctx, add_instruments, rm_instruments)
            for instrument in add_instruments:
                for feature in output_features:
                    self.instrument_features[instrument][provider.name][feature] = (
                        instr_to_features[instrument][feature]
                        if instrument in instr_to_features and feature in instr_to_features[instrument]
                        else TimeSeries(feature, provider.timeframe, self._max_series_length)
                    )

        # - remove the features of the removed instruments
        for instrument in rm_instruments:
            self.instrument_features.pop(instrument)

    def get_feature(self, instrument: Instrument, feature_name: str) -> TimeSeries:
        """
        Retrieve feature time series by instrument and feature name.

        Examples:
        - manager.get_feature(instrument, "ATR(14,1h,sma,pct=True)") -> returns the ATR feature for instrument
        """
        for features in self.instrument_features[instrument].values():
            if feature_name in features:
                return features[feature_name]
        raise KeyError(f"Feature '{feature_name}' not found for instrument '{instrument}'")

    def __getitem__(self, keys) -> dict[str, TimeSeries]:
        """
        Retrieve feature time series by provider and feature name.

        Examples:
        - manager["BTCUSDT", "ATR"] -> returns all ATR features for BTCUSDT (dict[str, TimeSeries])
        - manager["BTCUSDT"] -> returns all features for BTCUSDT (dict[str, TimeSeries])
        - manager[instrument] -> returns all features for instrument (dict[str, TimeSeries])
        """
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and (isinstance(keys[0], str) or isinstance(keys[0], Instrument))
            and isinstance(keys[1], str)
        ):
            symbol, provider_name = keys
            instrument = self._get_instrument(symbol) if isinstance(symbol, str) else symbol
            if provider_name in self.instrument_features[instrument]:
                return self.instrument_features[instrument][provider_name]
            raise KeyError(f"Feature '{provider_name}' not found for instrument '{instrument}'")
        elif isinstance(keys, str) or isinstance(keys, Instrument):
            instrument = self._get_instrument(keys) if isinstance(keys, str) else keys
            return {
                feature_name: ts
                for provider_features in self.instrument_features[instrument].values()
                for feature_name, ts in provider_features.items()
            }
        else:
            raise KeyError("Invalid key format. Expected a tuple (instrument, feature_name) or a string (instrument).")

    def on_market_data(self, ctx: IStrategyContext, event: MarketEvent) -> None:
        # - route the event to the subscribed feature providers
        providers = self.subscription_to_providers.get(event.type, [])
        for provider in providers:
            assert event.instrument is not None
            _time = time_as_nsec(event.time)
            output_names = provider.outputs()
            feature_values = provider.calculate(ctx, event.instrument, event.data)
            if feature_values is not None and event.instrument in self.instrument_features:
                if isinstance(feature_values, dict):
                    for feature_name, value in feature_values.items():
                        feature_series = self.instrument_features[event.instrument][provider.name][feature_name]
                        if len(feature_series) == 0 or feature_series.times[0] < _time:
                            feature_series.update(_time, value)
                elif len(output_names) == 1:
                    feature_series = self.instrument_features[event.instrument][provider.name][output_names[0]]
                    if len(feature_series) == 0 or feature_series.times[0] < _time:
                        feature_series.update(_time, feature_values)
                else:
                    raise ValueError(f"Invalid output from feature provider '{provider.name}': {feature_values}")

    def _subscribe_instrument(self, ctx: IStrategyContext, instrument: Instrument) -> None:
        for dtype in self._subscriptions:
            if not ctx.has_subscription(instrument, dtype):
                ctx.subscribe(dtype, instrument)

    def _get_instrument(self, symbol: str) -> Instrument:
        if symbol not in self._symbol_to_instrument:
            raise KeyError(f"Instrument '{symbol}' not found.")
        return self._symbol_to_instrument[symbol]


class FeatureProvider:
    timeframe: str = "1h"

    def __init__(self, **kwargs):
        set_parameters_to_object(self, **kwargs)
        if not hasattr(self, "name"):
            self.name = self.__class__.__name__

    def inputs(self) -> list[str]:
        """
        The required subscriptions for the feature provider.
        """
        return []

    def outputs(self) -> list[str]:
        """
        The list of features that will be calculated by the provider.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def on_start(self, ctx: IMarketManager) -> None:
        """
        Called when the strategy is started.
        """
        pass

    def on_universe_change(
        self,
        ctx: IMarketManager,
        add_instruments: list[Instrument],
        rm_instruments: list[Instrument],
    ) -> dict[Instrument, dict[str, TimeSeries]]:
        """
        Called when the universe of instruments changes.
        """
        for instrument in rm_instruments:
            self.on_instrument_removed(ctx, instrument)

        instr_to_feature_series = defaultdict(dict)
        for instrument in add_instruments:
            _res = self.on_instrument_added(ctx, instrument)
            if isinstance(_res, TimeSeries):
                _outputs = self.outputs()
                if len(_outputs) != 1:
                    raise ValueError(f"[{self.name}] The number of outputs must be one.")
                instr_to_feature_series[instrument][_outputs[0]] = _res
            elif isinstance(_res, dict):
                instr_to_feature_series[instrument].update(_res)

        return instr_to_feature_series

    def on_instrument_added(
        self, ctx: IMarketManager, instrument: Instrument
    ) -> TimeSeries | dict[str, TimeSeries] | None:
        """
        Called when a new instrument is added to the universe.
        """
        pass

    def on_instrument_removed(self, ctx: IMarketManager, instrument: Instrument) -> None:
        """
        Called when an instrument is removed from the universe.
        """
        pass

    def calculate(self, ctx: IStrategyContext, instrument: Instrument, event: Any) -> dict[str, float] | None:
        raise NotImplementedError("Subclasses must implement this method.")

    def on_subscribe(self, manager: "FeatureManager") -> None:
        """
        Called when the feature provider is subscribed to the manager.
        """
        pass

    def get_output_name(self, *args, **kwargs) -> str:
        params = [f"{v}" for v in args] + [f"{k}={v}" for k, v in kwargs.items()]
        return f"{self.name}({','.join(params)})"

    def __hash__(self):
        return hash((self.name, tuple(self.outputs())))

    def __eq__(self, other):
        if isinstance(other, FeatureProvider):
            return self.name == other.name and self.outputs() == other.outputs()
        return False
