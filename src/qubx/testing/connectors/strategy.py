"""Instrumented strategy for connector verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qubx import logger
from qubx.core.basics import Signal, TriggerEvent
from qubx.core.interfaces import IStrategy, IStrategyContext, IStrategyInitializer, MarketEvent

if TYPE_CHECKING:
    from qubx.testing.connectors.collector import EventCollector
    from qubx.testing.connectors.spec import TestCaseSpec


class ConnectorVerificationStrategy(IStrategy):
    """Bare-minimum strategy that configures subscriptions and hooks into the collector."""

    _spec: TestCaseSpec
    _collector: EventCollector

    def __init__(self, spec: TestCaseSpec, collector: EventCollector):
        super().__init__()
        self._spec = spec
        self._collector = collector

    def on_init(self, initializer: IStrategyInitializer):
        initializer.set_base_subscription(self._spec.subscription)
        if self._spec.warmup:
            initializer.set_warmup(self._spec.warmup)

    def on_start(self, ctx: IStrategyContext):
        # Install on start only if no warmup (otherwise install on warmup_finished for the live context)
        if not self._spec.warmup:
            self._collector.install(ctx)
        self._collector.record_lifecycle("strategy_started")
        logger.info(f"[ConnectorTest:{self._spec.name}] Started with {len(ctx.instruments)} instruments")

    def on_warmup_finished(self, ctx: IStrategyContext):
        # Install on the live context (after warmup, ctx is the live context)
        self._collector.install(ctx)
        self._collector.record_lifecycle("warmup_finished")
        logger.info(f"[ConnectorTest:{self._spec.name}] Warmup finished")

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        return None

    def on_event(self, ctx: IStrategyContext, event: TriggerEvent):
        return None

    def on_stop(self, ctx: IStrategyContext):
        self._collector.record_lifecycle("strategy_stopped")
        self._collector.uninstall()
