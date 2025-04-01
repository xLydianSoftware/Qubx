"""
Metric emitters for exporting metrics to external monitoring systems.
"""

from qubx.emitters.base import BaseMetricEmitter
from qubx.emitters.composite import CompositeMetricEmitter
from qubx.emitters.csv import CSVMetricEmitter
from qubx.emitters.prometheus import PrometheusMetricEmitter

__all__ = ["BaseMetricEmitter", "CompositeMetricEmitter", "PrometheusMetricEmitter", "CSVMetricEmitter"]

try:
    from qubx.emitters.questdb import QuestDBMetricEmitter

    __all__.append("QuestDBMetricEmitter")
except ImportError:
    # QuestDB is optional
    pass
