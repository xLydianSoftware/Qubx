"""Guards against ccxt has[] regressions on Qubx's binance UM exchange wrappers.

Background: ccxt 4.5.52 (PR #28493) made ``cxp.binanceusdm.describe()`` merge
the REST async_support describe on top of pro's describe. The REST class
sets every ``watch*`` flag to ``None``, which clobbers pro's ``True`` values
via ``deep_extend``. Result: any ccxt-pro binanceusdm-based class loses its
watch capability advertisements, and Qubx handlers that gate on
``has["watchOHLCV"]`` raise ``NotSupported`` during subscription setup —
bots get stuck in warmup forever.

``BinanceQVUSDM.describe()`` restores the flags Qubx handlers check
(handlers/{ohlc,orderbook,trade,quote}.py). Subclasses inherit the fix via
``super().describe()``. This test fails fast if a future ccxt release
reintroduces a similar regression OR if the Qubx restore is removed
prematurely.

Fully offline — runs in CI without binance network access.
"""

import ccxt.pro as cxp

# Side-effect: registers Qubx wrappers (cxp.binanceqv_usdm = BinanceQVUSDM, etc.)
import qubx.connectors.ccxt.exchanges  # noqa: F401

# Source: handlers/{ohlc,orderbook,trade,quote}.py — flags Qubx code gates on.
HANDLER_FLAGS = [
    "watchOHLCV",
    "watchOHLCVForSymbols",
    "watchOrderBookForSymbols",
    "watchTradesForSymbols",
    "watchBidsAsks",
]

# Three classes that must carry the restored flags. BinanceQVUSDM is the
# direct fix; the other two inherit via super().describe() in their own
# describe() overrides.
PATCHED_CLASS_NAMES = [
    "binanceqv_usdm",  # BinanceQVUSDM
    "binancepm",  # BinancePortfolioMargin(BinanceQVUSDM)
    "binance_um_mm",  # BINANCE_UM_MM(BinanceQVUSDM)
]


# Implemented as a single test that iterates internally rather than two
# stacked @pytest.mark.parametrize decorators because qubx pins
# pytest-lazy-fixture, which has an incompatibility with pytest 8.x that
# breaks parametrized test collection.
def test_handler_watch_flags_are_advertised():
    failures: list[str] = []
    for cls_name in PATCHED_CLASS_NAMES:
        instance = getattr(cxp, cls_name)({"enableRateLimit": True})
        for flag in HANDLER_FLAGS:
            actual = instance.has.get(flag)
            if actual is not True:
                failures.append(f"  {cls_name}.has[{flag!r}] = {actual!r}; expected True")

    assert not failures, (
        "Watch capability flags missing on patched binance classes:\n"
        + "\n".join(failures)
        + "\n\nThis usually means a ccxt upgrade reintroduced the binanceusdm "
        "describe() REST-on-pro merge regression, or the qubx patch was removed."
    )
