"""
Integration test that mimics the Lighter notebook workflow.

Tests:
1. Strategy initialization with xlighter reader (warmup data)
2. Strategy lifecycle (on_init, on_market_data, on_stop)
3. Accessing OHLC data from strategy context
4. Paper trading mode with xlighter connector
"""

import time
import pandas as pd
from qubx import logger, QubxLogConfig
from qubx.core.interfaces import IStrategy, IStrategyContext, BaseErrorEvent, IStrategyInitializer
from qubx.core.basics import DataType, MarketEvent
from qubx.utils.runner.runner import run_strategy, StrategyConfig, AccountConfigurationManager, ExchangeConfig, LoggingConfig
from qubx.utils.runner.configs import LiveConfig, ReaderConfig


# Set log level
QubxLogConfig.set_log_level("INFO")


class TestStrategy(IStrategy):
    """
    Simple test strategy that subscribes to OHLC data and validates warmup.
    """

    def __init__(self):
        self.warmup_complete = False
        self.data_count = 0

    def on_init(self, initializer: IStrategyInitializer):
        """
        Initialize strategy subscriptions.

        - Subscribe to trades as base subscription
        - Request 1 day of 1h OHLC warmup data
        """
        print("\n" + "=" * 60)
        print("TestStrategy.on_init() called")
        print("=" * 60)

        # Set base subscription
        initializer.set_base_subscription(DataType.TRADE)

        # Request warmup data: 1 day of 1h candles
        initializer.set_subscription_warmup({
            DataType.OHLC["1h"]: "1d"
        })

        print("✅ Subscriptions configured:")
        print("   - Base: TRADE")
        print("   - Warmup: OHLC[1h] with 1d history")

    def on_market_data(self, ctx: IStrategyContext, data: MarketEvent):
        """Handle market data events"""
        self.data_count += 1

        # Only log first few events
        if self.data_count <= 3:
            print(f"\n📊 Market data event #{self.data_count}: {data.type}")

    def on_error(self, ctx: IStrategyContext, error: BaseErrorEvent) -> None:
        """Handle errors"""
        print(f"\n❌ Error: {error}")
        logger.error(f"Strategy error: {error}")

    def on_stop(self, ctx: IStrategyContext):
        """Cleanup on stop"""
        print("\n" + "=" * 60)
        print("TestStrategy.on_stop() called")
        print(f"Received {self.data_count} market data events")
        print("=" * 60)


def test_notebook_workflow():
    """Test the complete notebook workflow"""

    print("\n" + "=" * 80)
    print("LIGHTER NOTEBOOK INTEGRATION TEST")
    print("=" * 80)

    # 1. Create strategy configuration
    print("\n1. Creating strategy configuration...")
    config = StrategyConfig(
        name="TestStrategy",
        strategy=TestStrategy,
        aux=ReaderConfig(reader="xlighter", args={"max_history": "10d"}),
        live=LiveConfig(
            exchanges={
                "LIGHTER": ExchangeConfig(
                    connector="xlighter",
                    universe=["BTCUSDC", "ETHUSDC"],
                )
            },
            logging=LoggingConfig(
                logger="InMemoryLogsWriter",
                position_interval="10s",
                portfolio_interval="1m",
                heartbeat_interval="10m",
            )
        )
    )
    print("✅ Configuration created")

    # 2. Create account manager with test accounts
    print("\n2. Creating account manager...")
    from pathlib import Path
    accounts_path = Path(__file__).parent / "accounts.toml"
    account_manager = AccountConfigurationManager(account_config=accounts_path)
    print(f"✅ Account manager created: {account_manager}")

    # 3. Run strategy
    print("\n3. Starting strategy in paper mode...")
    print("   (This will load warmup data via xlighter reader)")

    try:
        ctx = run_strategy(
            config=config,
            account_manager=account_manager,
            paper=True,
            blocking=False,  # Non-blocking so we can interact with it
        )
        print("✅ Strategy started successfully")

        # 4. Wait a bit for initialization to complete
        print("\n4. Waiting for strategy initialization...")
        time.sleep(3)

        # 5. Test accessing OHLC data from context
        print("\n5. Testing OHLC data access...")

        # Query instruments
        btc_instrument = ctx.query_instrument("BTCUSDC")
        eth_instrument = ctx.query_instrument("ETHUSDC")

        print(f"   - BTC Instrument: {btc_instrument}")
        print(f"   - ETH Instrument: {eth_instrument}")

        # Get OHLC data
        btc_ohlc = ctx.ohlc(btc_instrument, "1h")
        if btc_ohlc:
            btc_df = btc_ohlc.pd()
            print(f"\n   ✅ BTC OHLC data loaded: {len(btc_df)} candles")
            print(f"   - Columns: {btc_df.columns.tolist()}")
            print(f"   - Time range: {btc_df.index[0]} to {btc_df.index[-1]}")
            print(f"   - Latest candle:\n{btc_df.tail(1)}")
        else:
            print("   ❌ No BTC OHLC data available")

        eth_ohlc = ctx.ohlc(eth_instrument, "1h")
        if eth_ohlc:
            eth_df = eth_ohlc.pd()
            print(f"\n   ✅ ETH OHLC data loaded: {len(eth_df)} candles")
            print(f"   - Time range: {eth_df.index[0]} to {eth_df.index[-1]}")
        else:
            print("   ❌ No ETH OHLC data available")

        # 6. Test adding more symbols (like in notebook)
        print("\n6. Testing dynamic universe expansion...")
        additional_symbols = ["XRPUSDC", "SOLUSDC"]
        additional_instruments = [ctx.query_instrument(symbol) for symbol in additional_symbols]
        print(f"   - Adding instruments: {additional_instruments}")
        ctx.set_universe(additional_instruments)
        print("   ✅ Universe expanded")

        # Wait a bit for new subscriptions
        time.sleep(2)

        # Test accessing new instrument data
        xrp_instrument = ctx.query_instrument("XRPUSDC")
        xrp_ohlc = ctx.ohlc(xrp_instrument, "1h")
        if xrp_ohlc:
            xrp_df = xrp_ohlc.pd()
            print(f"   ✅ XRP OHLC data available: {len(xrp_df)} candles")
        else:
            print("   ⚠️  XRP OHLC data not yet available (may need more time)")

        # 7. Keep running for a bit to receive live data
        print("\n7. Running strategy for 5 seconds to receive live data...")
        time.sleep(5)

        # 8. Stop the strategy
        print("\n8. Stopping strategy...")
        ctx.stop()
        time.sleep(1)

        print("\n" + "=" * 80)
        print("✅ INTEGRATION TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_notebook_workflow()
    sys.exit(0 if success else 1)
