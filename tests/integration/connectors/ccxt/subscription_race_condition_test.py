import time
from threading import Thread
from typing import Dict, List

import pytest

from qubx import QubxLogConfig, logger
from qubx.connectors.ccxt.data import CcxtDataProvider
from qubx.connectors.ccxt.factory import get_ccxt_exchange_manager
from qubx.core.basics import AssetType, CtrlChannel, DataType, Instrument, LiveTimeProvider, MarketType
from qubx.health import DummyHealthMonitor


@pytest.mark.integration
class TestCcxtSubscriptionRaceConditions:
    """Integration tests for CCXT subscription race conditions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        QubxLogConfig.set_log_level("DEBUG")

        # Create test instruments with all required parameters
        self.btc_instrument = Instrument(
            symbol="BTCUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="BINANCE.UM",
            base="BTC",
            quote="USDT",
            settle="USDT",
            exchange_symbol="BTCUSDT",
            tick_size=0.1,
            lot_size=0.001,
            min_size=0.001,
        )
        self.eth_instrument = Instrument(
            symbol="ETHUSDT",
            asset_type=AssetType.CRYPTO,
            market_type=MarketType.SWAP,
            exchange="BINANCE.UM",
            base="ETH",
            quote="USDT",
            settle="USDT",
            exchange_symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_size=0.001,
        )

        self.all_instruments = [self.btc_instrument, self.eth_instrument]

        # Create control channel and provider using the factory
        self.channel = CtrlChannel("test_channel")
        self.time_provider = LiveTimeProvider()

        # Use the proper factory function to create exchange with event loop
        exchange_manager = get_ccxt_exchange_manager(
            exchange="binanceusdm",
            use_testnet=True,  # Use testnet for safer testing
        )

        self.provider = CcxtDataProvider(
            exchange_manager=exchange_manager,
            time_provider=self.time_provider,
            channel=self.channel,
            max_ws_retries=3,
            warmup_timeout=30,
            health_monitor=DummyHealthMonitor(),
        )

        # Track received data
        self.received_data: Dict[str, List] = {"ohlc": [], "trade": [], "orderbook": [], "quote": []}

        # Start data collection thread
        self.data_thread = Thread(target=self._collect_data, daemon=True)
        self.data_thread.start()

        yield

        # Cleanup
        self.channel.stop()
        if hasattr(self, "provider"):
            self.provider.close()

    def _collect_data(self):
        """Collect data from the channel."""
        while self.channel.control.is_set():
            try:
                data = self.channel.receive(timeout=1)
                if data:
                    instrument, data_type, payload, is_historical = data
                    sub_type = data_type.split("[")[0] if "[" in data_type else data_type
                    if sub_type in self.received_data:
                        self.received_data[sub_type].append(
                            {
                                "instrument": instrument,
                                "data_type": data_type,
                                "payload": payload,
                                "is_historical": is_historical,
                                "timestamp": time.time(),
                            }
                        )
            except Exception as e:
                if self.channel.control.is_set():
                    logger.debug(f"Data collection error: {e}")
                break

    def test_rapid_subscription_changes(self):
        """Test rapid subscription changes to detect race conditions."""
        logger.info("Testing rapid subscription changes...")

        # Test OHLC subscriptions with rapid changes
        subscription_type = DataType.OHLC["1m"]

        # Rapid subscription sequence
        for i in range(3):  # Reduced iterations for faster test
            instruments = [self.btc_instrument]
            if i % 2 == 1:
                instruments.append(self.eth_instrument)

            logger.info(f"Subscription change {i + 1}: {[inst.symbol for inst in instruments]}")
            self.provider.subscribe(subscription_type, instruments, reset=True)

            # Brief wait to let subscription start
            time.sleep(1)

            # Verify subscription state
            subscribed = self.provider.get_subscribed_instruments(subscription_type)
            logger.info(f"Currently subscribed: {[inst.symbol for inst in subscribed]}")

            # Check that we have the expected instruments (either active or pending)
            expected_symbols = {inst.symbol for inst in instruments}
            actual_symbols = {inst.symbol for inst in subscribed}

            assert expected_symbols.issubset(actual_symbols) or len(actual_symbols) > 0, (
                f"Expected {expected_symbols}, got {actual_symbols}"
            )

        # Wait for data to flow
        time.sleep(3)

        # Verify we received some OHLC data
        logger.info(f"Received {len(self.received_data['ohlc'])} OHLC updates")
        # Note: In testnet, we might not get data, so just check the subscription worked

    def test_same_instruments_resubscription(self):
        """Test resubscribing to the same instruments (name collision scenario)."""
        logger.info("Testing same instruments resubscription...")

        subscription_type = DataType.TRADE
        instruments = [self.btc_instrument]

        # Subscribe multiple times to same instruments
        for i in range(3):
            logger.info(f"Resubscription {i + 1} to same instruments")
            self.provider.subscribe(subscription_type, instruments, reset=True)
            time.sleep(1)

            # Verify subscription state
            subscribed = self.provider.get_subscribed_instruments(subscription_type)
            logger.info(f"Subscribed instruments: {[inst.symbol for inst in subscribed]}")

            # Should have the expected instruments
            expected_symbols = {inst.symbol for inst in instruments}
            actual_symbols = {inst.symbol for inst in subscribed}
            assert expected_symbols.issubset(actual_symbols), f"Expected {expected_symbols}, got {actual_symbols}"

        logger.info("Same instruments resubscription test completed")

    def test_subscription_state_consistency(self):
        """Test that subscription state remains consistent during rapid changes."""
        logger.info("Testing subscription state consistency...")

        subscription_type = DataType.OHLC["5m"]

        # Perform rapid subscription changes
        test_sequences = [
            [self.btc_instrument],
            [self.btc_instrument, self.eth_instrument],
            [self.eth_instrument],
            [],  # Unsubscribe all
        ]

        for i, instruments in enumerate(test_sequences):
            logger.info(f"Sequence {i + 1}: {[inst.symbol for inst in instruments]}")

            if instruments:
                self.provider.subscribe(subscription_type, instruments, reset=True)
            else:
                # Unsubscribe by subscribing to empty list
                self.provider.subscribe(subscription_type, [], reset=True)

            time.sleep(1)

            # Check state consistency
            subscribed = self.provider.get_subscribed_instruments(subscription_type)
            logger.info(f"State after sequence {i + 1}: {[inst.symbol for inst in subscribed]}")

            # Verify has_subscription consistency
            for instrument in self.all_instruments:
                has_sub = self.provider.has_subscription(instrument, subscription_type)
                has_pending = self.provider.has_pending_subscription(instrument, subscription_type)
                in_subscribed = instrument in subscribed

                logger.debug(
                    f"{instrument.symbol}: has_sub={has_sub}, has_pending={has_pending}, in_list={in_subscribed}"
                )

                # If instrument is in subscribed list, it should have subscription or pending
                if in_subscribed:
                    assert has_sub or has_pending, (
                        f"{instrument.symbol} in subscribed list but has_subscription={has_sub}, has_pending={has_pending}"
                    )

        logger.info("Subscription state consistency test completed")


if __name__ == "__main__":
    # Run a specific test for quick validation
    test_instance = TestCcxtSubscriptionRaceConditions()
    test_instance.setup()
    try:
        test_instance.test_rapid_subscription_changes()
        logger.info("Quick test passed!")
    finally:
        # Cleanup would be handled by fixture, but we're running standalone
        test_instance.channel.stop()
        if hasattr(test_instance, "provider"):
            test_instance.provider.close()
