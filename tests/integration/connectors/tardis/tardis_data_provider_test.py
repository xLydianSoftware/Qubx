import time
from threading import Event

import numpy as np
import pandas as pd
import pytest

from qubx import QubxLogConfig, logger
from qubx.connectors.tardis.data import TardisDataProvider
from qubx.core.basics import CtrlChannel, DataType, LiveTimeProvider
from qubx.core.lookups import lookup
from qubx.core.series import Bar, Quote


@pytest.mark.integration
class TestTardisDataProvider:
    """Integration tests for the TardisDataProvider class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test environment."""
        QubxLogConfig.set_log_level("DEBUG")

        # Set up a time provider and control channel
        self.time_provider = LiveTimeProvider()
        self.control = Event()
        self.control.set()  # Set the event to allow processing
        self.channel = CtrlChannel(name="test_channel")

        # Look up the instrument
        instrument = lookup.find_symbol("BITFINEX.F", "BTCUSDT")
        assert instrument is not None, "Could not find instrument BITFINEX.F:BTCUSDT"
        self.instrument = instrument

        # Set up the data provider
        self.data_provider = TardisDataProvider(
            host="quantlab",
            port=8011,  # WebSocket port - HTTP port will be 8010
            exchange="bitfinex-derivatives",
            time_provider=self.time_provider,
            channel=self.channel,
        )

        yield

        # Clean up after the test
        if hasattr(self, "data_provider") and self.data_provider:
            self.data_provider.close()

    @pytest.mark.integration
    def test_orderbook_subscription(self):
        """Test subscribing to orderbook data for Bitfinex BTCUSDT."""
        # Subscribe to orderbook
        subscription_type = "orderbook(0.01, 10)"  # 10 levels with 0.01% tick size

        # We've already asserted that self.instrument is not None in setup
        instrument = self.instrument
        self.data_provider.subscribe(subscription_type, {instrument})

        # Process data for 10 seconds
        received_data = []

        # Wait for data
        timeout = time.time() + 15  # 15 seconds timeout
        while time.time() < timeout:
            try:
                # Check for data received
                data = self.channel.receive(timeout=1)  # Use int timeout to avoid type error
                if data[0] is not None:  # Skip sentinel
                    instrument, data_type, payload, is_historical = data
                    logger.info(f"Received {data_type} data for {instrument.symbol}, historical: {is_historical}")
                    received_data.append(data)

                    # Break if we've received at least ten orderbook updates
                    if len(received_data) >= 5:
                        break
            except Exception:
                # Continue if we timeout waiting for data
                pass

        # Unsubscribe
        self.data_provider.unsubscribe(subscription_type, {self.instrument})

        # Assertions
        assert len(received_data) > 0, "Should receive at least one orderbook update"

        # Verify the received data
        for instrument, data_type, payload, is_historical in received_data:
            assert instrument == self.instrument, "Instrument should match the subscribed one"
            assert data_type.startswith("orderbook"), "Data type should be orderbook"
            assert is_historical is False, "Data should not be historical"

            # Check orderbook structure
            assert hasattr(payload, "bids"), "Orderbook should have bids"
            assert hasattr(payload, "asks"), "Orderbook should have asks"
            assert len(payload.bids) > 0, "Orderbook should have at least one bid"
            assert len(payload.asks) > 0, "Orderbook should have at least one ask"

            # Log some orderbook details
            logger.info(f"Top bid: {payload.bids[0]}, Top ask: {payload.asks[0]}")
            logger.info(f"Bid-ask spread: {payload.top_ask - payload.top_bid}")

        logger.info(f"Received {len(received_data)} orderbook updates in total")

    @pytest.mark.integration
    def test_warmup(self):
        """Test the warmup functionality for historical data."""
        # Define warmup configurations
        configs = {
            (f"{DataType.ORDERBOOK}(0.01, 10)", self.instrument): "1h",  # 1 hour of orderbook data
            (f"{DataType.TRADE}", self.instrument): "1h",  # 1 hour of trade data
            (f"{DataType.OHLC}(1m)", self.instrument): "1d",  # 1 day of 1-minute OHLC data
        }

        # Subscribe to data types first
        self.data_provider.subscribe(DataType.ORDERBOOK[0.01, 10], {self.instrument})
        self.data_provider.subscribe(DataType.TRADE, {self.instrument})
        self.data_provider.subscribe(DataType.OHLC["1m"], {self.instrument})

        # Perform warmup
        self.data_provider.warmup(configs)

        # Process historical data
        received_data = {"orderbook": [], "trade": [], "ohlc": []}

        # Wait for historical data
        timeout = time.time() + 30  # 30 seconds timeout
        while time.time() < timeout:
            try:
                # Check for data received
                data = self.channel.receive(timeout=1)
                if data[0] is not None:  # Skip sentinel
                    instrument, data_type, payload, is_historical = data

                    # Only process historical data from warmup
                    if is_historical:
                        logger.info(f"Received historical {data_type} data for {instrument.symbol}")

                        # Store data by type
                        if data_type.startswith("orderbook"):
                            received_data["orderbook"].append(data)
                        elif data_type == DataType.TRADE:
                            received_data["trade"].append(data)
                        elif data_type.startswith("ohlc"):
                            received_data["ohlc"].append(data)

                        # If we have received data for all types, we can stop
                        if received_data["orderbook"] and received_data["trade"] and received_data["ohlc"]:
                            break
            except Exception as e:
                # Continue if we timeout waiting for data
                logger.debug(f"Exception while receiving data: {e}")
                pass

        # Unsubscribe from all data types
        self.data_provider.unsubscribe(None, {self.instrument})

        # Check that we received at least some historical data
        logger.info(f"Received {len(received_data['orderbook'])} historical orderbook messages")
        logger.info(f"Received {len(received_data['trade'])} historical trade messages")
        logger.info(f"Received {len(received_data['ohlc'])} historical OHLC messages")

        # We might not receive all types of historical data depending on what's
        # available in the Tardis database, so just check that we got something
        assert any(len(data) > 0 for data in received_data.values()), "Should receive some historical data"

    @pytest.mark.integration
    def test_get_ohlc(self):
        """Test fetching historical OHLC data."""
        # Get 10 bars of 1-minute data
        bars = self.data_provider.get_ohlc(self.instrument, "1m", 10)

        # Validate returned bars
        assert isinstance(bars, list), "Should return a list of bars"
        assert len(bars) <= 10, "Should return at most 10 bars"

        if bars:
            logger.info(f"Received {len(bars)} OHLC bars")

            # Check structure of the first bar
            first_bar = bars[0]
            assert isinstance(first_bar, Bar), "Should return Bar objects"
            assert np.issubdtype(type(first_bar.time), np.integer), "Bar time should be a numpy integer type"
            assert isinstance(first_bar.open, float), "Bar open should be a float"
            assert isinstance(first_bar.high, float), "Bar high should be a float"
            assert isinstance(first_bar.low, float), "Bar low should be a float"
            assert isinstance(first_bar.close, float), "Bar close should be a float"
            assert isinstance(first_bar.volume, float), "Bar volume should be a float"

            # Log some bar details
            for bar in bars:
                bar_time = pd.Timestamp(bar.time)
                logger.info(
                    f"Bar: {bar_time.strftime('%Y-%m-%d %H:%M:%S')} - O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:.2f}"
                )
        else:
            logger.warning("No OHLC bars returned. This could be normal if data is not available.")

    @pytest.mark.integration
    def test_get_quote(self):
        """Test getting the latest quote for an instrument."""
        # First, we need to subscribe to orderbook or quote to ensure we have data
        self.data_provider.subscribe(DataType.ORDERBOOK[0.01, 10], {self.instrument})

        # Wait for some data to arrive
        logger.info("Waiting for orderbook data to populate quotes...")
        has_data = False
        timeout = time.time() + 15  # 15 seconds timeout

        while time.time() < timeout and not has_data:
            try:
                data = self.channel.receive(timeout=1)
                if data[0] is not None:  # Skip sentinel
                    instrument, data_type, payload, is_historical = data
                    if data_type.startswith("orderbook") and not is_historical:
                        logger.info("Received orderbook data, should now have a quote")
                        has_data = True
                        break
            except Exception:
                # Continue if we timeout waiting for data
                pass

        # Now try to get a quote
        quote = self.data_provider.get_quote(self.instrument)

        # Unsubscribe
        self.data_provider.unsubscribe(DataType.ORDERBOOK, {self.instrument})

        # Check the quote
        if has_data:
            assert quote is not None, "Should have a quote after receiving orderbook data"
            assert isinstance(quote, Quote), "Should return a Quote object"
            logger.info(f"Got quote - bid: {quote.bid:.2f}, ask: {quote.ask:.2f}, spread: {quote.ask - quote.bid:.2f}")

            # Validate quote properties
            assert quote.bid > 0, "Quote bid should be positive"
            assert quote.ask > 0, "Quote ask should be positive"
            assert quote.ask >= quote.bid, "Quote ask should be greater than or equal to bid"
        else:
            logger.warning("No orderbook data received, cannot test get_quote properly")
