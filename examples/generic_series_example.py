import numpy as np

from qubx.core.series import GenericSeries, IndicatorGeneric, Quote


# - Example 1: Bid-Ask Spread Indicator
class BidAskSpread(IndicatorGeneric):
    """Calculate bid-ask spread from Quote data"""

    def calculate(self, time, quote, new_item_started):
        if quote is None:
            return np.nan
        return quote.ask - quote.bid


class MidPrice(IndicatorGeneric):
    """Calculate mid price from Quote data"""

    def calculate(self, time, quote, new_item_started):
        if quote is None:
            return np.nan
        return quote.mid_price()


class SpreadBps(IndicatorGeneric):
    """Calculate spread in basis points"""

    def calculate(self, time, quote, new_item_started):
        if quote is None:
            return np.nan
        mid = quote.mid_price()
        if mid == 0:
            return np.nan
        spread = quote.ask - quote.bid
        return (spread / mid) * 10000  # - in basis points


def main():
    # - Create a GenericSeries for 5-minute Quote data
    quotes = GenericSeries("BTCUSDT_quotes", "5Min")

    # - Attach indicators
    spread = BidAskSpread("spread", quotes)
    mid_price = MidPrice("mid_price", quotes)
    spread_bps = SpreadBps("spread_bps", quotes)

    # - Simulate quote updates
    base_time = np.datetime64("2024-01-01T00:00:00", "ns")
    minute_ns = 60 * 10**9

    print("Updating quotes...")
    for i in range(10):
        time = base_time.item() + i * minute_ns
        bid = 50000 + i * 10
        ask = bid + 5 + i * 0.5  # - spread increases slightly
        bid_size = 10 + i
        ask_size = 12 + i

        q = Quote(time, bid, ask, bid_size, ask_size)
        quotes.update(q)

        print(f"\nTime: {np.datetime64(time, 'ns')}")
        print(f"  Quote: {q}")
        print(f"  Spread: {spread[0]:.2f}")
        print(f"  Mid Price: {mid_price[0]:.2f}")
        print(f"  Spread (bps): {spread_bps[0]:.4f}")

    print("\n\nGenericSeries summary:")
    print(quotes)

    print("\n\nSpread indicator summary:")
    print(spread)

    # - Convert to pandas
    print("\n\nAs pandas Series:")
    print(spread.pd())


if __name__ == "__main__":
    main()
