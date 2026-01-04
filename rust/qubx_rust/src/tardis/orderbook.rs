//! Orderbook state machine with BTreeMap-based storage

use std::collections::BTreeMap;

use super::types::{DepthUpdate, OrderbookSnapshotInner, Side};

/// Orderbook state machine that maintains bid/ask levels
///
/// Uses BTreeMap for efficient sorted access to price levels.
/// Prices are converted to integer ticks for precision.
pub struct Orderbook {
    /// Price tick size (e.g., 0.0001 for MANAUSDT)
    pub tick_size: f64,
    /// Bid levels: price_tick -> quantity (sorted by key, we iterate in reverse for best bid)
    pub bids: BTreeMap<i64, f64>,
    /// Ask levels: price_tick -> quantity (sorted by key, first is best ask)
    pub asks: BTreeMap<i64, f64>,
    /// Last update timestamp
    pub timestamp: i64,
}

impl Orderbook {
    /// Create a new empty orderbook
    pub fn new(tick_size: f64) -> Self {
        Self {
            tick_size,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            timestamp: 0,
        }
    }

    /// Convert price to tick (integer representation)
    #[inline]
    fn price_to_tick(&self, price: f64) -> i64 {
        (price / self.tick_size).round() as i64
    }

    /// Convert tick back to price
    #[inline]
    fn tick_to_price(&self, tick: i64) -> f64 {
        tick as f64 * self.tick_size
    }

    /// Apply a depth update to the orderbook
    pub fn update(&mut self, update: &DepthUpdate) {
        let price_tick = self.price_to_tick(update.price);

        let book = match update.side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };

        if update.amount == 0.0 {
            // Remove the price level
            book.remove(&price_tick);
        } else {
            // Update or insert the price level
            book.insert(price_tick, update.amount);
        }

        self.timestamp = update.timestamp;
    }

    /// Clear all levels from the orderbook
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    /// Get a snapshot of the top N levels from each side
    pub fn snapshot(&self, depth: usize) -> OrderbookSnapshotInner {
        // Bids: highest prices first (reverse iterator)
        let bids: Vec<(f64, f64)> = self
            .bids
            .iter()
            .rev()
            .take(depth)
            .map(|(&tick, &qty)| (self.tick_to_price(tick), qty))
            .collect();

        // Asks: lowest prices first (forward iterator)
        let asks: Vec<(f64, f64)> = self
            .asks
            .iter()
            .take(depth)
            .map(|(&tick, &qty)| (self.tick_to_price(tick), qty))
            .collect();

        OrderbookSnapshotInner {
            timestamp: self.timestamp,
            bids,
            asks,
        }
    }

    /// Get the best bid (highest bid price and quantity)
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids
            .iter()
            .next_back()
            .map(|(&tick, &qty)| (self.tick_to_price(tick), qty))
    }

    /// Get the best ask (lowest ask price and quantity)
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks
            .iter()
            .next()
            .map(|(&tick, &qty)| (self.tick_to_price(tick), qty))
    }

    /// Get the mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_basic() {
        let mut ob = Orderbook::new(0.01);

        // Add some bids
        ob.update(&DepthUpdate {
            timestamp: 1000,
            local_timestamp: 1000,
            is_snapshot: false,
            side: Side::Bid,
            price: 100.00,
            amount: 1.0,
        });
        ob.update(&DepthUpdate {
            timestamp: 1001,
            local_timestamp: 1001,
            is_snapshot: false,
            side: Side::Bid,
            price: 99.99,
            amount: 2.0,
        });

        // Add some asks
        ob.update(&DepthUpdate {
            timestamp: 1002,
            local_timestamp: 1002,
            is_snapshot: false,
            side: Side::Ask,
            price: 100.01,
            amount: 1.5,
        });

        assert_eq!(ob.best_bid(), Some((100.00, 1.0)));
        assert_eq!(ob.best_ask(), Some((100.01, 1.5)));

        // Remove a level
        ob.update(&DepthUpdate {
            timestamp: 1003,
            local_timestamp: 1003,
            is_snapshot: false,
            side: Side::Bid,
            price: 100.00,
            amount: 0.0,
        });

        let best_bid = ob.best_bid().unwrap();
        assert!((best_bid.0 - 99.99).abs() < 0.001);
        assert!((best_bid.1 - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_orderbook_snapshot() {
        let mut ob = Orderbook::new(0.01);

        // Add 5 bids
        for i in 0..5 {
            ob.update(&DepthUpdate {
                timestamp: i as i64,
                local_timestamp: i as i64,
                is_snapshot: false,
                side: Side::Bid,
                price: 100.0 - i as f64 * 0.01,
                amount: 1.0,
            });
        }

        // Add 5 asks
        for i in 0..5 {
            ob.update(&DepthUpdate {
                timestamp: i as i64,
                local_timestamp: i as i64,
                is_snapshot: false,
                side: Side::Ask,
                price: 100.01 + i as f64 * 0.01,
                amount: 1.0,
            });
        }

        let snap = ob.snapshot(3);
        assert_eq!(snap.bids.len(), 3);
        assert_eq!(snap.asks.len(), 3);

        // Bids should be sorted desc (best first)
        assert!((snap.bids[0].0 - 100.00).abs() < 0.001);
        assert!((snap.bids[1].0 - 99.99).abs() < 0.001);
        assert!((snap.bids[2].0 - 99.98).abs() < 0.001);

        // Asks should be sorted asc (best first)
        assert!((snap.asks[0].0 - 100.01).abs() < 0.001);
        assert!((snap.asks[1].0 - 100.02).abs() < 0.001);
        assert!((snap.asks[2].0 - 100.03).abs() < 0.001);
    }
}
