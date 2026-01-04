//! Core types for orderbook processing

use pyo3::prelude::*;

/// Side of the orderbook (bid or ask)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// A single orderbook update from tardis incremental_book_L2 data
#[derive(Debug, Clone)]
pub struct DepthUpdate {
    pub timestamp: i64,
    pub local_timestamp: i64,
    pub is_snapshot: bool,
    pub side: Side,
    pub price: f64,
    pub amount: f64,
}

/// An orderbook snapshot at a point in time
#[derive(Debug, Clone)]
pub struct OrderbookSnapshotInner {
    pub timestamp: i64,
    pub bids: Vec<(f64, f64)>, // (price, qty) sorted desc by price
    pub asks: Vec<(f64, f64)>, // (price, qty) sorted asc by price
}

/// PyO3 wrapper for OrderbookSnapshot exposed to Python
#[pyclass(name = "OrderbookSnapshot")]
#[derive(Debug, Clone)]
pub struct PyOrderbookSnapshot {
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub bids: Vec<(f64, f64)>,
    #[pyo3(get)]
    pub asks: Vec<(f64, f64)>,
}

impl From<OrderbookSnapshotInner> for PyOrderbookSnapshot {
    fn from(snap: OrderbookSnapshotInner) -> Self {
        Self {
            timestamp: snap.timestamp,
            bids: snap.bids,
            asks: snap.asks,
        }
    }
}

#[pymethods]
impl PyOrderbookSnapshot {
    fn __repr__(&self) -> String {
        format!(
            "OrderbookSnapshot(ts={}, bids={}, asks={})",
            self.timestamp,
            self.bids.len(),
            self.asks.len()
        )
    }

    /// Get best bid price and quantity
    #[getter]
    fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.first().copied()
    }

    /// Get best ask price and quantity
    #[getter]
    fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.first().copied()
    }

    /// Get mid price (average of best bid and best ask)
    #[getter]
    fn mid_price(&self) -> Option<f64> {
        match (self.bids.first(), self.asks.first()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread (best ask - best bid)
    #[getter]
    fn spread(&self) -> Option<f64> {
        match (self.bids.first(), self.asks.first()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }
}
