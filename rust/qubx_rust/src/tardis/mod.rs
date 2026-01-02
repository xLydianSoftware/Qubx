//! Tardis orderbook data processing module
//!
//! This module provides efficient Rust-based processing of Tardis
//! incremental_book_L2 data, converting orderbook updates into
//! fixed-interval snapshots.
//!
//! Key components:
//! - `OrderbookSnapshotTransformer`: Stateful transformer implementing IDataTransformer
//! - `OrderbookSnapshot`: Snapshot result with bids, asks, and metadata
//! - `Orderbook`: Internal orderbook state machine

mod orderbook;
mod transform;
mod types;

// Re-export public types
pub use transform::OrderbookSnapshotTransformer;
pub use types::PyOrderbookSnapshot;
