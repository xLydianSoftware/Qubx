//! Orderbook snapshot transformer implementing IDataTransformer interface

use pyo3::prelude::*;
use pyo3::types::PyString;

use super::orderbook::Orderbook;
use super::types::{DepthUpdate, PyOrderbookSnapshot, Side};

/// Stateful transformer that converts orderbook updates to snapshots
///
/// Implements IDataTransformer interface for seamless integration with
/// Qubx data pipeline. Processes raw tuple data directly without Python
/// iteration.
///
/// Usage:
/// ```python
/// parser = OrderbookSnapshotTransformer(tick_size=0.0001, depth=25, interval_ms=100)
/// for chunk in reader.read("MANAUSDT", DataType.ORDERBOOK, "2024-11-01", "2024-11-03", chunksize=1):
///     snapshots = chunk.transform(parser)  # Raw data goes directly to Rust!
/// ```
#[pyclass(name = "OrderbookSnapshotTransformer")]
pub struct OrderbookSnapshotTransformer {
    /// Internal orderbook state
    orderbook: Orderbook,
    /// Last snapshot timestamp
    last_snapshot_ts: i64,
    /// Number of levels per side to include in snapshot
    snapshot_depth: usize,
    /// Minimum interval between snapshots (milliseconds)
    snapshot_interval_ms: i64,
    /// Column indices (resolved on first call)
    ts_idx: Option<usize>,
    is_snapshot_idx: Option<usize>,
    side_idx: Option<usize>,
    price_idx: Option<usize>,
    amount_idx: Option<usize>,
}

#[pymethods]
impl OrderbookSnapshotTransformer {
    /// Create a new transformer
    ///
    /// Args:
    ///     tick_size: Price tick size (e.g., 0.0001 for MANAUSDT)
    ///     depth: Number of levels per side in snapshots
    ///     interval_ms: Minimum interval between snapshots in milliseconds
    #[new]
    #[pyo3(signature = (tick_size, depth, interval_ms))]
    fn new(tick_size: f64, depth: usize, interval_ms: i64) -> Self {
        Self {
            orderbook: Orderbook::new(tick_size),
            last_snapshot_ts: 0,
            snapshot_depth: depth,
            snapshot_interval_ms: interval_ms,
            ts_idx: None,
            is_snapshot_idx: None,
            side_idx: None,
            price_idx: None,
            amount_idx: None,
        }
    }

    /// IDataTransformer.process_data() - receives raw tuples directly
    ///
    /// Args:
    ///     data_id: Symbol/instrument identifier
    ///     dtype: Data type (e.g., "ORDERBOOK")
    ///     raw_data: List of tuples, each tuple is a row from CSV
    ///     names: Column names
    ///     index: Timestamp column index
    ///
    /// Returns:
    ///     List of OrderbookSnapshot objects
    fn process_data(
        &mut self,
        _data_id: &str,
        _dtype: &Bound<'_, PyAny>,
        raw_data: &Bound<'_, PyAny>,
        names: Vec<String>,
        _index: usize,
    ) -> PyResult<Vec<PyOrderbookSnapshot>> {
        // Resolve column indices on first call
        if self.ts_idx.is_none() {
            self.resolve_indices(&names)?;
        }

        let ts_idx = self.ts_idx.unwrap();
        let is_snapshot_idx = self.is_snapshot_idx.unwrap();
        let side_idx = self.side_idx.unwrap();
        let price_idx = self.price_idx.unwrap();
        let amount_idx = self.amount_idx.unwrap();

        let mut snapshots = Vec::new();

        // Iterate over raw_data (list of tuples/lists)
        for row in raw_data.try_iter()? {
            let row = row?;

            // Parse row fields
            let timestamp: i64 = row.get_item(ts_idx)?.extract()?;
            let is_snapshot = self.parse_bool(&row.get_item(is_snapshot_idx)?)?;
            let side = self.parse_side(&row.get_item(side_idx)?)?;
            let price: f64 = row.get_item(price_idx)?.extract()?;
            let amount: f64 = row.get_item(amount_idx)?.extract()?;

            let update = DepthUpdate {
                timestamp,
                local_timestamp: timestamp, // Not using local_timestamp in core logic
                is_snapshot,
                side,
                price,
                amount,
            };

            // Clear orderbook on SOD snapshot
            if update.is_snapshot {
                self.orderbook.clear();
            }

            // Apply update
            self.orderbook.update(&update);

            // Emit snapshot at interval
            if update.timestamp - self.last_snapshot_ts >= self.snapshot_interval_ms {
                let snap = self.orderbook.snapshot(self.snapshot_depth);
                snapshots.push(snap.into());
                self.last_snapshot_ts = update.timestamp;
            }
        }

        Ok(snapshots)
    }

    /// Reset transformer state (for new symbol or fresh start)
    fn reset(&mut self) {
        self.orderbook.clear();
        self.last_snapshot_ts = 0;
        // Keep column indices - they're likely the same
    }

    /// IDataTransformer.combine_data() - returns transformed data as-is
    fn combine_data(&self, transformed: PyObject) -> PyObject {
        transformed
    }

    fn __repr__(&self) -> String {
        format!(
            "OrderbookSnapshotTransformer(tick_size={}, depth={}, interval_ms={})",
            self.orderbook.tick_size, self.snapshot_depth, self.snapshot_interval_ms
        )
    }

    /// Get current orderbook statistics
    #[pyo3(name = "stats")]
    fn get_stats(&self) -> (usize, usize, i64) {
        (
            self.orderbook.bids.len(),
            self.orderbook.asks.len(),
            self.last_snapshot_ts,
        )
    }
}

impl OrderbookSnapshotTransformer {
    /// Resolve column indices from names
    fn resolve_indices(&mut self, names: &[String]) -> PyResult<()> {
        let find_idx = |name: &str| -> PyResult<usize> {
            names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Column '{}' not found in names: {:?}",
                        name, names
                    ))
                })
        };

        self.ts_idx = Some(find_idx("timestamp")?);
        self.is_snapshot_idx = Some(find_idx("is_snapshot")?);
        self.side_idx = Some(find_idx("side")?);
        self.price_idx = Some(find_idx("price")?);
        self.amount_idx = Some(find_idx("amount")?);

        Ok(())
    }

    /// Parse boolean value from various Python types
    fn parse_bool(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Try direct bool extraction first
        if let Ok(b) = value.extract::<bool>() {
            return Ok(b);
        }

        // Try string parsing
        if let Ok(s) = value.extract::<String>() {
            return Ok(matches!(s.to_lowercase().as_str(), "true" | "1" | "yes"));
        }

        // Try int
        if let Ok(i) = value.extract::<i64>() {
            return Ok(i != 0);
        }

        Ok(false)
    }

    /// Parse side from string or other types
    fn parse_side(&self, value: &Bound<'_, PyAny>) -> PyResult<Side> {
        if let Ok(s) = value.downcast::<PyString>() {
            let s = s.to_string_lossy();
            return Ok(match s.to_lowercase().as_str() {
                "bid" | "buy" => Side::Bid,
                "ask" | "sell" => Side::Ask,
                _ => Side::Bid, // Default to bid for unknown
            });
        }

        // Try extracting as string
        if let Ok(s) = value.extract::<String>() {
            return Ok(match s.to_lowercase().as_str() {
                "bid" | "buy" => Side::Bid,
                "ask" | "sell" => Side::Ask,
                _ => Side::Bid,
            });
        }

        Ok(Side::Bid)
    }
}
