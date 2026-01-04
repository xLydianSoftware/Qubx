use pyo3::prelude::*;

mod math;
mod tardis;

/// A Python module implemented in Rust.
#[pymodule]
fn qubx_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register functions from submodules
    m.add_function(wrap_pyfunction!(math::fibonacci, m)?)?;

    // Register tardis classes
    m.add_class::<tardis::OrderbookSnapshotTransformer>()?;
    m.add_class::<tardis::PyOrderbookSnapshot>()?;

    Ok(())
}
