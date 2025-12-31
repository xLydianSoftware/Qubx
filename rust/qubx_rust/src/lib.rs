use pyo3::prelude::*;

mod math;

/// A Python module implemented in Rust.
#[pymodule]
fn qubx_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register functions from submodules
    m.add_function(wrap_pyfunction!(math::fibonacci, m)?)?;
    Ok(())
}
