//! Rust kernels for quantibbler.
//!
//! Each kernel mirrors a NumPy reference implementation in the Python
//! package; parity is enforced by probe tests (float64, rel 1e-12) pinned
//! in the openspec specs.

use pyo3::prelude::*;

#[pyfunction]
fn backend_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pymodule]
fn quantibbler_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backend_version, m)?)?;
    Ok(())
}
