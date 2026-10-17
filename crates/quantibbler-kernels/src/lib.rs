//! Rust kernels for quantibbler.
//!
//! ONE kernel implementation, TWO faces: the plain-Rust rlib face is
//! [`core`]'s public API (linked directly by the pure-Rust server); the
//! PyO3 face below is a thin binding that only converts boundary types and
//! maps errors — no kernel arithmetic lives here (design D9).
//!
//! Parity with the NumPy reference is enforced by probe tests: pytest
//! across backends in-process, and cargo tests against committed JSON
//! fixtures (float64 rel 1e-12, packed outputs byte-identical).

pub mod core;

use numpy::{Element, PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyTuple};

use crate::core::KernelError;

fn to_value_error(e: KernelError) -> PyErr {
    PyValueError::new_err(e.0)
}

/// Extract any-dim f64 view plus its flat row-major buffer and shape.
fn dyn_f64<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let arr = obj.extract::<PyReadonlyArrayDyn<'py, f64>>()?;
    let view = arr.as_array();
    let shape: Vec<usize> = view.shape().to_vec();
    Ok((view.iter().copied().collect(), shape))
}

/// Extract any-dim u8 view plus its flat row-major buffer and shape.
fn dyn_u8<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(Vec<u8>, Vec<usize>)> {
    let arr = obj.extract::<PyReadonlyArrayDyn<'py, u8>>()?;
    let view = arr.as_array();
    let shape: Vec<usize> = view.shape().to_vec();
    Ok((view.iter().copied().collect(), shape))
}

/// Extract a ±1 factor as flat f64 (accepts float64 or int8 input per D2).
fn dyn_factor_f64<'py>(obj: &Bound<'py, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'py, f64>>() {
        let view = arr.as_array();
        let shape: Vec<usize> = view.shape().to_vec();
        return Ok((view.iter().copied().collect(), shape));
    }
    let arr = obj.extract::<PyReadonlyArrayDyn<'py, i8>>()?;
    let view = arr.as_array();
    let shape: Vec<usize> = view.shape().to_vec();
    Ok((view.iter().map(|&v| v as f64).collect(), shape))
}

/// Row/column counts of a shape flattened to 2-D (row-major).
fn rows_cols(shape: &[usize]) -> (usize, usize) {
    let n_cols = shape.last().copied().unwrap_or(0);
    let n_rows: usize = shape[..shape.len().saturating_sub(1)].iter().product();
    (n_rows, n_cols)
}

fn build_arr2<'py, T: Element>(
    py: Python<'py>,
    data: &[T],
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    let flat = PyArray1::<T>::from_slice(py, data);
    let reshaped: Bound<'py, PyArray<T, numpy::Ix2>> =
        PyArrayMethods::reshape(&flat, (n_rows, n_cols))?;
    Ok(reshaped)
}

#[pyfunction]
fn backend_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// pack_bits: ±1 factor -> uint8 bitplanes (row-major, LSB-first, pad 0).
#[pyfunction]
fn pack_bits<'py>(
    py: Python<'py>,
    factor: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let (flat, shape) = dyn_factor_f64(factor)?;
    let (n_rows, n_cols) = rows_cols(&shape);
    let as_i8: Vec<i8> = flat.iter().map(|&v| v as i8).collect();
    let packed = core::pack_bits(&as_i8, n_rows, n_cols).map_err(to_value_error)?;
    let n_words = n_cols.div_ceil(8);
    build_arr2(py, &packed, n_rows, n_words)
}

/// unpack_bits: uint8 bitplanes + explicit logical width -> ±1 factor.
#[pyfunction]
fn unpack_bits<'py>(
    py: Python<'py>,
    packed: &Bound<'py, PyAny>,
    n_bits: usize,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let (pflat, pshape) = dyn_u8(packed)?;
    let (n_rows, n_words) = rows_cols(&pshape);
    if n_bits > n_words * 8 {
        return Err(PyValueError::new_err(format!(
            "n_bits={n_bits} exceeds the packed width {n_words} ({} bits)",
            n_words * 8
        )));
    }
    let unpacked = core::unpack_bits(&pflat, n_rows, n_bits).map_err(to_value_error)?;
    match dtype {
        "int8" => build_arr2(py, &unpacked, n_rows, n_bits).map(|a| a.into_any().unbind()),
        "float64" => {
            let as_f: Vec<f64> = unpacked.iter().map(|&v| v as f64).collect();
            build_arr2(py, &as_f, n_rows, n_bits).map(|a| a.into_any().unbind())
        }
        other => Err(PyValueError::new_err(format!(
            "unpack_bits dtype must be 'int8' or 'float64', got {other:?}"
        ))),
    }
}

/// sign_estimate: int8 sign, exact zeros -> +1 (documented tie-break).
#[pyfunction]
fn sign_estimate<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let (flat, shape) = dyn_f64(x)?;
    let signs = core::sign_estimate(&flat);
    let total: usize = shape.iter().product();
    let flat_arr = PyArray1::<i8>::from_slice(py, &signs);
    let reshaped: Bound<'py, PyArray<i8, numpy::Ix1>> = PyArrayMethods::reshape(&flat_arr, total)?;
    Ok(reshaped)
}

/// packed_matvec: unpack(packed) @ vec without materializing the matrix.
#[pyfunction]
fn packed_matvec<'py>(
    py: Python<'py>,
    packed: &Bound<'py, PyAny>,
    n_bits: usize,
    vec: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (pflat, pshape) = dyn_u8(packed)?;
    let (n_rows, _n_words) = rows_cols(&pshape);
    let (vflat, _vshape) = dyn_f64(vec)?;
    let out = core::packed_matvec(&pflat, n_rows, n_bits, &vflat).map_err(to_value_error)?;
    Ok(PyArray1::from_slice(py, &out))
}

/// packed_matmul: column-batched packed_matvec (packed x dense-float only).
#[pyfunction]
fn packed_matmul<'py>(
    py: Python<'py>,
    packed: &Bound<'py, PyAny>,
    n_bits: usize,
    mat: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (pflat, pshape) = dyn_u8(packed)?;
    let (n_rows, _n_words) = rows_cols(&pshape);
    let (mflat, mshape) = dyn_f64(mat)?;
    let n_cols = mshape.last().copied().unwrap_or(0);
    let out =
        core::packed_matmul(&pflat, n_rows, n_bits, &mflat, n_cols).map_err(to_value_error)?;
    let k_cols = mflat
        .len()
        .checked_div(n_bits.max(1))
        .unwrap_or(mflat.len());
    build_arr2(py, &out, n_rows, k_cols)
}

/// joint_itq: alternating Procrustes over Z = [u_hat; v_hat] ([LB2] 4.4).
#[pyfunction]
#[pyo3(signature = (u_hat, v_hat, *, max_iter = None, tol = None, seed = None))]
fn joint_itq<'py>(
    py: Python<'py>,
    u_hat: &Bound<'py, PyAny>,
    v_hat: &Bound<'py, PyAny>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    seed: Option<u64>,
) -> PyResult<Py<PyTuple>> {
    let (uflat, ushape) = dyn_f64(u_hat)?;
    let (vflat, vshape) = dyn_f64(v_hat)?;
    let (dout, rank) = rows_cols(&ushape);
    let (din, rank_v) = rows_cols(&vshape);
    if rank != rank_v {
        return Err(PyValueError::new_err(format!(
            "joint_itq rank mismatch: u_hat r={rank}, v_hat r={rank_v}"
        )));
    }
    let (u_t, v_t, n_iter, converged) = core::joint_itq(
        &uflat,
        &vflat,
        dout,
        din,
        rank,
        max_iter.unwrap_or(core::DEFAULT_MAX_ITER),
        tol.unwrap_or(core::DEFAULT_TOL),
        seed.unwrap_or(0),
    )
    .map_err(to_value_error)?;
    let u_arr = build_arr2(py, &u_t, dout, rank)?;
    let v_arr = build_arr2(py, &v_t, din, rank)?;
    let u_obj: Py<PyAny> = u_arr.into_any().unbind();
    let v_obj: Py<PyAny> = v_arr.into_any().unbind();
    let n_obj: Py<PyAny> = n_iter.into_pyobject(py)?.unbind().into();
    let c_obj: Py<PyAny> = PyBool::new(py, converged).to_owned().into_any().unbind();
    Ok(PyTuple::new(py, [u_obj, v_obj, n_obj, c_obj])?.unbind())
}

#[pymodule]
fn quantibbler_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backend_version, m)?)?;
    m.add_function(wrap_pyfunction!(pack_bits, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bits, m)?)?;
    m.add_function(wrap_pyfunction!(sign_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(packed_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(packed_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(joint_itq, m)?)?;
    Ok(())
}
