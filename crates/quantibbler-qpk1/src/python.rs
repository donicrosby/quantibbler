//! PyO3 face (design B10): blob READING for the host tools and the cross-face
//! parity tests.
//!
//! The qpk1 crate owns the blob reader, and this module is the only way Python
//! reaches it — so there is exactly one blob parser in the repo. The WRITER is
//! host-side Python (`src/quantibbler/blob.py`, design B11); reading a blob in
//! Python must go through here, never through a second Python implementation.
//!
//! Importable as ``quantibbler_qpk1`` once built with the ``extension-module``
//! feature (``maturin`` does that via crates/quantibbler-qpk1/pyproject.toml).

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

/// Parse and validate a qpk1 blob, or raise ``ValueError`` with the refusal.
///
/// Returns ``{"header": {...}, "sections": [{"name", "tag", "data"}, ...]}``:
/// every declared header field at its parsed value, and each payload section's
/// name, dtype tag (0 qpk1_v1, 1 qpk1_v2, 2 fp32_le, 3 bf16_le, 4 raw) and raw
/// bytes. Payload bytes are copied out of the mapping, so the returned objects
/// outlive it.
#[pyfunction]
fn read_blob<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let file = std::fs::File::open(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    // Phase 1: read and validate the metadata prefix from the file with pread,
    // so a misaligned blob refuses before any mmap call (BL-4.2).
    let mut reader = std::io::BufReader::new(&file);
    let blob = crate::blob::Blob::parse(&mut reader)
        .map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
    drop(reader);
    // Phase 2: metadata passed; map the file and run the payload checks on it.
    // SAFETY: the map is read-only and outlives no reference returned here
    // (every payload is copied into a Python bytes object below).
    let map = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| PyIOError::new_err(format!("{path}: mmap failed: {e}")))?;
    let bytes: &[u8] = &map;
    blob.validate_payload(bytes)
        .map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;

    let header = PyDict::new(py);
    let h = &blob.header;
    header.set_item("model_id", &h.model_id)?;
    header.set_item("vocab_size", h.vocab_size)?;
    header.set_item("d_model", h.d_model)?;
    header.set_item("n_layers", h.n_layers)?;
    header.set_item("n_heads", h.n_heads)?;
    header.set_item("n_kv_heads", h.n_kv_heads)?;
    header.set_item("head_dim", h.head_dim)?;
    header.set_item("eos_token_id", h.eos_token_id)?;
    header.set_item("bos_token_id", h.bos_token_id)?;
    header.set_item("rms_norm_eps", h.rms_norm_eps)?;
    header.set_item("rope_theta", h.rope_theta)?;
    header.set_item("norm_slots", h.norm_slots.clone())?;
    header.set_item("layers", h.layers.clone())?;
    header.set_item("weight_tying", h.weight_tying.clone())?;
    header.set_item("remainder_dtype", h.remainder_dtype.as_str())?;
    if let Some(created_by) = &h.created_by {
        header.set_item("created_by", created_by)?;
    }

    let sections = PyList::empty(py);
    for section in &blob.sections {
        let entry = PyDict::new(py);
        entry.set_item("name", &section.name)?;
        entry.set_item("tag", section.tag.as_u8())?;
        entry.set_item("offset", section.offset)?;
        entry.set_item("length", section.length)?;
        entry.set_item("data", PyBytes::new(py, blob.slice(bytes, section)))?;
        sections.append(entry)?;
    }

    let out = PyDict::new(py);
    out.set_item("header", header)?;
    out.set_item("sections", sections)?;
    Ok(out)
}

/// Load a blob through the crate's OWN loader and return a summary, or raise
/// ``ValueError`` with the loader's refusal.
///
/// This is the same code path the server takes (BL-1.1 + BL-2.1), so a host
/// test can assert an exported blob passes every loader gate without a second
/// implementation of those gates.
#[pyfunction]
fn load_blob<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let model =
        crate::loader::load_blob(std::path::Path::new(path)).map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("model_id", &model.model_id)?;
    out.set_item(
        "layers",
        model
            .layers
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>(),
    )?;
    out.set_item("d_model", model.d_model)?;
    out.set_item("vocab_size", model.vocab_size)?;
    out.set_item("n_layers", model.n_layers)?;
    out.set_item("remainder_dtype", model.remainder_dtype.as_str())?;
    out.set_item("has_lm_head", model.lm_head.is_some())?;
    out.set_item("has_tokenizer", model.tokenizer_json.is_some())?;
    Ok(out)
}

#[pymodule]
fn quantibbler_qpk1(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_blob, m)?)?;
    m.add_function(wrap_pyfunction!(load_blob, m)?)?;
    Ok(())
}
