//! Dtype-tagged dispatch: the boundary form of "which instantiation".
//!
//! The kernels live in [`crate::core`], generic over
//! [`crate::dtype::KernelScalar`]. This module is where a dtype that arrives as
//! DATA — the PyO3 boundary, and any rlib caller whose dtype is not a static
//! choice — is turned into exactly one instantiation.
//!
//! It is deliberately NOT in `core`: an algorithm body that names a concrete
//! float type is the thing the dtype-genericity requirement forbids, and a
//! `match` over [`DenseOperand`] is the enumeration itself rather than a dtype
//! assumption inside a kernel. Keeping the two apart also keeps
//! `core::packed_matvec` / `core::packed_matmul` the single implementation of
//! each product, with this module delegating rather than re-deriving.

use crate::core::KernelError;
use crate::core::{packed_matmul, packed_matvec};
use crate::dtype::Dtype;
use half::bf16;

/// A dense operand carried WITH its dtype.
#[derive(Debug, Clone, Copy)]
pub enum DenseOperand<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    Bf16(&'a [bf16]),
}

impl DenseOperand<'_> {
    pub fn dtype(&self) -> Dtype {
        match self {
            DenseOperand::F64(_) => Dtype::F64,
            DenseOperand::F32(_) => Dtype::F32,
            DenseOperand::Bf16(_) => Dtype::Bf16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DenseOperand::F64(v) => v.len(),
            DenseOperand::F32(v) => v.len(),
            DenseOperand::Bf16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A product result carrying its dtype; exactly one variant is populated.
#[derive(Debug, Clone, PartialEq)]
pub enum DenseResult {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Bf16(Vec<bf16>),
}

impl DenseResult {
    pub fn dtype(&self) -> Dtype {
        match self {
            DenseResult::F64(_) => Dtype::F64,
            DenseResult::F32(_) => Dtype::F32,
            DenseResult::Bf16(_) => Dtype::Bf16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            DenseResult::F64(v) => v.len(),
            DenseResult::F32(v) => v.len(),
            DenseResult::Bf16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// [`crate::core::packed_matvec`] with the dtype carried as data.
pub fn packed_matvec_dyn(
    packed: &[u8],
    n_rows: usize,
    n_bits: usize,
    operand: DenseOperand<'_>,
) -> Result<DenseResult, KernelError> {
    Ok(match operand {
        DenseOperand::F64(v) => DenseResult::F64(packed_matvec(packed, n_rows, n_bits, v)?),
        DenseOperand::F32(v) => DenseResult::F32(packed_matvec(packed, n_rows, n_bits, v)?),
        DenseOperand::Bf16(v) => DenseResult::Bf16(packed_matvec(packed, n_rows, n_bits, v)?),
    })
}

/// [`crate::core::packed_matmul`] with the dtype carried as data.
pub fn packed_matmul_dyn(
    packed: &[u8],
    n_rows: usize,
    n_bits: usize,
    operand: DenseOperand<'_>,
    n_cols: usize,
) -> Result<DenseResult, KernelError> {
    Ok(match operand {
        DenseOperand::F64(v) => DenseResult::F64(packed_matmul(packed, n_rows, n_bits, v, n_cols)?),
        DenseOperand::F32(v) => DenseResult::F32(packed_matmul(packed, n_rows, n_bits, v, n_cols)?),
        DenseOperand::Bf16(v) => {
            DenseResult::Bf16(packed_matmul(packed, n_rows, n_bits, v, n_cols)?)
        }
    })
}
