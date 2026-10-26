//! Scalar dtype abstraction for the kernel surface.
//!
//! The FP-remainder products are written ONCE, against [`KernelScalar`], so a
//! second precision is a new impl block rather than a second algorithm body.
//! Monomorphization is the right shape here: two consumers, three types, and
//! an `f64` instantiation whose bytes must stay the v0.1 bytes.
//!
//! The accumulator is a DECLARED associated type, not the operand type. A
//! `bf16` row product rounded to `bf16` at every term grows its error with the
//! row width; accumulating `bf16` in `f32` keeps the error at one output
//! rounding, which is what makes the parity tolerance an output-precision
//! bound instead of a width-dependent one. The `f64` instantiation's
//! accumulator is `f64`, so its arithmetic is unchanged term for term.

use crate::core::KernelError;
use half::bf16;

/// One float dtype the kernels can be instantiated at.
///
/// `Acc` is the type that dtype's packed products sum in; `to_acc`/`from_acc`
/// are the only conversion points, so no kernel body branches on the dtype.
pub trait KernelScalar:
    Copy
    + Send
    + Sync
    + PartialEq
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
{
    /// The accumulator this dtype's products sum in.
    type Acc: KernelScalar;

    /// The dtype's name on the wire (the `Dtype` tag's spelling).
    const NAME: &'static str;

    const ZERO: Self;
    const ONE: Self;
    const NEG_ONE: Self;

    fn to_acc(self) -> Self::Acc;
    fn from_acc(acc: Self::Acc) -> Self;
}

impl KernelScalar for f64 {
    type Acc = f64;
    const NAME: &'static str = "float64";
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_ONE: Self = -1.0;

    fn to_acc(self) -> f64 {
        self
    }

    fn from_acc(acc: f64) -> f64 {
        acc
    }
}

impl KernelScalar for f32 {
    type Acc = f32;
    const NAME: &'static str = "float32";
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_ONE: Self = -1.0;

    fn to_acc(self) -> f32 {
        self
    }

    fn from_acc(acc: f32) -> f32 {
        acc
    }
}

impl KernelScalar for bf16 {
    // bf16 STORAGE, f32 accumulation (see the module note).
    type Acc = f32;
    const NAME: &'static str = "bfloat16";
    const ZERO: Self = bf16::ZERO;
    const ONE: Self = bf16::ONE;
    const NEG_ONE: Self = bf16::NEG_ONE;

    fn to_acc(self) -> f32 {
        self.to_f32()
    }

    fn from_acc(acc: f32) -> bf16 {
        bf16::from_f32(acc)
    }
}

/// The dtype tag: the boundary form of "which instantiation".
///
/// Resolution is fail-closed and names the offending value, so an unknown tag
/// can never silently pick a different numerical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    Bf16,
}

impl Dtype {
    /// Every accepted tag, in the order the refusal message lists them.
    pub const NAMES: [&'static str; 3] = ["float64", "float32", "bfloat16"];

    /// Resolve a tag name; an unknown name is an error naming the value.
    pub fn from_name(name: &str) -> Result<Dtype, KernelError> {
        match name {
            "float64" => Ok(Dtype::F64),
            "float32" => Ok(Dtype::F32),
            "bfloat16" => Ok(Dtype::Bf16),
            other => crate::core::err(format!(
                "unknown kernel dtype {other:?}; expected one of \"float64\", \"float32\", \"bfloat16\""
            )),
        }
    }

    /// The tag's wire name.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "float64",
            Dtype::F32 => "float32",
            Dtype::Bf16 => "bfloat16",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scenario: dtype_tag_resolution — every accepted name resolves, and an
    /// unknown name is refused BY NAME, not folded onto a default.
    #[test]
    fn dtype_tag_resolution() {
        assert_eq!(Dtype::from_name("float64").unwrap(), Dtype::F64);
        assert_eq!(Dtype::from_name("float32").unwrap(), Dtype::F32);
        assert_eq!(Dtype::from_name("bfloat16").unwrap(), Dtype::Bf16);
        // round-trip: the tag's own name resolves back to it
        for d in [Dtype::F64, Dtype::F32, Dtype::Bf16] {
            assert_eq!(Dtype::from_name(d.name()).unwrap(), d);
        }
        for bad in ["float16", "f32", "bfloat32", ""] {
            let e = Dtype::from_name(bad).expect_err("unknown tag must be refused");
            assert!(
                e.0.contains(&format!("{bad:?}")) && e.0.contains("bfloat16"),
                "refusal must name the value and the accepted set: {}",
                e.0
            );
        }
        assert_eq!(Dtype::NAMES.len(), 3);
    }

    /// Scenario: scalar_trait_covers_three_dtypes — exactly three impls, each
    /// declaring its own accumulator, and the `f64` pair is the identity so
    /// the v0.1 arithmetic is untouched.
    #[test]
    fn scalar_trait_covers_three_dtypes() {
        assert_eq!(<f64 as KernelScalar>::NAME, "float64");
        assert_eq!(<f32 as KernelScalar>::NAME, "float32");
        assert_eq!(<bf16 as KernelScalar>::NAME, "bfloat16");

        assert_eq!(<f64 as KernelScalar>::ONE.to_acc(), 1.0f64);
        assert_eq!(<f32 as KernelScalar>::ONE.to_acc(), 1.0f32);
        assert_eq!(<bf16 as KernelScalar>::ONE.to_acc(), 1.0f32);
        assert_eq!(<bf16 as KernelScalar>::NEG_ONE.to_acc(), -1.0f32);
        assert_eq!(<bf16 as KernelScalar>::ZERO.to_acc(), 0.0f32);

        // f64 conversions are the identity in both directions.
        assert_eq!(<f64 as KernelScalar>::from_acc(2.5), 2.5);
        assert_eq!(<f64 as KernelScalar>::to_acc(2.5), 2.5);
        // a bf16 output is the f32 accumulator rounded once, and a bf16
        // operand reaches the accumulator exactly.
        assert_eq!(<bf16 as KernelScalar>::from_acc(1.0f32), bf16::ONE);
        assert_eq!(<bf16 as KernelScalar>::to_acc(bf16::ONE), 1.0f32);
        let rounded = <bf16 as KernelScalar>::from_acc(1.0f32 + 2f32.powi(-12));
        assert_eq!(rounded.to_f32(), 1.0f32, "half an 8-bit ULP rounds away");
    }
}
