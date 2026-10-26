//! qpk1 container: stream reader + model-directory loader.
//!
//! This crate owns the serialized qpk1 contract end to end: the per-layer
//! stream format below, and the model-directory acquisition convention
//! (manifest/config/sidecar, qk-server design D3) in [`loader`]. Both the
//! Python face (src/quantibbler/qpk1.py) and the server consume this crate;
//! there is exactly one reader implementation per language, and the server
//! no longer carries a private copy.
//!
//! Byte layout (qk-representation design D1/D2, extended by the bias-carry
//! decisions D10/D11): 24-byte u32 LE header — magic `QPK1`, version, dtype
//! tag (1 = float16, 2 = float32), dims `dout`, `din`, rank `r` — then the
//! scale vectors in the tagged dtype, then (version 2 onward) the `(dout,)`
//! bias vector in that same dtype, then the packed uint8 `Ub`/`Vb`
//! bitplanes. Bitplanes are row-major with ONE continuous bit stream per
//! matrix (element `(i, j)` at stream bit `i*r + j`, LSB-first within
//! bytes; `+1 -> bit 1`, `-1 -> bit 0`, zero pad). Scale and bias vectors
//! are plain LE floats.
//!
//! Version semantics: version 2 is what the compressor writes and carries a
//! bias vector; version 1 is the pre-bias artifact. A version-1 stream is
//! read with an EMPTY `Layer::bias` — the declared legacy rule (a biasless
//! linear's map is the zero vector), never a silent default for a missing
//! field on a current stream.
//!
//! Fail-closed reader rules (SF-1.4 / SR-1.2): bad magic, a version outside
//! {1, 2}, unknown dtype tag, non-positive dims, truncation, and trailing
//! bytes are all hard rejections; no partial layer is ever returned. Rank
//! feasibility (`r <= min(dout, din)`) is a reader-side gate applied by the
//! model loader (matching `qk-representation`'s deserialize) — this module
//! exposes the parsed header so the caller can name the file in that
//! diagnostic.

pub mod loader;

pub use loader::{ModelDirectory, NormSlot, inventory_block_floats};

/// Parsed scale dtype tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDtype {
    /// dtype tag 1: scales are IEEE binary16 LE.
    Float16,
    /// dtype tag 2: scales are IEEE binary32 LE.
    Float32,
}

impl ScaleDtype {
    /// Itemsize of the scale vectors in bytes.
    pub fn itemsize(self) -> usize {
        match self {
            ScaleDtype::Float16 => 2,
            ScaleDtype::Float32 => 4,
        }
    }
}

/// Declared storage dtype of the model-directory remainder tensors — the FP
/// tensors the per-layer qpk1 streams do not carry: the embedding, an untied
/// `lm_head`, and the norm weights.
///
/// One dtype covers the whole remainder set; the manifest's `[sidecar]` block
/// declares it (optional `dtype` key, absent = fp32, the meaning every
/// pre-existing artifact already has), and `weight_tying`'s own suffix must
/// agree with it. The declaration ALONE selects the itemsize — a byte length
/// is never used to infer a dtype. bf16 carries 8 mantissa bits, so a bf16
/// remainder is lossy by declaration and is never compared against its fp32
/// original as if the two were the same dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemainderDtype {
    /// IEEE binary32 LE: 4 bytes per element.
    Float32,
    /// bfloat16 LE: 2 bytes per element, the high half of an f32 word.
    Bf16,
}

impl RemainderDtype {
    /// Itemsize of one remainder element in bytes.
    pub fn itemsize(self) -> usize {
        match self {
            RemainderDtype::Float32 => 4,
            RemainderDtype::Bf16 => 2,
        }
    }

    /// The spelling a declaration carries (`fp32` / `bf16`).
    pub fn as_str(self) -> &'static str {
        match self {
            RemainderDtype::Float32 => "fp32",
            RemainderDtype::Bf16 => "bf16",
        }
    }

    /// Resolve a declaration token; `None` for anything undeclared.
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "fp32" => Some(RemainderDtype::Float32),
            "bf16" => Some(RemainderDtype::Bf16),
            _ => None,
        }
    }

    /// Widen one little-endian word to f64. bf16 -> f32 is exact (the value is
    /// the high half of the f32 word, low bits zero) and f32 -> f64 is exact,
    /// so no widening step invents a bit.
    pub fn widen_le(self, bytes: &[u8]) -> f64 {
        match self {
            RemainderDtype::Float32 => {
                f64::from(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            RemainderDtype::Bf16 => {
                let word = u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16;
                f64::from(f32::from_bits(word))
            }
        }
    }
}

/// One deserialized qpk1 layer, in serialized layout.
#[derive(Debug, Clone)]
pub struct Layer {
    /// The stream's representation version (2 carries a bias vector;
    /// 1 is the pre-bias legacy reading).
    pub version: u32,
    pub dout: usize,
    pub din: usize,
    pub r: usize,
    /// The stream's scale dtype (informational post-parse in v0.1: the
    /// scales below are already widened).
    #[allow(dead_code)]
    pub dtype: ScaleDtype,
    /// float64 scales (float16 values widened exactly; f16 -> f64 is exact).
    pub s1: Vec<f64>,
    pub s2: Vec<f64>,
    /// float64 bias, `dout` entries. EMPTY on a legacy version-1 stream:
    /// consumers read an empty vector as "no bias term" (the declared
    /// legacy rule), never as a missing field on a current stream.
    pub bias: Vec<f64>,
    /// Canonical (continuous-bit) bitplane for Ub, exactly
    /// `(dout*r + 7) / 8` bytes.
    pub ub_packed: Vec<u8>,
    /// Canonical (continuous-bit) bitplane for Vb, exactly
    /// `(din*r + 7) / 8` bytes.
    pub vb_packed: Vec<u8>,
}

const HEADER_LEN: usize = 24;
const MAGIC: [u8; 4] = *b"QPK1";
/// Version the compressor writes: scales + a `(dout,)` bias vector.
pub const VERSION: u32 = 2;
/// Pre-bias artifacts. Read-only legacy: parsed with an empty bias vector.
pub const LEGACY_VERSION: u32 = 1;
const TAG_FLOAT16: u32 = 1;
const TAG_FLOAT32: u32 = 2;

fn u32le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Exact stream size for `(dout, din, r)` at the tagged scale itemsize
/// (qk-representation SF-1.1 formula, version 2). `with_bias` adds the
/// `(dout,)` bias vector's floats; pass `false` for a version-1 stream.
pub fn expected_size(
    dout: usize,
    din: usize,
    r: usize,
    dtype: ScaleDtype,
    with_bias: bool,
) -> usize {
    let floats = if with_bias {
        2 * dout + din
    } else {
        dout + din
    };
    HEADER_LEN + floats * dtype.itemsize() + (dout * r).div_ceil(8) + (din * r).div_ceil(8)
}

/// IEEE binary16 -> float64, exact (std-only bit twiddling).
fn f16_to_f64(bits: u16) -> f64 {
    let sign = f64::from((bits >> 15) & 1) * -2.0 + 1.0;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = f64::from(bits & 0x03ff);
    match exponent {
        0 => sign * fraction * 2.0_f64.powi(-24), // subnormal f16
        0x1f => {
            if fraction == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => sign * (1.0 + fraction / 1024.0) * 2.0_f64.powi(e - 15),
    }
}

/// IEEE binary16 LE decode of `bytes`.
fn f16_vec(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|p| f16_to_f64(u16::from_le_bytes([p[0], p[1]])))
        .collect()
}

/// IEEE binary32 LE decode of `bytes`.
fn f32_vec(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(4)
        .map(|p| f64::from(f32::from_le_bytes([p[0], p[1], p[2], p[3]])))
        .collect()
}

/// Parse exactly one qpk1 layer from `bytes`, or reject with a
/// human-readable reason (fail-closed; no partial layer).
pub fn parse(bytes: &[u8]) -> Result<Layer, String> {
    if bytes.len() < HEADER_LEN {
        return Err(format!(
            "stream is {} bytes; shorter than the {HEADER_LEN}-byte header",
            bytes.len()
        ));
    }
    if bytes[0..4] != MAGIC {
        return Err(format!("bad magic {:?}; expected {MAGIC:?}", &bytes[0..4]));
    }
    let version = u32le(bytes, 4);
    if version != LEGACY_VERSION && version != VERSION {
        return Err(format!(
            "unsupported representation version {version}; this reader implements version \
             {LEGACY_VERSION} (pre-bias legacy) and version {VERSION} (current, with bias)"
        ));
    }
    let tag = u32le(bytes, 8);
    let dtype = match tag {
        TAG_FLOAT16 => ScaleDtype::Float16,
        TAG_FLOAT32 => ScaleDtype::Float32,
        other => {
            return Err(format!(
                "unknown dtype tag {other}; expected 1 (float16) or 2 (float32)"
            ));
        }
    };
    let dout = u32le(bytes, 12) as usize;
    let din = u32le(bytes, 16) as usize;
    let r = u32le(bytes, 20) as usize;
    if dout == 0 || din == 0 || r == 0 {
        return Err(format!(
            "dout, din and r must all be >= 1, got ({dout}, {din}, {r})"
        ));
    }
    let has_bias = version >= VERSION;
    let expected = expected_size(dout, din, r, dtype, has_bias);
    if bytes.len() < expected {
        return Err(format!(
            "truncated stream: {} bytes; header declares {expected}",
            bytes.len()
        ));
    }
    if bytes.len() > expected {
        return Err(format!(
            "trailing garbage: {} bytes; one layer at this header is {expected}",
            bytes.len()
        ));
    }

    let mut offset = HEADER_LEN;
    let s1_len = dout * dtype.itemsize();
    let s2_len = din * dtype.itemsize();
    let s1 = match dtype {
        ScaleDtype::Float16 => f16_vec(&bytes[offset..offset + s1_len]),
        ScaleDtype::Float32 => f32_vec(&bytes[offset..offset + s1_len]),
    };
    offset += s1_len;
    let s2 = match dtype {
        ScaleDtype::Float16 => f16_vec(&bytes[offset..offset + s2_len]),
        ScaleDtype::Float32 => f32_vec(&bytes[offset..offset + s2_len]),
    };
    offset += s2_len;
    let bias = if has_bias {
        let bias_len = dout * dtype.itemsize();
        let decoded = match dtype {
            ScaleDtype::Float16 => f16_vec(&bytes[offset..offset + bias_len]),
            ScaleDtype::Float32 => f32_vec(&bytes[offset..offset + bias_len]),
        };
        offset += bias_len;
        decoded
    } else {
        Vec::new()
    };
    let ub_len = (dout * r).div_ceil(8);
    let vb_len = (din * r).div_ceil(8);
    let ub_packed = bytes[offset..offset + ub_len].to_vec();
    offset += ub_len;
    let vb_packed = bytes[offset..offset + vb_len].to_vec();
    Ok(Layer {
        version,
        dout,
        din,
        r,
        dtype,
        s1,
        s2,
        bias,
        ub_packed,
        vb_packed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SF-1.3 known layer (dout=3, din=5, r=7, float16): the pinned 48-byte
    /// VERSION-1 stream from tests/test_representation.py (qk-representation
    /// lane). A version-1 stream is the pre-bias legacy artifact: it parses
    /// to an EMPTY bias vector, i.e. the zero map a biasless linear applies.
    /// The writer-only fixture is rank-infeasible (7 > min(3, 5)) so the
    /// full model loader rejects it; the byte-level parse must still match
    /// it field-for-field (IA5 / e2e anchor, task 6.1).
    #[test]
    fn parses_sf_1_3_known_layer_bytes() {
        const EXPECTED: &str = "51504b31\
             01000000\
             01000000\
             03000000\
             05000000\
             07000000\
             003c003800c0\
             003400b80040003c0030\
             2b7307\
             55e6413503";
        let bytes = hex_bytes(EXPECTED);
        let layer = parse(&bytes).expect("byte-legal stream parses");
        assert_eq!(layer.version, LEGACY_VERSION);
        assert_eq!(layer.dout, 3);
        assert_eq!(layer.din, 5);
        assert_eq!(layer.r, 7);
        assert_eq!(layer.dtype, ScaleDtype::Float16);
        assert_eq!(layer.s1, vec![1.0, 0.5, -2.0]);
        assert_eq!(layer.s2, vec![0.25, -0.5, 2.0, 1.0, 0.125]);
        // A pre-bias stream carries no bias, and an empty vector IS the zero
        // map: the declared legacy reading, not a missing-field default.
        assert!(layer.bias.is_empty());
        assert_eq!(layer.ub_packed, vec![0x2b, 0x73, 0x07]);
        assert_eq!(layer.vb_packed, vec![0x55, 0xe6, 0x41, 0x35, 0x03]);
        // Reader-side rank feasibility (qk-representation deserialize rule)
        // is applied by the model loader; assert the raw header here so the
        // loader's rejection is provably about the rank gate.
        assert!(layer.r > layer.dout.min(layer.din));
    }

    /// The version-2 pinned vector, byte-identical to what
    /// `qpk1.serialize(...)` produces in tests/test_representation.py for the
    /// same inputs — one bias-carrying vector, both faces, same bytes.
    /// dout=3, din=5, r=3, float16, bias `[1.0, -0.5, 2.0]`. 50 bytes:
    /// 24 + (2*3 + 5)*2 + ceil(9/8) + ceil(15/8).
    #[test]
    fn parses_the_v2_bias_carrying_vector_from_the_python_face() {
        const EXPECTED: &str = "51504b31\
             02000000\
             01000000\
             03000000\
             05000000\
             03000000\
             003c003800c0\
             003400b80040003c0030\
             003c00b80040\
             7301\
             e535";
        let bytes = hex_bytes(EXPECTED);
        assert_eq!(bytes.len(), 50);
        assert_eq!(
            expected_size(3, 5, 3, ScaleDtype::Float16, true),
            bytes.len()
        );
        let layer = parse(&bytes).expect("v2 stream parses");
        assert_eq!(layer.version, VERSION);
        assert_eq!(layer.dout, 3);
        assert_eq!(layer.din, 5);
        assert_eq!(layer.r, 3);
        assert_eq!(layer.dtype, ScaleDtype::Float16);
        assert_eq!(layer.s1, vec![1.0, 0.5, -2.0]);
        assert_eq!(layer.s2, vec![0.25, -0.5, 2.0, 1.0, 0.125]);
        // The bias vector sits between the scales and the bitplanes.
        assert_eq!(layer.bias, vec![1.0, -0.5, 2.0]);
        assert_eq!(layer.ub_packed, vec![0x73, 0x01]);
        assert_eq!(layer.vb_packed, vec![0xe5, 0x35]);
        // The same header at version 1 is one bias vector shorter.
        let mut legacy = bytes.clone();
        legacy.drain(40..46);
        legacy[4..8].copy_from_slice(&LEGACY_VERSION.to_le_bytes());
        let legacy = parse(&legacy).expect("v1 sibling of the same layer parses");
        assert_eq!(legacy.version, LEGACY_VERSION);
        assert!(legacy.bias.is_empty());
        assert_eq!(legacy.ub_packed, layer.ub_packed);
        assert_eq!(legacy.vb_packed, layer.vb_packed);
    }

    /// A version-2 header whose body is short by the bias vector's bytes is
    /// a truncation; one byte past the extent is trailing garbage. Neither
    /// may be read as a silent zero bias.
    #[test]
    fn rejects_a_v2_stream_missing_its_bias_vector() {
        const V2: &str = "51504b31\
             02000000\
             01000000\
             03000000\
             05000000\
             03000000\
             003c003800c0\
             003400b80040003c0030\
             003c00b80040\
             7301\
             e535";
        let bytes = hex_bytes(V2);
        assert!(parse(&bytes[..44]).is_err(), "v2 truncation must reject");
        let mut long = bytes.clone();
        long.push(0);
        assert!(parse(&long).is_err(), "v2 trailing byte must reject");
    }

    /// Decode a hex string to bytes (test helper).
    #[allow(dead_code)]
    fn hex_bytes(hex: &str) -> Vec<u8> {
        let clean: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
        (0..clean.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).expect("valid hex"))
            .collect()
    }

    #[test]
    fn rejects_every_sr_1_2_class() {
        // Truncated header.
        assert!(parse(&MAGIC).is_err());
        // Bad magic.
        let mut b = vec![0u8; 44];
        b[0..4].copy_from_slice(b"XPK1");
        assert!(parse(&b).is_err());
        // Wrong version.
        let mut b = vec![0u8; 44];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..8].copy_from_slice(&7u32.to_le_bytes());
        assert!(parse(&b).is_err());
        // Unknown dtype tag.
        let mut b = vec![0u8; 44];
        b[0..4].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&9u32.to_le_bytes());
        assert!(parse(&b).is_err());
        // r = 0.
        let mut b = vec![0u8; 44];
        b[0..4].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&1u32.to_le_bytes());
        b[12..16].copy_from_slice(&4u32.to_le_bytes());
        b[16..20].copy_from_slice(&4u32.to_le_bytes());
        b[20..24].copy_from_slice(&0u32.to_le_bytes());
        assert!(parse(&b).is_err());
        // Truncated body (valid header, short stream).
        let mut b = vec![0u8; 43];
        b[0..4].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&1u32.to_le_bytes());
        b[12..16].copy_from_slice(&4u32.to_le_bytes());
        b[16..20].copy_from_slice(&4u32.to_le_bytes());
        b[20..24].copy_from_slice(&4u32.to_le_bytes());
        assert!(parse(&b).is_err());
        // Trailing bytes.
        let mut b = vec![0u8; 44];
        b[0..4].copy_from_slice(&MAGIC);
        b[8..12].copy_from_slice(&1u32.to_le_bytes());
        b[12..16].copy_from_slice(&4u32.to_le_bytes());
        b[16..20].copy_from_slice(&4u32.to_le_bytes());
        b[20..24].copy_from_slice(&4u32.to_le_bytes());
        b.push(0);
        assert!(parse(&b).is_err());
    }

    #[test]
    fn f16_decode_matches_known_values() {
        assert_eq!(f16_to_f64(0x3c00), 1.0);
        assert_eq!(f16_to_f64(0x3800), 0.5);
        assert_eq!(f16_to_f64(0xc000), -2.0);
        assert_eq!(f16_to_f64(0x3400), 0.25);
        assert_eq!(f16_to_f64(0xb800), -0.5);
        assert_eq!(f16_to_f64(0x4200), 3.0);
        assert_eq!(f16_to_f64(0xbc00), -1.0);
        assert_eq!(f16_to_f64(0x0000), 0.0);
        assert!(f16_to_f64(0x7c01).is_nan());
        assert_eq!(f16_to_f64(0x7c00), f64::INFINITY);
    }
}
