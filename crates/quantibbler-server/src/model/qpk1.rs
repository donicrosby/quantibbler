//! qpk1 stream reader (server side of the `qk-representation` contract).
//!
//! Byte layout (qk-representation design D1/D2, pinned by that capability's
//! spec): 24-byte u32 LE header — magic `QPK1`, version 1, dtype tag
//! (1 = float16, 2 = float32), dims `dout`, `din`, rank `r` — then the
//! scale vectors in the tagged dtype, then the packed uint8 `Ub`/`Vb`
//! bitplanes. Bitplanes are row-major with ONE continuous bit stream per
//! matrix (element `(i, j)` at stream bit `i*r + j`, LSB-first within
//! bytes; `+1 -> bit 1`, `-1 -> bit 0`, zero pad). Scale vectors are plain
//! LE floats.
//!
//! Fail-closed reader rules (SF-1.4 / SR-1.2): bad magic, version != 1,
//! unknown dtype tag, non-positive dims, truncation, and trailing bytes
//! are all hard rejections; no partial layer is ever returned. Rank
//! feasibility (`r <= min(dout, din)`) is a reader-side gate applied by the
//! model loader (matching `qk-representation`'s deserialize) — this module
//! exposes the parsed header so the caller can name the file in that
//! diagnostic.

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

/// One deserialized qpk1 layer, in serialized layout.
#[derive(Debug, Clone)]
pub struct Layer {
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
    /// Canonical (continuous-bit) bitplane for Ub, exactly
    /// `(dout*r + 7) / 8` bytes.
    pub ub_packed: Vec<u8>,
    /// Canonical (continuous-bit) bitplane for Vb, exactly
    /// `(din*r + 7) / 8` bytes.
    pub vb_packed: Vec<u8>,
}

const HEADER_LEN: usize = 24;
const MAGIC: [u8; 4] = *b"QPK1";
const VERSION: u32 = 1;
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
/// (qk-representation SF-1.1 formula).
pub fn expected_size(dout: usize, din: usize, r: usize, dtype: ScaleDtype) -> usize {
    HEADER_LEN + (dout + din) * dtype.itemsize() + (dout * r).div_ceil(8) + (din * r).div_ceil(8)
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
    if version != VERSION {
        return Err(format!(
            "unsupported representation version {version}; this reader implements version {VERSION} only"
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
    let expected = expected_size(dout, din, r, dtype);
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
    let ub_len = (dout * r).div_ceil(8);
    let vb_len = (din * r).div_ceil(8);
    let ub_packed = bytes[offset..offset + ub_len].to_vec();
    offset += ub_len;
    let vb_packed = bytes[offset..offset + vb_len].to_vec();
    Ok(Layer {
        dout,
        din,
        r,
        dtype,
        s1,
        s2,
        ub_packed,
        vb_packed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SF-1.3 known layer (dout=3, din=5, r=7, float16): the pinned 48-byte
    /// stream from tests/test_representation.py (qk-representation lane).
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
        assert_eq!(layer.dout, 3);
        assert_eq!(layer.din, 5);
        assert_eq!(layer.r, 7);
        assert_eq!(layer.dtype, ScaleDtype::Float16);
        assert_eq!(layer.s1, vec![1.0, 0.5, -2.0]);
        assert_eq!(layer.s2, vec![0.25, -0.5, 2.0, 1.0, 0.125]);
        assert_eq!(layer.ub_packed, vec![0x2b, 0x73, 0x07]);
        assert_eq!(layer.vb_packed, vec![0x55, 0xe6, 0x41, 0x35, 0x03]);
        // Reader-side rank feasibility (qk-representation deserialize rule)
        // is applied by the model loader; assert the raw header here so the
        // loader's rejection is provably about the rank gate.
        assert!(layer.r > layer.dout.min(layer.din));
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
