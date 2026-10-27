//! qpk1 single-file blob: `QKPB` magic, JSON header, binary section table,
//! page-aligned payloads (change `qk-artifact-blob`, design B1-B9; spec
//! `qk-representation` BF-1..BF-7).
//!
//! Byte layout, all integers little-endian:
//!
//! ```text
//! 0   magic  b"QKPB"
//! 4   u32    format version (1)
//! 8   u32    header byte length
//! 12  ...    UTF-8 JSON header (header byte length bytes)
//!     ...    zero padding to the next 4096-byte boundary
//!     u32    section count
//!     ...    count entries: u8 name_len | name | u8 dtype tag | u64 offset | u64 length
//!     ...    zero padding to the next 4096-byte boundary
//!     ...    payloads, each at a 4096-aligned offset, in canonical order
//! ```
//!
//! The table's start is DERIVED (`next_page(12 + header_len)`), never stored,
//! so a header length cannot describe two different tables. Every section
//! range must sit entirely after the table's last byte, the greatest section
//! end must equal the file size (BF-7.2), and every byte of the metadata and
//! inter-payload regions must be zero (BF-4.2).
//!
//! The blob is a wrapping CONTAINER: its magic is distinct from the per-layer
//! stream's `QPK1` (B1), so neither parser can silently read the other's bytes.

use std::collections::BTreeMap;
use std::io::Read;

use crate::RemainderDtype;

/// Blob magic (design B1) — deliberately distinct from the stream's `QPK1`.
pub const BLOB_MAGIC: [u8; 4] = *b"QKPB";
/// The only supported blob format version (BF-1.3).
pub const BLOB_VERSION: u32 = 1;
/// Section payload alignment: the host page size (B4). A page-isolated
/// section is what makes a page-granular mmap view contain only that section.
pub const PAGE_SIZE: u64 = 4096;
/// Cap on the header JSON (bytes). Real headers are a few KiB; the cap keeps a
/// crafted length from driving a multi-GB allocation at parse time.
const MAX_HEADER_BYTES: u32 = 16 << 20;

/// One section entry's dtype tag (design B3, the closed vocabulary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobTag {
    /// 0: a version-1 (pre-bias) qpk1 layer stream.
    Qpk1V1,
    /// 1: a version-2 (current, bias-carrying) qpk1 layer stream.
    Qpk1V2,
    /// 2: IEEE binary32 little-endian remainder bytes.
    Fp32Le,
    /// 3: IEEE bfloat16 little-endian remainder bytes.
    Bf16Le,
    /// 4: opaque bytes (tokenizer files).
    Raw,
}

impl BlobTag {
    /// Decode the tag byte, or refuse naming it (BF-3.1).
    pub fn from_u8(tag: u8) -> Result<Self, String> {
        match tag {
            0 => Ok(Self::Qpk1V1),
            1 => Ok(Self::Qpk1V2),
            2 => Ok(Self::Fp32Le),
            3 => Ok(Self::Bf16Le),
            4 => Ok(Self::Raw),
            other => Err(format!(
                "unknown section dtype tag {other}; the vocabulary is 0 qpk1_v1, 1 qpk1_v2, \
                 2 fp32_le, 3 bf16_le, 4 raw"
            )),
        }
    }

    /// The tag byte this variant writes.
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Qpk1V1 => 0,
            Self::Qpk1V2 => 1,
            Self::Fp32Le => 2,
            Self::Bf16Le => 3,
            Self::Raw => 4,
        }
    }

    /// The stream version a layer tag declares (`None` for a non-stream tag).
    pub fn stream_version(self) -> Option<u32> {
        match self {
            Self::Qpk1V1 => Some(crate::LEGACY_VERSION),
            Self::Qpk1V2 => Some(crate::VERSION),
            _ => None,
        }
    }

    /// The itemsize a remainder tag declares (`None` for a non-float tag).
    pub fn remainder_dtype(self) -> Option<RemainderDtype> {
        match self {
            Self::Fp32Le => Some(RemainderDtype::Float32),
            Self::Bf16Le => Some(RemainderDtype::Bf16),
            _ => None,
        }
    }

    /// The declared spelling (`fp32` / `bf16` / the stream and raw labels).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qpk1V1 => "qpk1_v1",
            Self::Qpk1V2 => "qpk1_v2",
            Self::Fp32Le => "fp32_le",
            Self::Bf16Le => "bf16_le",
            Self::Raw => "raw",
        }
    }
}

/// The blob's declared metadata (design B2). Every field below is REQUIRED in
/// the JSON header; `weight_tying` may be JSON `null` (untied head, no suffix)
/// but the key itself must be present.
#[derive(Debug, Clone, PartialEq)]
pub struct BlobHeader {
    pub model_id: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub norm_slots: Vec<String>,
    pub layers: Vec<String>,
    /// The `weight_tying` value exactly as declared: `Some(string)` for the
    /// documented grammar, `None` for a JSON null (untied, no suffix).
    pub weight_tying: Option<String>,
    pub remainder_dtype: RemainderDtype,
    /// Optional provenance string (never a timestamp, BF-2.2).
    pub created_by: Option<String>,
}

impl BlobHeader {
    /// True when the declaration names an UNTIED head: a JSON null, or a path
    /// component other than the tied sentinel `e1` (B2/B8). The tied sentinel
    /// means the head IS the embedding; every other declaration expects its
    /// own `"lm_head"` section.
    pub fn head_is_untied(&self) -> bool {
        match self.weight_tying.as_deref() {
            None => true,
            Some(wt) => match tied_path(wt) {
                Some(path) => path != "e1",
                None => true,
            },
        }
    }

    /// The dtype `weight_tying`'s own suffix declares, or `None` when the
    /// field is a JSON null. An unrecognized grammar is refused by the loader.
    pub fn tying_dtype(&self) -> Option<Result<RemainderDtype, String>> {
        let wt = self.weight_tying.as_deref()?;
        let (path, token) = match wt.strip_prefix("lm_head=").and_then(|r| r.rsplit_once('@')) {
            Some(pair) => pair,
            None => {
                return Some(Err(format!(
                    "unknown weight_tying {wt:?}; documented forms: null or \
                     \"lm_head=<path>@<dtype>\" with dtype fp32 or bf16"
                )));
            }
        };
        if path.is_empty() {
            return Some(Err(format!(
                "weight_tying {wt:?} names an empty path; the form is \
                 \"lm_head=<path>@<dtype>\""
            )));
        }
        Some(RemainderDtype::from_token(token).ok_or_else(|| {
            format!(
                "weight_tying {wt:?} declares an undeclared dtype token {token:?}; \
                 declared dtypes are fp32 and bf16"
            )
        }))
    }
}

/// The path component of the tied sentinel grammar, when the declaration has
/// the documented shape.
fn tied_path(weight_tying: &str) -> Option<&str> {
    weight_tying
        .strip_prefix("lm_head=")
        .and_then(|rest| rest.rsplit_once('@'))
        .map(|(path, _)| path)
}

/// One parsed section-table entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobSection {
    pub name: String,
    pub tag: BlobTag,
    pub offset: u64,
    pub length: u64,
}

impl BlobSection {
    /// The entry's half-open byte range.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset as usize..(self.offset + self.length) as usize
    }
}

/// A fully parsed and validated blob: header plus the section table.
#[derive(Debug, Clone, PartialEq)]
pub struct Blob {
    pub header: BlobHeader,
    /// Offset of the header JSON's first byte (always 12).
    pub header_start: usize,
    /// Offset one past the header JSON's last byte.
    pub header_end: usize,
    /// Offset of the section table's first byte (`next_page(header_end)`).
    pub table_start: usize,
    /// Offset one past the table's last entry byte.
    pub table_end: usize,
    /// Entries, in the file's canonical order.
    pub sections: Vec<BlobSection>,
}

impl Blob {
    /// Parse and validate a blob's METADATA from a reader. This reads only the
    /// prefix — magic, version, header JSON, header padding, and the section
    /// table — and never the payload. The alignment assertion (BL-4.2) runs
    /// here, so a caller can refuse a misaligned blob before mapping the file.
    /// Payload checks run later, on the map, via [`Blob::validate_payload`].
    /// Fail-closed: a blob that trips any metadata rule yields `Err` and no
    /// partial `Blob`.
    pub fn parse(reader: &mut (impl Read + ?Sized)) -> Result<Blob, String> {
        // --- BF-1: magic + version ---
        let mut magic = [0u8; 4];
        if let Err(e) = reader.read_exact(&mut magic) {
            return Err(format!(
                "bad blob magic: the file is shorter than 4 bytes ({e})"
            ));
        }
        if magic != BLOB_MAGIC {
            return Err(format!(
                "bad blob magic {magic:?}; expected QKPB {BLOB_MAGIC:?}"
            ));
        }
        let mut version_bytes = [0u8; 4];
        read_prefix(reader, &mut version_bytes, "blob version")?;
        let version = u32::from_le_bytes(version_bytes);
        if version != BLOB_VERSION {
            return Err(format!(
                "unsupported blob format version {version}; this reader implements {BLOB_VERSION}"
            ));
        }
        let mut header_len_bytes = [0u8; 4];
        read_prefix(reader, &mut header_len_bytes, "blob header length")?;
        let header_len = u32::from_le_bytes(header_len_bytes);
        if header_len > MAX_HEADER_BYTES {
            return Err(format!(
                "blob header declares {header_len} bytes, above the {MAX_HEADER_BYTES}-byte cap"
            ));
        }
        let header_start = 12usize;
        let header_end = header_start + header_len as usize;
        let mut header_raw = vec![0u8; header_len as usize];
        read_prefix(reader, &mut header_raw, "blob header")?;
        let header_json = std::str::from_utf8(&header_raw)
            .map_err(|e| format!("blob header is not valid UTF-8: {e}"))?;
        let value: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| format!("blob header is not valid JSON: {e}"))?;
        let header = parse_header(&value)?;

        // --- section table ---
        let table_start = next_page(header_end as u64) as usize;
        let padding_len = table_start - header_end;
        let mut padding = vec![0u8; padding_len];
        read_prefix(reader, &mut padding, "blob header padding")?;
        if let Some(pos) = padding.iter().position(|b| *b != 0) {
            return Err(format!(
                "non-zero header padding byte at offset {}; padding must be 0x00",
                header_end + pos
            ));
        }
        let mut count_bytes = [0u8; 4];
        read_prefix(reader, &mut count_bytes, "section table count")?;
        let count = u32::from_le_bytes(count_bytes);
        let mut table_end = table_start + 4;
        let mut sections: Vec<BlobSection> = Vec::with_capacity(count.min(64) as usize);
        for index in 0..count {
            let mut name_len_byte = [0u8; 1];
            read_prefix(reader, &mut name_len_byte, "section table name length")?;
            let name_len = name_len_byte[0] as usize;
            let mut name_raw = vec![0u8; name_len];
            read_prefix(reader, &mut name_raw, "section table name")?;
            let name = std::str::from_utf8(&name_raw)
                .map_err(|e| format!("section table entry {index} has a non-UTF-8 name: {e}"))?
                .to_string();
            let mut tag_byte = [0u8; 1];
            read_prefix(reader, &mut tag_byte, "section table tag")?;
            let tag =
                BlobTag::from_u8(tag_byte[0]).map_err(|e| format!("section {name:?}: {e}"))?;
            let mut offset_bytes = [0u8; 8];
            read_prefix(reader, &mut offset_bytes, "section table offset")?;
            let offset = u64::from_le_bytes(offset_bytes);
            let mut length_bytes = [0u8; 8];
            read_prefix(reader, &mut length_bytes, "section table length")?;
            let length = u64::from_le_bytes(length_bytes);
            table_end += 1 + name_len + 1 + 8 + 8;
            sections.push(BlobSection {
                name,
                tag,
                offset,
                length,
            });
        }

        // --- metadata validation (BF-3/BF-5/BF-7); no payload read ---
        validate_section_metadata(table_end, &header, &sections)?;
        Ok(Blob {
            header,
            header_start,
            header_end,
            table_start,
            table_end,
            sections,
        })
    }

    /// Parse a blob from an in-memory slice: metadata via a `Cursor`, then the
    /// payload checks on the same bytes. Convenience for callers that already
    /// hold the whole file (tests, and the PyO3 face when it maps first).
    pub fn parse_bytes(bytes: &[u8]) -> Result<Blob, String> {
        let mut cursor = std::io::Cursor::new(bytes);
        let blob = Self::parse(&mut cursor)?;
        blob.validate_payload(bytes)?;
        Ok(blob)
    }

    /// Payload checks on the map: every section range must fit the file, every
    /// inter-payload gap must be zero, and the last section's end must equal
    /// the file size (BF-4.2, BF-7.2). This runs AFTER the caller maps the
    /// file, so metadata is validated before any map is requested (BL-4.2).
    pub fn validate_payload(&self, bytes: &[u8]) -> Result<(), String> {
        let file_len = bytes.len() as u64;
        let table_end = self.table_end as u64;
        for section in &self.sections {
            let end = section.offset.checked_add(section.length).ok_or_else(|| {
                format!(
                    "section {:?}: offset {} + length {} overflows u64",
                    section.name, section.offset, section.length
                )
            })?;
            if end > file_len {
                return Err(format!(
                    "section {:?}: range [{} , {end}) runs past the {file_len}-byte file",
                    section.name, section.offset
                ));
            }
        }
        // Overlap (BF-3.3), named by both sections. Runs after the file-length
        // bound so a section that runs past EOF is refused by THAT rule first.
        let mut ordered: Vec<&BlobSection> = self.sections.iter().collect();
        ordered.sort_by_key(|s| s.offset);
        for pair in ordered.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if a.offset + a.length > b.offset {
                return Err(format!(
                    "sections {:?} and {:?} overlap: [{}, {}) intersects [{}, {})",
                    a.name,
                    b.name,
                    a.offset,
                    a.offset + a.length,
                    b.offset,
                    b.offset + b.length
                ));
            }
        }
        let mut cursor = table_end;
        for section in &ordered {
            if section.offset > cursor {
                require_zero(
                    bytes,
                    cursor as usize,
                    section.offset as usize,
                    "inter-payload padding",
                )?;
            }
            cursor = cursor.max(section.offset + section.length);
        }
        if cursor < file_len {
            return Err(format!(
                "trailing bytes: {} byte(s) begin at offset {cursor}, after the last section's end",
                file_len - cursor
            ));
        }
        Ok(())
    }

    /// The entry with this exact name, if any.
    pub fn section(&self, name: &str) -> Option<&BlobSection> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// The payload bytes of one section, sliced from the whole-file buffer.
    pub fn slice<'a>(&self, bytes: &'a [u8], section: &BlobSection) -> &'a [u8] {
        &bytes[section.range()]
    }

    /// Tokenizer section names, in file order.
    pub fn tokenizer_names(&self) -> Vec<&str> {
        self.sections
            .iter()
            .filter(|s| s.name.starts_with("tokenizer."))
            .map(|s| s.name.as_str())
            .collect()
    }
}

/// `x` rounded up to the next [`PAGE_SIZE`] boundary (already-aligned `x`
/// stays put). Saturates rather than wrapping on absurd inputs.
fn next_page(x: u64) -> u64 {
    let rem = x % PAGE_SIZE;
    if rem == 0 { x } else { x + (PAGE_SIZE - rem) }
}

/// Read exactly `buf.len()` bytes of the blob prefix, or refuse with the named
/// region. A short read is a truncation refusal (the stream cannot be a blob),
/// not an I/O panic. The message deliberately names the region so a caller can
/// tell a short magic from a short section table.
fn read_prefix(
    reader: &mut (impl Read + ?Sized),
    buf: &mut [u8],
    what: &str,
) -> Result<(), String> {
    reader
        .read_exact(buf)
        .map_err(|e| format!("truncated {what}: needs {} byte(s): {e}", buf.len()))
}

/// Every byte in `[start, end)` must be zero (BF-4.2).
fn require_zero(bytes: &[u8], start: usize, end: usize, what: &str) -> Result<(), String> {
    if let Some(pos) = bytes[start..end].iter().position(|b| *b != 0) {
        return Err(format!(
            "non-zero {what} byte at offset {}; padding must be 0x00",
            start + pos
        ));
    }
    Ok(())
}

/// The closed section-name vocabulary (BF-3.7): a declared layer name, one of
/// the fixed remainder names, a `tokenizer.*` file, or the canonical
/// chat-template sidecar.
fn name_is_declared(name: &str, header: &BlobHeader) -> bool {
    header.layers.iter().any(|l| l == name)
        || matches!(name, "embedding" | "lm_head" | "norms")
        || name.starts_with("tokenizer.")
        || name == crate::CHAT_TEMPLATE_FILE
}

/// Canonical-order key (B5): layers in declared order, then remainder
/// (embedding, lm_head, norms), then `tokenizer.*` by name, then the
/// chat-template sidecar.
fn order_key(name: &str, header: &BlobHeader) -> (u8, String) {
    if let Some(index) = header.layers.iter().position(|l| l == name) {
        return (0, format!("{index:08}"));
    }
    match name {
        "embedding" => (1, "0".into()),
        "lm_head" => (1, "1".into()),
        "norms" => (1, "2".into()),
        _ if name.starts_with("tokenizer.") => (2, name.to_string()),
        _ if name == crate::CHAT_TEMPLATE_FILE => (3, name.to_string()),
        _ => (9, name.to_string()),
    }
}

/// BF-3 / BF-5 / BF-7 metadata rules: every section-table rule that does not
/// read payload bytes. The file-length bound and the zero-gap checks move to
/// [`Blob::validate_payload`], which runs on the map after BL-4.2's alignment
/// assertion has passed.
fn validate_section_metadata(
    table_end: usize,
    header: &BlobHeader,
    sections: &[BlobSection],
) -> Result<(), String> {
    let table_end = table_end as u64;
    let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
    let mut previous_key: Option<(u8, String)> = None;
    for section in sections {
        if section.name.is_empty() || !name_is_declared(&section.name, header) {
            return Err(format!(
                "unknown section name {:?}; expected a declared layer, embedding, lm_head, norms, \
                 or tokenizer.*",
                section.name
            ));
        }
        if seen.insert(section.name.as_str(), ()).is_some() {
            return Err(format!("duplicate section name {:?}", section.name));
        }
        // Checked offset+length: u64 overflow is a refusal, not a wrap
        // (BF-3.2). The file-length bound runs on the map (validate_payload).
        section.offset.checked_add(section.length).ok_or_else(|| {
            format!(
                "section {:?}: offset {} + length {} overflows u64",
                section.name, section.offset, section.length
            )
        })?;
        if section.offset % PAGE_SIZE != 0 {
            return Err(format!(
                "section {:?}: offset {} is not {PAGE_SIZE}-aligned",
                section.name, section.offset
            ));
        }
        if section.offset < table_end {
            return Err(format!(
                "section {:?}: offset {} aliases the metadata region (the section table ends at \
                 {table_end}); a section cannot start inside the magic/header/table bytes",
                section.name, section.offset
            ));
        }
        let key = order_key(&section.name, header);
        if let Some(previous) = &previous_key {
            if *previous >= key {
                return Err(format!(
                    "section {:?} is out of canonical order (after {:?}); the order is layers, \
                     then embedding/lm_head/norms, then tokenizer.*, then the chat-template \
                     sidecar",
                    section.name, previous.1
                ));
            }
        }
        previous_key = Some(key);
    }
    Ok(())
}

/// Parse the header JSON into typed fields, refusing missing and wrong-typed
/// required fields by name (BF-2.4) and any undeclared `remainder_dtype`.
fn parse_header(value: &serde_json::Value) -> Result<BlobHeader, String> {
    let map = value
        .as_object()
        .ok_or_else(|| "blob header JSON is not an object".to_string())?;
    let model_id = req_str(map, "model_id")?;
    let vocab_size = req_usize(map, "vocab_size")?;
    let d_model = req_usize(map, "d_model")?;
    let n_layers = req_usize(map, "n_layers")?;
    let n_heads = req_usize(map, "n_heads")?;
    let n_kv_heads = req_usize(map, "n_kv_heads")?;
    let head_dim = req_usize(map, "head_dim")?;
    let eos_token_id = req_u32(map, "eos_token_id")?;
    let bos_token_id = req_u32(map, "bos_token_id")?;
    let rms_norm_eps = req_f64(map, "rms_norm_eps")?;
    let rope_theta = req_f64(map, "rope_theta")?;
    let norm_slots = req_str_list(map, "norm_slots")?;
    let layers = req_str_list(map, "layers")?;
    let weight_tying = req_nullable_str(map, "weight_tying")?;
    let token = req_str(map, "remainder_dtype")?;
    let remainder_dtype = RemainderDtype::from_token(&token).ok_or_else(|| {
        format!("header field `remainder_dtype` is {token:?}; declared dtypes are fp32 and bf16")
    })?;
    let created_by = match map.get("created_by") {
        None => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(_) => {
            return Err(
                "header field `created_by` must be a JSON string; found another type".to_string(),
            );
        }
    };
    Ok(BlobHeader {
        model_id,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        eos_token_id,
        bos_token_id,
        rms_norm_eps,
        rope_theta,
        norm_slots,
        layers,
        weight_tying,
        remainder_dtype,
        created_by,
    })
}

fn missing(field: &str) -> String {
    format!("blob header is missing the required field {field:?}")
}

fn req_str(
    map: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<String, String> {
    match map.get(field) {
        None => Err(missing(field)),
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(format!(
            "header field {field:?} must be a JSON string; found another type"
        )),
    }
}

/// A required field that may be JSON `null` but whose KEY must be present.
fn req_nullable_str(
    map: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<Option<String>, String> {
    match map.get(field) {
        None => Err(missing(field)),
        Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!(
            "header field {field:?} must be a JSON string or null; found another type"
        )),
    }
}

fn req_usize(
    map: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<usize, String> {
    match map.get(field) {
        None => Err(missing(field)),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("header field {field:?} must be a non-negative integer")),
        Some(_) => Err(format!(
            "header field {field:?} must be a JSON integer; found another type"
        )),
    }
}

fn req_u32(map: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<u32, String> {
    match map.get(field) {
        None => Err(missing(field)),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("header field {field:?} must be an integer in u32 range")),
        Some(_) => Err(format!(
            "header field {field:?} must be a JSON integer; found another type"
        )),
    }
}

fn req_f64(map: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<f64, String> {
    match map.get(field) {
        None => Err(missing(field)),
        Some(serde_json::Value::Number(n)) => n
            .as_f64()
            .ok_or_else(|| format!("header field {field:?} must be a JSON number")),
        Some(_) => Err(format!(
            "header field {field:?} must be a JSON number; found another type"
        )),
    }
}

fn req_str_list(
    map: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<Vec<String>, String> {
    match map.get(field) {
        None => Err(missing(field)),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                serde_json::Value::String(s) => Ok(s.clone()),
                _ => Err(format!(
                    "header field {field:?} must be an array of strings"
                )),
            })
            .collect(),
        Some(_) => Err(format!(
            "header field {field:?} must be a JSON array; found another type"
        )),
    }
}
