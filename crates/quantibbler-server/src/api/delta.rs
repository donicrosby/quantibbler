//! Incremental detokenization for SSE streaming (qk-server SR-4).
//!
//! Decoding one token at a time breaks on two real classes of tokenizer
//! output: (1) a codepoint split across two BPE tokens yields invalid
//! UTF-8 in any single-token chunk, and (2) bytes that begin a stop
//! sequence must not be emitted before the sequence can be ruled out —
//! otherwise the stop string leaks into the visible stream and cannot be
//! retracted. `DeltaDecoder` accumulates token ids, decodes the whole
//! sequence so far, and releases only the byte range that is provably
//! safe: complete UTF-8 characters that are not a prefix of any stop
//! sequence. At session end the caller flushes (`FinishReason::Stop` via
//! EOS releases the remainder; a stop-sequence hit drops the withheld
//! suffix).
//!
//! v0.1 honesty note: the released-prefix computation is byte-granular
//! with character-boundary snapping, correct for UTF-8 encoded text and
//! for stop sequences searched in decoded-string space (matching the
//! synchronous path).

/// Incremental, stop-aware, character-safe detokenizer.
///
/// The decoder owns the token sequence: callers push one token at a time
/// and never handle accumulation themselves (an earlier "pass the whole
/// sequence each call" shape was error-prone enough to misfire in its own
/// unit tests within an hour, so the buffer moved inside).
pub struct DeltaDecoder {
    stops: Vec<String>,
    skip_special_tokens: bool,
    tokens: Vec<u32>,
    released_len: usize,
    finished: bool,
}

impl DeltaDecoder {
    pub fn new(stops: Vec<String>, skip_special_tokens: bool) -> Self {
        Self {
            stops,
            skip_special_tokens,
            tokens: Vec::new(),
            released_len: 0,
            finished: false,
        }
    }

    /// Append one token and return the newly releasable text delta, or
    /// `None` when nothing new is safe to emit (withheld stop prefix,
    /// incomplete character, or a special token dropped by
    /// `skip_special_tokens`). Emptiness is not an error.
    pub fn push(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        token: u32,
    ) -> Result<Option<String>, String> {
        assert!(!self.finished, "DeltaDecoder used after finish()");
        self.tokens.push(token);
        let tokens: &[u32] = &self.tokens;
        let text = tokenizer
            .decode(tokens, self.skip_special_tokens)
            .map_err(|e| format!("decode: {e}"))?;
        let bytes = text.as_bytes();
        if bytes.len() <= self.released_len {
            return Ok(None);
        }
        // Release limit, in bytes of `text`:
        // 1. never past the EARLIEST complete stop occurrence (a stop that
        //    fit inside already-decoded text must not leak on a later push);
        // 2. never within the last (longest_stop_len - 1) bytes, which could
        //    still complete into a stop on the next token (rolling window);
        // 3. snapped down to a UTF-8 character boundary (a codepoint split
        //    across tokens never emits invalid UTF-8).
        let mut limit = bytes.len();
        for stop in &self.stops {
            if let Some(pos) = text.find(stop.as_str()) {
                limit = limit.min(pos);
            }
        }
        let max_stop = self.stops.iter().map(String::len).max().unwrap_or(0);
        if max_stop > 0 {
            limit = limit.saturating_sub(max_stop - 1);
        }
        while limit > self.released_len && !text.is_char_boundary(limit) {
            limit -= 1;
        }
        if limit <= self.released_len {
            return Ok(None);
        }
        let delta = text[self.released_len..limit].to_string();
        self.released_len = limit;
        Ok(Some(delta))
    }

    /// End of generation. `stop_hit` = the stop sequence that matched (the
    /// withheld suffix holds its beginning and is dropped); `None` = EOS or
    /// length finish (the remainder is complete text and is released).
    pub fn finish(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        stop_hit: Option<&str>,
    ) -> Result<String, String> {
        self.finished = true;
        let tokens: &[u32] = &self.tokens;
        let text = tokenizer
            .decode(tokens, self.skip_special_tokens)
            .map_err(|e| format!("decode: {e}"))?;
        if stop_hit.is_some() {
            return Ok(String::new());
        }
        let bytes = text.as_bytes();
        if bytes.len() <= self.released_len {
            return Ok(String::new());
        }
        Ok(text[self.released_len..].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/server-model/tokenizer.json"),
        )
        .expect("fixture tokenizer loads")
    }

    /// NOTE on expectations: the fixture tokenizer decodes with an inserted
    /// space between tokens ("abc" encodes to ["ab", "c"] and decodes to
    /// "ab c"), so assertions are against decode-space text, not the
    /// encode() input.

    #[test]
    fn whole_text_released_without_stops() {
        let tok = tokenizer();
        let ids = tok.encode("abc ab", false).unwrap().get_ids().to_vec();
        let whole = tok.decode(&ids, false).unwrap();
        let mut d = DeltaDecoder::new(vec![], false);
        let mut acc = String::new();
        for t in &ids {
            if let Some(p) = d.push(&tok, *t).unwrap() {
                acc.push_str(&p);
            }
        }
        acc.push_str(&d.finish(&tok, None).unwrap());
        assert_eq!(acc, whole);
        assert!(!acc.is_empty());
    }

    #[test]
    fn stop_prefix_withheld_then_dropped() {
        let tok = tokenizer();
        let ids = tok.encode("abc ab", false).unwrap().get_ids().to_vec();
        let mut d = DeltaDecoder::new(vec!["ab".to_string()], false);
        let mut acc = String::new();
        for t in &ids {
            if let Some(p) = d.push(&tok, *t).unwrap() {
                acc.push_str(&p);
            }
        }
        // Stop hit: nothing at/after the earliest "ab" occurrence may have
        // leaked, and finish() drops the withheld remainder.
        acc.push_str(&d.finish(&tok, Some("ab")).unwrap());
        assert!(
            !acc.contains("ab"),
            "stop sequence leaked into stream: {acc:?}"
        );
    }

    #[test]
    fn single_token_stop_never_leaks() {
        // The first token IS the stop string: the engine would terminate on
        // it; the decoder must not have released any of it either.
        let tok = tokenizer();
        let ids = tok.encode("abc", false).unwrap().get_ids().to_vec();
        let mut d = DeltaDecoder::new(vec!["ab".to_string()], false);
        let mut acc = String::new();
        for t in &ids {
            if let Some(p) = d.push(&tok, *t).unwrap() {
                acc.push_str(&p);
            }
        }
        acc.push_str(&d.finish(&tok, Some("ab")).unwrap());
        assert!(!acc.contains("ab"), "stop leaked: {acc:?}");
    }

    #[test]
    fn withheld_prefix_released_when_no_match() {
        let tok = tokenizer();
        let ids = tok.encode("b ab", false).unwrap().get_ids().to_vec();
        let whole = tok.decode(&ids, false).unwrap();
        // "az" never occurs; the withholding window must not eat the output.
        let mut d = DeltaDecoder::new(vec!["az".to_string()], false);
        let mut acc = String::new();
        for t in &ids {
            if let Some(p) = d.push(&tok, *t).unwrap() {
                acc.push_str(&p);
            }
        }
        acc.push_str(&d.finish(&tok, None).unwrap());
        assert_eq!(acc, whole);
    }

    #[test]
    fn skip_special_drops_eos_token() {
        let tok = tokenizer();
        let mut all = tok.encode("abc", false).unwrap().get_ids().to_vec();
        all.push(2); // <eos>
        let mut d = DeltaDecoder::new(vec![], true);
        let mut acc = String::new();
        for &t in &all {
            if let Some(p) = d.push(&tok, t).unwrap() {
                acc.push_str(&p);
            }
        }
        acc.push_str(&d.finish(&tok, None).unwrap());
        assert!(!acc.contains("<eos>"), "special token leaked: {acc:?}");
        assert_eq!(acc, tok.decode(&all, true).unwrap());
    }
}
