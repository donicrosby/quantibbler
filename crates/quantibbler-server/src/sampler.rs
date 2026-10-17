//! Sampler (design D6): greedy argmax is the ONLY v0.1 sampler. Sampling
//! parameters in requests are accepted and honestly ignored (OQ4/IA4).

/// Greedy argmax with first-max tie-break (design D6).
pub fn greedy(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}
