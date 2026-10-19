# qk-flash-attention — Tasks

Rules honored: tests precede implementation per capability (RED observed
before code); every task independently verifiable; final task is the full
gate suite.

## 1. Reference first (Python face)

- [ ] 1.1 RED: author `tests/test_attention_tiled.py` covering scenario IDs
      `ref-matches-formula` and `ref-causal-mask` (fixtures per
      specs/qk-kernels delta: seq 8/10/64, d 4/128, heads (2,1)/(8,2));
      run `uv run pytest tests/test_attention_tiled.py` and OBSERVE the
      failure (module absent). Record the RED output in the task note.
- [ ] 1.2 GREEN: implement `src/quantibbler/kernels/attention_tiled.py`
      ([FA] §3.1 Algorithm 1, f64, ragged-tail safe); same command exits
      0 with all scenarios passing.

## 2. CubeCL GPU kernel (kernels crate, `gpu` feature)

- [ ] 2.1 RED: author `crates/quantibbler-kernels/tests/gpu_attention_parity.rs`
      (hardware-gated per house pattern: unset env = clean skip notice, set
      = device run) with scenario `gpu-parity-budget` (rel ≤ 1e-5 vs the
      NumPy reference via the PyO3 face or fixture vectors) and
      `fail-closed-construction`; OBSERVE compile/run failure before the
      kernel exists; record it.
- [ ] 2.2 GREEN: implement `src/gpu_attention.rs` — tiled online-softmax
      kernel, working forms from landed gpu.rs (0.11 idioms, bounds
      guards, u32 ceiling); hardware gate green on gfx1151 in BOTH env
      states; clippy/fmt green.
- [ ] 2.3 Re-sync the PyO3 face (`uv sync --reinstall-package
      quantibbler-kernels`) and confirm the CHANGED behavior (new symbol
      computes, not just imports).

## 3. Server attention face (dispatch + parity)

- [ ] 3.1 RED: author `crates/quantibbler-server/tests/attention_dispatch.rs`
      (scenario `face-selection-health`, `soft-fallback-init-failure` —
      the fallback test must run WITHOUT hardware); observe RED.
- [ ] 3.2 GREEN: engine wiring mirroring MatvecFace: AttentionFace enum,
      select-once at load, env opt-in, soft fallback, /health
      `attention_face` label; default-feature suite green (scalar path
      untouched).
- [ ] 3.3 Hardware parity: scenario `cross-face-token-identity` on gfx1151
      (fixture pipeline greedy generation, both faces, tokens identical,
      logits rel ≤ 1e-5).

## 4. Perf probe (OQ-1 — resolved only by measurement)

- [ ] 4.1 Timed decode-attention benchmark, CPU scalar face vs GPU tiled
      face at (seq 512, seq 2048) × d=128 × GQA 8:1 on gfx1151; record
      the table in design.md crystallization; OQ-1 answered WITH numbers.
      Outcome gate: ≥ 1.5× at seq 2048 keeps the face opt-in-recommended;
      a miss keeps it opt-in-experimental and records why.

## 5. Landing

- [ ] 5.1 Full gate suite: `uv run pytest`, `cargo test` (kernels + server,
      both feature states), `cargo clippy --all-targets -D warnings`,
      `cargo fmt --all -- --check`, `uv run maturin build`, and
      `openspec validate qk-flash-attention --strict`; scoped commit
      (spec + code + crystallization together).
