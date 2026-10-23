# qk-geometry-conformance — design

## Unit decision (resolved)

The knobs keep their names and their protocol values and come to MEAN sequences. Rationale: the paper's protocol is 128 sequences ([NQ] Sec 3.4/App. C); the CLI surface (`--calibration-seqs 128`) is the one the campaign already uses; changing the unit means changing every recorded launch line. The packed-batch structure (BATCH=8) is an implementation detail of the tokenizer and SHALL live below the knob boundary.

## Fix placement (resolved, Coder requirement)

The knob→chunks conversion lands at the **same boundary as the counter** — before the `tune_seq_len != seq_len` slice at `quantize_hf_model.py:971`, which splits each (8, 2048) batch into four (8, 512) windows and quadruples `len(calib)`. A conversion placed after that line computes a knob value 4× too large under `TUNE_SEQ_LEN=512`, a geometry the campaign actually runs. The existing comment already requires the counter "counted BEFORE the slice"; the fix rides the same line of reasoning.

## Constant taxonomy (Bug Hunter spec line, incorporated)

- **(a) Pedestals** (baseline PPL, fp-prefix PPL): geometry-dependent — re-measured per run at that run's geometry (already provenance-recorded); no code movement.
- **(b) COLLAPSE_LINE = 40.0**: citation-only. Derived `ceil(241.96 / 6.10)` from [LB2] Sec. 5.2 Table 1 (241.96 = documented method-collapse PPL at 0.55 bpp; 6.10 = FP16 reference), asserted in `tests/test_qk_report.py`. Ratio form (`probe_ratio` divides by the same-depth fp prefix) makes it transportable across geometries. Zero code movement.
- **(c) Depth-2 fp-prefix probe**: same-object-same-depth by construction; geometry-consistent automatically.
- **(d) Eval-window gate**: names its own unit (`full_local_window_batches`) and compares like units; re-expresses in sequences with the fix and re-derives automatically.

## Known limitations

- The bpp-axis inference (our ~1 bpp stored form sits above the anchor pair's 0.55 bpp collapse point, so the line is lenient for our operating point) is read off two anchor points — inference, not a paper claim. Recorded here so it is never cited as such.
- v2's 339× verdict (8.5× above the line) survives the G geometry correction: no plausible window-size effect spans that margin.
