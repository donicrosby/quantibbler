## ADDED Requirements

### Requirement: The pipeline SHALL be held to a paper-recreation bar pinned to [NQ] Table 2 rows for Qwen3-0.6B

An acceptance run SHALL demonstrate, on Qwen3-0.6B over [NQ]'s WikiText-2 sequence-length-2048 protocol, a perplexity at or below `1.15 * row` for one of the three NanoQuant rows of [NQ] Sec. 4.2 Table 2 (27.56 @1.00 bits, 33.79 @0.80 bits, 52.94 @0.55 bits, against the 12.66 baseline), with byte-derived `effective_bpw` at or below the targeted row's bit budget.

#### Scenario: bar_constants_asserted

- **WHEN** the suite runs
- **THEN** a test asserts the four row constants and the tolerance arithmetic (`1.15 * row` for each targeted row) are the values pinned in this delta, and that no runtime path recomputes them

#### Scenario: bar_undefined_below_lowest_row

- **WHEN** an artifact's bit budget is below 0.55 bits
- **THEN** no recreation row exists for it; it is beyond-paper research — never recorded under the word recreation, never granted ledger status on interpolation

#### Scenario: bar_tolerance_provisional_until_measured

- **WHEN** artifacts are still far from any row
- **THEN** the 1.15 tolerance stands as the pre-registered provisional, and the first artifact landing within 2x of a targeted row obligates re-deriving the tolerance from that artifact's measured run-to-run eval variance and re-pinning it as a spec change

#### Scenario: bar_never_recalibrates_the_band

- **WHEN** a band-edge recalibration is ever proposed
- **THEN** moving an edge so a paper-reference artifact (including [NQ]'s own rows, e.g. @0.55 ratio 4.18) reads `ok` is prohibited — the band is never tuned to flatter a reference; it recalibrates only from production healthy-extreme evidence

#### Scenario: bar_not_purchased_with_bits

- **WHEN** an artifact claims a bar row
- **THEN** its serialized-byte `effective_bpw` is at or below that row's bit budget — a lower PPL at a higher bit budget meets no row

#### Scenario: bar_requires_paper_protocol

- **WHEN** an acceptance run executes
- **THEN** it evaluates at sequence length 2048 over the WT2 test partition per [NQ] Sec. 4.2; seq-512 numbers alone are recorded alongside, never substituted for the bar

#### Scenario: bar_recorded_not_gated

- **WHEN** an acceptance run completes
- **THEN** a ledger entry records model id, commit hash, targeted row, eval protocol, measured PPL, ratio, effective_bpw, and the operational-gate verdict — the bar is a documented human-gated acceptance criterion, not a per-run gate verdict

#### Scenario: bar_distinct_from_landing_gate

- **WHEN** any artifact is classified
- **THEN** the `ok/degraded/failed` band remains the only per-run verdict; an `ok` artifact may still be below the bar, and no artifact above the bar receives `ok` on bar evidence alone
### Requirement: The PTQ driver SHALL support independent tune and evaluation sequence lengths

The driver's `--seq-len` SHALL remain the evaluation geometry, and a `--tune-seq-len` flag SHALL bound the per-block reconstruction windows independently. When `--tune-seq-len` is unset it SHALL default to `--seq-len`, preserving single-geometry behavior exactly.

#### Scenario: split_mode_preserves_eval_geometry

- **WHEN** the driver runs with `--seq-len 2048 --tune-seq-len 512`
- **THEN** evaluation PPL runs over 2048-token WT2 test sequences while block reconstruction tunes on 512-token windows, and the report records both lengths

#### Scenario: default_single_geometry_unchanged

- **WHEN** `--tune-seq-len` is unset
- **THEN** tuning windows use `--seq-len` exactly as before the flag existed — existing run behavior is unchanged and no silent geometry change occurs
