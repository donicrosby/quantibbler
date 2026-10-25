# Acceptance-run ledger format — `bar_recorded_not_gated`

Task 28 (orchestrator). The scenario's contract: when an acceptance run
completes, a ledger entry records **model id, commit hash, targeted row, eval
protocol, measured PPL, ratio, effective_bpw, and the operational-gate
verdict** — "the bar is a documented human-gated acceptance criterion, not a
per-run gate verdict". This note pins what that entry IS. It is a format note,
deliberately not code: the ledger's tooling is authored when the first
acceptance run is actually scheduled (proposal, Impact), and that run is a
@user-gated fire (lane authority rule). A format reviewable before the first
entry exists is the point of authoring it now.

## The file

- **`docs/acceptance-ledger.jsonl`** — JSON Lines, UTF-8, LF endings: one JSON
  object per line, one line per acceptance evaluation (one artifact × one
  targeted row × one eval protocol).
- **Append-only.** Never edit, never delete, never reorder. A correction or
  re-measure is a NEW line whose `supersedes` carries the earlier entry's
  `entry_id`; the current judgment for a run is its last unsuperseded line.
  An acceptance record that can be quietly rewritten is not a record.
- **Authored by a human at gate time**, from the run's `quantize_report.json`
  (which stays the measurement of record). Nothing in the pipeline writes or
  reads this file — it is the acceptance record, not an input to anything.
  Until the first acceptance run fires, the file does not exist, and that is
  the honest state.

## Required fields

The scenario's eight, in contract order, plus the identity field that makes
append-only corrections possible.

| field | type | rule |
|---|---|---|
| `entry_id` | string | unique; `<model-slug>-<commit7>-<row-label>-<UTC date>` |
| `model_id` | string | base model HF id the artifact quantizes (e.g. `Qwen/Qwen3-0.6B`) |
| `commit` | string | full 40-hex `git rev-parse HEAD` of the tree that produced the artifact |
| `targeted_row` | enum | exactly one of the `qk_bar.BAR_ROWS` labels: `nanoquant_1.00_bits`, `nanoquant_0.80_bits`, `nanoquant_0.55_bits`. Nothing else is a row: the FP16 baseline is not a target ("a run does not meet the baseline"), and below 0.55 bits there is no row (`bar_undefined_below_lowest_row` — beyond-paper research is never recorded under the word recreation, never granted ledger status on interpolation). The format names the labels but never re-lists the values; the constants' single source of truth stays `qk_bar` and its paper-line audit |
| `eval_protocol` | object | `corpus` (`wikitext-2-raw-v1`), `partition` (`test`, per [NQ] Sec. 4.2), `seq_len` (int; 2048 = paper protocol), `tune_seq_len` (int), `calibration_samples` (int; 128 in [NQ]) |
| `measured` | object | `ppl` (float; quantized eval), `ratio` (= `ppl / row_ppl`, the bar-relevant distance to the targeted row), `ratio_vs_baseline` (= `ppl / FP16_BASELINE_PPL`; [NQ] Table 2's own ratio column is baseline-relative), `effective_bpw` (float; byte-derived serialized figure, as the report prints it), plus COPIES of the numbers judged against: `row_ppl`, `row_bits`, `bar_ceiling` (= `1.15 * row_ppl`) |
| `tolerance_provenance` | string | `"provisional 1.15 (qk-nq-recreation-bar spec pin)"` until the spec re-pins it; a re-derived tolerance arrives as a SPEC change (`bar_tolerance_provisional_until_measured`) and later entries name the new pin. Never edited inside a ledger line |
| `bar_met` | bool | `qk_bar.row_meets_bar` semantics (`ppl <= bar_ceiling` AND `effective_bpw <= row_bits`, fail-closed on missing/non-finite/non-numeric) AND paper protocol (`corpus` wikitext-2, `partition` test, `seq_len == 2048`). A surrogate-only entry is `false`: seq-512 numbers are recorded alongside, never substituted for the bar (`bar_requires_paper_protocol`) |
| `operational_gate_verdict` | object | COPY from the run report: `band` (`ok`/`degraded`/`failed`) + `report_path`. The ledger records the verdict, never computes it; the band stays the ONLY per-run verdict (`bar_distinct_from_landing_gate`) |

## Required integrity fields

| field | type | rule |
|---|---|---|
| `recorded_at` | string | UTC ISO-8601 |
| `run_command` | string | the exact driver command line |
| `report_path` | string | the run's `quantize_report.json` |
| `artifact_path` | string | the exported qpk1 model directory |
| `acceptance` | object | the human gate: `decision` (`met` / `not_met` / `equivalence`), `decided_by`, `decided_at` |

## Optional fields

| field | type | rule |
|---|---|---|
| `protocol_deviation` | object | present iff the eval geometry deviates from [NQ]'s: `deviation` (what differs) + `equivalence_argument` (why a surrogate number is being accepted — the proposal requires this argument recorded in the ledger, not assumed) |
| `notes` | string | anything else the human gate wants preserved |
| `supersedes` | string | `entry_id` of the line this one replaces |

## Rules the format enforces

1. `decision: "met"` requires `bar_met: true`. `decision: "equivalence"`
   requires `bar_met: false` AND a non-empty `equivalence_argument`: it is a
   recorded human acceptance of a protocol deviation, never a flip of
   `bar_met` — the spec's bar stays met on the paper protocol only.
2. The ledger never feeds `qk_report.verdict()`, and no pipeline code imports
   it: a doc the run path cannot see cannot gate a run (the structural
   separation `qk_bar`'s landing pinned, preserved at the ledger layer).
3. Values, never renderings: the design table's 2dp ceilings (31.69 / 38.86 /
   60.88) are display; entries carry full-precision floats.
4. One row per line: an artifact claiming two rows writes two entries.

## Example (illustrative only — every number below is placeholder, not a run)

```json
{"entry_id": "qwen3-0p6b-a1b2c3d-nanoquant_0.80_bits-20261102",
 "model_id": "Qwen/Qwen3-0.6B",
 "commit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
 "targeted_row": "nanoquant_0.80_bits",
 "eval_protocol": {"corpus": "wikitext-2-raw-v1", "partition": "test",
                   "seq_len": 2048, "tune_seq_len": 512,
                   "calibration_samples": 128},
 "measured": {"ppl": 34.12, "ratio": 1.0098, "ratio_vs_baseline": 2.6954,
              "effective_bpw": 0.79, "row_ppl": 33.79, "row_bits": 0.80,
              "bar_ceiling": 38.8585},
 "tolerance_provenance": "provisional 1.15 (qk-nq-recreation-bar spec pin)",
 "bar_met": true,
 "operational_gate_verdict": {"band": "ok",
   "report_path": "artifacts/qwen3-0p6b-r08/quantize_report.json"},
 "recorded_at": "2026-11-02T18:40:11Z",
 "run_command": "uv run python tools/quantize_hf_model.py --model Qwen/Qwen3-0.6B --rank 8 --seq-len 2048 --tune-seq-len 512 --out artifacts/qwen3-0p6b-r08",
 "report_path": "artifacts/qwen3-0p6b-r08/quantize_report.json",
 "artifact_path": "artifacts/qwen3-0p6b-r08/qpk1",
 "acceptance": {"decision": "met", "decided_by": "@user",
                "decided_at": "2026-11-02T19:02:00Z"},
 "notes": "format illustration only; not a measurement"}
```

## Validation convenience (doc snippet, not tooling)

```bash
uv run python - <<'PY'
import json
REQ = {"entry_id", "model_id", "commit", "targeted_row", "eval_protocol",
       "measured", "tolerance_provenance", "bar_met",
       "operational_gate_verdict", "recorded_at", "run_command",
       "report_path", "artifact_path", "acceptance"}
ROWS = {"nanoquant_1.00_bits", "nanoquant_0.80_bits", "nanoquant_0.55_bits"}
n = 0
for i, line in enumerate(open("docs/acceptance-ledger.jsonl"), 1):
    e = json.loads(line)
    missing = REQ - e.keys()
    assert not missing, (i, sorted(missing))
    assert e["targeted_row"] in ROWS, (i, e["targeted_row"])
    if e.get("acceptance", {}).get("decision") == "met":
        assert e["bar_met"] is True, i
    if e.get("acceptance", {}).get("decision") == "equivalence":
        assert e["bar_met"] is False, i
        assert e.get("protocol_deviation", {}).get("equivalence_argument"), i
    n += 1
print(f"ledger ok: {n} entries")
PY
```
