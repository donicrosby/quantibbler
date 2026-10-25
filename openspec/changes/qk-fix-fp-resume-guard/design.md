# Design: qk-fix-fp-resume-guard

## The class fix

A guard must be driven from **what the block is owed** (the live model's
expected quantizable names for that block), never from **what the archive
contained**. Concretely in `apply_layer_states`:

- expected := the block's quantizable linear names (from the live model, the
  same predicate `_quantizable_names` uses);
- present := names actually reconstructed from the archive;
- assert `expected == present` (set equality) and raise with the symmetric
  difference on mismatch. An empty archive then raises because
  `expected ≠ ∅`.

Plus the two upstream pins so empty archives stop existing at all:
`save_block_checkpoint` raises on `layers == []`, and `completed_blocks`
rejects zero-key archives (defense in depth — a hand-copied empty archive
can no longer masquerade as done).

## Failure semantics after the fix

A resumed fp run hits the raise at the first empty archive: loud abort with
a message naming the block and the archive path. No pristine prefix advances
silently. Fresh runs are unaffected (they write real archives or, in fp mode,
now fail loudly at save time — which is the honest signal that fp mode is a
fresh-run-only configuration until the extraction path is fixed separately).

## Class-fix sweep (sec-eng's ruling)

Grep the compressor for guards whose membership set is derived from loaded
content (`by_rel`, `missed`, anything built from `npz.files` / archive keys):
each must be rewritten to expected-vs-present or carry a recorded waiver in
this design. `qk-divergence-guards` is a sibling change with live deltas —
this change cites it and does not modify it.

## Discarded

- **Size-threshold heuristics on the archive bytes.** Rejected: 22 bytes is
  incidental; the defect is "no keys", and key-set equality is the correct
  predicate. A threshold can be gamed by an archive with one key.
- **Fixing `apply_layer_states` alone.** Rejected: empty archives would still
  be *written* and still counted as *done*; the failure would move around
  instead of being killed at the source.

## Implementation crystallization

### 2026-10-25 (landing disposition of the two review passes)

- **Scan-side key grammar (the y_b asymmetry).** `completed_blocks` now
  applies the loader's rule — only ``|``-separated keys derive layer
  names — so a y_b-only archive (the reserved teacher target carries no
  separator) is excluded from the done set exactly as a zero-key archive
  is. This closes the spec-letter gap both reviews flagged against
  scenario ``empty_archive_is_never_counted_done`` ("no readable
  quantized-layer names"). RED→GREEN observed:
  ``test_yb_only_archive_is_not_counted_done`` (2 failed / 9 passed
  before the fix, 15 passed after, under ``SOLVER_BACKEND=numpy``).
- **`apply_layer_states.expected_names` is required, not defaulted.**
  The guard this change exists for cannot be disarmed by omission: a
  caller that cannot name what the live block is owed has no business
  restoring into it. All in-tree callers (the one production call site
  and every test) already pass it. RED→GREEN observed:
  ``test_expected_names_cannot_be_omitted``.
- **Ruling requested by review (doctrine vs landed bytes):** the landed
  record-only digest CONFORMS. This change's spec requires resume to
  fail closed on empty/incomplete archives; content-hash drift refusal
  per write is integrity hardening beyond this delta and belongs with
  the fingerprint surface (model-identity binding, the shared driver
  ``--checkpoint-dir`` default), which is routed to follow-up tickets,
  not folded in here.
- **Findings routed out of this change** (reviews t_a76ffbcd /
  t_bde3c34c): two-phase validate-before-write apply loop (MEDIUM);
  fingerprint binds no model identity + shared default (should-fix);
  archive byte hash, int()-parse notice-vs-crash consistency, and the
  reconstruct_model fresh-start forensic clobber (LOW). Filed on the
  board; neither review blocked this change's delta on them.
- **Foreign gate blocker (not this change):** the torch-default backend
  refusal at the unthreaded `magnitude_balance` call site (board ticket
  for it already exists; sec-eng filed it). On this CPU-only host the
  ptq solve-bearing tiers run green under ``SOLVER_BACKEND=numpy``, the
  configuration both reviews certified; that env is recorded with every
  gate run in this landing.

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." Consequence: this fix and its siblings (`qk-fix-tunefp-teacher`,
`qk-fix-ste-sign-adaptation`) land before any further diagnostic runs; the
ladder fork fires only after a paper-faithful run exists.
