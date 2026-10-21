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

## User directive (2026-10-21)

User: "Follow the paper TO THE LETTER; prove the paper's method is possible
first." Consequence: this fix and its siblings (`qk-fix-tunefp-teacher`,
`qk-fix-ste-sign-adaptation`) land before any further diagnostic runs; the
ladder fork fires only after a paper-faithful run exists.
