# qk-wrapper-contract — Design

## Facts checked before authoring (2026-10-25)

- `tools/run-ptq-container.sh` exists and carries the wrapper contract — witness capture, guard chains, launch.env provenance, DRY_RUN seam oracle, and dead-mode cleanup (the retired `lane0-rehearsal`/`lane0-fire` modes were removed at `t_df5bfe7e`).
- `tests/test_run_ptq_wrapper.py` exercises the wrapper's fail-closed branches hermetically.
- No `qk-wrapper` capability spec exists in `openspec/specs/` — the wrapper contract has no first-class spec home.
- The 7 requirements being extracted were authored during the Lane 0 incident response (chunking/checkpointing/post-mortems) and landed in the lane0 delta alongside measurement experiment requirements that have nothing to do with the wrapper.

## Decisions

1. **New capability spec `qk-wrapper`, not a bolt-on to `qk-ptq`.** The wrapper is host-side launch infrastructure; it wraps the PTQ container but is not itself a PTQ algorithm concern. A separate capability spec means future wrapper changes (new guard chains, witness format changes) don't clutter the PTQ spec.

2. **As-built — no code changes.** The 7 requirements already have implementation in the surviving files. This change captures them as an as-built spec; no new code is owed.

3. **Requirements extracted verbatim.** The requirement text and scenarios are copied from the lane0 delta unchanged (minus the delta header). The lane0-specific `LANE0_IMAGE_DIGEST_ALLOW` env var reference in `seam_oracle_is_dry_run_rc` is kept — it's an instance of the pattern, and the requirement (guard chain with DRY_RUN seam oracle) is universal.

4. **Archive immediately.** Since no code changes are owed and the spec is as-built, this change can be archived immediately (or left active for review). The lane0 archive sweep is the triggering event; the wrapper-contract change is the extraction target.

## Discarded alternatives

- **Fold into `qk-launch-attribution`.** Rejected: `qk-launch-attribution` is about driver-side crumb trails, not host-side wrapper discipline. The wrapper requirements cover witness capture, guard chains, and provenance — a different contract surface.
- **Fold into `qk-divergence-guards`.** Rejected: that change is about PTQ guard band thresholds, not launch infrastructure.
- **Attach to `qk-ptq` main spec.** Rejected: the wrapper is not a PTQ algorithm concern.

## Open Questions

- OQ-1 (owner: planner): should `LANE0_IMAGE_DIGEST_ALLOW` be renamed to a mode-agnostic name (e.g. `PTQ_IMAGE_DIGEST_ALLOW`)? The env var is lane0-specific in name but universal in function. Status: OPEN — keep as-is for now; rename when a non-lane0 mode actually uses the wrapper digest pin.
