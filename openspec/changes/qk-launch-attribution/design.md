# qk-launch-attribution — Design

## Facts checked before authoring (2026-10-21)

- The crumb start line ALREADY carries pid + argv
  (`tools/quantize_hf_model.py:558`) — sec-eng's read of the trail as
  "no argv, no pid" is stale against the current tree. What is
  genuinely absent: the run's code identity (git rev), the resolved
  output target, and any guard against the shared default path.
- The shared-path hazard is real: the default trail is
  `$QUANTIBBLER_KD_TMPDIR/quantibbler-breadcrumbs.log`, so concurrent
  sandbox runs append interleaved lines — part of why the corpse
  reconciliation had to choose between "two attempts" and "unknown
  death."
- The container contract must not change: the wrapper sets
  `QUANTIBBLER_BREADCRUMBS=/tmp/ptq/quantibbler-breadcrumbs.log` and
  appends dated run headers; forensics reads that exact path. The
  per-launch suffix therefore applies ONLY to the defaulted path; an
  explicit value is honored exactly. A container is single-run by
  construction, so the interleaving hazard does not exist there.

## Decisions

1. **Identity fields on the launch line, not a new file.** One line,
   first in every trail: `start pid=… argv=… rev=<git rev|unknown>
   out=<resolved out dir>`. A corpse then self-attributes regardless of
   which other mechanism (announcements, docker metadata, kernel
   records) failed.
2. **git rev at runtime, `unknown` when unresolvable.** The driver may
   legitimately run outside a checkout (CI smoke harness copies, bare
   scripts); the field must exist in every trail with an honest value.
3. **Pid-suffixed default, exact configured path.** Covers the
   concurrent-sandbox case without touching the container/forensics
   contract.
4. **Structural pin rides the same commit** — the launch-line
   construction joins the driver-wiring tripwire set
   (`tests/test_qk_report.py::_driver_wiring_findings`), same-edit
   discipline: the pin and the behavior move together, and a
   downstream edit that drops a field reds the tripwire test.

## Discarded alternatives

- **Always pid-suffix the trail file.** Rejected: breaks the wrapper's
  header/forensics contract for the container case, which has no
  concurrency to defend against.
- **A launch-registration service.** Rejected: network + a new
  component to secure; the trail file is already crash-proof and the
  corpse proved it survives the kill.
- **Prose-only rule.** Rejected: the protocol already exists in
  AGENTS.md and was insufficient — mechanism is the point of this
  change.

## Open Questions

- OQ-1 (owner: orchestrator): should the watchdog lines also carry the
  pid? No — the identity line plus per-launch files already binds
  every line to its launch; adding pid to each 5 s sample buys
  nothing. Status: CLOSED at authoring.

## Implementation crystallization

### 2026-10-25 — task 1.1: the RED tests read the contract at a doomed start

`tests/ptq/test_launch_attribution.py` takes every reading from a driver
process that STARTS for real and then aborts at model load (a nonexistent
`--model` path, resolved offline), with `--device cpu` passed by name. That
trigger is deliberate: it IS the corpse case this change exists for, and it is
reachable whether the launch line is written before `argparse` or immediately
after it. Two consequences for task 1.2, decided here rather than left silent:

- The tests do NOT require a trail from a `--help` / parse-error exit, so moving
  the launch-line write below `parser.parse_args()` — the natural way to know
  the resolved `out` dir — is permitted. What is pinned is that the launch line
  is line 1 of its trail: the `memory.events@start` crumb and the watchdog must
  not be able to reach the file first. Field order is pinned as decision 1
  states it (epoch crumb field, then `pid`, `argv`, `rev`, `out`).
- `rev=unknown` is asserted by running the driver from a COPY placed outside any
  checkout, with `cwd` outside it too, and the test first asserts that
  `git -C <dir> rev-parse --show-toplevel` fails there — so the literal cannot
  be satisfied by whichever directory the resolver happens to consult.

Pre-implementation the whole file was RED (4 failed): the identity line carried
`pid` and `argv`, as the "facts checked" section above already said, and
neither `rev` nor `out`; the defaulted trail was still the single shared
`quantibbler-breadcrumbs.log`.

### 2026-10-25 — tasks 1.2/1.3: which directory `rev` comes from, and why the launch line moved below the parse

Four choices the spec delta and the decisions above left silent, settled here
rather than left for a reader of the diff:

- **`rev` is read from the DRIVER'S OWN location, never the cwd.** `_resolve_rev()`
  runs `git -C <driver's own dir> rev-parse HEAD`, because the field is the run's
  CODE identity: a fire launched from another directory still belongs to the tree
  its code came from, and that is the tree a post-mortem wants. The outside-a-checkout
  test above is deliberately blind to this choice — it makes BOTH readings (cwd and
  the driver's own dir) fail there — so the literal `unknown` is not evidence for
  either directory, and the choice is recorded here instead of being inferred from a
  green test.
- **The launch line is written AFTER `parser.parse_args()`, and the watchdog thread
  starts BELOW it.** The resolved `out` dir is only known after the parse, and the
  invariant that matters is "the launch line is line 1 of its trail" — so the
  watchdog thread start and the `memory.events@start` crumb both moved down with the
  launch line. Leaving the thread started above it would have made the ordering true
  only by the watchdog's 5 s first sleep; writing first and starting the sampler
  afterwards makes it structural. Nothing writes to the trail before the parse, so
  no crumb can beat the launch line now by any path.
- **The pin reads the crumb f-string's own field markers, not the local names.**
  `_launch_line_findings` flattens the `JoinedStr` handed to `_bread` and requires
  ` pid=`, ` argv=`, ` rev=`, ` out=` in that order, so renaming the locals
  (`out_dir`, `rev`) cannot evade the pin while dropping or reordering a field reds
  it. Three poisons ship beside it in
  `test_report_driver_tripwire_rejects_poisoned_drivers`
  (`launch_line_dropped_rev`, `launch_line_dropped_out`,
  `launch_line_fields_reordered`), each asserted against the real driver source.
- **`out_dir` is bound once, as `args.out.resolve()`.** The launch line needs the
  value before the export stage does, so the later `out_dir = args.out` rebind was
  folded into that single binding, and the field records the RESOLVED target the
  requirement names rather than whatever spelling argv carried.

### 2026-10-25 — task 3.1: which interpreter produced which gate number

Three measurement facts this change's landing rests on, recorded because each one
is a way to read a false green off the same tree:

- **`uv run pytest` bare is NOT the gate here.** This checkout's project
  environment is a single shared `/opt/venv` that already carries the extras;
  `uv sync --dry-run` (measured) shows a switch-like bare sync would UNINSTALL 31
  packages — torch, transformers, pyarrow, `quantibbler-kernels` — after which the
  ptq tier is silently ignored and the suite reports a smaller green with no
  failures. The gate forms used are `uv run --no-sync … --extra torch --extra
  kernels --extra hf pytest` (full extras) and the CI recipe's fresh
  `.venv-lean` interpreter (lean), never a sync-shaped bare run.
- **The lean tier's 16 skips are the contract, not noise.** Lean
  (`/opt/data/quantibbler/.venv-lean`: fresh `uv venv` + `pip install -e . pytest
  pyyaml`, torch/transformers/tokenizers proven absent BEFORE any count is
  believed) → `718 passed, 16 skipped`, rc=0, and
  `tools/ci/pytest_summary.py --expect-skipped 16` → `skip discipline holds`. The
  16 are `tests/test_kernels_parity.py`'s Rust leg (extension not importable
  without the kernels extra).
- **A full-tier green number cannot exist on the dev interpreter.** With the full
  extras (`/opt/venv`, torch 2.14.1+cpu, no accelerator) the tier reads `968
  passed, 14 failed, 3 skipped`; all 14 are the fail-closed `resolve_device`
  refusal in device-unpinned files, and an A/B on a pristine-HEAD worktree with
  the same interpreter reproduces exactly those 14 in 4.66 s — an interpreter ×
  device property, not this change and not this tree's diff. The number that can
  be green belongs to the accelerator image the fire runs in
  (`quantibbler-ptq:rocm-official`); it was not launched in this lane (container
  launches are announced first under AGENTS.md, and the box was co-tenant
  saturated), and it is left as row 3.2 rather than papered over.
