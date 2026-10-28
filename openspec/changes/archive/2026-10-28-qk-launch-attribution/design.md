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

### 2026-10-25 — launch ownership: where the refusal lives, and why the two gates demand different things

Incident: at 18:10:32Z an unattributed paper-faithful fire ran on this box
(container `quantibbler-ptq-run`, image `quantibbler-ptq:rocm-official`, launch
id `qk-pertarget-v4-20261025T1815Z`). It carried no owner stamp, no
announcement, no GO, and it bypassed the launch-attribution infrastructure
entirely — a direct `docker run` never touches the wrapper. Six choices the
card and the requirements left open, settled here rather than left for a reader
of the diff:

- **Two gates, because one cannot see the other's launch.** The WRAPPER refuses
  an ownerless fire on the host, before the name is taken and before 32g is
  spent; the DRIVER refuses it from inside the container, which is the only
  place a direct `docker run` can be stopped at all. Neither is redundant: the
  wrapper's refusal is invisible to a bypass, and the driver's is invisible to a
  launcher that never reaches the host-side checks. Both refusals are named,
  and both leave nothing attributable behind (the wrapper resolves no argv; the
  driver writes no trail).
- **The wrapper demands owner AND card; the driver demands the owner.** A board
  fire is born in the wrapper, so that is where the owning kanban card is a real
  fact and is required. The driver is also run by things that legitimately own
  no card — a CI smoke leg, a local diagnostic, the test suite spawning a
  subprocess — and demanding a card there would force a placeholder value, which
  is how a required field rots into ceremony. So the driver requires the owner
  and records the card as the literal `unset`, and the launch line always
  carries both MARKERS: an absent field reads as "not measured", which is the
  shape this change exists to keep out of a trail. CI sets
  `QKB_LAUNCH_OWNER=ci-harness`; the tier's driver-spawning legs set `pytest`;
  the two delegated fires set both variables (the exact line is in the card's
  completion comment).
- **The refusal lives OUTSIDE `main()`.** `main()` has exactly one exit, the
  verdict path, and `_driver_wiring_findings` flags any `raise SystemExit` or
  `sys.exit()` inside it — correctly, because a run that exits outside the
  verdict writes no report. `_launch_owner()` therefore sits beside
  `_resolve_device()`, the file's existing pre-flight refusal, and is named as a
  pre-flight rather than folded into the pin's exception list. It is called
  before the launch-line write, so a refused run is not recorded as an anonymous
  one.
- **Owner evidence is env, and labels are not readable from inside a container.**
  The incident ruling named the container LABELS (`qkb_launch_owner` /
  `qkb_launch_card`), but docker never shows a container its own labels — there
  is no channel from a label to the process inside it. So the label vocabulary
  is accepted as ENV under its own spellings (`qkb_launch_owner`,
  `QKB_LAUNCH_LABEL_OWNER`, `QKB_LAUNCH_CARD`, `qkb_launch_card`,
  `QKB_LAUNCH_LABEL_CARD`) and the wrapper mirrors the contract variables into
  the container, which is what makes a wrapper-launched fire compliant at both
  gates. That is a tolerance with a hard edge, and it is named as one: a direct
  `docker run` MUST pass `-e QKB_LAUNCH_OWNER=…`, because the labels alone reach
  the process nowhere. The spelling tolerance exists so a launcher that read the
  ruling rather than this contract is not hard-refused mid-program; it is the
  same evidence under a second name, never a second, weaker contract.
- **The owner line is machine-written at fire time, in three surfaces the
  post-mortem already reads**: the trail's first line (`owner=… card=…`, in the
  identity order `pid, argv, rev, out, owner, card`), the wrapper's `=== run …`
  run header (`owner=… card=…` before the closing `===`, so the leading
  `=== run <ISO> rank=…` shape the watchdog reads is untouched), and `launch.env`
  beside `qkb_launch_id`. The "owner launch" announcement is thereby enforced by
  the launcher instead of remembered by whoever fires — the failure mode of the
  2026-10-20 corpse, restated by this incident.
- **Attribution, not authentication.** Any non-empty value is accepted: the
  contract's job is that a fire cannot be ANONYMOUS. Verifying the card id
  against the board, or refusing on a value the board does not recognise, would
  put a live external dependency on the launch path (and a board outage would
  stop work) for a guarantee this change does not claim. Named as a
  non-goal rather than left to be rediscovered.

RED records, taken against the pre-change bytes at `HEAD` before this cut
(`git show HEAD:<path>` into `/opt/data/tmp/prefix-red/`, so the observation is
of the shipped tree, not of a hand-mutated copy):

- Wrapper, no owner: `DRY_RUN=1 WATCHDOG=0 sh <pre-change wrapper>` → **RC=0**
  and the full `'docker' 'run' …` argv printed — an ownerless fire launches.
- Driver, no owner: `<pre-change driver> --device cpu --model
  /nonexistent-model-xyz --out …` → starts for real and writes the trail's first
  line with **no owner field** (`… rev=unknown out=…`), exiting 1 only because
  the model does not exist.

Post-change the same two commands exit nonzero/exit 2 with
`launch ownership missing`, resolve no argv, and write no trail. The AGENTS.md
card cites a "2026-10-27 precedent" alongside which this paragraph was to land;
the only precedent in `AGENTS.md` at this revision is the 2026-10-20 corpse, so
the paragraph lands there and the discrepancy is reported rather than invented
around.
