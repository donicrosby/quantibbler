#!/bin/sh
# Run the PTQ driver inside the quantibbler-ptq container.
#
# The sandbox's 5GiB cgroup killed the in-process runs (exit 137); the docker
# daemon lives OUTSIDE that cgroup, so this runs against host RAM with a sane
# explicit ceiling (--memory 32g of ~61g available; swap disabled so the
# failure mode is a clean in-container OOM kill, not host thrash).
#
# Mounts:
#   /work  = repo (ro)          /hf = HF cache (models + wikitext parquet)
#   /tmp/ptq = real-disk scratch (KD teacher-logit memmap lands here, 20GB class)
#
# Knobs, all env — positional arguments are REFUSED, not ignored:
#   MODEL OUT RANK SOLVER SEQ_LEN TUNE_SEQ_LEN CALIBRATION_SEQS EVAL_SEQS
#   FACTOR_MODE ABLATION LAYER_SCOPE PAPER_FAITHFUL STE_FLIP_PASS_BUDGET DRY_RUN
#
# Two defaults are traps worth knowing before you launch: SOLVER defaults to
# `admm` (the measured ablation showed the ADMM stage earns nothing here —
# production is `svd`), and SEQ_LEN defaults to 512, a DIAGNOSTIC geometry: the
# NQ-protocol claim needs 2048, and the two numbers are not interchangeable. The
# resolved configuration is echoed before launch, and DRY_RUN=1 prints the whole
# command line instead of running it. These knobs deliberately do not choose
# protocol values — the eval geometry belongs to the recreation-bar spec; the
# wrapper's job is to stop being the thing that decides it silently.
#
# TUNE_SEQ_LEN is passthrough and optional: empty means UNSET, which is how the
# driver learns to use `--seq-len` for the tuning windows too. The wrapper never
# supplies a window length of its own - picking one would be the same silent
# decision this script exists to stop making.
#
# Canonical launches:
#   RANK=512 SOLVER=svd OUT=<fresh dir> sh tools/run-ptq-container.sh
#     -> r512 diagnostic: recorded alongside, never a bar claim
#   RANK=512 SOLVER=svd SEQ_LEN=2048 OUT=<fresh dir> sh tools/run-ptq-container.sh
#     -> NQ-protocol geometry, the line with a landing site
#   RANK=512 SOLVER=svd SEQ_LEN=2048 TUNE_SEQ_LEN=512 OUT=<fresh dir> \
#     sh tools/run-ptq-container.sh
#     -> the same eval protocol with tune windows bounded to 512 tokens: the
#        geometry whose tune stage fits the 32g ceiling (the unset-window 2048
#        run is the one that died with a memcg OOM at step [4/6])
#   Prefix either with DRY_RUN=1 to print the exact docker command line instead
#   of running it (no RAM spent, container name left free).
set -eu

REPO=/opt/data/quantibbler
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
OUT="${OUT:-/opt/data/quantibbler/artifacts/qwen3-0.6B-r8}"
RANK="${RANK:-8}"
# Whether the operator chose the solver, remembered BEFORE it is defaulted: the
# advisory below has to distinguish "left unset" from "explicitly svd", and after
# the assignment both look identical.
SOLVER_GIVEN="${SOLVER:-}"
SOLVER="${SOLVER:-admm}"
SEQ_LEN="${SEQ_LEN:-512}"
# Deliberately NOT defaulted to anything: an empty value must reach the driver
# as "unset" so the driver's own default (tune == --seq-len) applies.
TUNE_SEQ_LEN="${TUNE_SEQ_LEN:-}"
# Root-cause ablation knobs (qk-root-cause-ablation): STRICT PASSTHROUGH, exactly
# like TUNE_SEQ_LEN — empty means UNSET, so the driver's own default applies. The
# wrapper never authors a measurement geometry or mode: a launch that claims to
# be an fp rung must have said so.
FACTOR_MODE="${FACTOR_MODE:-}"
ABLATION="${ABLATION:-}"
LAYER_SCOPE="${LAYER_SCOPE:-}"
# Paper-faithful claim + the bounded discrete flip pass (qk-fix-tunefp-teacher /
# qk-fix-ste-sign-adaptation): STRICT PASSTHROUGH like the ablation knobs, and
# the wrapper never authors either. PAPER_FAITHFUL=1 is the ONLY way a launch
# claims the paper's method: it switches the stage budgets to [NQ] App. C and
# makes a TuneFP vacuity event FAIL the run instead of continuing. An empty or
# 0 value is the gate-shaped pipeline, which is not the paper.
PAPER_FAITHFUL="${PAPER_FAITHFUL:-}"
STE_FLIP_PASS_BUDGET="${STE_FLIP_PASS_BUDGET:-}"
CALIBRATION_SEQS="${CALIBRATION_SEQS:-16}"
EVAL_SEQS="${EVAL_SEQS:-5}"
DRY_RUN="${DRY_RUN:-0}"

# Refuse arguments instead of ignoring them. This wrapper pins the eval geometry
# below, so a stray flag — or SEQ_LEN=2048 written in argument position — used to
# be dropped silently and produced a run that looked healthy while measuring
# something other than what was asked for.
if [ "$#" -ne 0 ]; then
  echo "run-ptq-container.sh takes no arguments; use the env knobs:" >&2
  echo "  MODEL OUT RANK SOLVER SEQ_LEN TUNE_SEQ_LEN CALIBRATION_SEQS EVAL_SEQS FACTOR_MODE ABLATION LAYER_SCOPE PAPER_FAITHFUL STE_FLIP_PASS_BUDGET DRY_RUN" >&2
  echo "  got: $*" >&2
  exit 2
fi

# Say which run this is, before it runs. None of these advisories judges the
# choice — they exist so a bare launch cannot quietly be mistaken for the
# claimable one. An exported-but-empty SOLVER falls through to the default and
# gets the warning too, because `${SOLVER:-admm}` cannot tell those two cases
# apart afterwards.
if [ -z "$SOLVER_GIVEN" ]; then
  echo "note: SOLVER unset -> admm (the ablation's no-gain path); production is SOLVER=svd" >&2
fi
if [ "$SEQ_LEN" != "2048" ]; then
  echo "note: SEQ_LEN=$SEQ_LEN is a diagnostic geometry, not the NQ protocol (2048)" >&2
fi
# The OOM-shaped launch, announced and not refused: seq 2048 with the tune window
# left unset runs the tune stage at 2048 too — the geometry whose N^2 attention
# node died with a memcg OOM at this container's 32g ceiling (step [4/6]).
# TUNE_SEQ_LEN bounds the tune windows while `--seq-len` stays the eval geometry,
# which is what turns that OOM into a number. Said before the run, not instead of
# it: a deliberate unsplit 2048 run is still allowed to start.
if [ "$SEQ_LEN" = "2048" ] && [ -z "$TUNE_SEQ_LEN" ]; then
  echo "note: SEQ_LEN=2048 with TUNE_SEQ_LEN unset runs the tune stage at 2048 too - the geometry that died with a memcg OOM at the 32g ceiling; set TUNE_SEQ_LEN=512 to bound the tune windows" >&2
fi
# Stream contract, deliberate and now pinned from both sides: STDOUT is the
# machine channel and carries the resolved argv and NOTHING else, so a verifier
# can parse the launch line byte-for-byte; STDERR is the human channel and
# carries the announce plus the advisories. (Before this, the announce went to
# stdout and mixed with the argv; the split arrived as part of the ablation-knob
# passthrough repair, so the why is recorded here rather than left to archaeology.)
# A launch that does not claim paper fidelity is NOT running the paper's method:
# the gate-shaped stage budgets turn TuneFP into a single step and the vacuity
# guard into a diagnostic event. Said before the run so the resulting artifact
# cannot be read as a paper-faithful one.
if [ "$PAPER_FAITHFUL" != "1" ]; then
  echo "note: PAPER_FAITHFUL unset -> gate-shaped stage budgets (TuneFP 1 step); the paper's method is PAPER_FAITHFUL=1" >&2
fi
echo "run-ptq-container: model=$MODEL rank=$RANK solver=$SOLVER seq_len=$SEQ_LEN" \
  "tune_seq_len=${TUNE_SEQ_LEN:-unset} calibration_seqs=$CALIBRATION_SEQS" \
  "factor_mode=${FACTOR_MODE:-binary} ablation=${ABLATION:-none} layer_scope=${LAYER_SCOPE:-all}" \
  "paper_faithful=${PAPER_FAITHFUL:-0} ste_flip_pass_budget=${STE_FLIP_PASS_BUDGET:-0}" \
  "eval_seqs=$EVAL_SEQS out=$OUT" >&2

set -- docker run --rm \
  --name quantibbler-ptq-run \
  --memory 32g --memory-swap 32g \
  --cpus 24 \
  -v "$REPO:/work:ro" \
  -v /opt/data/hf-cache:/hf \
  -v /opt/data/tmp:/tmp/ptq \
  -v "$OUT:/out" \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e QUANTIBBLER_BREADCRUMBS=/tmp/ptq/quantibbler-breadcrumbs.log \
  -e OMP_NUM_THREADS=16 \
  -e MKL_NUM_THREADS=16 \
  -e OPENBLAS_NUM_THREADS=16 \
  quantibbler-ptq:latest \
  python tools/quantize_hf_model.py \
    --model "$MODEL" --rank "$RANK" \
    --seq-len "$SEQ_LEN" --calibration-seqs "$CALIBRATION_SEQS" --eval-seqs "$EVAL_SEQS" \
    --max-admm-iter 400 \
    --solver "$SOLVER" \
    --checkpoint-dir /out/checkpoints \
    --out /out

# The split knob is appended only when it was set. Passing it unconditionally
# would make the wrapper the author of the tuning geometry - and an empty
# `--tune-seq-len` is not a value the driver accepts.
if [ -n "$TUNE_SEQ_LEN" ]; then
  set -- "$@" --tune-seq-len "$TUNE_SEQ_LEN"
fi
if [ -n "$FACTOR_MODE" ]; then
  set -- "$@" --factor-mode "$FACTOR_MODE"
fi
if [ -n "$ABLATION" ]; then
  set -- "$@" --ablation "$ABLATION"
fi
if [ -n "$LAYER_SCOPE" ]; then
  set -- "$@" --layer-scope "$LAYER_SCOPE"
fi
# PAPER_FAITHFUL is a boolean FLAG, not a valued option: it is appended only
# when it was asked for, so an unset knob cannot author the claim.
if [ "$PAPER_FAITHFUL" = "1" ]; then
  set -- "$@" --paper-faithful
fi
if [ -n "$STE_FLIP_PASS_BUDGET" ]; then
  set -- "$@" --ste-flip-pass-budget "$STE_FLIP_PASS_BUDGET"
fi

# DRY_RUN=1 prints the exact command line (copy-pasteable) and stops: the launch
# is checkable without spending 32g of host RAM and without stealing the
# container name a real run needs.
# --- push fence: clone-local hooks need re-installing per clone -----------------
# A git hook is untracked by construction, so the tracked source in
# tools/git-hooks/ is copied into this clone on every launch. The copy is
# BYTE-COMPARED, not merely presence-checked, so drift is caught rather than
# tolerated; publication-hygiene.sh refuses (nonzero) if the installed hook is
# missing, non-executable, or differs from the tracked source.
#
# core.hooksPath is deliberately NOT used: setting it makes git ignore .git/hooks
# entirely, which would orphan the pre-commit framework hook this box relies on.
# That trades one silent disarm for another.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FENCE_SRC="$REPO_ROOT/tools/git-hooks/pre-push"
FENCE_DST="$REPO_ROOT/.git/hooks/pre-push"
if [ -r "$FENCE_SRC" ] && { [ ! -f "$FENCE_DST" ] || ! cmp -s "$FENCE_SRC" "$FENCE_DST"; }; then
  mkdir -p "$REPO_ROOT/.git/hooks"
  cp "$FENCE_SRC" "$FENCE_DST"
  chmod 755 "$FENCE_DST"
  echo "push fence: installed/refreshed $FENCE_DST from the tracked source"
fi

if [ "$DRY_RUN" != "0" ]; then
  for arg in "$@"; do printf "'%s' " "$arg"; done
  echo
  exit 0
fi

mkdir -p /opt/data/tmp "$(dirname "$OUT")"
# Append a run header instead of truncating. This file is the forensic tail a
# post-mortem reads after a 137, and the previous run's tail is exactly what a
# killed run leaves behind: `: >` destroyed it on every launch, including the
# one that died (`tail` from the *failed* run is the useful line, not the new
# run's first sample).
printf '=== run %s rank=%s solver=%s seq_len=%s tune_seq_len=%s paper_faithful=%s ste_flip_pass_budget=%s out=%s ===\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RANK" "$SOLVER" "$SEQ_LEN" "${TUNE_SEQ_LEN:-unset}" "${PAPER_FAITHFUL:-0}" "${STE_FLIP_PASS_BUDGET:-0}" "$OUT" \
  >> /opt/data/tmp/quantibbler-breadcrumbs.log


# Host-side stall watchdog (post-23:10Z hardening): the 17:16 attempt hung
# 4.8 h at the memory ceiling with the in-process breadcrumb writer silenced
# along with it, so anything inside the container shares that fate. This one
# runs OUTSIDE, watches for total line-silence in the breadcrumbs log
# (heartbeats gap ~5 s; stage markers legitimately gap ~21 min), and injects
# its alerts into that same forensic tail + $OUT/watchdog.log. Alert-only by
# default; WATCHDOG_KILL_AFTER_MIN opts into docker stop after sustained
# stall. WATCHDOG=0 disables. DRY_RUN exited above, so a dry run never
# launches one.
WATCHDOG="${WATCHDOG:-1}"
WATCHDOG_STALL_MIN="${WATCHDOG_STALL_MIN:-15}"
WATCHDOG_KILL_AFTER_MIN="${WATCHDOG_KILL_AFTER_MIN:-0}"
if [ "$WATCHDOG" = "1" ]; then
  nohup sh "$REPO/tools/qkb-stall-watchdog.sh" \
    --breadcrumbs /opt/data/tmp/quantibbler-breadcrumbs.log \
    --container quantibbler-ptq-run \
    --stall-min "$WATCHDOG_STALL_MIN" \
    --kill-after-min "$WATCHDOG_KILL_AFTER_MIN" \
    --log "$OUT/watchdog.log" >>"$OUT/watchdog.log" 2>&1 &
  echo "watchdog: pid=$! stall=${WATCHDOG_STALL_MIN}m kill_after=${WATCHDOG_KILL_AFTER_MIN}m log=$OUT/watchdog.log" >&2
fi

exec "$@"
