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
# Root-cause ablation knobs: STRICT PASSTHROUGH, exactly like TUNE_SEQ_LEN —
# empty means UNSET, so the driver's own default applies. The wrapper never
# authors a measurement geometry or mode: a launch that claims to be an fp rung
# must have said so.
FACTOR_MODE="${FACTOR_MODE:-}"
ABLATION="${ABLATION:-}"
LAYER_SCOPE="${LAYER_SCOPE:-}"
# Paper-faithful claim + the bounded discrete flip pass: STRICT PASSTHROUGH
# like the ablation knobs, and
# the wrapper never authors either. PAPER_FAITHFUL=1 is the ONLY way a launch
# claims the paper's method: it switches the stage budgets to [NQ] App. C and
# makes a TuneFP vacuity event FAIL the run instead of continuing. An empty or
# 0 value is the gate-shaped pipeline, which is not the paper.
PAPER_FAITHFUL="${PAPER_FAITHFUL:-}"
STE_FLIP_PASS_BUDGET="${STE_FLIP_PASS_BUDGET:-}"
# The data knobs count SEQUENCES, so the defaults are
# the [NQ] protocol's own numbers: 128 calibration / 40 eval sequences. The
# packed-batch size is the driver's business, not the launcher's.
CALIBRATION_SEQS="${CALIBRATION_SEQS:-128}"
EVAL_SEQS="${EVAL_SEQS:-40}"
DRY_RUN="${DRY_RUN:-0}"
# Lane 0 modes (lane0-rehearsal, lane0-fire) are retired: the harness is gone.
MODE="${MODE:-ptq}"

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

# --- launch ownership: an unattributable fire is refused, not announced --------
# A fire with no owner stamp, no announcement and no GO is not
# attributable after the fact, and nothing in the launch path would say whose it
# was. It is REFUSED here, on the host, before the name is taken and before 32g
# is spent; the driver refuses the same launch from inside
# (tools/quantize_hf_model.py::_launch_owner), which is what closes the
# direct-`docker run` path this wrapper never sees.
#
# The contract is QKB_LAUNCH_OWNER (who is launching) + QKB_LAUNCH_CARD (the
# kanban card that owns the fire). The label-named spellings are the SAME
# evidence, not a second contract: docker never shows a container its own labels,
# so the launched process can only carry that vocabulary as env. Read by the
# wrapper, stamped below as labels AND mirrored into the container env, so the
# driver's gate sees the same evidence -- this script is the launcher's
# owner-of-record, never a place to author one.
OWNER="${QKB_LAUNCH_OWNER:-${QKB_LAUNCH_LABEL_OWNER:-${qkb_launch_owner:-}}}"
CARD="${QKB_LAUNCH_CARD:-${QKB_LAUNCH_LABEL_CARD:-${qkb_launch_card:-}}}"
if [ -z "$OWNER" ] || [ -z "$CARD" ]; then
  echo "error: launch ownership missing - refusing to launch an unattributable fire" >&2
  echo "       set QKB_LAUNCH_OWNER=<who is launching> and QKB_LAUNCH_CARD=<kanban card id>" >&2
  echo "       (label spellings qkb_launch_owner / qkb_launch_card carry the same evidence)" >&2
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
# Image identity is run identity. `quantibbler-ptq:rocm` and
# `quantibbler-ptq:latest` are different images that coexist under two names, so
# GPU=1 against the default tag would hand ROCm hardware a CUDA-wheel image --
# the exact failure the rocm Dockerfile's own build assertion exists to make
# unbuildable. Required, not defaulted: a wrong tag is refused before the launch,
# because the run's record cannot repair a wrong binary after the fact.
IMAGE="${IMAGE:-quantibbler-ptq:latest}"
if [ "${GPU:-0}" = "1" ]; then
  case "$IMAGE" in
    *rocm*) ;;
    *) echo "error: GPU=1 requires a rocm-tagged image (IMAGE=$IMAGE); refusing to hand ROCm devices to a non-ROCm torch" >&2; exit 2 ;;
  esac
fi
# Only MODE=ptq remains. The lane0 modes were retired with their harness
# (e5c6939), and are now refused by name so a stale launcher learns why.
case "$MODE" in
  ptq) ;;
  lane0-rehearsal|lane0-fire)
    echo "error: MODE retired: $MODE — lane0 harness deleted at e5c6939; only MODE=ptq remains" >&2
    exit 2 ;;
  *) echo "error: MODE=$MODE unknown (only ptq)" >&2; exit 2 ;;
esac
CONTAINER_NAME="quantibbler-ptq-run"
# The digest, not just the tag: a tag is a mutable name, so a rebuild changes the
# bytes while the name stays put -- the same "two arms indistinguishable" gap as
# compute_dtype and rho_end, only quieter. Unknown is stated rather than implied
# (a DRY_RUN has no image to inspect).
IMAGE_DIGEST="$(docker inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
[ -n "$IMAGE_DIGEST" ] || IMAGE_DIGEST=unknown

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
  "compute_dtype=${COMPUTE_DTYPE:-bf16} gpu=${GPU:-0} image=$IMAGE image_digest=$IMAGE_DIGEST" \
  "eval_seqs=$EVAL_SEQS out=$OUT owner=$OWNER card=$CARD" >&2

mkdir -p "$OUT" 2>/dev/null || true
# Launch provenance, written into the artifact directory BEFORE the container
# starts. The summary above goes to stderr, and stderr does not travel with the
# artifact -- a run's provenance has to outlive the terminal it was launched
# from. `rho_end_override` is here for the same reason the report records
# rho_end_source: an attribution pass must be identifiable from its own
# directory, not from somebody's scrollback.
{
  printf 'model=%s\nrank=%s\nsolver=%s\nseq_len=%s\ntune_seq_len=%s\n' \
    "$MODEL" "$RANK" "$SOLVER" "$SEQ_LEN" "${TUNE_SEQ_LEN:-}"
  printf 'calibration_seqs=%s\neval_seqs=%s\nfactor_mode=%s\nablation=%s\n' \
    "$CALIBRATION_SEQS" "$EVAL_SEQS" "${FACTOR_MODE:-}" "${ABLATION:-}"
  printf 'layer_scope=%s\npaper_faithful=%s\nste_flip_pass_budget=%s\ncompute_dtype=%s\n' \
    "${LAYER_SCOPE:-}" "${PAPER_FAITHFUL:-0}" "${STE_FLIP_PASS_BUDGET:-0}" "${COMPUTE_DTYPE:-bf16}"
  # SOLVER_BACKEND is stamped beside the rho override for the same reason: the
  # solver arm is run identity, and "torch" alone does not say which device ran.
  # Unset resolves to the config default (numpy); the driver re-resolves and
  # reports the effective value in the run report and solver-probe rows.
  printf 'rho_end_override=%s\nsolver_backend=%s\nqk_admm_reg_override=%s\nsolver_probe=%s\nopenblas_num_threads=%s\nlaunched_at=%s\n' \
    "${QK_RHO_END:-unset}" "${SOLVER_BACKEND:-unset}" "${QK_ADMM_REG:-unset}" "${QK_SOLVER_PROBE:-unset}" "${OPENBLAS_NUM_THREADS:-driver-sized}" "$(date -u +%FT%TZ)"
  # The commit and the tree state are part of the run's identity: a run
  # launched from a dirty tree cannot be reproduced from its SHA, which is the
  # whole reason commit-then-fire is a rule here. `grep -vc '^??'` counts
  # tracked modifications; untracked files (the owner-held wordlist) are
  # reported by their presence separately and are not dirt.
  printf 'gpu=%s\nimage=%s\nimage_digest=%s\ncommit=%s\ntree_dirty_files=%s\n' \
    "${GPU:-0}" "$IMAGE" "$IMAGE_DIGEST" \
    "$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    "$(git -C "$REPO" status --porcelain 2>/dev/null | grep -vc '^??' || true)"
  # Who owns the fire, recorded with the rest of the run's identity: the owner
  # line has to outlive the terminal it was launched from, exactly like the
  # commit and the image digest. The owner is required (refused above); the card
  # is required here too, because this is the board's launch path.
  printf 'owner=%s\ncard=%s\n' "$OWNER" "$CARD"
  printf 'mode=%s\n' "$MODE"
} > "$OUT/launch.env" 2>/dev/null || echo "warning: could not write $OUT/launch.env" >&2
# Pre-launch host free-RAM baseline, beside the artifact: the fence decision
# reads the kfd-visible peak against what the host actually had going in.
{ echo "--- free -g baseline ---"; free -g; } >> "$OUT/launch.env" 2>/dev/null || true

# --- launch identity: a name is shared, an id is not ---------------------------
# The container name is fixed per mode, so every fire reuses it. Teardown by name
# is therefore a claim about *a* container, not about *this* one: a stale
# wall-stop timer or mem-floor net from a dead fire can docker-stop a later fire
# that inherited the freed name. Near-miss on record -- a test's orphan held the
# production name and `docker stop` printed the name without error, and no one
# could say from the outside which container it had actually stopped.
#
# So: stamp an id on the container and into launch.env, and make every
# teardown-by-name verify the id before stopping. A held name is REFUSED, never
# auto-removed -- a container someone else started is not ours to reap, and
# "helpfully" removing it is how a live fire gets killed by a bystander.
LAUNCH_ID="${QKB_LAUNCH_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
_holder="$(docker inspect --format '{{.Config.Image}} running={{.State.Running}} launch_id={{index .Config.Labels "qkb_launch_id"}}' "$CONTAINER_NAME" 2>/dev/null || true)"
if [ -n "$_holder" ]; then
  echo "error: container name $CONTAINER_NAME is already held: $_holder" >&2
  echo "       refusing to launch; a held name is not ours to reap -- inspect it and stop it deliberately" >&2
  exit 2
fi

set -- docker run --rm \
  --name "$CONTAINER_NAME" \
  --memory 32g --memory-swap 32g \
  --cpus 24 \
  -v "$REPO:/work:ro" \
  -v /opt/data/hf-cache:/hf \
  -v /opt/data/tmp:/tmp/ptq \
  -v "$OUT:/out" \
  --label qkb_launch_id="$LAUNCH_ID" \
  --label qkb_launch_owner="$OWNER" \
  --label qkb_launch_card="$CARD" \
  -e QKB_LAUNCH_ID="$LAUNCH_ID" \
  -e QKB_LAUNCH_OWNER="$OWNER" \
  -e QKB_LAUNCH_CARD="$CARD" \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e QUANTIBBLER_BREADCRUMBS=/tmp/ptq/quantibbler-breadcrumbs.log
# No thread-count variables are passed: the driver sizes its pools at startup
# from the container's visible cores and the host's 1-minute load average (its
# good-guest block, which must run above the BLAS import). A hardcoded
# OMP_NUM_THREADS=16 on a shared 48-core host is the recorded anti-pattern -- it
# left 0 free cores and drove the host's load average to 47-70 during a
# BLAS-heavy solve. `--max-threads` on the driver is the operator's explicit cap.
# QK_RHO_END is a DIAGNOSTIC override (lb_admm._rho_end) and a DOCKER flag, so
# it must precede the image name. Appended after the image it lands in the
# container's command argv, where the driver rejects it as an unknown option --
# and a presence-grep of the command line does not catch that, only position
# does. Only-when-set: an unset knob cannot author the claim.
if [ -n "${QK_RHO_END:-}" ]; then
  set -- "$@" -e "QK_RHO_END=$QK_RHO_END"
fi

# DEVICE_AUDIT turns on the call-time device auditor (fail-closed, names the
# call site). Same idiom as QK_RHO_END: passed only when set, so a run that does
# not ask for it is unaffected -- the campaign fire runs with it off.
if [ -n "${DEVICE_AUDIT:-}" ]; then
  set -- "$@" -e "QUANTIBBLER_DEVICE_AUDIT=$DEVICE_AUDIT"
fi
set -- "$@" -e "QUANTIBBLER_IMAGE=$IMAGE" -e "QUANTIBBLER_IMAGE_DIGEST=$IMAGE_DIGEST"

# GPU access is opt-in and explicit. These are host kernel devices, so passing
# them into a container is a privilege decision -- never a silent default, and
# never inferred from the image being a ROCm one. The GIDs are read off the
# device nodes rather than named: this host has no `render` group (kfd and
# renderD128 are GID 993, card1 is video 44), so matching GIDs is the portable
# form and needs no host user change and no reboot.
#
# The presence check binds REAL launches, and DRY_RUN keeps its documented
# reading: a dry run inspects a launch without executing it, so on a host with
# no accelerator nodes it prints the plan (minus --group-add, whose gid is only
# readable off the nodes) and says the real launch refuses here -- the same shape
# the unresolvable-digest path above already uses. Measured: a GPU-less ubuntu
# runner (no /dev/kfd, /dev/dri) reddened every GPU=1 guard leg with
# "GPU=1 but /dev/kfd is absent on this host" before any argv was printed, which
# is the DRY_RUN seam being unable to inspect a launch it does not run.
if [ "${GPU:-0}" = "1" ]; then
  _gpu_absent=""
  for node in /dev/kfd /dev/dri; do
    [ -e "$node" ] || _gpu_absent="$_gpu_absent $node"
  done
  if [ -z "$_gpu_absent" ]; then
    set -- "$@" --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)"
  elif [ "${DRY_RUN:-0}" = "1" ]; then
    echo "note: DRY_RUN without$_gpu_absent - the real launch refuses here (GPU=1 needs the accelerator devices); the argv below is the plan for a GPU host and carries no --group-add, whose gid is read off the device nodes" >&2
    set -- "$@" --device /dev/kfd --device /dev/dri
  else
    for node in $_gpu_absent; do
      echo "error: GPU=1 but $node is absent on this host" >&2
    done
    exit 2
  fi
fi

# QK_SOLVER_PROBE enables the read-only per-linear diagnostic
# (block_recon._record_solver_probe) that writes the attribution numbers. These
# are docker flags, so they belong here, not after the image.
if [ -n "${QK_SOLVER_PROBE:-}" ]; then
  set -- "$@" -e "QK_SOLVER_PROBE=$QK_SOLVER_PROBE"
fi
if [ -n "${QK_SOLVER_PROBE_PATH:-}" ]; then
  set -- "$@" -e "QK_SOLVER_PROBE_PATH=$QK_SOLVER_PROBE_PATH"
fi
# ...and the container command continues the same argument list.
set -- "$@" \
  "$IMAGE" \
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
# Same rule as TUNE_SEQ_LEN: the wrapper passes the precision only when it was
# asked for, so the wrapper is never the author of the run's numerics. The
# driver default (float64) is what launch.env records when it is unset.
if [ -n "${COMPUTE_DTYPE:-}" ]; then
  set -- "$@" --compute-dtype "$COMPUTE_DTYPE"
fi

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
# The hooks DIRECTORY is resolved by git, never spelled `$REPO_ROOT/.git/hooks`:
# in a git LINKED WORKTREE -- which is how every lane on this board is told to
# work -- `.git` is a FILE (a gitdir pointer), so `mkdir -p` on the literal path
# dies with `Not a directory` and `set -eu` takes the whole launch down before
# the dry run can even print argv (measured: nine legs of
# tests/test_run_ptq_wrapper.py red in every worktree while the full clone, where
# `.git` IS a directory, stayed green). `git rev-parse --git-path hooks` is the
# worktree-aware form; it lands in the COMMON dir, which is where git actually
# reads hooks from, so the fence is installed where it is honoured -- once, for
# the clone and every linked worktree of it.
# (core.hooksPath is still deliberately NOT used; see the note above.)
HOOKS_DIR="$(git -C "$REPO_ROOT" rev-parse --git-path hooks 2>/dev/null || true)"
case "$HOOKS_DIR" in
  /*) ;;                                   # absolute (linked worktree: the common dir)
  "") HOOKS_DIR="$REPO_ROOT/.git/hooks" ;; # git unavailable: the full-clone path
  *) HOOKS_DIR="$REPO_ROOT/$HOOKS_DIR" ;;  # relative to the root we asked from
esac
FENCE_DST="$HOOKS_DIR/pre-push"
if [ -r "$FENCE_SRC" ] && { [ ! -f "$FENCE_DST" ] || ! cmp -s "$FENCE_SRC" "$FENCE_DST"; }; then
  mkdir -p "$HOOKS_DIR"
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
# Append a run header instead of truncating. This file is the forensic tail read
# after a killed run, and the previous run's tail is exactly what a kill leaves
# behind: `: >` destroyed it on every launch, including the one that died (`tail`
# from the *failed* run is the useful line, not the new run's first sample).
# The owner line rides the SAME header the forensics tail already reads, in the
# tail's own field syntax, so the announcement of who owns a fire is written by
# the launcher rather than remembered by the person who fired it. The leading
# `=== run <ISO> rank=…` shape is unchanged -- the watchdog's line-silence scan
# and the header greps read it.
printf '=== run %s rank=%s solver=%s seq_len=%s tune_seq_len=%s paper_faithful=%s ste_flip_pass_budget=%s compute_dtype=%s image=%s out=%s owner=%s card=%s ===\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RANK" "$SOLVER" "$SEQ_LEN" "${TUNE_SEQ_LEN:-unset}" "${PAPER_FAITHFUL:-0}" "${STE_FLIP_PASS_BUDGET:-0}" "${COMPUTE_DTYPE:-bf16}" "$IMAGE" "$OUT" "$OWNER" "$CARD" \
  >> /opt/data/tmp/quantibbler-breadcrumbs.log


# Host-side stall watchdog: a run hung at the memory ceiling silences the
# in-process breadcrumb writer along with it, so anything inside the container
# shares that fate. This one runs OUTSIDE, watches for total line-silence in the
# breadcrumbs log
# (heartbeats gap ~5 s; stage markers legitimately gap ~21 min), and injects
# its alerts into that same forensic tail + $OUT/watchdog.log. Alert-only by
# default; WATCHDOG_KILL_AFTER_MIN opts into docker stop after sustained
# stall. WATCHDOG=0 disables. DRY_RUN exited above, so a dry run never
# launches one.
WATCHDOG="${WATCHDOG:-1}"
WATCHDOG_STALL_MIN="${WATCHDOG_STALL_MIN:-15}"
WATCHDOG_KILL_AFTER_MIN="${WATCHDOG_KILL_AFTER_MIN:-0}"
if [ "$WATCHDOG" = "1" ]; then
  # shellcheck disable=SC2094  # deliberate: the redirect forwards the watchdog's console output into the same file its own --log already writes
  nohup sh "$REPO/tools/qkb-stall-watchdog.sh" \
    --breadcrumbs /opt/data/tmp/quantibbler-breadcrumbs.log \
    --container "$CONTAINER_NAME" \
    --stall-min "$WATCHDOG_STALL_MIN" \
    --kill-after-min "$WATCHDOG_KILL_AFTER_MIN" \
    --log "$OUT/watchdog.log" >>"$OUT/watchdog.log" 2>&1 &
  echo "watchdog: pid=$! stall=${WATCHDOG_STALL_MIN}m kill_after=${WATCHDOG_KILL_AFTER_MIN}m log=$OUT/watchdog.log" >&2
fi

# --- container stdout/stderr witness (container_log_witness, spec law) ---------
# Four launches have died with their account destroyed: `--rm` removes the
# container, nothing captured what it printed, and the artifact held a telemetry
# csv and a launch.env and no reason. The telemetry names the *phase*; only
# stdout names the line, and for a fault that lives in the real-model/device
# surface the fixture matrix cannot supply it -- two of the last two real-path
# deaths were green on CPU. Tee on the host side, deliberately: no in-container
# change and no image rebuild, so the log survives `--rm` and teardown stays
# clean.
#
# POSIX sh has no PIPESTATUS. `$?` after a pipeline is *tee's* status, not the
# container's, so the status is captured inside the pipeline into a file and read
# back. Without that the wrapper reports success for every failure -- a witness
# that cannot testify.
CONTAINER_LOG="$OUT/container.log"
RC_FILE="$OUT/.container.rc"
rm -f "$RC_FILE"
# `set -eu` is on, so a bare `"$@"; echo "$?"` never writes the status: the
# failing container kills the subshell before the echo runs, and the wrapper
# then reports `unknown` for a run that failed for a perfectly legible reason.
# Measured, not reasoned -- the missing-image check printed `exit=unknown` and
# the -x trace stopped dead at the docker failure. A command in a `||` list is
# exempt from errexit, which is what makes this the correct POSIX form.
{ rc=0; "$@" || rc=$?; echo "$rc" > "$RC_FILE"; } 2>&1 | tee "$CONTAINER_LOG"
CONTAINER_RC="$(cat "$RC_FILE" 2>/dev/null || true)"
rm -f "$RC_FILE"
case "$CONTAINER_RC" in
  ''|*[!0-9]*) CONTAINER_RC=unknown ;;
esac
{ echo "qkb_launch_id=$LAUNCH_ID"; echo "--- container exit ---"; echo "container_exit_code=$CONTAINER_RC"; echo "container_log=$CONTAINER_LOG"; } >> "$OUT/launch.env" 2>/dev/null || true
# Fail closed, for ANY mode: a run whose witness is absent cannot be read
# after the fact, and "it died with no reason" is the exact failure this lane
# keeps paying for. The `MODE != ptq` qualifier that used to gate this was
# dead twice over -- the law is unconditional (`container_log_witness`) and
# the lane0 modes that motivated it are retired, so `MODE` can only be `ptq`
# and the check never ran. Recorded loudly rather than discovered later.
if [ ! -s "$CONTAINER_LOG" ]; then
  echo "error: container_log_witness missing or empty at $CONTAINER_LOG; the run's account was not captured" >&2
  exit 2
fi
echo "container: exit=$CONTAINER_RC log=$CONTAINER_LOG" >&2
if [ "$CONTAINER_RC" = "unknown" ]; then
  exit 1
fi
exit "$CONTAINER_RC"
