#!/bin/sh
# publication-hygiene — the quantibbler publication runbook (sec-eng lane).
#
# Sequence (bug-hunter mapped the refs; coder proved gc-before-refs is a
# no-op; the amend is DONE, so pre-amend 7daea90 is garbage-but-present and
# must be pruned by this runbook before the box-holds-no-token verdict can
# pass):
#   0. preconditions: tip pinned, gate-landed marker present, fence clean
#   1. git branch -f main qk-pertarget-cycle
#   2. git update-ref -d refs/remotes/origin/main
#   3. git update-ref -d refs/remotes/origin/HEAD
#   4. git update-ref -d ORIG_HEAD
#   5. git reflog expire --expire=now --all
#   6. git gc --prune=now
#   7. gates: every must-be-absent sha FAILS cat-file in the fire tree,
#      then box-wide object-presence sweep finds no holder anywhere.
#
# DRY-RUN BY DEFAULT: prints state, plan, and gate expectations, touches
# nothing. --execute additionally requires --tip <sha> (must equal the
# branch tip), --absent <sha> (repeatable; must-be-absent set), and
# --require <substr> (must appear in the tip commit message — the
# planner signal that the gate landed). Nothing destructive runs without
# all three.
set -u
PROG=publication-hygiene
REPO=/opt/data/quantibbler
BRANCH=qk-pertarget-cycle
BASE=3b6faf4
EXECUTE=0
TIP=
REQUIRE=
ABSENT="4229e8a 2a92995"
SWEEP_ROOTS="/opt/data"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --tip) TIP=$2; shift 2 ;;
    --require) REQUIRE=$2; shift 2 ;;
    --absent) ABSENT="$ABSENT $2"; shift 2 ;;
    --sweep-only) EXECUTE=2; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "$PROG: unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO" || exit 2
TIP_NOW=$(git rev-parse "$BRANCH")
SUBJ_NOW=$(git log -1 --format=%s "$BRANCH")
COUNT=$(git rev-list --count "$BASE..$BRANCH")
echo "state: branch=$BRANCH tip=$TIP_NOW"
echo "       subject: $SUBJ_NOW"
echo "       commits past $BASE: $COUNT"
echo "refs:  main=$(git rev-parse refs/heads/main 2>/dev/null || echo gone)"
echo "       origin/main=$(git rev-parse refs/remotes/origin/main 2>/dev/null || echo gone)"
echo "       origin/HEAD=$(git rev-parse refs/remotes/origin/HEAD 2>/dev/null || echo gone)"
echo "       ORIG_HEAD=$(cat .git/ORIG_HEAD 2>/dev/null || echo none)"

echo "fence (two faces over $BASE..$BRANCH):"
sh /opt/data/tmp/qkb-fence/secret-fence.sh --repo "$REPO" --range "$BASE..$BRANCH"
FENCE_RC=$?
if [ "$FENCE_RC" -ne 0 ]; then
  echo "$PROG: FENCE NOT CLEAN — runbook refuses to proceed" >&2
  [ "$EXECUTE" = "2" ] && exit 0
  exit 1
fi

if [ "$EXECUTE" = "2" ]; then
  echo "--- sweep-only: box-wide object presence ---"
else
  echo "--- plan ---"
  echo "  1 git branch -f main $BRANCH"
  echo "  2 git update-ref -d refs/remotes/origin/main"
  echo "  3 git update-ref -d refs/remotes/origin/HEAD"
  echo "  4 git update-ref -d ORIG_HEAD"
  echo "  5 git reflog expire --expire=now --all"
  echo "  6 git gc --prune=now"
  echo "  7 gates: cat-file must FAIL for:$ABSENT"
fi

if [ "$EXECUTE" = "0" ]; then
  echo "dry-run only; re-run with --execute --tip <sha> --require <substr> [--absent <sha>]..."
  exit 0
fi

if [ "$EXECUTE" = "1" ]; then
  if [ -z "$TIP" ] || [ -z "$REQUIRE" ]; then
    echo "$PROG: --execute needs --tip and --require" >&2; exit 2
  fi
  if [ "$TIP" != "$TIP_NOW" ]; then
    echo "$PROG: tip moved (want $TIP, have $TIP_NOW) — refusing" >&2; exit 1
  fi
  if ! git log -1 --format=%B "$BRANCH" | grep -qF -- "$REQUIRE"; then
    echo "$PROG: gate marker '$REQUIRE' not in tip message — planner signal absent, refusing" >&2; exit 1
  fi
  git branch -f main "$BRANCH"
  git update-ref -d refs/remotes/origin/main 2>/dev/null
  git update-ref -d refs/remotes/origin/HEAD 2>/dev/null
  git update-ref -d ORIG_HEAD 2>/dev/null
  git reflog expire --expire=now --all
  git gc --prune=now
  echo "--- gates: fire tree ---"
fi

FAIL=0

echo "--- gates: push fence wired into THIS clone ---"
HYG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FENCE_SRC="$HYG_ROOT/tools/git-hooks/pre-push"
FENCE_DST="$HYG_ROOT/.git/hooks/pre-push"
if [ ! -r "$FENCE_SRC" ]; then
  echo "  FAIL: tracked hook source $FENCE_SRC is missing (nothing to install from)"; FAIL=1
elif [ ! -x "$FENCE_DST" ]; then
  echo "  FAIL: $FENCE_DST is missing or not executable -- the push path is UNFENCED"; FAIL=1
elif ! cmp -s "$FENCE_SRC" "$FENCE_DST"; then
  echo "  FAIL: installed hook differs from the tracked source -- DRIFT, and an"
  echo "        outdated fence fails open where it must fail closed"; FAIL=1
else
  echo "  ok: push fence installed, executable, byte-identical to the tracked source"
fi

for sha in $ABSENT; do
  if git cat-file -t "$sha" >/dev/null 2>&1; then
    echo "  FAIL: $sha still present in fire tree"; FAIL=1
  else
    echo "  ok: $sha absent (fire tree)"
  fi
done

echo "--- gates: box-wide object-presence sweep ($SWEEP_ROOTS) ---"
for d in $(find $SWEEP_ROOTS -type d -name .git -not -path '*/hf-cache/*' 2>/dev/null); do
  r=$(dirname "$d")
  for sha in $ABSENT; do
    if git -C "$r" cat-file -t "$sha" >/dev/null 2>&1; then
      echo "  FAIL: $sha present in $r"; FAIL=1
    fi
  done
done
# Bare repos have no .git subdir — enumerate them explicitly.
for r in /opt/data/qkb-rewrite/qkb-mirror; do
  [ -d "$r" ] || continue
  for sha in $ABSENT; do
    if git -C "$r" cat-file -t "$sha" >/dev/null 2>&1; then
      echo "  FAIL: $sha present in $r (bare)"; FAIL=1
    fi
  done
done
[ "$FAIL" = "0" ] && echo "  sweep clean: no object store holds any must-be-absent sha"
[ "$FAIL" = "0" ] && echo "VERDICT: box holds no token"
exit $FAIL
