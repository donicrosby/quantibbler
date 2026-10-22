#!/bin/sh
# secret-fence — two-face pre-push secret detector for quantibbler.
#
# Face 1 (tree):   every TRACKED file at HEAD.
# Face 2 (range):  every commit message AND every diff in the outgoing range.
#   The original leak lived in a file AND a commit message body; a tree-only
#   scan sees one face. Both are pinned here.
#
# Patterns: built-in stock literals + a wordlist read from an UNTRACKED file
# (default <repo>/.git-secret-fence-wordlist, override SECRET_FENCE_WORDLIST).
# The wordlist is NEVER committed — a tracked list of site identifiers is
# this repo's original failure class wearing a badge.
#
# Output discipline: hits are reported as file / commit + counts ONLY. A
# fence that echoes the matching line re-publishes the payload into CI logs;
# the three-reproduction rule applies to tooling too. --show prints content
# for local debugging, never in CI.
#
# Exit 0 = clean. Exit 1 = hits. Exit 2 = usage/probe fault.
set -u
PROG=secret-fence
RANGE=
SHOW=0
REPO=.
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)  REPO=$2; shift 2 ;;
    --range) RANGE=$2; shift 2 ;;
    --show)  SHOW=1; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "$PROG: unknown arg: $1" >&2; exit 2 ;;
  esac
done

WL="${SECRET_FENCE_WORDLIST:-$REPO/.git-secret-fence-wordlist}"
PATTERNS='-----BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}'
if [ -f "$WL" ]; then
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    PATTERNS="$PATTERNS|$line"
  done < "$WL"
else
  echo "$PROG: note: no wordlist at $WL (site-specific strings inactive)" >&2
fi

TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT
printf '%s\n' "$PATTERNS" > "$TMP/patterns"

hits=0

# --- Face 1: tracked tree at HEAD (masked listing: file:line only) ---
git -C "$REPO" grep -nE -f "$TMP/patterns" HEAD -- . > "$TMP/tree" 2>/dev/null
tc=$(wc -l < "$TMP/tree")
hits=$((hits + tc))
echo "face1 tree: $tc hit(s)"
if [ "$tc" -gt 0 ]; then
  # Masked by construction: git grep prints REF:path:line:content — print
  # only the first three colon fields, content never survives. (A regex
  # mask that fails open re-publishes the payload; this one cannot.)
  awk -F':' '{ print "  tree-hit " $1 ":" $2 ":" $3 }' "$TMP/tree" | head -10
  [ "$SHOW" = "1" ] && sed -n '1,10p' "$TMP/tree" >&2
fi

# --- Face 2: outgoing range — messages and diffs, per commit, masked ---
if [ -n "$RANGE" ]; then
  for sha in $(git -C "$REPO" rev-list "$RANGE"); do
    if git -C "$REPO" log -1 --format=%B "$sha" 2>/dev/null | grep -qE -f "$TMP/patterns"; then
      echo "  message-hit $sha"
      hits=$((hits + 1))
      [ "$SHOW" = "1" ] && git -C "$REPO" log -1 --format=%B "$sha" | grep -nE -f "$TMP/patterns" >&2
    fi
    if git -C "$REPO" show --format= "$sha" 2>/dev/null | grep -qE -f "$TMP/patterns"; then
      echo "  diff-hit $sha"
      hits=$((hits + 1))
      [ "$SHOW" = "1" ] && git -C "$REPO" show --format= "$sha" | grep -nE -f "$TMP/patterns" | head -5 >&2
    fi
  done
  echo "face2 range $RANGE: cumulative hits included above"
fi

echo "total: $hits hit(s)"
[ "$hits" -gt 0 ] && exit 1
exit 0
