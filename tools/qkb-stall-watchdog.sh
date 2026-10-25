#!/bin/sh
# qkb-stall-watchdog — host-side liveness watchdog for a quantibbler PTQ run.
#
# Why host-side: a process that wedges at the memory ceiling silences its own
# in-process breadcrumb writer — an in-process
# watchdog shares fate with the thing it watches. This runs OUTSIDE the
# container: it reads the breadcrumbs file the driver appends to and asks
# the docker daemon whether the container is still alive.
#
# Stall predicate: no new line of ANY species (stage marker or ~5 s memory
# heartbeat) in --stall-min minutes. Stage markers alone legitimately gap
# ~21 min (measured 1294 s); heartbeats do not, so total
# line-silence is the honest signal. Default 15 min = ~180 heartbeat
# intervals of margin, and still converts a 4.8 h silent hang into a
# 15-minute alarm.
#
# ALERT-ONLY by default. Killing a 20 h run is an operator decision, not a
# heuristic's; alerts land in the breadcrumbs file itself (the forensic tail
# a forensic tail reads) and in --log. --kill-after-min N opts into
# `docker stop` after N minutes of sustained post-alert silence.
#
# Self-identifies at start (argv, pid, thresholds) per the crash-trail rule:
# a watchdog whose own state cannot be tied to its launch is attribution
# noise, not instrumentation. Fails loud, never exits silently: a missing or
# unparseable breadcrumbs file WHILE the container runs is a probe fault and
# is alerted as such, not swallowed.
set -u

PROG=qkb-stall-watchdog
BC=/opt/data/tmp/quantibbler-breadcrumbs.log
CONTAINER=quantibbler-ptq-run
STALL_MIN=15
INTERVAL=30
KILL_AFTER_MIN=0
# 90% of the wrapper's --memory 32g ceiling; advisory early-warning line.
# Memory alert threshold, calibrated from the recorded breadcrumb log rather
# than from a fraction of the ceiling:
#   healthy peak at the paper geometry (seq 2048, tune 2048, faithful): 17.86G
#   next-highest healthy observation (seq 512):                         18.38G
#   30.14G and 31.07G are NOT levels. Both are runs escalating to death at
#   seq 2048 -- the log ends, or jumps to an unrelated run, at the top of the
#   ramp (6.21 -> 12.07 -> 22.18 -> 31.07G inside one TuneFP fire). Reading
#   them as normal would push the alert above the real 32G ceiling.
# 30.5G sits above every healthy observation and below the ceiling, so a trip
# means something new. The actionable channel stays stall/silence: a true
# memory breach ends as a cgroup kill, and a killed container reads as silence.
MEM_ALERT_GB=30.5
# Seconds after watchdog start before container-absence or empty-file states
# are trusted: the container takes seconds to create+run and ~a minute to
# emit its first breadcrumb line.
PROBE_GRACE=120
LOG=

while [ $# -gt 0 ]; do
  case "$1" in
    --breadcrumbs)     BC=$2; shift 2 ;;
    --container)       CONTAINER=$2; shift 2 ;;
    --stall-min)       STALL_MIN=$2; shift 2 ;;
    --interval)        INTERVAL=$2; shift 2 ;;
    --kill-after-min)  KILL_AFTER_MIN=$2; shift 2 ;;
    --mem-alert-gb)    MEM_ALERT_GB=$2; shift 2 ;;
    --probe-grace)     PROBE_GRACE=$2; shift 2 ;;
    --log)             LOG=$2; shift 2 ;;
    -h|--help)         sed -n '2,31p' "$0"; exit 0 ;;
    *) echo "$PROG: unknown arg: $1" >&2; exit 2 ;;
  esac
done

START=$(date +%s)
STALL_SEC=$((STALL_MIN * 60))
KILL_SEC=$((KILL_AFTER_MIN * 60))
STALL_ALERTED=0
STALL_SINCE=0
MEM_ALERTED=0
FAULT_ALERTED=0
NOT_RUNNING_SEEN=0

note() { # note <text> — human/watchdog-log channel
  if [ -n "$LOG" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $PROG[$$]: $1" >> "$LOG" 2>/dev/null
  fi
  echo "$PROG: $1" >&2
}

inject() { # inject <kind> <text> — becomes part of the breadcrumbs forensic tail
  printf '%s :: WATCHDOG %s %s\n' "$(date +%s)" "$1" "$2" >> "$BC" 2>/dev/null
}

note "start: container=$CONTAINER breadcrumbs=$BC stall=${STALL_MIN}m interval=${INTERVAL}s mem_alert=${MEM_ALERT_GB}G kill_after=${KILL_AFTER_MIN}m probe_grace=${PROBE_GRACE}s"

while :; do
  sleep "$INTERVAL"
  age=$(( $(date +%s) - START ))

  # --- container liveness (two consecutive non-running polls to exit) ---
  if docker ps --filter "name=^${CONTAINER}$" --filter status=running -q 2>/dev/null | grep -q .; then
    NOT_RUNNING_SEEN=0
  elif [ "$age" -ge "$PROBE_GRACE" ]; then
    NOT_RUNNING_SEEN=$((NOT_RUNNING_SEEN + 1))
  fi
  if [ "$NOT_RUNNING_SEEN" -ge 2 ]; then
    note "exit: container $CONTAINER not running on 2 consecutive polls after ${age}s — run over, watchdog done"
    inject exit "container-not-running after ${age}s"
    exit 0
  fi

  # --- breadcrumbs liveness (any line species counts) ---
  ts=$(tail -n 60 "$BC" 2>/dev/null | awk '$1 ~ /^[0-9]+$/ { if ($1+0 > m) m = $1+0 } END { print m+0 }')
  if [ "${ts:-0}" -le 0 ]; then
    if [ "$age" -ge "$PROBE_GRACE" ] && [ "$FAULT_ALERTED" -eq 0 ]; then
      FAULT_ALERTED=1
      note "ALERT probe_fault: no parseable breadcrumb line while container runs — file missing/empty/malformed: $BC"
      inject probe_fault "no-parseable-line at ${age}s"
    fi
  else
    gap=$(( $(date +%s) - ts ))
    if [ "$gap" -gt "$STALL_SEC" ]; then
      if [ "$STALL_ALERTED" -eq 0 ]; then
        STALL_ALERTED=1
        STALL_SINCE=$(date +%s)
        note "ALERT stall: no breadcrumb line for ${gap}s (threshold ${STALL_SEC}s) on $CONTAINER"
        inject stall "gap=${gap}s threshold=${STALL_SEC}s"
      fi
      if [ "$KILL_SEC" -gt 0 ] && [ "$(( $(date +%s) - STALL_SINCE ))" -ge "$KILL_SEC" ]; then
        note "ALERT kill: stall sustained ${KILL_SEC}s past alert — docker stop $CONTAINER"
        inject kill "sustained-stall ${KILL_SEC}s"
        docker stop "$CONTAINER" >/dev/null 2>&1
      fi
    fi
    # --- memory ceiling early warning (advisory, re-arms when it recedes) ---
    peak=$(tail -n 60 "$BC" 2>/dev/null | grep -o 'peak=[0-9.]*G' | tail -n 1 | tr -dc '0-9.')
    if [ -n "$peak" ]; then
      if awk -v p="$peak" -v t="$MEM_ALERT_GB" 'BEGIN { exit !(p >= t) }'; then
        if [ "$MEM_ALERTED" -eq 0 ]; then
          MEM_ALERTED=1
          note "ALERT memory: cgroup peak ${peak}G >= ${MEM_ALERT_GB}G on $CONTAINER"
          inject memory "peak=${peak}G threshold=${MEM_ALERT_GB}G"
        fi
      else
        MEM_ALERTED=0
      fi
    fi
  fi
done
