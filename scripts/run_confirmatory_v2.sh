#!/usr/bin/env bash
# Methodology-v2 confirmatory campaign: N blocks, strictly sequential, stop at the first incomplete block.
#
# v2.2: no external `timeout`. The runner enforces unit timeouts and its watchdog itself, and cleans up every
# owned process on SIGINT/SIGTERM/SIGHUP (src/ncclenergy/v2/procctl.py). This script forwards those signals to
# the runner, waits for its cleanup, and stops the campaign; it never kills anything else.
set -u
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
CAMPAIGN="${CAMPAIGN:-configs/v2/confirmatory.yaml}"
N=$(python3 -c "import yaml;print(yaml.safe_load(open('$CAMPAIGN'))['max_blocks'])")
ALLOC=$(python3 -c "import yaml;print(','.join(map(str, yaml.safe_load(open('$CAMPAIGN'))['allocation'])))")
LOGDIR="${LOGDIR:-logs/v2}"
mkdir -p "$LOGDIR"
RUNNER_PID=""
STOP=0
# optional campaign wall-time budget (v2.2): no block starts unless BLOCK_EST_S fits in the remaining budget; at the
# budget the running block's runner gets SIGTERM (it cleans up its owned processes and records INTERRUPTED).
WALL_BUDGET_S="${WALL_BUDGET_S:-0}"
BLOCK_EST_S="${BLOCK_EST_S:-0}"
T0=$(date +%s)
# our runner is still a running job of this shell (never true for a recycled pid of another process)
running() { [ -n "$RUNNER_PID" ] && jobs -rp | grep -qx "$RUNNER_PID"; }
forward() {
  STOP=1
  if running; then kill -TERM "$RUNNER_PID" 2>/dev/null; fi
}
trap forward INT TERM HUP
rc=0
for b in $(seq 1 "$N"); do
  # cool-down: all allocated GPUs (from the campaign config) below 42 C (read-only temperature query), at most 10 min
  for i in $(seq 1 60); do
    [ "$STOP" -eq 1 ] && break
    hot=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | awk -F', ' -v a="$ALLOC" 'BEGIN{n=split(a,x,",");for(i=1;i<=n;i++)m[x[i]]=1} ($1 in m) && $2>=42' | wc -l)
    [ "$hot" -eq 0 ] && break; sleep 10
  done
  [ "$STOP" -eq 1 ] && { echo "campaign stopped by signal before block $b $(date -u +%FT%TZ)" >> "$LOGDIR/confirmatory_progress.log"; exit 130; }
  if [ "$WALL_BUDGET_S" -gt 0 ] && [ $(( $(date +%s) - T0 + BLOCK_EST_S )) -gt "$WALL_BUDGET_S" ]; then
    echo "campaign stopped: wall budget ${WALL_BUDGET_S}s does not allow block $b $(date -u +%FT%TZ)" >> "$LOGDIR/confirmatory_progress.log"
    exit 3
  fi
  "$PYTHON" -m ncclenergy.v2.runner block --campaign "$CAMPAIGN" > "$LOGDIR/confirmatory_block$b.log" 2>&1 &
  RUNNER_PID=$!
  rc=0
  BUDGET_HIT=0
  while running; do
    if [ "$WALL_BUDGET_S" -gt 0 ] && [ "$BUDGET_HIT" -eq 0 ] && [ $(( $(date +%s) - T0 )) -ge "$WALL_BUDGET_S" ]; then
      BUDGET_HIT=1; STOP=1
      echo "campaign wall budget ${WALL_BUDGET_S}s reached in block $b $(date -u +%FT%TZ)" >> "$LOGDIR/confirmatory_progress.log"
      running && kill -TERM "$RUNNER_PID" 2>/dev/null
    fi
    sleep 1
  done
  wait "$RUNNER_PID"; rc=$?
  RUNNER_PID=""
  echo "block $b exit $rc $(date -u +%FT%TZ)" >> "$LOGDIR/confirmatory_progress.log"
  [ "$STOP" -eq 1 ] && exit 130
  if [ "$rc" -ne 0 ]; then break; fi
done
# the first failing block's runner exit code, or 0 when every block completed (v2.2: was the status of the last
# `[ ... ] && break` test, i.e. 1 after a fully successful campaign)
exit "$rc"
