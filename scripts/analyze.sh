#!/usr/bin/env bash
# Raw -> processed tables -> figures for a complete local run directory.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
RUN="${1:-full}"
if [ ! -f "data/raw/$RUN/launches.jsonl" ]; then
  echo "data/raw/$RUN/launches.jsonl is missing; the public checkout cannot rerun raw-to-processed analysis." >&2
  exit 2
fi
"$PYTHON" -m ncclenergy analyze --run "data/raw/$RUN" --out "data/processed/$RUN"
"$PYTHON" -m ncclenergy plot --processed "data/processed/$RUN" --out "figures/$RUN"
"$PYTHON" -m ncclenergy report --processed "data/processed/$RUN"
