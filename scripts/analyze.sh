#!/usr/bin/env bash
# Raw -> processed tables -> figures for a run (default: full).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
RUN="${1:-full}"
"$PYTHON" -m ncclenergy analyze --run "data/raw/$RUN" --out "data/processed/$RUN"
"$PYTHON" -m ncclenergy plot --processed "data/processed/$RUN" --out "figures/$RUN"
"$PYTHON" -m ncclenergy report --processed "data/processed/$RUN"
