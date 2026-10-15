#!/usr/bin/env bash
# Full sweep. Resumable: re-running the same command skips launches already
# recorded in data/raw/full/launches.jsonl.
set -euo pipefail
echo "Historical v1 GPU launcher disabled; analyze retained data instead." >&2
exit 2
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
mkdir -p logs
"$PYTHON" -m ncclenergy run --config configs/full.yaml 2>&1 | tee -a logs/full_run.log
