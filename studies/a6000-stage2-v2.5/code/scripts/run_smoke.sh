#!/usr/bin/env bash
# Smoke test: run, analyze, validate, plot. Exit code != 0 if any check fails.
set -euo pipefail
echo "Historical v1 GPU launcher disabled; analyze retained data instead." >&2
exit 2
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"
"$PYTHON" tests/run_tests.py
"$PYTHON" -m ncclenergy run --config configs/smoke.yaml
"$PYTHON" -m ncclenergy smoke-check --run data/raw/smoke --out data/processed/smoke
"$PYTHON" -m ncclenergy plot --processed data/processed/smoke --out figures/smoke
