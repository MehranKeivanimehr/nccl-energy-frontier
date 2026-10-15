#!/usr/bin/env bash
# Pack a raw run directory (all launches, logs, power traces) into one archive.
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN="${1:-full}"
cd "$REPO_ROOT/data/raw"
tar -czf "$RUN.tar.gz" "$RUN"
sha256sum "$RUN.tar.gz" | tee "$RUN.tar.gz.sha256"
