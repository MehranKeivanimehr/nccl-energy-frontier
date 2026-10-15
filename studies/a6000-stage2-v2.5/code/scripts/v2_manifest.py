#!/usr/bin/env python
"""SHA-256 manifest of all methodology-v2 data (raw, processed, figures) and the frozen inputs.

    python scripts/v2_manifest.py provenance/v2_raw_sha256_<date>.txt
Refuses to overwrite an existing manifest. Paths are relative to the repository root.
"""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = ["data/v2/raw", "data/v2/processed", "figures/v2", "docs/v2/prereg.json", "configs/v2",
         "patches", "native/nccl_profiler_algo/plugin.c", "native/nccl_tuner_example/plugin.c"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = REPO / sys.argv[1]
    if out.exists():
        raise SystemExit(f"{out} exists")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    files = []
    for r in ROOTS:
        p = REPO / r
        files += [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    lines = [f"# SHA-256 of methodology-v2 data, computed {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
             f"at git HEAD {head} (working tree may include uncommitted files listed here)",
             f"# Roots: {', '.join(ROOTS)}. {len(files)} files. Paths relative to the repository root."]
    lines += [f"{sha(p)}  {p.relative_to(REPO)}" for p in files]
    out.write_text("\n".join(lines) + "\n")
    (out.parent / (out.name + ".sha256")).write_text(f"{sha(out)}  {out.relative_to(REPO)}\n")
    print(f"{len(files)} files -> {out.relative_to(REPO)}; manifest sha256 {sha(out)}")


if __name__ == "__main__":
    main()
