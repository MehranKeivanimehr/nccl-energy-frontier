"""Shared constants and provenance helpers (no NVML, no GPU)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
UNAVAILABLE = "UNAVAILABLE_DUE_TO_PERMISSIONS"

# NVML return codes that matter for the energy counter
NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4

COLL_FUNC = {"all_reduce": "AllReduce", "all_gather": "AllGather", "reduce_scatter": "ReduceScatter"}
HAS_OP = {"all_reduce": True, "all_gather": False, "reduce_scatter": True}


def norm_pci(bus_id: str) -> str:
    """Canonical PCI bus id: NVML '00000000:81:00.0' and CUDA '0000:81:00.0' -> '0000:81:00.0'."""
    b = bus_id.strip().lower()
    parts = b.split(":")
    if len(parts) == 3:
        dom, bus, devfn = parts
        return f"{int(dom, 16):04x}:{bus}:{devfn}"
    return b


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def gpu_alias(uuid: str) -> str:
    """Stable, non-reversible alias for a GPU UUID."""
    return "A6000-" + sha256_bytes(uuid.encode())[:8] if uuid else "unknown"


_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def redact(text: str) -> str:
    """Replace raw GPU UUIDs with their stable aliases."""
    return _UUID.sub(lambda m: gpu_alias(m.group(0)), text)


def git_provenance(repo: Path = REPO) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""
    diff = run("diff", "HEAD")
    untracked = run("ls-files", "--others", "--exclude-standard")
    return {"head": run("rev-parse", "HEAD").strip() or None,
            "dirty": bool(diff.strip() or untracked.strip()),
            "diff_sha256": sha256_bytes(diff.encode()) if diff.strip() else None,
            "untracked": [x for x in untracked.splitlines() if x][:50]}


def append_jsonl(path: Path, rec: dict) -> None:
    """Append one record and fsync; never rewrites earlier lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = redact(json.dumps(rec, default=_default, sort_keys=False))
    if "\n" in line:
        raise ValueError("record serialized with a newline")
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> tuple[list[dict], bool]:
    """Return (records, partial_last_line). A malformed line other than the last raises."""
    path = Path(path)
    if not path.exists():
        return [], False
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    out, partial = [], False
    for i, ln in enumerate(lines):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                partial = True
                break
            raise
    return out, partial


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
    except ImportError:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)
