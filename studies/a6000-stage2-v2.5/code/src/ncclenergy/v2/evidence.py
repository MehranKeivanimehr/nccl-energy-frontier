"""Actual NCCL configuration evidence from the nccle profiler plugin."""

from __future__ import annotations

import json
from pathlib import Path

ALGO_NAMES = {"ring": "RING", "tree": "TREE"}
PROTO_NAMES = {"simple": "SIMPLE", "ll": "LL", "ll128": "LL128"}
# NCCL re-types some collectives internally (AllGather runs as ncclInt8 with count x elsize), so
# collectives are matched on bytes = count x datatype size, not on element counts.
DTYPE_BYTES = {"ncclInt8": 1, "ncclUint8": 1, "ncclFloat16": 2, "ncclBfloat16": 2, "ncclFloat8e4m3": 1,
               "ncclFloat8e5m2": 1, "ncclInt32": 4, "ncclUint32": 4, "ncclFloat32": 4, "ncclInt64": 8,
               "ncclUint64": 8, "ncclFloat64": 8}


def load_profiler_dir(d: Path) -> tuple[list[dict], list[str]]:
    files, flags = [], []
    for p in sorted(Path(d).glob("prof_*.json")):
        try:
            files.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            flags.append(f"PROFILER_FILE_UNREADABLE:{p.name}")
    return files, flags


def verify(files: list[dict], func: str, count: int, nranks: int, req_algo: str, req_proto: str,
           min_expected: int, exact_expected: int | None = None, elem_bytes: int = 4) -> dict:
    """Decide VERIFIED/UNVERIFIED for one process.

    func: NCCL function name ("AllReduce", ...); count: per-rank element count of the benchmark
    (nccl-tests' count column) and elem_bytes its element size; req_algo/req_proto: requested values
    ("default" = not set);
    min_expected: collectives of this size that must have run per rank (>= R);
    exact_expected: warm-up + R + data-check, reported (not required) if given.
    """
    reasons: list[str] = []
    if not files:
        return {"status": "UNVERIFIED", "reasons": ["NO_PROFILER_FILE"], "actual": None}
    ranks = sorted(f["rank"] for f in files)
    if ranks != list(range(nranks)):
        reasons.append("RANK_FILES_MISSING")
    actual = set()
    per_rank_n = {}
    for f in files:
        if f.get("overflow"):
            reasons.append("PROFILER_OVERFLOW")
        nbytes = int(count) * int(elem_bytes)
        hits = [c for c in f["colls"] if c["func"] == func and DTYPE_BYTES.get(c["datatype"], -1) > 0
                and int(c["count"]) * DTYPE_BYTES[c["datatype"]] == nbytes]
        if any(c["func"] == func and c["datatype"] not in DTYPE_BYTES for c in f["colls"]):
            reasons.append("UNKNOWN_DATATYPE")
        per_rank_n[f["rank"]] = sum(c["n"] for c in hits)
        for c in hits:
            actual.add((c["algo"].upper(), c["proto"].upper(), c["nChannels"]))
    if any(n < min_expected for n in per_rank_n.values()):
        reasons.append("COUNT_BELOW_R")
    algos = {a for a, _, _ in actual}
    protos = {p for _, p, _ in actual}
    if len(algos) > 1 or len(protos) > 1:
        reasons.append("MIXED_CONFIG")
    if not actual:
        reasons.append("NO_MATCHING_COLLECTIVES")
    if req_algo != "default" and algos and algos != {ALGO_NAMES.get(req_algo.lower(), req_algo.upper())}:
        reasons.append("FORCED_ALGO_NOT_USED")
    if req_proto != "default" and protos and protos != {PROTO_NAMES.get(req_proto.lower(), req_proto.upper())}:
        reasons.append("FORCED_PROTO_NOT_USED")
    act = sorted(actual)
    out = {"status": "VERIFIED" if not reasons else "UNVERIFIED", "reasons": sorted(set(reasons)),
           "actual": [{"algo": a, "proto": p, "nChannels": ch} for a, p, ch in act],
           "actual_key": "|".join(f"{a}/{p}/{ch}" for a, p, ch in act) or None,
           "per_rank_n": per_rank_n}
    if exact_expected is not None:
        out["n_matches_expected"] = all(n == exact_expected for n in per_rank_n.values())
    return out
