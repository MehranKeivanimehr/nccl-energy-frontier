"""Profile scope of a measurement campaign, derived only from its recorded raw evidence (methodology v2.2).

For each placement, every compatibility-relevant fact is read from the raw
files and its source is named. A fact that is not recorded, or recorded with
conflicting values, is None and listed under "missing"/"conflicts"; it is never
filled in. The recommender and the tuner export treat a None fact as
incompatible (fall back to NCCL automatic).

Sources:
* GPU model, driver, CUDA driver version: blocks/*/preflight_pre.json (NVML at block start)
* NCCL version string, nRanks/nNodes: units/*/nccl.log (NCCL INFO lines)
* NCCL library version code: units/*/stdout.txt (nccl-tests header "nccl-library=")
* dtype, reduction op, in-place flag: units/*/region.jsonl (patched nccl-tests region record)
* fingerprint, placement env, PCI bus ids, ranks: units/*/unit.json "placement"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

NCCL_TESTS_TYPE = {"float": "float32", "half": "float16", "bfloat16": "bfloat16", "double": "float64",
                   "int8": "int8", "uint8": "uint8", "int32": "int32", "uint32": "uint32"}
NCCL_OP = {0: "sum", 1: "prod", 2: "max", 3: "min", 4: "avg"}
SCOPE_FIELDS = ("gpu_model", "driver_version", "cuda_driver_version", "nccl_version", "nccl_version_code",
                "nranks", "nnodes", "fingerprint", "env", "pci_bus_ids", "dtype", "in_place")
RE_VER = re.compile(r"NCCL version (\S+)")
RE_NODES = re.compile(r"nRanks (\d+) nNodes (\d+)")
RE_LIB = re.compile(r"nccl-library=(\d+)")


def env_string(env: dict | None) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted((env or {}).items()))


def _one(values: set, field: str, missing: list, conflicts: dict):
    vals = {v for v in values if v is not None}
    if not vals:
        missing.append(field)
        return None
    if len(vals) > 1:
        conflicts[field] = sorted(map(str, vals))
        return None
    return vals.pop()


def campaign_scope(campaign_dir: Path, max_logs_per_placement: int | None = None) -> dict:
    """{placement_id: scope dict} for one campaign directory (e.g. data/v2/raw/confirmatory)."""
    campaign_dir = Path(campaign_dir)
    pre = {}
    for f in sorted(campaign_dir.glob("blocks/block_*/preflight_pre.json")):
        pre[f.parent.name] = json.loads(f.read_text())
    acc: dict = {}
    for uj in sorted(campaign_dir.glob("blocks/block_*/units/*/unit.json")):
        u = json.loads(uj.read_text())
        pl = u.get("placement") or {}
        if not isinstance(pl, dict) or "id" not in pl:
            continue
        a = acc.setdefault(pl["id"], {k: set() for k in ("gpu_model", "driver_version", "cuda_driver_version",
                                                         "nccl_version", "nccl_version_code", "nranks", "nnodes",
                                                         "fingerprint", "env", "pci_bus_ids", "dtype", "in_place")})
        a.setdefault("ops", {})
        a.setdefault("n_units", 0)
        a.setdefault("n_logs", 0)
        a["n_units"] += 1
        gpus = list(pl.get("gpus") or [])
        a["nranks"].add(len(gpus) if gpus else None)
        a["fingerprint"].add((pl.get("fingerprint") or {}).get("fingerprint"))
        a["env"].add(env_string(pl.get("env")))
        a["pci_bus_ids"].add(",".join(pl.get("pci") or []) or None)
        p = pre.get(uj.parents[2].name)
        if p:
            names = {g.get("name") for g in p.get("gpus", []) if g.get("index") in gpus}
            a["gpu_model"].add(names.pop() if len(names) == 1 else None)
            a["driver_version"].add(p.get("driver"))
            a["cuda_driver_version"].add(p.get("cuda_driver"))
        d = uj.parent
        if max_logs_per_placement is None or a["n_logs"] < max_logs_per_placement:
            a["n_logs"] += 1
            log = (d / "nccl.log").read_text(errors="replace") if (d / "nccl.log").exists() else ""
            m = RE_VER.search(log)
            a["nccl_version"].add(m.group(1) if m else None)
            nn = {int(x[1]) for x in RE_NODES.findall(log)}
            a["nnodes"].add(nn.pop() if len(nn) == 1 else None)
            so = (d / "stdout.txt").read_text(errors="replace") if (d / "stdout.txt").exists() else ""
            m = RE_LIB.search(so)
            a["nccl_version_code"].add(int(m.group(1)) if m else None)
        rf = d / "region.jsonl"
        if rf.exists():
            ln = rf.read_text().splitlines()
            if ln:
                r = json.loads(ln[0])
                a["dtype"].add(NCCL_TESTS_TYPE.get(r.get("type")))
                a["in_place"].add(int(r["in_place"]) if "in_place" in r else None)
                a["ops"].setdefault(u.get("coll"), set()).add(NCCL_OP.get(r.get("op")))
    out = {}
    for pid, a in acc.items():
        missing, conflicts = [], {}
        sc = {f: _one(a[f], f, missing, conflicts) for f in SCOPE_FIELDS}
        ops = {}
        for coll, vals in a["ops"].items():
            if coll == "all_gather":
                continue                      # AllGather has no reduction op
            ops[coll] = _one(vals, f"op[{coll}]", missing, conflicts)
        sc.update(ops_by_coll=ops, missing=missing, conflicts=conflicts, n_units=a["n_units"],
                  n_logs_read=a["n_logs"], source=str(campaign_dir))
        out[pid] = sc
    return out


def declared_scope_string(scope: dict, coll: str | None = None) -> str:
    """Assumption string the controlled harness must export as NCCLE_TUNER_DECLARED_SCOPE (not verifiable by NCCL's
    v4 tuner interface, which does not pass dtype or reduction op to the plugin)."""
    ops = scope.get("ops_by_coll") or {}
    op = ",".join(f"{c}:{o}" for c, o in sorted(ops.items()))
    return f"dtype={scope.get('dtype')};ops={op};in_place={scope.get('in_place')}"
