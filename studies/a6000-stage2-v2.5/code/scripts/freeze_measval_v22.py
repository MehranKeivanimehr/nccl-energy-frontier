"""Freeze the gate 3b schedule (v2.2-measval-01) before collection. CPU only; refuses to overwrite.

Conditions: P2_NVL AllGather 2 MiB FP32 out-of-place; A auto, B forced Ring/Simple, C forced Ring/LL128; target
region lengths 2.5, 10, 20, 40 s (docs/v2/v22_audit/stage3_plan_rev1.md §7). R = ceil(T / t), w = max(5, ceil(0.2/t)),
t = median historical per-op time of the same cell and configuration (confirmatory for auto and Ring/LL128,
exploratory for Ring/Simple). Run from the repository root:  python scripts/freeze_measval_v22.py
"""
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CELL = ("P2_NVL", "all_gather", 2097152)
LENGTHS = (2.5, 10.0, 20.0, 40.0)


def t_median(csv, config):
    u = pd.read_csv(REPO / csv)
    u = u[(u.qc_status == "OK") & (u.evidence == "VERIFIED") & (u.placement == CELL[0]) & (u.coll == CELL[1])
          & (u.size_bytes == CELL[2]) & (u.config == config)]
    return float(u.t_gpu_s.median()), len(u)


T = {"auto": t_median("data/v2/processed/confirmatory/units.csv", "auto"),
     "Ring/LL128": t_median("data/v2/processed/confirmatory/units.csv", "Ring/LL128"),
     "Ring/Simple": t_median("data/v2/processed/exploratory/units.csv", "Ring/Simple")}
entries = {}
for label, (t, n) in T.items():
    for L in LENGTHS:
        uid = f"{CELL[0]}__{CELL[1]}__{label.replace('/', '-')}__{CELL[2]}__T{L:g}s"
        entries[uid] = {"R": math.ceil(L / t), "w": max(5, math.ceil(0.2 / t)), "t_gpu_cal_s": t, "target_region_s": L,
                        "source": f"median t_gpu of {n} accepted historical units"}
out = REPO / "configs/v2.2/measval01_frozen.json"
if out.exists():
    raise SystemExit(f"{out} exists; frozen schedules are never overwritten")
blob = json.dumps(entries, sort_keys=True).encode()
out.write_text(json.dumps({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "method": "v2.2",
                           "campaign": "v2.2-measval-01", "entries_sha256": hashlib.sha256(blob).hexdigest(),
                           "entries": entries}, indent=1))
for k, v in entries.items():
    print(k, v["R"], v["w"])
