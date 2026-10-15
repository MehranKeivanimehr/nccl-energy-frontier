"""Freeze the v2.2 smoke-test schedules (gate 3a) before collection. CPU only; refuses to overwrite.

R = ceil(T / t), w = max(5, ceil(0.2 s / t)); t = median historical per-op time of the same cell and configuration
(P2_NVL AllGather 2 MiB): confirmatory (12 blocks) for auto and Ring/LL128, exploratory (2 blocks) for Ring/Simple,
which was not in the confirmatory matrix. Run from the repository root:  python scripts/freeze_smoke_v22.py
"""
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CELL = ("P2_NVL", "all_gather", 2097152)


def t_median(csv, config):
    u = pd.read_csv(REPO / csv)
    u = u[(u.qc_status == "OK") & (u.evidence == "VERIFIED") & (u.placement == CELL[0]) & (u.coll == CELL[1])
          & (u.size_bytes == CELL[2]) & (u.config == config)]
    assert len(u) >= 2, (csv, config, len(u))
    return float(u.t_gpu_s.median()), len(u)


T = {"auto": t_median("data/v2/processed/confirmatory/units.csv", "auto"),
     "Ring/LL128": t_median("data/v2/processed/confirmatory/units.csv", "Ring/LL128"),
     "Ring/Simple": t_median("data/v2/processed/exploratory/units.csv", "Ring/Simple")}


def entry(label, region_s):
    t, n = T[label]
    return {"R": math.ceil(region_s / t), "w": max(5, math.ceil(0.2 / t)), "count": CELL[2] // 4 // 2,
            "t_gpu_cal_s": t, "target_region_s": region_s,
            "source": f"median t_gpu of {n} accepted historical units ({'confirmatory' if label != 'Ring/Simple' else 'exploratory'})"}


def uid(label):
    return f"{CELL[0]}__{CELL[1]}__{label.replace('/', '-')}__{CELL[2]}"


plans = {"smoke01_frozen.json": {uid(l): entry(l, 2.5) for l in ("auto", "Ring/Simple", "Ring/LL128")},
         "smoke01_s3_frozen.json": {uid("auto"): entry("auto", 20.0)},
         "smoke01_s4_frozen.json": {uid("auto"): entry("auto", 20.0)}}
for name, entries in plans.items():
    out = REPO / "configs/v2.2" / name
    if out.exists():
        raise SystemExit(f"{out} exists; frozen schedules are never overwritten")
    blob = json.dumps(entries, sort_keys=True).encode()
    out.write_text(json.dumps({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "method": "v2.2",
                               "purpose": "gate 3a smoke (SMOKE; not used for inference)",
                               "entries_sha256": hashlib.sha256(blob).hexdigest(), "entries": entries}, indent=1))
    print(name, {k: (v["R"], v["w"], round(v["t_gpu_cal_s"] * 1e6, 3)) for k, v in entries.items()})
