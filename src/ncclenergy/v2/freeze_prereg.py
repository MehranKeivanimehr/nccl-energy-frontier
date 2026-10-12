"""Phase 6: freeze the confirmatory protocol from exploratory results.

    python -m ncclenergy.v2.freeze_prereg --exploratory data/v2/processed/exploratory \
        --setup data/v2/raw/setup --pilot data/processed/full --out docs/v2/prereg.json \
        --campaign-yaml configs/v2/confirmatory.yaml

Writes the frozen matrix (cells, candidate configurations), endpoints, block count
and the hashes of code, configuration, calibration and exploratory data. Refuses to
overwrite an existing pre-registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .analysis import EFFECT, LAT_SLO, required_blocks
from .common import REPO, git_provenance, sha256_file, utc_now

PILOT_MAP = {"P2_NVL": "2gpu_nvlink", "P4_MIX": "4gpu"}   # analogous pilot placements (hypothesis/variance only)


def pilot_sd(pilot_dir: Path) -> pd.DataFrame:
    r = pd.read_csv(pilot_dir / "regions.csv")
    r = r[r.placement == "oop"].copy()
    r["logE"] = np.log(r.e_op_j)
    out = []
    for (c, g, s), cell in r.groupby(["coll", "gpu_set", "size_bytes"]):
        a = cell[cell.config == "auto/auto"].set_index("trial")["logE"]
        for cfg, grp in cell[cell.config != "auto/auto"].groupby("config"):
            d = grp.set_index("trial")["logE"] - a
            if d.notna().sum() >= 3:
                out.append({"coll": c, "gpu_set": g, "size_bytes": s, "config": cfg.replace("auto/", "").replace("/auto", ""),
                            "sd_pilot": float(d.std(ddof=1))})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exploratory", type=Path, required=True)
    ap.add_argument("--setup", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--campaign-yaml", type=Path, required=True)
    ap.add_argument("--n-max", type=int, default=12)
    a = ap.parse_args()
    for k in ("exploratory", "setup", "pilot", "out", "campaign_yaml"):
        setattr(a, k, getattr(a, k).resolve())
    if a.out.exists() or a.campaign_yaml.exists():
        raise SystemExit("pre-registration already frozen; refusing to overwrite")
    E = a.exploratory
    pol = pd.read_csv(E / "policies.csv")
    pv = pd.read_csv(E / "paired_vs_auto.csv")
    cand = json.loads((E / "candidates.json").read_text())
    expl_summary = json.loads((E / "summary.json").read_text())

    # RQ1: cells where the Energy-SLO pick executes something other than NCCL automatic
    rq1 = pol[pol.energy_slo_actual != pol.auto_actual]
    # RQ2: cells where latency- and energy-optimal executed configurations differ
    rq2 = pol[pol.lat_opt_actual != pol.en_opt_actual]
    cells = sorted({f"{r.placement}|{r.coll}|{r.size_bytes}" for r in pd.concat([rq1, rq2]).itertuples()})
    cell_configs = {c: cand[c] for c in cells}

    # block count: larger of exploratory-v2 and pilot SD of the paired log-energy difference (RQ1 cells)
    ps = pilot_sd(a.pilot)
    sds = []
    for r in rq1.itertuples():
        m = pv[(pv.placement == r.placement) & (pv.coll == r.coll) & (pv.size_bytes == r.size_bytes) & (pv.config == r.energy_slo)]
        sd_v2 = float(m.dE_sd.iloc[0]) if len(m) and np.isfinite(m.dE_sd.iloc[0]) else np.nan
        sd_p = np.nan
        if r.placement in PILOT_MAP:
            q = ps[(ps.gpu_set == PILOT_MAP[r.placement]) & (ps.coll == r.coll) & (ps.size_bytes == r.size_bytes) &
                   (ps.config == r.energy_slo)]
            sd_p = float(q.sd_pilot.max()) if len(q) else np.nan
        sds.append(np.nanmax([sd_v2, sd_p]) if not (np.isnan(sd_v2) and np.isnan(sd_p)) else np.nan)
    rule = required_blocks(np.array(sds), n_max=a.n_max) if sds else {"n": 5, "note": "no RQ1 cells"}
    n_blocks = max(5, rule["n"])

    frozen = json.loads((a.setup / "calibration_frozen.json").read_text())
    expl_files = sorted(E.glob("*.csv")) + sorted(E.glob("*.json"))
    prereg = {
        "frozen_utc": utc_now(), "git": git_provenance(),
        "hashes": {"calibration_entries_sha256": frozen["entries_sha256"],
                   "patch_sha256": sha256_file(REPO / "patches/nccl-tests-afd59ab-v2.1-region.patch"),
                   "plugin_sha256": sha256_file(REPO / "native/nccl_profiler_algo/build/libnccl-profiler-nccle.so"),
                   "exploratory_files": {p.name: sha256_file(p) for p in expl_files}},
        "research_questions": {
            "RQ1": "Does the Energy-SLO policy reduce raw participating-GPU board energy per collective relative to NCCL "
                   "automatic while meeting the latency SLO?",
            "RQ2": "Do latency-optimal and raw-energy-optimal configurations differ after block-level uncertainty?",
            "RQ3": "Does the best microbenchmark policy change energy per step of the synthetic PyTorch workload?",
            "RQ4": "Sensitivity to placement (3 fingerprints), temperature, clocks; NCCL version not tested."},
        "endpoints": {"primary": "ln(raw J per collective), block-paired difference policy - auto",
                      "latency": "ln(t_gpu) block-paired difference", "meaningful_effect": f">= {EFFECT:.0%} lower raw energy",
                      "latency_slo": f"policy latency <= {LAT_SLO} x auto latency (95 % upper bound)",
                      "secondary": ["idle-adjusted J", "EDP", "bytes/J", "counter vs integration"]},
        "tests": {"RQ1": "one-sided paired t on block log-differences, Holm across RQ1 cells (alpha 0.025); confirmed if "
                         "Holm p < 0.025 and latency-ratio 95 % upper bound <= 1.05; meaningful if also ratio <= 0.95",
                  "RQ2": "two one-sided paired t (energy of lat-opt > en-opt; latency of en-opt > lat-opt), Holm per "
                         "family; DIFFER if both < 0.05; else TOST +-3 % on energy; else INCONCLUSIVE",
                  "robustness": ["Wilcoxon signed-rank", "Hodges-Lehmann", "block bootstrap", "idle-adjusted",
                                 "order/temperature regression"]},
        "qc_rules": "docs/v2/protocol.md sections 9 and 12; QC_SHORT / MEASUREMENT_INVALID / UNVERIFIED / "
                    "EXCLUDED_DEVIATION are excluded and counted; no outcome-dependent re-runs",
        "missing": "complete pairs per cell; missing counts reported per cell",
        "outliers": "none removed",
        "randomization": "per block: random.Random('confirmatory-<seed>-<block>') shuffle of all units",
        "block_count_rule": {"rule": "smallest N <= 12 with predicted 95 % CI half-width <= ln(1.025) in >= 80 % of "
                                     "RQ1 cells, using max(exploratory-v2 SD, pilot SD); minimum 5",
                             "result": rule, "n_blocks": n_blocks},
        "rq1_cells": [{"placement": r.placement, "coll": r.coll, "size_bytes": int(r.size_bytes),
                       "energy_slo": r.energy_slo, "energy_slo_actual": r.energy_slo_actual,
                       "auto_actual": r.auto_actual, "exploratory_E_vs_auto": r.energy_slo_E_vs_auto,
                       "exploratory_T_vs_auto": r.energy_slo_T_vs_auto} for r in rq1.itertuples()],
        "rq2_cells": [{"placement": r.placement, "coll": r.coll, "size_bytes": int(r.size_bytes), "lat_opt": r.lat_opt,
                       "en_opt": r.en_opt, "lat_opt_actual": r.lat_opt_actual, "en_opt_actual": r.en_opt_actual}
                      for r in rq2.itertuples()],
        "confirmatory_cells": cell_configs,
        "exploratory_qc": {k: expl_summary.get(k) for k in ("units_run", "final_status", "evidence", "inst_over_counter")},
    }
    # Confirmatory iteration schedule: same R/w in every confirmatory block, sized from the exploratory blocks'
    # fastest measured latency per unit (sustained runs) with T* = 3.0 s, so that regions stay >= T_min.
    units = pd.read_csv(E / "units.csv")
    okv = units[(units.qc_status.isin(["OK", "QC_SHORT"])) & units.t_gpu_s.notna()]
    tmin = okv.groupby("unit_id").t_gpu_s.min()
    T_conf = 3.0
    entries = {}
    for cell, cfgs in cell_configs.items():
        pl, coll, size = cell.split("|")
        for cfg in cfgs:
            uid = f"{pl}__{coll}__{cfg.replace('/', '-')}__{size}"
            base = frozen["entries"].get(uid)
            if base is None or uid not in tmin.index:
                raise SystemExit(f"no exploratory latency for {uid}")
            t = float(tmin[uid])
            entries[uid] = {"R": max(10, math.ceil(T_conf / t)), "w": max(5, math.ceil(0.2 / t)), "t_gpu_cal_s": t,
                            "count": base["count"], "source": "exploratory min t_gpu over blocks"}
    body = json.dumps(entries, sort_keys=True)
    conf_frozen = {"frozen_utc": utc_now(), "T_star_s": T_conf, "warmup_s": 0.2, "R_min": 10, "w_min": 5,
                   "entries_sha256": hashlib.sha256(body.encode()).hexdigest(), "entries": entries}
    conf_path = a.setup / "calibration_frozen_confirmatory.json"
    if conf_path.exists():
        raise SystemExit(f"{conf_path} exists; refusing to overwrite")
    conf_path.write_text(json.dumps(conf_frozen, indent=1, sort_keys=True))
    prereg["hashes"]["confirmatory_schedule_sha256"] = conf_frozen["entries_sha256"]
    prereg["confirmatory_schedule"] = {"file": str(conf_path.relative_to(REPO)), "T_star_s": T_conf,
                                       "rule": "R = max(10, ceil(3.0 s / min exploratory t_gpu)), w = max(5, ceil(0.2 s / t))"}
    sizes = sorted({int(c.split("|")[2]) for c in cells})
    configs = sorted({x for v in cell_configs.values() for x in v})
    camp = yaml.safe_load((REPO / "configs/v2/exploratory.yaml").read_text())
    camp.update({"campaign": "confirmatory", "mode": "CONFIRMATORY", "seed": 20261009, "sizes": {"list": sizes},
                 "frozen_file": str(conf_path.relative_to(REPO)),
                 "configs": configs, "cells": cells, "cell_configs": cell_configs, "max_blocks": n_blocks})
    a.campaign_yaml.write_text("# FROZEN confirmatory campaign generated by ncclenergy.v2.freeze_prereg; do not edit.\n"
                               + yaml.safe_dump(camp, sort_keys=False))
    prereg["hashes"]["confirmatory_yaml_sha256"] = sha256_file(a.campaign_yaml)
    a.out.write_text(json.dumps(prereg, indent=1, default=str))
    print(json.dumps({"rq1_cells": len(prereg["rq1_cells"]), "rq2_cells": len(prereg["rq2_cells"]),
                      "confirmatory_cells": len(cells), "units_per_block": sum(len(v) for v in cell_configs.values()),
                      "block_rule": rule, "n_blocks": n_blocks}, indent=1, default=str))


if __name__ == "__main__":
    main()
