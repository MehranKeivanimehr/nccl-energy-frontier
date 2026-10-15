"""v2 analysis entry point.

    python -m ncclenergy.v2.analyze_cli exploratory --campaign data/v2/raw/exploratory --out data/v2/processed/exploratory
    python -m ncclenergy.v2.analyze_cli confirmatory --campaign data/v2/raw/confirmatory --prereg docs/v2/prereg.json \
        --out data/v2/processed/confirmatory
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (CELL, EFFECT, LAT_SLO, block_bootstrap_ci, candidates, cell_config_table, choose_policies, collapse_effective, holm,
                       load_blocks, paired_test, paired_vs_auto, required_blocks, tost, valid)


def qc_summary(units: pd.DataFrame, atts: pd.DataFrame, summaries: list[dict]) -> dict:
    run = units[~units.qc_status.str.startswith("NOT_RUN")]
    ok = run[run.qc_status == "OK"]
    q = {"blocks": [{k: s.get(k) for k in ("block", "status", "stop_reason", "utc_start", "utc_end", "n_units",
                                            "final_status_counts", "evidence_counts", "health_events")} for s in summaries],
         "units_planned": int(len(units)), "units_run": int(len(run)),
         "final_status": run.qc_status.value_counts().to_dict(),
         "not_run": units[units.qc_status.str.startswith("NOT_RUN")].qc_status.value_counts().to_dict(),
         "evidence": run.evidence.value_counts(dropna=False).to_dict(),
         "attempts": int(len(atts)), "retried_units": int((units.n_attempts > 1).sum())}
    if len(ok):
        r = ok.inst_over_counter.dropna()
        q["inst_over_counter"] = {"n": int(len(r)), "median": float(r.median()), "p05": float(r.quantile(.05)),
                                  "p95": float(r.quantile(.95)), "min": float(r.min()), "max": float(r.max())}
        q["T_region_s"] = {"min": float(ok.T_region_s.min()), "median": float(ok.T_region_s.median()),
                           "max": float(ok.T_region_s.max())}
        q["temp_max_c"] = float(ok.temp_max_c.max())
        q["clk_gr_mean_range"] = [float(ok.clk_gr_mean.min()), float(ok.clk_gr_mean.max())]
        q["negative_idle_adjusted"] = int((ok.J_idle_adj_per_coll <= 0).sum())
        q["region_flags"] = sorted(set(";".join(ok.region_flags.fillna("")).split(";")) - {""})
        q["qc_reasons"] = sorted(set(";".join(ok.qc_reasons.fillna("")).split(";")) - {""})
    return q


def cmd_exploratory(campaign: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    units, atts, summaries = load_blocks(campaign)
    units.to_csv(out / "units.csv", index=False)
    atts.to_csv(out / "attempts.csv", index=False)
    v = valid(units)
    tab = cell_config_table(v)
    tab.to_csv(out / "cell_config.csv", index=False)
    pv = paired_vs_auto(v)
    pv.to_csv(out / "paired_vs_auto.csv", index=False)
    eff = collapse_effective(tab)
    eff.to_csv(out / "cell_effective.csv", index=False)
    pol = choose_policies(eff)
    pol.to_csv(out / "policies.csv", index=False)
    cand = candidates(eff, pol)
    (out / "candidates.json").write_text(json.dumps(cand, indent=1))
    q = qc_summary(units, atts, summaries)
    nb = int(units.block.nunique())
    if nb >= 2 and len(pv):
        # RQ1 cells: Energy-SLO differs from auto in its executed configuration
        rq1 = pol[pol.energy_slo_actual != pol.auto_actual]
        sd = []
        for r in rq1.itertuples():
            m = pv[(pv.placement == r.placement) & (pv.coll == r.coll) & (pv.size_bytes == r.size_bytes) &
                   (pv.config == r.energy_slo)]
            if len(m):
                sd.append(float(m.dE_sd.iloc[0]))
        q["rq1_cells"] = int(len(rq1))
        q["rq1_sd_dE"] = {"n": len(sd), "median": float(np.median(sd)) if sd else None,
                          "p80": float(np.quantile(sd, .8)) if sd else None, "max": float(max(sd)) if sd else None}
        q["block_count_rule"] = required_blocks(np.array(sd)) if sd else None
    (out / "summary.json").write_text(json.dumps(q, indent=1, default=str))
    print(json.dumps({k: q[k] for k in ("units_run", "final_status", "evidence", "inst_over_counter", "rq1_cells",
                                         "rq1_sd_dE", "block_count_rule") if k in q}, indent=1, default=str))


def cmd_confirmatory(campaign: Path, prereg: Path, out: Path):
    """Pre-registered analysis (docs/v2/prereg.md). Writes per-block values for every test."""
    out.mkdir(parents=True, exist_ok=True)
    P = json.loads(prereg.read_text())
    units, atts, summaries = load_blocks(campaign)
    units.to_csv(out / "units.csv", index=False)
    atts.to_csv(out / "attempts.csv", index=False)
    v = valid(units)
    q = qc_summary(units, atts, summaries)
    rows_rq1, rows_rq2 = [], []
    for cell in P["rq1_cells"]:
        pl, coll, size = cell["placement"], cell["coll"], int(cell["size_bytes"])
        c = v[(v.placement == pl) & (v.coll == coll) & (v.size_bytes == size)]
        a = c[c.config == "auto"].set_index("block")
        p = c[c.config == cell["energy_slo"]].set_index("block")
        j = p[["logE", "logT", "actual"]].join(a[["logE", "logT", "actual"]], rsuffix="_auto", how="inner")
        dE, dT = (j.logE - j.logE_auto).to_numpy(), (j.logT - j.logT_auto).to_numpy()
        te, tt = paired_test(dE, "less"), paired_test(dT, "two-sided")
        bs = block_bootstrap_ci(dE)
        rows_rq1.append({"placement": pl, "coll": coll, "size_bytes": size, "policy": cell["energy_slo"],
                         "policy_actual": "|".join(sorted(set(j.actual))), "auto_actual": "|".join(sorted(set(j.actual_auto))),
                         "n_blocks": te.get("n"), "E_ratio": math.exp(te["mean"]) if te.get("n", 0) >= 2 else None,
                         "E_ratio_ci_lo": math.exp(te["ci_lo"]) if te.get("n", 0) >= 2 else None,
                         "E_ratio_ci_hi": math.exp(te["ci_hi"]) if te.get("n", 0) >= 2 else None,
                         "E_ratio_boot_lo": math.exp(bs[0]), "E_ratio_boot_hi": math.exp(bs[1]),
                         "p_one_sided": te.get("p"), "wilcoxon_p": te.get("wilcoxon_p_two_sided"),
                         "hl_ratio": math.exp(te["hodges_lehmann"]) if te.get("n", 0) >= 2 else None,
                         "T_ratio": math.exp(tt["mean"]) if tt.get("n", 0) >= 2 else None,
                         "T_ratio_ci_hi": math.exp(tt["ci_hi"]) if tt.get("n", 0) >= 2 else None,
                         "per_block_E_ratio": json.dumps([round(math.exp(x), 5) for x in dE]),
                         "per_block_T_ratio": json.dumps([round(math.exp(x), 5) for x in dT])})
    r1 = pd.DataFrame(rows_rq1)
    if len(r1):
        r1["p_holm"] = holm(r1.p_one_sided.fillna(1).tolist())
        k = len(r1)
        # Holm-adjusted (one-sided) upper bound: Bonferroni-style bound at alpha/k as a conservative report
        from scipy import stats
        his = []
        for row in r1.itertuples():
            d = np.log(np.array(json.loads(row.per_block_E_ratio)))
            n = len(d)
            se = d.std(ddof=1) / math.sqrt(n) if n >= 2 else np.nan
            his.append(math.exp(d.mean() + stats.t.ppf(1 - 0.025 / k, n - 1) * se) if n >= 2 else np.nan)
        r1["E_ratio_upper_bonf"] = his
        r1["saving_confirmed"] = (r1.p_holm < 0.025) & (r1.T_ratio_ci_hi <= LAT_SLO)
        r1["saving_meaningful"] = r1.saving_confirmed & (r1.E_ratio <= 1 - EFFECT)
    r1.to_csv(out / "rq1.csv", index=False)
    for cell in P["rq2_cells"]:
        pl, coll, size = cell["placement"], cell["coll"], int(cell["size_bytes"])
        c = v[(v.placement == pl) & (v.coll == coll) & (v.size_bytes == size)]
        L = c[c.config == cell["lat_opt"]].set_index("block")
        E = c[c.config == cell["en_opt"]].set_index("block")
        j = L[["logE", "logT"]].join(E[["logE", "logT"]], rsuffix="_en", how="inner")
        dE = (j.logE - j.logE_en).to_numpy()      # > 0 : latency-optimal costs more energy
        dT = (j.logT_en - j.logT).to_numpy()      # > 0 : energy-optimal is slower
        te, tt = paired_test(dE, "greater"), paired_test(dT, "greater")
        rows_rq2.append({"placement": pl, "coll": coll, "size_bytes": size, "lat_opt": cell["lat_opt"],
                         "en_opt": cell["en_opt"], "n_blocks": te.get("n"),
                         "energy_penalty_of_lat_opt": math.exp(te["mean"]) - 1 if te.get("n", 0) >= 2 else None,
                         "pen_ci_lo": math.exp(te["ci_lo"]) - 1 if te.get("n", 0) >= 2 else None,
                         "pen_ci_hi": math.exp(te["ci_hi"]) - 1 if te.get("n", 0) >= 2 else None,
                         "latency_penalty_of_en_opt": math.exp(tt["mean"]) - 1 if tt.get("n", 0) >= 2 else None,
                         "lat_ci_lo": math.exp(tt["ci_lo"]) - 1 if tt.get("n", 0) >= 2 else None,
                         "lat_ci_hi": math.exp(tt["ci_hi"]) - 1 if tt.get("n", 0) >= 2 else None,
                         "pE": te.get("p"), "pT": tt.get("p"), "tost_energy_p": tost(dE).get("p"),
                         "per_block_dE": json.dumps([round(x, 6) for x in dE]),
                         "per_block_dT": json.dumps([round(x, 6) for x in dT])})
    r2 = pd.DataFrame(rows_rq2)
    if len(r2):
        r2["pE_holm"] = holm(r2.pE.fillna(1).tolist())
        r2["pT_holm"] = holm(r2.pT.fillna(1).tolist())
        r2["verdict"] = np.where((r2.pE_holm < 0.05) & (r2.pT_holm < 0.05), "DIFFER",
                                 np.where(r2.tost_energy_p < 0.05, "EQUIVALENT_ENERGY_+-3%", "INCONCLUSIVE"))
    r2.to_csv(out / "rq2.csv", index=False)
    (out / "summary.json").write_text(json.dumps(q, indent=1, default=str))
    print(json.dumps({"qc": {k: q[k] for k in ("units_run", "final_status", "evidence")},
                      "rq1": {"cells": len(r1), "confirmed": int(r1.saving_confirmed.sum()) if len(r1) else 0,
                              "meaningful": int(r1.saving_meaningful.sum()) if len(r1) else 0},
                      "rq2": r2.verdict.value_counts().to_dict() if len(r2) else {}}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["exploratory", "confirmatory"])
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prereg", type=Path)
    a = ap.parse_args()
    if a.cmd == "exploratory":
        cmd_exploratory(a.campaign, a.out)
    else:
        cmd_confirmatory(a.campaign, a.prereg, a.out)


if __name__ == "__main__":
    main()
