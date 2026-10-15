"""Phase 8 (scientific analysis extras) and Phase 9 (recommender validation) on confirmatory data.

    python -m ncclenergy.v2.phase89 --campaign data/v2/raw/confirmatory --prereg docs/v2/prereg.json \
        --processed data/v2/processed/confirmatory --figures figures/v2/confirmatory
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .analysis import load_blocks, paired_test, valid
from .common import REPO
from .recommender import build_profiles, lobo_validate, query_from_meta, recommend, tuner_conf


def paired_frame(v: pd.DataFrame, cell: dict, a_cfg: str, b_cfg: str) -> pd.DataFrame:
    c = v[(v.placement == cell["placement"]) & (v.coll == cell["coll"]) & (v.size_bytes == int(cell["size_bytes"]))]
    A = c[c.config == a_cfg].set_index("block")
    B = c[c.config == b_cfg].set_index("block")
    cols = ["logE", "logT", "J_idle_adj_per_coll", "exec_pos", "temp_mean_c", "clk_gr_mean", "inst_over_counter"]
    return A[cols].join(B[cols], lsuffix="_a", rsuffix="_b", how="inner")


def robustness(v: pd.DataFrame, P: dict) -> pd.DataFrame:
    rows = []
    for cell in P["rq1_cells"]:
        j = paired_frame(v, cell, cell["energy_slo"], "auto")
        if len(j) < 2:
            continue
        dE = (j.logE_a - j.logE_b).to_numpy()
        ok = (j.J_idle_adj_per_coll_a > 0) & (j.J_idle_adj_per_coll_b > 0)
        dI = np.log(j.J_idle_adj_per_coll_a[ok] / j.J_idle_adj_per_coll_b[ok]).to_numpy()
        t1, t2 = paired_test(dE, "less"), paired_test(dI, "less") if len(dI) >= 2 else {}
        rng = np.random.default_rng(0)
        bs = dE[rng.integers(0, len(dE), (10000, len(dE)))].mean(axis=1)
        rows.append({"placement": cell["placement"], "coll": cell["coll"], "size_bytes": int(cell["size_bytes"]),
                     "policy": cell["energy_slo"], "n_blocks": len(dE),
                     "raw_ratio": math.exp(t1["mean"]), "raw_ci": [math.exp(t1["ci_lo"]), math.exp(t1["ci_hi"])],
                     "idle_adj_ratio": math.exp(t2["mean"]) if t2.get("n", 0) >= 2 else None,
                     "idle_adj_ci": [math.exp(t2["ci_lo"]), math.exp(t2["ci_hi"])] if t2.get("n", 0) >= 2 else None,
                     "negative_idle_adjusted": int((~ok).sum()),
                     "hodges_lehmann_ratio": math.exp(t1["hodges_lehmann"]), "wilcoxon_p_two_sided": t1["wilcoxon_p_two_sided"],
                     "boot_ci": [math.exp(np.quantile(bs, .025)), math.exp(np.quantile(bs, .975))]})
    return pd.DataFrame(rows)


def order_temperature(v: pd.DataFrame, P: dict) -> dict:
    """Pooled regression of within-cell-centred paired log-energy differences on the paired differences in
    execution position (fraction of the block) and mean temperature."""
    xs = []
    for cell in P["rq1_cells"]:
        j = paired_frame(v, cell, cell["energy_slo"], "auto")
        if len(j) < 3:
            continue
        d = (j.logE_a - j.logE_b)
        xs.append(pd.DataFrame({"y": d - d.mean(), "dpos": (j.exec_pos_a - j.exec_pos_b) - (j.exec_pos_a - j.exec_pos_b).mean(),
                                "dtemp": (j.temp_mean_c_a - j.temp_mean_c_b) - (j.temp_mean_c_a - j.temp_mean_c_b).mean()}))
    if not xs:
        return {}
    X = pd.concat(xs).dropna()
    A = np.column_stack([X.dpos / 82.0, X.dtemp])
    beta, res, rank, _ = np.linalg.lstsq(A, X.y, rcond=None)
    resid = X.y - A @ beta
    dof = len(X) - A.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    q = stats.t.ppf(0.975, dof)
    return {"n_pairs": int(len(X)), "per_block_fraction_slope_logE": float(beta[0]),
            "ci": [float(beta[0] - q * se[0]), float(beta[0] + q * se[0])],
            "per_degC_slope_logE": float(beta[1]), "ci_temp": [float(beta[1] - q * se[1]), float(beta[1] + q * se[1])],
            "note": "slopes are on ln(energy ratio); position is the difference in execution position as a fraction of 82 units"}


def placement_meta(units_dir: Path) -> dict:
    """v2.2: profile scope per placement, derived from the campaign's raw evidence (profile_scope.py). Facts that
    are not recorded stay None (the recommender then falls back to NCCL automatic)."""
    from .profile_scope import campaign_scope
    return campaign_scope(units_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--prereg", type=Path, required=True)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    a = ap.parse_args()
    a.processed.mkdir(parents=True, exist_ok=True)
    a.figures.mkdir(parents=True, exist_ok=True)
    P = json.loads(a.prereg.read_text())
    units, atts, summaries = load_blocks(a.campaign)
    v = valid(units)
    rob = robustness(v, P)
    rob.to_csv(a.processed / "robustness.csv", index=False)
    ot = order_temperature(v, P)
    (a.processed / "order_temperature.json").write_text(json.dumps(ot, indent=1))
    meta = placement_meta(a.campaign)
    (a.processed / "placement_meta.json").write_text(json.dumps(meta, indent=1))
    lobo = lobo_validate(v, meta)
    lobo.to_csv(a.processed / "recommender_lobo.csv", index=False)
    lsum = {}
    if len(lobo):
        lsum = {"rows": int(len(lobo)), "heldout_blocks": int(lobo.heldout_block.nunique()),
                "status_counts": lobo.status.value_counts().to_dict(),
                "regret_median": float(lobo.regret.median()), "regret_p90": float(lobo.regret.quantile(.9)),
                "regret_max": float(lobo.regret.max()), "auto_regret_median": float(lobo.auto_regret.median()),
                "measured_saving_vs_auto_median": float(lobo.measured_saving_vs_auto.median()),
                "slo_violation_rate": float(lobo.slo_violated.mean())}
        rec = lobo[lobo.status == "RECOMMEND"]
        if len(rec):
            lsum["recommended_rows"] = int(len(rec))
            lsum["recommended_measured_saving_median"] = float(rec.measured_saving_vs_auto.median())
            lsum["recommended_predicted_minus_measured_E_ratio_median"] = float((rec.predicted_E_ratio - rec.measured_E_ratio).median())
    (a.processed / "recommender_lobo_summary.json").write_text(json.dumps(lsum, indent=1))
    prof = build_profiles(v, meta)
    for pl in meta:
        (a.processed / f"nccl_tuner_{pl}.conf").write_text("\n".join(tuner_conf(prof, pl, meta)) + "\n")
        (a.processed / f"nccl_tuner_{pl}_latency.conf").write_text(
            "\n".join(tuner_conf(prof, pl, meta, objective="min_latency")) + "\n")
    lat = []
    for pl in meta:   # latency-optimal policy table (objective min_latency), for the Phase 10 A/B
        m = meta[pl]
        for (coll, s), g in v[v.placement == pl].groupby(["coll", "size_bytes"]):
            r = recommend(prof, query_from_meta(coll, int(s), m, objective="min_latency"))
            lat.append({"placement": pl, "coll": coll, "size_bytes": int(s), "status": r["status"],
                        "recommendation": r.get("recommendation")})
    pd.DataFrame(lat).to_csv(a.processed / "latency_policy.csv", index=False)
    from . import plots_v2
    rq1 = pd.read_csv(a.processed / "rq1.csv") if (a.processed / "rq1.csv").exists() else pd.DataFrame()
    plots_v2.rq1_blocks(rq1, a.figures)
    plots_v2.counter_vs_integration(units, a.figures)
    print(json.dumps({"robustness_cells": len(rob), "order_temperature": ot, "lobo": lsum}, indent=1, default=str))


if __name__ == "__main__":
    main()
