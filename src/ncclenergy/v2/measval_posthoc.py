"""Gate 3b POST-HOC (exploratory) analysis: GPU temperature as the dominant source of between-process variation.

NOT pre-specified. Written after the pre-specified decisions (measval.py) were computed; it changes none of them.
Its purpose is to explain those decisions and to inform a revised design. CPU only, no NVML.

    python -m ncclenergy.v2.measval_posthoc --campaign data/v2.2/raw/v2.2-measval-01 \
        --out data/v2.2/processed/v2.2-measval-01/posthoc_temperature.json

* Temperature: mean over both GPUs of the sampler's per-process mean temperature during the region.
* Slope: ln(J/op) ~ arm x length fixed effects + temperature, ordinary least squares, accepted processes only.
  Per-block slopes (one block alone) use the additive model arm + length + temperature.
* Adjusted contrasts: ln(J/op) - slope * (temp - 40 C), with the slope re-estimated leaving the contrast's own block
  out (leave-one-block-out), then the same within-block contrasts and block t-intervals as measval.py.
* Caveat: temperature is not randomized. It follows execution position (warm-up within a block) and is partly caused
  by the configuration's own power; adjusting for it can absorb part of a real configuration effect (C vs A).
  The model associates temperature with energy variation; it does not establish temperature as the sole cause.
  Adjusted contrast intervals are conditional/descriptive: overlapping leave-one-block-out fits and slope-estimation
  uncertainty are not propagated by the block t-interval. They are not confirmatory intervals or sensor calibration.

Second section, "window_gap": counter vs sampled energy over MATCHED windows. Per GPU and region, both estimators
cover [t0 = midpoint of the start read, t1 = loop end] on CLOCK_MONOTONIC, in joules per GPU: the counter estimate
(C1 - C0 minus context power x tail, region.py) and the trapezoid integral of POWER_INSTANT ("inst") or of
PowerUsage shifted by power_lag_s ("lag") over the same [t0, t1] (telemetry.window_summary). Per block, OLS of
gap = E_counter - E_sampled on [1, P_region * dstart, p_ctx * tail, duration], where dstart = t0 minus the last
counter commit before t0 (from the recorded phase and period). Hypothesis H_start ("the counter delta includes
the partial update window before the start read") predicts a P_region*dstart coefficient of 1 and an intercept
of 0. The duration coefficient (J/s) is the steady-state power difference (counter - sampled) per GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO, read_jsonl
from .measval import CFG, LENGTHS, summarize

T_REF_C = 40.0


def load(campaign: Path) -> pd.DataFrame:
    rows = []
    for bdir in sorted((campaign / "blocks").glob("block_*")):
        for a in read_jsonl(bdir / "attempts.jsonl")[0]:
            if a.get("qc_status") != "OK":
                continue
            m, t = a["region"], a["telemetry"]
            rows.append({"block": int(bdir.name.split("_")[1]), "exec_pos": a["exec_pos"], "arm": CFG[a["requested"]],
                         "length_s": float(a["length_s"]), "R": m["R"], "T_region_s": m["T_region_s"],
                         "E_counter": m["E_raw_J"], "power_W": m["E_raw_J"] / m["T_region_s"],
                         "us_per_op": 1e6 * m["T_region_s"] / m["R"],
                         "temp_c": float(np.mean([v["temp_mean_c"] for v in t.values()])),
                         "clk_gr_mhz": float(np.mean([v["clk_gr_mean"] for v in t.values()]))})
    d = pd.DataFrame(rows)
    d["y"] = np.log(d.E_counter / d.R)
    return d


def _design(d: pd.DataFrame, with_temp: bool, additive: bool = False) -> np.ndarray:
    if additive:   # intercept + arm + length (for one block: 12 processes cannot carry 12 cell means + a slope)
        arms, lens = sorted(d.arm.unique())[1:], sorted(d.length_s.unique())[1:]
        X = np.column_stack([np.ones(len(d))] + [(d.arm == a).to_numpy(float) for a in arms]
                            + [(d.length_s == L).to_numpy(float) for L in lens])
    else:
        cells = sorted({(a, L) for a, L in zip(d.arm, d.length_s)})
        X = np.array([[float((a, L) == c) for c in cells] for a, L in zip(d.arm, d.length_s)])
    return np.column_stack([X, d.temp_c.to_numpy()]) if with_temp else X


def fit(d: pd.DataFrame, with_temp: bool, additive: bool = False) -> tuple[float | None, float]:
    X = _design(d, with_temp, additive)
    beta, *_ = np.linalg.lstsq(X, d.y.to_numpy(), rcond=None)
    r = d.y.to_numpy() - X @ beta
    return (float(beta[-1]) if with_temp else None), float(r.std(ddof=X.shape[1]))


def adjusted(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for b, g in d.groupby("block"):
        slope, _ = fit(d[d.block != b], True)
        out.append(g.assign(y_adj=g.y - slope * (g.temp_c - T_REF_C), slope_lobo=slope))
    return pd.concat(out)


def _pair(d: pd.DataFrame, num: str, den: str, L: float, col: str) -> list[float]:
    v = []
    for _, g in d[d.length_s == L].groupby("block"):
        a, c = g[g.arm == num], g[g.arm == den]
        if len(a) == 1 and len(c) == 1:
            v.append(float(a[col].iloc[0] - c[col].iloc[0]))
    return v


def _len_ratio(d: pd.DataFrame, arm: str, col: str) -> list[float]:
    v = []
    for _, g in d[d.arm == arm].groupby("block"):
        s, l = g[g.length_s == 2.5], g[g.length_s == 40.0]
        if len(s) == 1 and len(l) == 1:
            v.append(float(s[col].iloc[0] - l[col].iloc[0]))
    return v


def analyse(d: pd.DataFrame) -> dict:
    res = {"label": "POST-HOC EXPLORATORY; not a gate decision", "n_processes": int(len(d)), "t_ref_c": T_REF_C}
    for lab, sub in (("length_ge_10s", d[d.length_s >= 10]), ("all_lengths", d)):
        slope, sd_t = fit(sub, True)
        _, sd_0 = fit(sub, False)
        res[f"fit_{lab}"] = {"n": int(len(sub)), "temp_slope_ln_per_c": slope, "resid_sd_ln_without_temp": sd_0,
                             "resid_sd_ln_with_temp": sd_t}
    per_block = []
    for b, g in d.groupby("block"):
        per_block.append(fit(g, True, additive=True)[0])
    res["temp_slope_within_block"] = summarize(per_block)
    res["corr_y_with"] = {c: float(np.corrcoef(d.y, d[c])[0, 1]) for c in ("temp_c", "exec_pos", "clk_gr_mhz", "power_W")}
    res["temp_by_block"] = {int(b): [float(g.temp_c.min()), float(g.temp_c.max())] for b, g in d.groupby("block")}
    res["temp_by_exec_pos"] = {int(p): float(g.temp_c.mean()) for p, g in d.groupby("exec_pos")}
    a = adjusted(d)
    res["slope_lobo"] = sorted(float(x) for x in a.slope_lobo.unique())
    res["contrasts"] = {
        "D1_counter_ln_Jop_2.5_over_40": {arm: {"raw": summarize(_len_ratio(a, arm, "y")),
                                                "temp_adjusted": summarize(_len_ratio(a, arm, "y_adj"))}
                                          for arm in ("A", "B", "C")},
        "D3_counter_B_over_A": {L: {"raw": summarize(_pair(a, "B", "A", L, "y")),
                                    "temp_adjusted": summarize(_pair(a, "B", "A", L, "y_adj"))} for L in LENGTHS},
        "D4_counter_C_over_A": {L: {"raw": summarize(_pair(a, "C", "A", L, "y")),
                                    "temp_adjusted": summarize(_pair(a, "C", "A", L, "y_adj"))} for L in LENGTHS},
        "temp_diff_C_minus_A_c": {L: summarize(_pair(a, "C", "A", L, "temp_c")) for L in LENGTHS},
    }
    return res


def window_gap(campaign: Path) -> dict:
    rows = []
    for bdir in sorted((campaign / "blocks").glob("block_*")):
        for a in read_jsonl(bdir / "attempts.jsonl")[0]:
            if a.get("qc_status") != "OK":
                continue
            rec = json.loads((REPO / a["dir"] / "region.jsonl").read_text().splitlines()[0])
            P = rec["commit_period_ns"]
            per = {p["rank"]: p for p in a["region"]["per_gpu"]}
            for g in rec["gpus"]:
                p = per[g["rank"]]
                tel = a["telemetry"][str(p["nvml_index"])]
                t0 = (g["t_start_a_ns"] + g["t_start_b_ns"]) / 2
                c_prev = g["phase_ns"] + math.floor((t0 - g["phase_ns"]) / P) * P
                rows.append({"block": int(bdir.name.split("_")[1]), "length_s": float(a["length_s"]),
                             "dstart_s": (t0 - c_prev) * 1e-9, "tail_s": p["tail_s"], "p_ctx_W": p["p_ctx_W"],
                             "p_region_W": p["p_avg_W"], "duration_s": p["duration_s"], "E_counter_J": p["delta_J"],
                             "E_inst_J": tel["energy_int_inst_J"], "E_lag_J": tel["energy_int_lag_J"]})
    d = pd.DataFrame(rows)
    bins = []      # start phases cluster at a few values; compare each cluster with the H_start prediction
    for lo, hi in ((0, 5), (5, 50), (50, 90), (90, 100.5)):
        b = d[(1e3 * d.dstart_s >= lo) & (1e3 * d.dstart_s < hi)]
        if len(b):
            bins.append({"dstart_ms": [lo, hi], "n": int(len(b)),
                         "gap_lag_J": float((b.E_counter_J - b.E_lag_J).mean()),
                         "gap_inst_J": float((b.E_counter_J - b.E_inst_J).mean()),
                         "H_start_prediction_J": float((b.p_region_W * b.dstart_s).mean())})
    out = {"label": "POST-HOC EXPLORATORY; matched windows [t0, t1] per GPU, joules per GPU",
           "by_start_phase": bins,
           "n_gpu_regions": int(len(d)), "mean_power_per_gpu_W": float(d.p_region_W.mean()),
           "dstart_ms_range": [float(1e3 * d.dstart_s.min()), float(1e3 * d.dstart_s.max())]}
    for est in ("inst", "lag"):
        gap = d.E_counter_J - d[f"E_{est}_J"]
        coefs, means = [], []
        for b, g in d.assign(gap=gap).groupby("block"):
            X = np.column_stack([np.ones(len(g)), g.p_region_W * g.dstart_s, g.p_ctx_W * g.tail_s, g.duration_s])
            beta, *_ = np.linalg.lstsq(X, g.gap.to_numpy(), rcond=None)
            coefs.append(beta)
            means.append(float(g.gap.mean()))
        c = np.array(coefs)
        slope = summarize(c[:, 3])
        out[est] = {"gap_J_per_gpu_region": summarize(means),
                    "gap_J_by_length_median": {float(L): float(v) for L, v in gap.groupby(d.length_s).median().items()},
                    "intercept_J": summarize(c[:, 0]), "coef_p_region_x_dstart": summarize(c[:, 1]),
                    "coef_p_ctx_x_tail": summarize(c[:, 2]), "per_second_J_per_s": slope,
                    "steady_state_power_diff_pct": {k: (100 * v / out["mean_power_per_gpu_W"] if isinstance(v, float)
                                                        else [100 * x / out["mean_power_per_gpu_W"] for x in v])
                                                    for k, v in slope.items() if k in ("mean", "ci")},
                    "H_start_prediction": {"coef_p_region_x_dstart": 1.0, "intercept_J": 0.0}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = analyse(load(a.campaign))
    res["window_gap"] = window_gap(a.campaign)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    f = res["fit_length_ge_10s"]
    print(f"temperature slope {100 * f['temp_slope_ln_per_c']:.3f} %/C; residual SD {100 * f['resid_sd_ln_without_temp']:.2f} %"
          f" -> {100 * f['resid_sd_ln_with_temp']:.2f} % (length >= 10 s, n={f['n']})")
    pct = lambda s: "n/a" if s.get("ci") is None else \
        f"{100 * (math.exp(s['mean']) - 1):+.2f}% [{100 * (math.exp(s['ci'][0]) - 1):+.2f}, {100 * (math.exp(s['ci'][1]) - 1):+.2f}]"
    for k, v in res["contrasts"].items():
        if k.startswith("temp_diff"):
            continue
        for kk, vv in v.items():
            print(f"{k:32s} {str(kk):4s} raw {pct(vv['raw']):28s} adjusted {pct(vv['temp_adjusted'])}")


if __name__ == "__main__":
    main()
