"""v2.2 audit reanalysis of the historical v2.1 data (CPU only; writes only into a new directory).

    python -m ncclenergy.v2.reanalysis --out data/v2/reanalysis/2026-10-09_v2.2-audit

Every step reads the frozen raw data and processed outputs and writes into --out. The frozen
outputs (data/v2/processed/*, figures/v2/*) are hashed before and after; a changed hash aborts.

Sections (files in --out):
  A  reproduced/             pre-registered RQ1/RQ2 analysis re-run with the current code; compared with frozen
  B  rq1_threshold.csv       separately identified analysis: ratio <= 0.95 supported by a corrected threshold test
  C  rq2_estimators.*        RQ2 under four energy estimators; exact paired inputs, definitions, TOST
  D  order_temperature.json  published iid fit + block-, cell- and two-way-clustered fits + wild cluster bootstrap
  E  read_latency.*          boundary-read timing uncertainty, derived from the recorded read brackets
  F  sampling_intervals.json measurement-window and complete-trace sample spacing
  G  qc_reclassification.*   historical units under the v2.2 fail-closed QC
  H  phase89_v22/            robustness, LOBO and tuner files with v2.2 recommender guards (exact-size rules)
  I  phase10_v22/            Phase 10 analysis with all-rank execution and numerics fields
  J  figures/                new figures
  K  ledger.csv / ledger.md  every number that differs from a frozen output, with the reason
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import REPO

CONF = REPO / "data/v2/raw/confirmatory"
EXPL = REPO / "data/v2/raw/exploratory"
FROZEN_DIRS = [REPO / "data/v2/processed", REPO / "figures/v2"]
PREREG = REPO / "docs/v2/prereg.json"
TOST_MARGIN = math.log(1.03)
THRESH = 0.95


# ------------------------------------------------------------------ provenance

def tree_hash(dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------ helpers

def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    k = len(p)
    order = np.argsort(p)
    adj, run = np.empty(k), 0.0
    for i, idx in enumerate(order):
        run = max(run, min(1.0, (k - i) * p[idx]))
        adj[idx] = run
    return adj.tolist()


def t_ci(d: np.ndarray, level: float = 0.95):
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return m, se, (m - q * se, m + q * se)


def tost_p(d: np.ndarray, margin: float = TOST_MARGIN) -> float:
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    return float(max(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1)))


def confirmatory_units() -> pd.DataFrame:
    u = pd.read_csv(REPO / "data/v2/processed/confirmatory/units.csv")
    return u[(u.qc_status == "OK") & (u.evidence == "VERIFIED")].copy()


def unit_records(campaign: Path) -> dict:
    """{(block, unit_id): final attempt record} straight from attempts.jsonl (the runner's append-only log)."""
    from .attempts import final_attempt
    out = {}
    for bdir in sorted((campaign / "blocks").glob("block_*")):
        b = int(bdir.name.split("_")[1])
        by = {}
        for ln in (bdir / "attempts.jsonl").read_text().splitlines():
            if ln.strip():
                a = json.loads(ln)
                by.setdefault(a["unit_id"], []).append(a)
        for uid, al in by.items():
            out[(b, uid)] = final_attempt(al)
    return out


# ------------------------------------------------------------------ A reproduce pre-registered analysis

def section_a(out: Path, ledger: list) -> dict:
    from .analyze_cli import cmd_confirmatory
    d = out / "reproduced"
    cmd_confirmatory(CONF, PREREG, d)
    res = {}
    for f in ("rq1.csv", "rq2.csv", "units.csv"):
        a = pd.read_csv(REPO / "data/v2/processed/confirmatory" / f)
        b = pd.read_csv(d / f)
        same_bytes = (REPO / "data/v2/processed/confirmatory" / f).read_bytes() == (d / f).read_bytes()
        num_same = a.shape == b.shape
        if num_same:
            for c in a.columns:
                if pd.api.types.is_numeric_dtype(a[c]):
                    num_same &= bool(np.allclose(a[c].fillna(-9e99), b[c].fillna(-9e99), rtol=1e-12, atol=0))
                else:
                    from .release_repro import _num_equal
                    num_same &= all(_num_equal(x, y)[0] for x, y in
                                    zip(a[c].fillna("").tolist(), b[c].fillna("").tolist()))
        res[f] = {"byte_identical": same_bytes, "numerically_identical": num_same}
        if not num_same:
            ledger.append({"item": f"reproduced {f}", "frozen": "data/v2/processed/confirmatory/" + f,
                           "new": str((d / f).relative_to(REPO)), "kind": "REPRODUCTION_DIFFERENCE",
                           "explanation": "see reproduced/ diff"})
    return res


# ------------------------------------------------------------------ B threshold-supported analysis

def section_b(out: Path, ledger: list) -> dict:
    r = pd.read_csv(REPO / "data/v2/processed/confirmatory/rq1.csv")
    u = confirmatory_units()
    P = json.loads(PREREG.read_text())
    rows = []
    for c in P["rq1_cells"]:
        s = u[(u.placement == c["placement"]) & (u.coll == c["coll"]) & (u.size_bytes == int(c["size_bytes"]))]
        a = s[s.config == c["energy_slo"]].set_index("block")
        b = s[s.config == "auto"].set_index("block")
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        d = np.log(j.J_per_coll_a / j.J_per_coll_b).to_numpy()
        n = len(d)
        m, se, _ = t_ci(d)
        p_thr = float(stats.t.cdf((m - math.log(THRESH)) / se, n - 1))      # H0: ratio >= 0.95
        rows.append({"placement": c["placement"], "coll": c["coll"], "size_bytes": int(c["size_bytes"]),
                     "policy": c["energy_slo"], "n_blocks": n, "E_ratio": math.exp(m), "p_threshold_one_sided": p_thr})
    t = pd.DataFrame(rows)
    t["p_threshold_holm"] = holm(t.p_threshold_one_sided.tolist())
    t = t.merge(r[["placement", "coll", "size_bytes", "E_ratio_upper_bonf", "saving_confirmed", "saving_meaningful"]],
                on=["placement", "coll", "size_bytes"])
    t["threshold_supported_holm_0.025"] = t.p_threshold_holm < 0.025
    t["threshold_supported_bonferroni_bound"] = t.E_ratio_upper_bonf < THRESH
    t.to_csv(out / "rq1_threshold.csv", index=False)
    res = {"preregistered_meaningful_point_estimate": int(t.saving_meaningful.sum()),
           "threshold_supported_holm": int(t["threshold_supported_holm_0.025"].sum()),
           "threshold_supported_bonferroni_bound": int(t.threshold_supported_bonferroni_bound.sum()),
           "point_but_not_threshold": t[t.saving_meaningful & ~t["threshold_supported_holm_0.025"]][
               ["placement", "coll", "size_bytes", "E_ratio", "E_ratio_upper_bonf"]].to_dict("records")}
    ledger.append({"item": "RQ1 cells with >= 5 % saving", "frozen": "25 (rq1.csv saving_meaningful; point estimate)",
                   "new": f"{res['threshold_supported_holm']} threshold-supported (rq1_threshold.csv)",
                   "kind": "NEW_SEPARATELY_IDENTIFIED_ANALYSIS",
                   "explanation": "Pre-registered 'meaningful' = confirmed and point estimate <= 0.95 (unchanged, 25). "
                                  "Separate analysis: one-sided test of H0 ratio >= 0.95, Holm over 39 cells, alpha "
                                  "0.025 (same count with the Bonferroni bound)."})
    return res


# ------------------------------------------------------------------ C RQ2 estimator sensitivity

ESTIMATORS = {
    "counter": "Commit-aligned NVML cumulative counter (patch v2.1) minus the idle tail, per collective "
               "(units.csv J_per_coll).",
    "integrated_instant": "Trapezoidal integral of NVML_FI_DEV_POWER_INSTANT sampled at 50 ms over the region, summed "
                          "over GPUs, per collective (unit telemetry energy_int_inst_J / R).",
    "lag_adjusted_powerusage": "Trapezoidal integral of nvmlDeviceGetPowerUsage (about 1 s average) shifted by the "
                               "configured lag 0.75 s, summed over GPUs, per collective (energy_int_lag_J / R).",
    "idle_adjusted_counter": "Counter estimator minus block idle power x region duration per GPU "
                             "(units.csv J_idle_adj_per_coll).",
}


def rq2_inputs() -> pd.DataFrame:
    recs = unit_records(CONF)
    u = confirmatory_units()
    P = json.loads(PREREG.read_text())
    rows = []
    for c in P["rq2_cells"]:
        s = u[(u.placement == c["placement"]) & (u.coll == c["coll"]) & (u.size_bytes == int(c["size_bytes"]))]
        for _, x in s.iterrows():
            if x.config not in (c["lat_opt"], c["en_opt"]):
                continue
            a = recs[(int(x.block), x.unit_id)]
            R = int(a["R"])
            tel = a["telemetry"]
            rows.append({"cell": f"{c['placement']} {c['coll']} {int(c['size_bytes'])}", "block": int(x.block),
                         "arm": "lat_opt" if x.config == c["lat_opt"] else "en_opt", "config": x.config,
                         "unit_id": x.unit_id, "R": R,
                         "counter": x.J_per_coll,
                         "integrated_instant": sum(t["energy_int_inst_J"] for t in tel.values()) / R,
                         "lag_adjusted_powerusage": sum(t["energy_int_lag_J"] for t in tel.values()) / R,
                         "idle_adjusted_counter": x.J_idle_adj_per_coll})
    return pd.DataFrame(rows)


def section_c(out: Path, ledger: list) -> dict:
    inp = rq2_inputs()
    inp.to_csv(out / "rq2_estimator_inputs.csv", index=False)
    rows = []
    for cell, g in inp.groupby("cell", sort=False):
        w = g.pivot(index="block", columns="arm")
        for est in ESTIMATORS:
            d = np.log(w[est]["lat_opt"] / w[est]["en_opt"]).to_numpy()
            m, se, (lo, hi) = t_ci(d)
            _, _, (lo90, hi90) = t_ci(d, 0.90)
            rows.append({"cell": cell, "estimator": est, "n_blocks": len(d), "ratio_lat_over_en": math.exp(m),
                         "ci95": [math.exp(lo), math.exp(hi)], "ci90": [math.exp(lo90), math.exp(hi90)],
                         "p_penalty_one_sided": float(1 - stats.t.cdf(m / se, len(d) - 1)),
                         "tost_3pct_p": tost_p(d), "per_block_log_ratio": [float(x) for x in d]})
    res = pd.DataFrame(rows)
    for est in ESTIMATORS:
        m = res.estimator == est
        res.loc[m, "p_penalty_holm_2cells"] = holm(res.loc[m, "p_penalty_one_sided"].tolist())
    res.to_csv(out / "rq2_estimators.csv", index=False)
    (out / "rq2_estimator_definitions.json").write_text(json.dumps(
        {"estimators": ESTIMATORS, "test": "paired over blocks; ln(E_lat_opt/E_en_opt); one-sided t for penalty > 0; "
                                           "Holm over the 2 RQ2 cells within an estimator; TOST +-3 % (90 % CI)",
         "caveat": "The four estimators share the board's power sensor and its averaging; agreement among them is "
                   "not an independent physical calibration."}, indent=1))
    ar = res[res.cell.str.contains("all_reduce 1073741824")]
    ag = res[res.cell.str.contains("all_gather 2097152")]
    (out / "ar1g_tost.json").write_text(json.dumps({"inputs_file": "rq2_estimator_inputs.csv",
                                                    "results": ar.to_dict("records")}, indent=1, default=float))
    ledger.append({"item": "RQ2 AllGather 2 MiB tradeoff", "frozen": "DIFFER (counter estimator; rq2.csv)",
                   "new": "counter 1.0123 [1.0028, 1.0218]; instant 1.0048 [0.9905, 1.0194]; PowerUsage-lag 1.0092 "
                          "(Holm p 0.069); idle-adjusted 1.0214",
                   "kind": "WORDING_AND_SENSITIVITY",
                   "explanation": "Verdict depends on the estimator: not an estimator-robust physical tradeoff."})
    return {"ag2m": ag[["estimator", "ratio_lat_over_en", "ci95", "p_penalty_holm_2cells"]].to_dict("records"),
            "ar1g": ar[["estimator", "ratio_lat_over_en", "ci90", "tost_3pct_p"]].to_dict("records")}


# ------------------------------------------------------------------ D clustered order/temperature regression

def order_temperature_frame() -> pd.DataFrame:
    from .phase89 import paired_frame
    u = confirmatory_units()
    u = u.assign(logE=np.log(u.J_per_coll), logT=np.log(u.t_gpu_s))
    P = json.loads(PREREG.read_text())
    xs = []
    for ci, c in enumerate(P["rq1_cells"]):
        j = paired_frame(u, c, c["energy_slo"], "auto")
        if len(j) < 3:
            continue
        d = j.logE_a - j.logE_b
        dp = j.exec_pos_a - j.exec_pos_b
        dt = j.temp_mean_c_a - j.temp_mean_c_b
        xs.append(pd.DataFrame({"y": d - d.mean(), "dpos": (dp - dp.mean()) / 82.0, "dtemp": dt - dt.mean(),
                                "block": j.index.astype(int), "cell": ci}))
    return pd.concat(xs).dropna().reset_index(drop=True)


def _ols(X: pd.DataFrame):
    A, y = X[["dpos", "dtemp"]].to_numpy(), X.y.to_numpy()
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A, y, b, y - A @ b, np.linalg.inv(A.T @ A)


def _meat(A, e, groups):
    M = np.zeros((A.shape[1], A.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        s = A[m].T @ e[m]
        M += np.outer(s, s)
    return M


def section_d(out: Path, ledger: list) -> dict:
    X = order_temperature_frame()
    A, y, b, e, AtAi = _ols(X)
    n, k = A.shape
    G_cell = X.cell.nunique()
    res = {"n_pairs": int(n), "n_blocks": int(X.block.nunique()), "n_cells": int(G_cell),
           "coef": {"per_block_position": float(b[0]), "per_degC": float(b[1])}, "fits": {}}

    def add(name, V, df, note):
        se = np.sqrt(np.diag(V))
        q = stats.t.ppf(0.975, df)
        res["fits"][name] = {"se": se.tolist(), "t_df": int(df), "note": note,
                             "ci_position": [b[0] - q * se[0], b[0] + q * se[0]],
                             "ci_degC": [b[1] - q * se[1], b[1] + q * se[1]]}

    add("iid_published", AtAi * (e @ e) / (n - k), n - k,
        "as published (phase89.order_temperature): iid errors, df n-k; ignores the 39 within-cell centrings")
    add("iid_df_corrected", AtAi * (e @ e) / (n - k - G_cell), n - k - G_cell,
        "iid errors, df reduced by the 39 cell means removed by centring")
    for cl in ("block", "cell"):
        g = X[cl].to_numpy()
        G = len(np.unique(g))
        V = AtAi @ _meat(A, e, g) @ AtAi * G / (G - 1) * (n - 1) / (n - k)
        add(f"cluster_{cl}_CR1", V, G - 1, f"cluster-robust by {cl} (CR1), t with G-1 = {G - 1} df")
    gb, gc = X.block.to_numpy(), X.cell.to_numpy()
    Gb, Gc = len(np.unique(gb)), len(np.unique(gc))
    pair = np.arange(n)
    Vb = AtAi @ _meat(A, e, gb) @ AtAi * Gb / (Gb - 1)
    Vc = AtAi @ _meat(A, e, gc) @ AtAi * Gc / (Gc - 1)
    Vi = AtAi @ _meat(A, e, pair) @ AtAi * n / (n - 1)     # block x cell intersection = single pairs
    V2 = Vb + Vc - Vi
    w, vec = np.linalg.eigh(V2)
    V2 = vec @ np.diag(np.clip(w, 0, None)) @ vec.T          # eigenvalue fix for a non-PSD two-way estimate
    add("twoway_block_cell_CGM", V2, min(Gb, Gc) - 1,
        "Cameron-Gelbach-Miller two-way (block, cell); non-PSD part clipped; t with min(G)-1 = 11 df")
    # wild cluster restricted bootstrap by block: all 2^12 Rademacher sign vectors (exact enumeration)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=Gb)))
    blocks = np.unique(gb)
    idx = np.searchsorted(blocks, gb)
    wcr = {}
    for j, name in enumerate(("per_block_position", "per_degC")):
        keep = [i for i in range(k) if i != j]
        Ar = A[:, keep]
        br, *_ = np.linalg.lstsq(Ar, y, rcond=None)
        er = y - Ar @ br
        fit = Ar @ br

        def tstat(yy):
            bb = AtAi @ (A.T @ yy)
            ee = yy - A @ bb
            V = AtAi @ _meat(A, ee, gb) @ AtAi * Gb / (Gb - 1)
            return bb[j] / math.sqrt(V[j, j])

        t0 = tstat(y)
        ts = np.array([tstat(fit + er * s[idx]) for s in signs])
        wcr[name] = {"t_observed": float(t0), "p_two_sided": float(np.mean(np.abs(ts) >= abs(t0) - 1e-12)),
                     "n_sign_vectors": int(len(signs))}
    res["wild_cluster_restricted_bootstrap_block"] = wcr
    res["limitations"] = (
        "12 blocks and 39 cells are few clusters. CR1 with G-1 df can still be anti-conservative with 12 clusters; "
        "the wild cluster bootstrap (exact enumeration of the 4,096 sign vectors) is the more reliable p-value. "
        "Blocks ran back to back in one ~2 h session on one host: they are within-session replicates, not "
        "independent days or hosts. Position and temperature are partly confounded within a block.")
    (out / "order_temperature.json").write_text(json.dumps(res, indent=1, default=float))
    ledger.append({"item": "order/temperature CIs", "frozen": "iid OLS (order_temperature.json)",
                   "new": "clustered + two-way + wild bootstrap (order_temperature.json)", "kind": "REANALYSIS",
                   "explanation": "Same point estimates; intervals now block-/cell-aware. See fits."})
    return res


# ------------------------------------------------------------------ E boundary-read timing uncertainty

def section_e(out: Path, ledger: list) -> dict:
    rows = []
    for (b, uid), a in unit_records(CONF).items():
        if a.get("qc_status") != "OK":
            continue
        d = REPO / a["dir"]
        ln = (d / "region.jsonl").read_text().splitlines()
        rec = json.loads(ln[0])
        m = a["region"]
        E_tot = m["E_raw_J"]
        P = rec["commit_period_ns"]
        for g, pm in zip(rec["gpus"], m["per_gpu"]):
            rs = (g["t_start_b_ns"] - g["t_start_a_ns"]) * 1e-9
            re_ = (g["t_end_b_ns"] - g["t_end_a_ns"]) * 1e-9
            post_s = (g["t_commit_ns"] + P - (g["t_end_a_ns"] + g["t_end_b_ns"]) / 2) * 1e-9
            p_ctx, tail_s = pm.get("p_ctx_W"), pm.get("tail_s")
            tail_J = (p_ctx or 0) * (tail_s or 0)
            u_pctx_rel = (re_ / 2) / post_s                       # reset instant anywhere inside the end read
            u_tail_J = tail_J * u_pctx_rel + (p_ctx or 0) * (g["bracket_ns"] * 1e-9 / 2)   # + commit-time bracket
            u_start_J = (p_ctx or 0) * rs / 2                     # start boundary inside the start read
            rows.append({"block": b, "unit_id": uid, "rank": g["rank"], "read_start_ms": rs * 1e3,
                         "read_end_ms": re_ * 1e3, "post_window_ms": post_s * 1e3, "bracket_ms": g["bracket_ns"] / 1e6,
                         "tail_ms": (tail_s or 0) * 1e3, "p_ctx_W": p_ctx, "tail_J": tail_J,
                         "tail_rel_uncertainty": u_pctx_rel, "u_tail_J": u_tail_J, "u_start_J": u_start_J,
                         "gpu_region_J": pm["delta_J"], "unit_region_J": E_tot})
    t = pd.DataFrame(rows)
    t.to_csv(out / "read_latency.csv.gz", index=False, compression="gzip")
    per_unit = t.groupby(["block", "unit_id"]).agg(u_J=("u_tail_J", "sum"), u_s=("u_start_J", "sum"),
                                                   tail_J=("tail_J", "sum"), E=("unit_region_J", "first"))
    per_unit["u_total_J"] = per_unit.u_J + per_unit.u_s
    q = lambda s: {"median": float(s.median()), "p95": float(s.quantile(.95)), "max": float(s.max())}
    res = {"n_gpu_regions": int(len(t)), "n_units": int(len(per_unit)),
           "read_start_ms": q(t.read_start_ms), "read_end_ms": q(t.read_end_ms), "bracket_ms": q(t.bracket_ms),
           "tail_correction_magnitude_rel_to_region": q(per_unit.tail_J / per_unit.E),
           "tail_correction_relative_uncertainty_from_end_read": q(t.tail_rel_uncertainty),
           "boundary_uncertainty_J_per_unit_worstcase": q(per_unit.u_total_J),
           "boundary_uncertainty_rel_to_region_energy": q(per_unit.u_total_J / per_unit.E),
           "method": "Half-widths: the counter-reset instant may lie anywhere inside a read call (start and end "
                     "reads), and the commit time inside the phase bracket. Worst-case sum over the GPUs of a "
                     "unit. This is the interpretation supported by the local Phase 4 experiments, not an NVML "
                     "guarantee; sensor error and the averaging of the counter itself are not included."}
    (out / "read_latency.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ F sample spacing

def section_f(out: Path, ledger: list) -> dict:
    win, full = [], []
    for (b, uid), a in unit_records(CONF).items():
        if a.get("qc_status") != "OK":
            continue
        for t in (a.get("telemetry") or {}).values():
            win.append((t.get("gaps") or {}).get("max_dt"))
        f = REPO / a["dir"] / "telemetry.csv.gz"
        if f.exists():
            df = pd.read_csv(f, compression="gzip", usecols=["t_mono", "gpu"])
            for _, g in df.groupby("gpu"):
                tt = np.sort(g.t_mono.to_numpy())
                if len(tt) > 1:
                    full.append(float(np.diff(tt).max()))
    win = np.array([x for x in win if x is not None], float)
    full = np.array(full, float)
    res = {"measurement_window": {"n_gpu_regions": int(len(win)), "median_max_interval_s": float(np.median(win)),
                                  "p99_s": float(np.quantile(win, .99)), "max_s": float(win.max())},
           "complete_trace": {"n_gpu_traces": int(len(full)), "median_max_interval_s": float(np.median(full)),
                              "p99_s": float(np.quantile(full, .99)), "max_s": float(full.max()),
                              "n_traces_max_interval_gt_0.15s": int((full > 0.15).sum())},
           "note": "measurement_window = samples inside [region start, loop end] per GPU (what the energy integrals "
                   "use); complete_trace = the whole sampler trace of the unit, including the launch and teardown."}
    (out / "sampling_intervals.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ G QC reclassification

def reclassify(a: dict, T_min: float, energy_source: str = "counter") -> dict:
    """Apply the v2.2 fail-closed classify() to a stored v2.1 attempt record."""
    from .qc import classify, telemetry_problems
    proc = a.get("proc") or {}
    tel = a.get("telemetry") or {}
    m = a.get("region")
    clock_or, temp_max, pfrac = 0, None, 0.0
    for t in tel.values():
        clock_or |= int(t.get("clock_events_or", 0) or 0)
        if t.get("temp_max_c") is not None:
            temp_max = max(temp_max or -1, t["temp_max_c"])
        lim = (t.get("enforced_limit_mw") or [0])[0] / 1000.0
        if lim and t.get("power_mean_w") is not None:
            pfrac = max(pfrac, t["power_mean_w"] / lim)
    rows = a.get("stdout_rows") or []
    smp = a.get("sampler") or {}
    tp = telemetry_problems({"errors": smp.get("errors", 0), "truncated": smp.get("truncated", False)},
                            tel if m else None, len(tel) if tel else 0, energy_source)
    if m and len(tel) != len((m.get("per_gpu") or [])):
        tp = tp + ["TELEMETRY_GPU_MISSING"]
    ev = a.get("evidence_reverified") or (a.get("evidence") or {}).get("status")
    return classify({"exit_code": proc.get("rc"), "timed_out": proc.get("timed_out"), "region": m,
                     "wrong": rows[0]["wrong"] if len(rows) == 1 else None, "foreign_pids": a.get("foreign_pids"),
                     "clock_events_or": clock_or & ~0x1, "temp_max_c": temp_max, "power_frac_max": pfrac,
                     "temp_stop_c": 80, "sampled_energy_J": a.get("sampled_energy_J"), "telemetry_problems": tp,
                     "evidence_status": ev}, T_min, energy_source=energy_source)


def section_g(out: Path, ledger: list) -> dict:
    from .analysis import load_blocks, valid
    res, rows = {}, []
    for name, camp, T_min in (("exploratory", EXPL, 2.0), ("confirmatory", CONF, 2.0)):
        units, _, _ = load_blocks(camp)
        hist_ok = valid(units)
        ev = {(int(r.block), r.unit_id): r.evidence for r in units.itertuples()}
        dev = {(int(r.block), r.unit_id) for r in units.itertuples() if r.qc_status == "EXCLUDED_DEVIATION"}
        recs = unit_records(camp)
        new_status = {}
        for key, a in recs.items():
            if str(a.get("qc_status", "")).startswith("NOT_RUN"):
                continue
            a = dict(a, evidence_reverified=ev.get(key))
            st = reclassify(a, T_min)["qc_status"]
            if key in dev:
                st = "EXCLUDED_DEVIATION"
            new_status[key] = st
            rows.append({"campaign": name, "block": key[0], "unit_id": key[1], "v21_qc": a.get("qc_status"),
                         "v21_evidence": ev.get(key), "v22_qc": st})
        acc_old = {(int(r.block), r.unit_id) for r in hist_ok.itertuples()}
        acc_new = {k for k, v in new_status.items() if v == "OK"}
        res[name] = {"v21_accepted_ok_verified": len(acc_old), "v22_accepted_ok": len(acc_new),
                     "only_v21": sorted(map(list, acc_old - acc_new))[:20],
                     "only_v22": sorted(map(list, acc_new - acc_old))[:20],
                     "v22_status_counts": pd.Series(list(new_status.values())).value_counts().to_dict()}
    pd.DataFrame(rows).to_csv(out / "qc_reclassification.csv.gz", index=False, compression="gzip")
    (out / "qc_reclassification.json").write_text(json.dumps(res, indent=1, default=str))
    for name, r in res.items():
        if r["v21_accepted_ok_verified"] != r["v22_accepted_ok"]:
            ledger.append({"item": f"{name} accepted units", "frozen": r["v21_accepted_ok_verified"],
                           "new": r["v22_accepted_ok"], "kind": "QC_RULE_CHANGE",
                           "explanation": "see qc_reclassification.json only_v21/only_v22"})
    return res


# ------------------------------------------------------------------ H/I recommender and Phase 10 with v2.2 code

def section_h(out: Path, ledger: list) -> dict:
    import subprocess
    import sys
    d, f = out / "phase89_v22", out / "figures/phase89_v22"
    d.mkdir(parents=True, exist_ok=True)
    for x in ("rq1.csv",):
        (d / x).write_bytes((REPO / "data/v2/processed/confirmatory" / x).read_bytes())
    p = subprocess.run([sys.executable, "-m", "ncclenergy.v2.phase89", "--campaign", str(CONF), "--prereg", str(PREREG),
                        "--processed", str(d), "--figures", str(f)], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    if p.returncode != 0:
        raise SystemExit(p.stderr[-3000:])
    res = {}
    old = json.loads((REPO / "data/v2/processed/confirmatory/recommender_lobo_summary.json").read_text())
    new = json.loads((d / "recommender_lobo_summary.json").read_text())
    res["lobo_summary_identical"] = old == new
    lo, ln_ = pd.read_csv(REPO / "data/v2/processed/confirmatory/recommender_lobo.csv"), pd.read_csv(d / "recommender_lobo.csv")
    res["lobo_rows_identical"] = bool(lo.shape == ln_.shape and lo.astype(str).equals(ln_.astype(str)))
    rob_o = pd.read_csv(REPO / "data/v2/processed/confirmatory/robustness.csv")
    rob_n = pd.read_csv(d / "robustness.csv")
    res["robustness_identical"] = bool(rob_o.astype(str).equals(rob_n.astype(str)))
    conf_changes = {}
    for cf in sorted(d.glob("nccl_tuner_*.conf")):
        o = (REPO / "data/v2/processed/confirmatory" / cf.name).read_text().splitlines()
        n = cf.read_text().splitlines()
        conf_changes[cf.name] = {"frozen_rules": [l for l in o if l and not l.startswith("#")],
                                 "v22_rules": [l for l in n if l and not l.startswith("#")]}
    res["tuner_files"] = conf_changes
    (out / "recommender_v22.json").write_text(json.dumps(res, indent=1))
    if not res["lobo_summary_identical"] or not res["lobo_rows_identical"]:
        ledger.append({"item": "recommender LOBO", "frozen": "recommender_lobo*.{csv,json}",
                       "new": "phase89_v22/recommender_lobo*", "kind": "CODE_CHANGE",
                       "explanation": "v2.2 recommender guards / interpolation; see recommender_v22.json"})
    ledger.append({"item": "tuner files", "frozen": "nccl_tuner_*.conf (range rules, no scope header)",
                   "new": "phase89_v22/nccl_tuner_*.conf (one rule per measured size + #@scope header)",
                   "kind": "CODE_CHANGE",
                   "explanation": "range rules covered unmeasured sizes; v2.2 exports measured sizes only, for the "
                                  "guarded plugin"})
    return res


def section_i(out: Path, ledger: list) -> dict:
    from .phase10 import cmd_analyze
    d = out / "phase10_v22"
    cmd_analyze(d, figs_dir=out / "figures/phase10_v22")
    old = json.loads((REPO / "data/v2/processed/phase10/phase10_summary.json").read_text())
    new = json.loads((d / "phase10_summary.json").read_text())
    same = {k: old.get(k) == new.get(k) for k in old}
    return {"unchanged_sections": same, "all_rank_execution": new.get("all_rank_execution"),
            "numerics": sorted({c["status"] for c in new.get("numerics_comparisons", [])})}


# ------------------------------------------------------------------ J figures

def section_j(out: Path, b_res: dict, c_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = out / "figures"
    f.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(out / "rq1_threshold.csv").sort_values("E_ratio")
    fig, ax = plt.subplots(figsize=(7.5, 9))
    y = np.arange(len(t))
    lab = [f"{r.placement} {r.coll} {r.size_bytes}" for r in t.itertuples()]
    col = ["#1b7837" if s else ("#e08214" if mn else "#878787")
           for s, mn in zip(t["threshold_supported_holm_0.025"], t.saving_meaningful)]
    ax.hlines(y, t.E_ratio, t.E_ratio_upper_bonf, color=col, lw=2)
    ax.scatter(t.E_ratio, y, color=col, zorder=3, s=18)
    ax.axvline(THRESH, color="k", ls="--", lw=1)
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("energy ratio policy/automatic: point estimate and Bonferroni one-sided upper bound")
    ax.set_title("RQ1 (12 blocks). Green: <= 0.95 supported by the threshold test (Holm); orange: point\n"
                 "estimate <= 0.95 only (pre-registered 'meaningful'); grey: neither. Dashed line: 0.95",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f / "rq1_threshold.png", dpi=150)
    plt.close(fig)
    r = pd.read_csv(out / "rq2_estimators.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (cell, g) in zip(axes, r.groupby("cell", sort=False)):
        for i, row in enumerate(g.itertuples()):
            ci = json.loads(row.ci95.replace("'", '"')) if isinstance(row.ci95, str) else row.ci95
            ax.errorbar(row.ratio_lat_over_en, i, xerr=[[row.ratio_lat_over_en - ci[0]], [ci[1] - row.ratio_lat_over_en]],
                        fmt="o", color="#2166ac", capsize=3)
        ax.axvline(1.0, color="k", lw=0.8)
        for v in (1 / 1.03, 1.03):
            ax.axvline(v, color="grey", ls=":", lw=0.8)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.estimator, fontsize=8)
        ax.set_title(f"RQ2 {cell}\npre-selected latency / energy candidates, not confirmed optima", fontsize=8.5)
        ax.set_xlabel("energy ratio, 95 % CI over 12 within-session blocks")
    fig.tight_layout()
    fig.savefig(f / "rq2_estimators.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else REPO / a.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; reanalysis outputs are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    before = tree_hash(FROZEN_DIRS)
    ledger: list = []
    summary = {"A_reproduction": section_a(out, ledger), "B_threshold": section_b(out, ledger),
               "C_rq2_estimators": section_c(out, ledger), "D_order_temperature": section_d(out, ledger),
               "E_read_latency": section_e(out, ledger), "F_sampling": section_f(out, ledger),
               "G_qc_reclassification": section_g(out, ledger), "H_recommender": section_h(out, ledger),
               "I_phase10": section_i(out, ledger)}
    section_j(out, summary["B_threshold"], out)
    after = tree_hash(FROZEN_DIRS)
    summary["frozen_outputs_unchanged"] = before == after
    summary["frozen_files_hashed"] = len(before)
    if before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        summary["frozen_changed"] = changed
    pd.DataFrame(ledger).to_csv(out / "ledger.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: (v if k in ("frozen_outputs_unchanged", "frozen_files_hashed", "A_reproduction",
                                     "B_threshold", "G_qc_reclassification") else "...") for k, v in summary.items()},
                     indent=1, default=float))
    if before != after:
        raise SystemExit("FROZEN OUTPUTS CHANGED: " + ", ".join(summary["frozen_changed"][:10]))


if __name__ == "__main__":
    main()
