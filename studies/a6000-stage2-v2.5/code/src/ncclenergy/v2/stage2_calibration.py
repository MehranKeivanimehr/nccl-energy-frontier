"""Stage 2 calibration analysis (K1), frozen at Freeze-C. CPU only; reads unit records and raw files, never NVML.

Inputs: the calibration session directory written by stage2_runner (attempts.jsonl, units/<id>/{unit.json, region.jsonl,
s2.jsonl, telemetry.csv.gz}, waits/<id>/wait_trace.csv.gz). Only ACCEPTED units enter the estimates; every exclusion
is listed with its reasons. Calibration decides only what packet v3 section K1 lists:
  T_launch and the R2 bands (Q3); M4c retained or an amendment proposed; the secondary delay-shifted estimator added
  or not (Q4). It REPORTS, without turning them into bounds: the M4d end-alignment timing contribution (conditional
  on the constant-period model and on p_ctx; an observed maximum over calibration launches is not a bound on future
  launches under other load) and the temperature association (not a physical thermal bound).
The region end is the synchronized GPU-work end recorded by v2.5 (before the dose poller stops); sampled energy uses
each POWER_INSTANT sample's own query-bracket midpoint as its time. NVML field timestamps are carried as reported
("CPU timestamp of this value") and are not treated as physical acquisition times.
R and w are never re-frozen here (A and C were not measured in calibration); a B t_op deviation > 3 % is reported
and leads to an amendment proposal only.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from .common import read_jsonl
from .region import parse_region_text
from .stage2_decisions import mean_summary, q1
from .stage2_energy import CoverageError, InvalidCounter, e_aligned_gpu_J, sampled_energy_trapezoid_J
from .stage2_phase import aligned_hat_period, end_alignment_from_post_probe, estimate_period_phase
from .stage2_readiness import derive_heldout_rules

P_NOM_NS = 100e6
M4C_MAX_DEV_MS, M4C_MAX_BAD = 2.0, 1
M4C_MAX_TEMP_EFFECT_C, M4C_MAX_CLOCK_EFFECT_MHZ = 0.5, 15.0
END_ALIGN_MIN_LAUNCHES = 18             # of 23, per GPU, with an identified M4d alignment
T_OP_HIST_B = {"2.5": 2.5 / 47799, "10": 10.0 / 191193, "20": 20.0 / 382385, "40": 40.0 / 764769}


def load_units(session_dir: Path) -> list[dict]:
    recs, partial = read_jsonl(Path(session_dir) / "attempts.jsonl")
    if partial:
        raise ValueError("attempt ledger has a partial line; analysis refuses to guess")
    return recs


def _telemetry(path: Path) -> np.ndarray:
    with gzip.open(path, "rt") as f:
        r = csv.reader(f)
        head = next(r)
        rows = [[float(x) if x not in ("", "nan") else np.nan for x in row[:14]] for row in r]
    if head[0] != "t_mono" or head[13] != "power_inst_w":
        raise ValueError(f"unexpected telemetry header in {path}")
    return np.array(rows, float) if rows else np.zeros((0, 14))


def _telemetry_v3(path: Path) -> dict:
    """Schema-v3 columns by name (raw values). Raises if the file is not schema v3."""
    with gzip.open(path, "rt") as f:
        r = csv.reader(f)
        head = next(r)
        if "t_inst_a_ns" not in head:
            raise ValueError("not telemetry schema v3")
        cols = {k: [] for k in head}
        for row in r:
            for k, x in zip(head, row):
                cols[k].append(x)
    num = {k: np.array([float(x) if x not in ("", "nan") else np.nan for x in v]) for k, v in cols.items()}
    return num


def unit_energies(rec: dict, repo: Path) -> dict:
    """Per-unit energies: E_counter_raw, E_aligned(P_hat) (primary), E_aligned(P_nom), E_samp,trap,inst; per-GPU
    pieces kept. Unavailable estimators are None with a reason; nothing is imputed."""
    d = repo / rec["dir"]
    regs, flags = parse_region_text((d / "region.jsonl").read_text())
    if len(regs) != 1 or flags:
        return {"status": "NO_REGION", "flags": flags}
    reg = regs[0]
    v3 = _telemetry_v3(d / "telemetry.csv.gz")
    gpus = rec["gpus"]                          # rank -> NVML index of this session's placement (recorded)
    s2 = {p["rank"]: p for p in (rec.get("s2") or {}).get("per_gpu", [])}
    per, out = [], {"status": "OK", "reasons": []}
    for g in reg["gpus"]:
        t_work_end = (s2.get(g["rank"]) or {}).get("t_work_end_ns")
        if t_work_end is None:
            out["reasons"].append(f"NO_WORK_END_RANK{g['rank']}")
        gi = {"c0_mJ": int(g["e_start_mJ"]), "c1_mJ": int(g["e_end_mJ"]), "cpost_mJ": int(g["e_post_mJ"]),
              "rc0": g["rc_start"], "rc1": g["rc_end"], "rcpost": g["rc_post"],
              "t_loop_end_ns": t_work_end if t_work_end is not None else g["t_loop_end_ns"],
              "t_commit_ns": g["t_commit_ns"], "t_end_a_ns": g["t_end_a_ns"], "t_end_b_ns": g["t_end_b_ns"]}
        est = estimate_period_phase([tuple(e) for e in g.get("edges") or []])
        p_hat, why_hat = aligned_hat_period(est)
        p = {"rank": g["rank"], "period_status": est["status"], "period_ns": est.get("period_ns"),
             "primary_period": est.get("primary_period")}
        try:
            p["raw_J"] = (gi["c1_mJ"] - gi["c0_mJ"]) / 1000.0 if gi["c1_mJ"] >= gi["c0_mJ"] and not gi["rc0"] and not gi["rc1"] else None
            p["aligned_nom"] = e_aligned_gpu_J(gi, P_NOM_NS)
            p["aligned_hat"] = e_aligned_gpu_J(gi, p_hat) if p_hat is not None else None
        except InvalidCounter as e:
            p["error"] = str(e)
        if p_hat is None:
            p["aligned_hat_unavailable_reason"] = why_hat
            out["reasons"].append(f"ALIGNED_HAT_UNAVAILABLE_RANK{g['rank']}: {why_hat}")
        t0 = (g["t_start_a_ns"] + g["t_start_b_ns"]) / 2e9
        m = v3["gpu"] == gpus[g["rank"]]
        t_q = (v3["t_inst_a_ns"][m] + v3["t_inst_b_ns"][m]) / 2e9         # each sample's own query-bracket midpoint
        ok = v3["inst_status"][m] == 0
        try:
            p["samp_inst_J"] = sampled_energy_trapezoid_J(t_q[ok], v3["power_inst_w"][m][ok], t0, gi["t_loop_end_ns"] / 1e9,
                                                          0.15)
        except CoverageError as e:
            p["samp_inst_J"], p["samp_error"] = None, str(e)
        br = (v3["t_inst_b_ns"][m] - v3["t_inst_a_ns"][m]) / 1e6
        p["query_bracket_ms"] = {"median": float(np.median(br)) if len(br) else None, "max": float(br.max()) if len(br) else None}
        p["field_timestamp_us"] = {"n_distinct": int(len(np.unique(v3["inst_field_ts_us"][m]))),
                                   "label": "NVML 'CPU timestamp of this value'; not established as acquisition time"}
        per.append(p)
    out["per_gpu"] = per

    def tot(key, sub=None):
        v = [(p.get(key) or {}).get(sub) if sub else p.get(key) for p in per]
        return sum(v) if v and all(x is not None for x in v) else None
    out.update(E_counter_raw_J=tot("raw_J"), E_aligned_hat_J=tot("aligned_hat", "energy_J"),
               E_aligned_nom_J=tot("aligned_nom", "energy_J"), E_samp_inst_J=tot("samp_inst_J"), R=reg["R"])
    return out


def q1_from_units(recs: list[dict], energies: dict, estimator: str = "E_aligned_hat_J") -> dict:
    """C2 rounds -> Q1 (difference and equivalence evidence for D100/E0 and D500/E0 separately)."""
    rounds = []
    for r in (1, 2, 3):
        rd = {}
        for D in ("E0", "D100", "D500"):
            u = [x for x in recs if x.get("accepted") and x["unit"].get("part") == f"C2r{r}" and x["unit"].get("dose") == D]
            if len(u) != 1:
                rd[D] = None
                continue
            e = energies[u[0]["unit_id"]]
            R = e["R"]
            c, s = e.get(estimator), e.get("E_samp_inst_J")
            rd[D] = {"counter": None if c is None else c / R, "sampled": None if s is None else s / R}
        rounds.append(rd)
    return q1(rounds)


def readiness_inputs(recs: list[dict], repo: Path) -> tuple[list[dict], list[list[dict]]]:
    """Accepted C1/C2 launches -> per-GPU {t_launch_c (last wait row), t_region_c, clk_mhz, pstate} and the full
    recorded 1 Hz rows of their fixed waits (for replaying the complete R1 gate)."""
    cal, cool = [], []
    for r in recs:
        part = r["unit"].get("part", "")
        if not r.get("accepted") or not part.startswith(("C1", "C2")):
            continue
        rd = r.get("readiness") or {}
        pr = r.get("pre_region_state") or {}
        g = {}
        for k, v in pr.items():
            g[int(k)] = {"t_launch_c": (rd.get("t_launch_c") or {}).get(str(k), (rd.get("t_launch_c") or {}).get(k)),
                         "t_region_c": v.get("t_region_c"), "clk_mhz": v.get("clk_mhz"), "pstate": v.get("pstate")}
        cal.append({"gpus": g})
        if rd.get("trace_rows_file"):
            cool.append(json.loads((repo / rd["trace_rows_file"]).read_text()))
    return cal, cool


def q3_readiness(recs: list[dict], repo: Path) -> dict:
    cal, traces = readiness_inputs(recs, repo)
    gpus = recs[0]["gpus"] if recs else []
    return derive_heldout_rules(cal, traces, gpus)


def m4c_decision(recs: list[dict]) -> dict:
    """M4c retained unless |d_act - d_req| > 2 ms in more than 1 launch, or the pre-region state is associated with
    d_act by more than 0.5 C or 15 MHz across 0-100 ms (OLS slope x 0.1 s). Calibration data only, never energies."""
    n_bad, xs, temps, clks = 0, [], [], []
    for r in recs:
        s2 = r.get("s2") or {}
        per = s2.get("per_gpu", [])
        if any(abs(p["pause_dev_ms"]) > M4C_MAX_DEV_MS for p in per):
            n_bad += 1                                            # counted per launch
        st = r.get("pre_region_state") or {}
        for p in per:
            g = st.get(str(r["gpus"][p["rank"]])) or st.get(r["gpus"][p["rank"]]) or {}
            if g.get("t_region_c") is not None and g.get("clk_mhz") is not None:
                xs.append(p["pause_act_ns"] / 1e9)
                temps.append(g["t_region_c"])
                clks.append(g["clk_mhz"])
    eff = {}
    for name, y, lim in (("temp_c", temps, M4C_MAX_TEMP_EFFECT_C), ("clock_mhz", clks, M4C_MAX_CLOCK_EFFECT_MHZ)):
        if len(xs) >= 3 and np.ptp(xs) > 0:
            slope = float(np.polyfit(xs, y, 1)[0])
            eff[name] = {"effect_over_100ms": slope * 0.1, "limit": lim, "ok": abs(slope * 0.1) <= lim}
        else:
            eff[name] = {"effect_over_100ms": None, "limit": lim, "ok": False}
    retained = n_bad <= M4C_MAX_BAD and all(e["ok"] for e in eff.values())
    return {"decision": "RETAINED" if retained else "AMENDMENT_PROPOSED", "launches_dev_gt_2ms": n_bad,
            "association": eff}


def end_alignment_timing(recs: list[dict], energies: dict) -> dict:
    """M4d end alignment (calibration only), reported as a CONDITIONAL timing contribution, never as a bound.

    Per accepted launch and GPU: the post-probe brackets identify, under the constant-period model and its prior, the
    first update after the synchronized work end, and check that C1 and Cpost bracket it. The tail timing
    contribution p_ctx * (t_commit_used - t_update) is an interval conditional on that model and on p_ctx.
    Summary: status CONDITIONAL if every launch is ALIGNED_UNDER_MODEL (>= 18 per GPU), with the OBSERVED MAXIMUM of
    the per-launch intervals relative to that GPU's E_aligned per length - an observed maximum over idle post-probes,
    not a guaranteed bound for future launches or other load. Otherwise ALIGNMENT_NOT_ESTABLISHED. Not included: the
    context-power model, sensor/counter accuracy, the denominator and extrapolation to held-out load."""
    per_gpu, rel_by_len, bad, denom_fallbacks = {}, {}, [], []
    for r in recs:
        if not r.get("accepted") or not r["unit"].get("post_probe"):
            continue
        s2 = r.get("s2") or {}
        e = energies.get(r["unit_id"]) or {}
        for p in s2.get("per_gpu", []):
            raw = _raw_gpu(r, p["rank"])
            if raw is None or not (p.get("post_probe") or {}).get("brackets") or p.get("t_work_end_ns") is None:
                bad.append((r["unit_id"], p["rank"], "MISSING_INPUTS"))
                continue
            eg = next((x for x in e.get("per_gpu", []) if x["rank"] == p["rank"]), {})
            pctx = (eg.get("aligned_nom") or {}).get("p_ctx_W")
            a = end_alignment_from_post_probe([tuple(b) for b in p["post_probe"]["brackets"]], p["t_work_end_ns"],
                                              raw["t_commit_ns"], raw["t_end_a_ns"], raw["t_end_b_ns"],
                                              raw["t_post_a_ns"], raw["t_post_b_ns"], pctx)
            per_gpu.setdefault(p["rank"], []).append(a["status"])
            if a["status"] != "ALIGNED_UNDER_MODEL":
                bad.append((r["unit_id"], p["rank"], a["status"]))
                continue
            # relative size uses E_aligned(P_hat); if it is unavailable the nominal-period sensitivity value is used
            # as the denominator and that substitution is recorded (never silent)
            if (eg.get("aligned_hat") or {}).get("energy_J"):
                eh = eg["aligned_hat"]["energy_J"]
            else:
                eh = (eg.get("aligned_nom") or {}).get("energy_J")
                denom_fallbacks.append((r["unit_id"], p["rank"], eg.get("aligned_hat_unavailable_reason")))
            iv = a.get("conditional_tail_timing_contribution_J")
            if eh and eh > 0 and iv:
                L = float(r["unit"]["length_s"])
                rel_by_len[L] = max(rel_by_len.get(L, 0.0), max(abs(iv[0]), abs(iv[1])) / eh)
    enough = len(per_gpu) == 2 and all(sum(1 for x in v if x == "ALIGNED_UNDER_MODEL") >= END_ALIGN_MIN_LAUNCHES
                                       for v in per_gpu.values())
    label = ("observed maximum over calibration launches of model-conditional per-launch intervals (relative to "
             "E_aligned); not a bound on future launches")
    if bad or not enough:
        return {"status": "ALIGNMENT_NOT_ESTABLISHED", "problems": bad[:20], "launches_per_gpu": {k: len(v) for k, v in per_gpu.items()},
                "by_length": {L: {"status": "UNKNOWN"} for L in (2.5, 10.0, 20.0, 40.0)},
                "denominator_nominal_period_substitutions": denom_fallbacks}
    return {"status": "CONDITIONAL", "label": label,
            "by_length": {L: ({"status": "CONDITIONAL", "observed_max_rel": rel_by_len[L], "label": label}
                              if L in rel_by_len else {"status": "UNKNOWN", "reason": "no launch at this length"})
                          for L in (2.5, 10.0, 20.0, 40.0)},
            "denominator_nominal_period_substitutions": denom_fallbacks}


def _raw_gpu(rec: dict, rank: int) -> dict | None:
    return (rec.get("_raw_region_gpus") or {}).get(rank)


def attach_raw(recs: list[dict], repo: Path) -> None:
    """Attach the raw v2.1 per-GPU times needed by the M4d alignment (region.jsonl), keyed by rank."""
    for r in recs:
        p = repo / r["dir"] / "region.jsonl" if r.get("dir") else None
        if p is not None and p.exists():
            regs, _ = parse_region_text(p.read_text())
            if len(regs) == 1:
                r["_raw_region_gpus"] = {g["rank"]: g for g in regs[0]["gpus"]}


def thermal_slope(recs: list[dict], energies: dict) -> dict:
    """ASSOCIATION of ln(J) with the mean pre-region temperature over accepted C1 (E0) launches, adjusted for length
    and round. Never a physical thermal bound (the thermal systematic stays UNKNOWN). Refused (NOT_ESTIMABLE /
    CONFOUNDED) when: fewer than 9 launches; rank-deficient or ill-conditioned design; residual degrees of freedom
    < 3; residual variance indistinguishable from zero (relative to the response); or temperature nearly collinear
    with launch order or round (|corr| > 0.8 or VIF > 5)."""
    rows = []
    for r in recs:
        part = r["unit"].get("part", "")
        if not r.get("accepted") or not part.startswith("C1"):
            continue
        e = energies.get(r["unit_id"]) or {}
        st = r.get("pre_region_state") or {}
        t = [v.get("t_region_c") for v in st.values()]
        if e.get("E_aligned_hat_J") and t and all(isinstance(x, (int, float)) and math.isfinite(x) for x in t):
            rows.append((float(r["unit"]["length_s"]), part[:4], math.log(e["E_aligned_hat_J"] / e["R"]),
                         float(np.mean(t)), float(r["unit"].get("pos", len(rows)))))
    if len(rows) < 9:
        return {"status": "NOT_ESTIMABLE", "reason": f"{len(rows)} launches (< 9)"}
    Ls, rounds = sorted({x[0] for x in rows}), sorted({x[1] for x in rows})
    T = np.array([x[3] for x in rows])
    pos = np.array([x[4] for x in rows])
    X = np.array([[1.0 if x[0] == L else 0.0 for L in Ls] + [1.0 if x[1] == rr else 0.0 for rr in rounds[1:]] + [x[3]]
                  for x in rows])
    y = np.array([x[2] for x in rows])
    rank = np.linalg.matrix_rank(X)
    dof = len(y) - X.shape[1]
    if rank < X.shape[1] or dof < 3 or np.linalg.cond(X) > 1e8:
        return {"status": "NOT_ESTIMABLE", "reason": f"rank {rank}/{X.shape[1]}, dof {dof}, cond {np.linalg.cond(X):.3g}"}
    corr_pos = float(np.corrcoef(T, pos)[0, 1]) if np.std(T) > 0 and np.std(pos) > 0 else None
    others = X[:, :-1]
    fit_T = others @ np.linalg.lstsq(others, T, rcond=None)[0]
    ss_tot = float(np.sum((T - T.mean()) ** 2))
    r2 = 1 - float(np.sum((T - fit_T) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    vif = 1 / (1 - r2) if r2 < 1 else math.inf
    if corr_pos is None or abs(corr_pos) > 0.8 or vif > 5:
        return {"status": "CONFOUNDED", "corr_T_launch_order": corr_pos, "vif_T": vif,
                "reason": "temperature not separable from launch order / round / length"}
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    s2 = float(np.sum(resid ** 2)) / dof
    if not s2 > (1e-12 * max(1.0, float(np.max(np.abs(y))))) ** 2:
        return {"status": "NOT_ESTIMABLE", "reason": "residual variance indistinguishable from zero"}
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[-1, -1])
    q = float(stats.t.ppf(0.975, dof))
    b = float(beta[-1])
    return {"status": "ASSOCIATION_ONLY", "slope_per_C": b, "ci95": (b - q * se, b + q * se), "n": len(rows), "dof": dof,
            "corr_T_launch_order": corr_pos, "vif_T": vif,
            "label": "association conditional on this design; not a causal effect and not a bound"}


def t_op_check(recs: list[dict]) -> dict:
    """B's observed t_op vs the historical value; > 3 % -> amendment proposal (R/w are never re-frozen here)."""
    dev = []
    for r in recs:
        reg = r.get("region") or {}
        if r.get("accepted") and reg.get("t_gpu_s"):
            L = f"{float(r['unit']['length_s']):g}"
            dev.append(reg["t_gpu_s"] / T_OP_HIST_B[L] - 1)
    s = mean_summary(dev, len(dev), "launches") if len(dev) >= 2 else {"status": "INSUFFICIENT"}
    worst = max((abs(d) for d in dev), default=None)
    return {"n": len(dev), "max_abs_dev": worst, "summary": s,
            "action": "AMENDMENT_PROPOSED" if worst is not None and worst > 0.03 else "NONE"}


def q2_channels(t_s: np.ndarray, inst_w: np.ndarray, field_ts_us: np.ndarray) -> dict:
    """Descriptive (Yang et al. sec. 4.1 style): intervals between value changes of POWER_INSTANT, run lengths of
    repeated values, repeated fraction, field-timestamp intervals and whether the field timestamp advances while the
    value repeats. No decision uses these."""
    ok = np.isfinite(inst_w) & np.isfinite(t_s)
    t, v, ts = t_s[ok], inst_w[ok], field_ts_us[ok]
    if len(v) < 3:
        return {"status": "TOO_FEW_ROWS", "n": int(len(v))}
    ch = np.flatnonzero(np.diff(v) != 0) + 1
    runs = np.diff(np.concatenate([[0], ch, [len(v)]]))
    rep = np.diff(v) == 0
    ts_adv = np.diff(ts) > 0
    out = {"status": "OK", "n": int(len(v)), "repeated_fraction": float(rep.mean()),
           "run_length_median": float(np.median(runs)), "run_length_max": int(runs.max())}
    if len(ch) >= 2:
        iv = np.diff(t[ch])
        out.update(change_interval_ms_median=float(np.median(iv)) * 1e3, change_interval_ms_p10=float(np.quantile(iv, 0.1)) * 1e3,
                   change_interval_ms_p90=float(np.quantile(iv, 0.9)) * 1e3)
    tsu = np.unique(ts)
    if len(tsu) >= 2:
        out["field_ts_interval_ms_median"] = float(np.median(np.diff(tsu))) / 1e3
    out["field_ts_advances_on_repeated_value_frac"] = float(ts_adv[rep].mean()) if rep.any() else None
    return out


def q4_delay(units: list[dict]) -> dict:
    """H_delay. Per accepted E0 launch: d_eff = mean over GPUs of (E_aligned(P_hat)_g - E_samp,trap,inst_g) /
    (P_region_g - P_before_g) [s], P_region_g = E_aligned_g / region duration, P_before_g = mean POWER_INSTANT over
    [C0 - 1 s, C0). units: [{"gpus": [{"e_aligned_J", "e_samp_J", "duration_s", "p_before_W"}]}]. A delay-shifted
    sampled estimator is added to held-out as SECONDARY only if the 95 % interval of d_eff over launches excludes 0."""
    vals = []
    for u in units:
        d = []
        for g in u["gpus"]:
            dp = g["e_aligned_J"] / g["duration_s"] - g["p_before_W"]
            if not dp > 0:
                d = None
                break
            d.append((g["e_aligned_J"] - g["e_samp_J"]) / dp)
        if d:
            vals.append(float(np.mean(d)))
    s = mean_summary(vals, len(vals), "launches") if len(vals) >= 3 else {"status": "INSUFFICIENT"}
    add = s.get("status") == "OK" and (s["ci95"][0] > 0 or s["ci95"][1] < 0)
    return {"d_eff_s": s, "decision": "ADD_SECONDARY_DELAY_SHIFTED_ESTIMATOR" if add else "NO_SECONDARY_ESTIMATOR",
            "n": len(vals)}


def q4_inputs(recs: list[dict], energies: dict, repo: Path) -> list[dict]:
    """Per accepted E0 launch and GPU: E_aligned, E_samp,trap,inst, the region duration C0 -> synchronized work end and
    P_before = mean POWER_INSTANT over [C0 - 1 s, C0) at each sample's own query-bracket midpoint."""
    out = []
    for r in recs:
        if not r.get("accepted") or r["unit"].get("dose", "E0") != "E0":
            continue
        e = energies.get(r["unit_id"]) or {}
        v3 = _telemetry_v3(repo / r["dir"] / "telemetry.csv.gz")
        s2 = {x["rank"]: x for x in (r.get("s2") or {}).get("per_gpu", [])}
        gl = []
        for p in (r.get("region") or {}).get("per_gpu", []):
            eg = next((x for x in e.get("per_gpu", []) if x["rank"] == p["rank"]), {})
            ea = (eg.get("aligned_hat") or {}).get("energy_J")
            t_end = (s2.get(p["rank"]) or {}).get("t_work_end_ns")
            m = v3["gpu"] == r["gpus"][p["rank"]]
            tq = (v3["t_inst_a_ns"][m] + v3["t_inst_b_ns"][m]) / 2e9
            pw = v3["power_inst_w"][m]
            c0 = p["t_start_ns"] / 1e9
            pb = pw[(tq < c0) & (tq >= c0 - 1.0) & (v3["inst_status"][m] == 0) & np.isfinite(pw)]
            if ea is None or eg.get("samp_inst_J") is None or len(pb) < 5 or t_end is None:
                gl = None
                break
            gl.append({"e_aligned_J": ea, "e_samp_J": eg["samp_inst_J"], "duration_s": t_end / 1e9 - c0,
                       "p_before_W": float(pb.mean())})
        if gl:
            out.append({"unit_id": r["unit_id"], "gpus": gl})
    return out


def q2_from_units(recs: list[dict], repo: Path) -> dict:
    out = {}
    for r in recs:
        if not r.get("accepted"):
            continue
        with gzip.open(repo / r["dir"] / "telemetry.csv.gz", "rt") as f:
            rd = csv.reader(f)
            head = next(rd)
            if "inst_field_ts_us" not in head:
                out[r["unit_id"]] = {"status": "NOT_SCHEMA_V3"}
                continue
            i_t, i_g, i_p, i_ts = (head.index(k) for k in ("t_mono", "gpu", "power_inst_w", "inst_field_ts_us"))
            rows = [(float(x[i_t]), int(x[i_g]), float(x[i_p]), float(x[i_ts])) for x in rd]
        a = np.array(rows, float)
        out[r["unit_id"]] = {str(g): q2_channels(a[a[:, 1] == g, 0], a[a[:, 1] == g, 2], a[a[:, 1] == g, 3])
                             for g in r["gpus"]}
    return out


def analyse(session_dir: Path, repo: Path) -> dict:
    if Path(session_dir).name == "smoke" or str(Path(session_dir).parent.name).endswith("-smoke"):
        raise ValueError("smoke data are a software gate only and never enter the calibration analysis")
    recs = load_units(session_dir)
    attach_raw(recs, repo)
    energies = {r["unit_id"]: unit_energies(r, repo) for r in recs if r.get("accepted")}
    out = {"n_units": len(recs), "n_accepted": len(energies),
           "excluded": {r["unit_id"]: r.get("not_accepted_reasons") or r.get("qc_status") for r in recs if not r.get("accepted")},
           "energies": energies,
           "q1_primary": q1_from_units(recs, energies, "E_aligned_hat_J"),
           "q1_counter_raw": q1_from_units(recs, energies, "E_counter_raw_J"),
           "q3_readiness": q3_readiness(recs, repo), "m4c": m4c_decision(recs),
           "end_alignment_timing": end_alignment_timing(recs, energies), "thermal_slope": thermal_slope(recs, energies),
           "t_op": t_op_check(recs), "q4_delay": q4_delay(q4_inputs(recs, energies, repo)),
           "q2_channels": q2_from_units(recs, repo)}
    for r in recs:
        r.pop("_raw_region_gpus", None)
    return json.loads(json.dumps(out, default=str))
