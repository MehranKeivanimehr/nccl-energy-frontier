"""Gate 3b measurement validation (v2.2-measval-01): pre-specified acceptance, decisions D1-D4 and descriptive outputs.
Written and committed before collection (docs/v2/v22_audit/stage3_plan_rev1.md §3, §4, §7). CPU only.

    python -m ncclenergy.v2.measval --campaign data/v2.2/raw/v2.2-measval-01 --out data/v2.2/processed/v2.2-measval-01

Fixed choices (where §7 left one open), recorded before collection:
* Unit of inference: the block. Every contrast/slope is formed within a block, then summarized over blocks with a
  t-interval (df = blocks - 1). A decision needs >= 3 blocks contributing, otherwise INCONCLUSIVE.
* D1 is decided per configuration (A auto, B forced Ring/Simple, C forced Ring/LL128) on the counter estimator;
  overall D1 = PASS if all three pass, LENGTH_EFFECT if any shows one, else INCONCLUSIVE.
* D2 is decided on the integrated POWER_INSTANT estimator (lag-adjusted PowerUsage reported with the same rule as
  secondary). Model per block: ln(E_est/E_counter) ~ 1 + 1/T + [protocol LL128].
* D3 is decided on the counter estimator at 10, 20 and 40 s; instant and lag estimators reported.
* D4: SUFFICIENT only if D1, D2 and D3 are PASS and the counter C/A half-width at 40 s <= 0.4 %. C/A values are
  descriptive measurement-validation output, not a test of the AllGather 2 MiB effect.
* Thresholds on the log scale: +-ln(1.003) (D1, D2), +-ln(1.005) (D3), ln(1.004) (D4).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import REPO, read_jsonl
from .evidence import load_profiler_dir

CFG = {"auto": "A", "Ring/Simple": "B", "Ring/LL128": "C"}
LENGTHS = (2.5, 10.0, 20.0, 40.0)
T_D1, T_D2, T_D3, T_D4 = math.log(1.003), math.log(1.003), math.log(1.005), math.log(1.004)
MIN_BLOCKS = 3
REL_LEN_TOL = 0.10


# ------------------------------------------------------------------ loading and acceptance

def _exec_set(d: Path):
    files, _ = load_profiler_dir(d / "prof")
    ex = sorted({(c["algo"], c["proto"], c["nChannels"], c["nWarps"]) for f in files for c in f["colls"]
                 if c["func"] == "AllGather"})
    return ex, sorted(f["rank"] for f in files)


def load(campaign: Path, frozen: dict) -> pd.DataFrame:
    rows = []
    for bdir in sorted((campaign / "blocks").glob("block_*")):
        b = int(bdir.name.split("_")[1])
        atts, _ = read_jsonl(bdir / "attempts.jsonl") if (bdir / "attempts.jsonl").exists() else ([], False)
        for a in atts:
            if str(a.get("qc_status", "")).startswith("NOT_RUN"):
                continue
            uid = a["unit_id"]
            m = a.get("region") or {}
            tel = a.get("telemetry") or {}
            ev = a.get("evidence") or {}
            fr = frozen["entries"].get(uid, {})
            d = REPO / a["dir"]
            ex, ranks = _exec_set(d) if (d / "prof").exists() else ([], [])
            rec = {}
            if (d / "region.jsonl").exists():
                lines = (d / "region.jsonl").read_text().splitlines()
                rec = json.loads(lines[0]) if lines else {}
            P = rec.get("commit_period_ns", 100_000_000)
            a1 = a2 = b1 = 0.0
            periods = []
            for g, pm in zip(rec.get("gpus") or [], m.get("per_gpu") or []):
                p_ctx, tail_s = pm.get("p_ctx_W"), pm.get("tail_s")
                if p_ctx is None or tail_s is None:
                    continue
                post_s = (g["t_commit_ns"] + P - (g["t_end_a_ns"] + g["t_end_b_ns"]) / 2) * 1e-9
                a1 += p_ctx * (g["t_start_b_ns"] - g["t_start_a_ns"]) * 1e-9 / 2
                a2 += p_ctx * tail_s * ((g["t_end_b_ns"] - g["t_end_a_ns"]) * 1e-9 / 2) / post_s
                b1 += p_ctx * g["bracket_ns"] * 1e-9 / 2
                mids = sorted((lo + hi) / 2 for lo, hi in g.get("edges") or [])
                periods += [(y - x) / max(1, round((y - x) / P)) / 1e6 for x, y in zip(mids, mids[1:])]
            e_inst = [t.get("energy_int_inst_J") for t in tel.values()]
            e_lag = [t.get("energy_int_lag_J") for t in tel.values()]
            ok_num = lambda v: len(v) == 2 and all(x is not None and np.isfinite(x) for x in v)
            rows.append({
                "block": b, "unit_id": uid, "config": a.get("requested"), "arm": CFG.get(a.get("requested")),
                "length_s": a.get("length_s"), "qc_status": a.get("qc_status"), "evidence": ev.get("status"),
                "n_matches_expected": ev.get("n_matches_expected"), "exec": ex, "ranks": ranks,
                "R": a.get("R"), "T_region_s": m.get("T_region_s"), "target_s": fr.get("target_region_s"),
                "E_counter": m.get("E_raw_J"), "E_inst": sum(e_inst) if ok_num(e_inst) else None,
                "E_lag": sum(e_lag) if ok_num(e_lag) else None,
                "tail_J": sum((p.get("tail_J") or 0) for p in m.get("per_gpu") or []),
                "A1_J": a1, "A2_J": a2, "B1_J": b1, "period_ms_median": float(np.median(periods)) if periods else None,
                "flags": ";".join(m.get("flags") or []), "exec_pos": a.get("exec_pos")})
    return pd.DataFrame(rows)


def accept(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified acceptance; adds 'accepted' and 'exclusion'."""
    reasons = []
    for r in df.itertuples():
        why = []
        if r.qc_status != "OK":
            why.append(f"QC_{r.qc_status}")
        if r.evidence != "VERIFIED":
            why.append("EVIDENCE")
        if r.n_matches_expected is not True:
            why.append("OP_COUNT_NOT_EXACT")
        if sorted(r.ranks) != [0, 1]:
            why.append("RANKS")
        if len(r.exec) != 1:
            why.append("NOT_SINGLE_EXECUTION")
        if r.T_region_s is None or r.target_s is None or abs(r.T_region_s / r.target_s - 1) > REL_LEN_TOL:
            why.append("REGION_LENGTH_OFF_TARGET")
        if r.E_counter is None:
            why.append("NO_COUNTER_ENERGY")
        reasons.append(";".join(why))
    return df.assign(exclusion=reasons, accepted=[not x for x in reasons])


def control_status(acc: pd.DataFrame) -> dict:
    """§3: per block, A and B are an identical-execution control only if every accepted A and B process executed the
    same single (algo, proto, channels, warps) on both ranks."""
    per = {}
    for b, g in acc.groupby("block"):
        ea = {tuple(x[0]) for x in g[g.arm == "A"].exec if len(x) == 1}
        eb = {tuple(x[0]) for x in g[g.arm == "B"].exec if len(x) == 1}
        per[int(b)] = bool(ea) and ea == eb and len(ea) == 1
    failing = [b for b, ok in per.items() if not ok]
    return {"per_block": per, "failing_blocks": failing,
            "status": "CONTROL_NOT_ESTABLISHED" if len(failing) >= 2 else "ESTABLISHED"}


# ------------------------------------------------------------------ statistics

def summarize(x) -> dict:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None, "ci": None}
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return {"n": n, "mean": float(m), "ci": [float(m - q * se), float(m + q * se)], "half_width": float(q * se),
            "per_block": [float(v) for v in x]}


def band_decision(s: dict, thr: float, inside: str, outside: str) -> str:
    if s["n"] < MIN_BLOCKS or s["ci"] is None:
        return "INCONCLUSIVE_INSUFFICIENT_BLOCKS"
    lo, hi = s["ci"]
    if -thr <= lo and hi <= thr:
        return inside
    if lo > thr or hi < -thr:
        return outside
    return "INCONCLUSIVE"


def d1(acc: pd.DataFrame) -> dict:
    out = {"per_config": {}}
    for arm in ("A", "B", "C"):
        g = acc[acc.arm == arm]
        lr, slopes, inters = {e: [] for e in ("counter", "inst", "lag")}, [], []
        for b, gb in g.groupby("block"):
            byL = {float(r.length_s): r for r in gb.itertuples()}
            if not all(L in byL for L in LENGTHS):
                continue
            R = np.array([byL[L].R for L in LENGTHS], float)
            E = np.array([byL[L].E_counter for L in LENGTHS], float)
            sl, ic = np.polyfit(R, E, 1)
            slopes.append(sl)
            inters.append(ic)
            for e, col in (("counter", "E_counter"), ("inst", "E_inst"), ("lag", "E_lag")):
                v25, v40 = getattr(byL[2.5], col), getattr(byL[40.0], col)
                if v25 is not None and v40 is not None:
                    lr[e].append(math.log((v25 / byL[2.5].R) / (v40 / byL[40.0].R)))
        s = summarize(lr["counter"])
        out["per_config"][arm] = {
            "ln_Jop_2.5_over_40": {e: summarize(v) for e, v in lr.items()},
            "slope_J_per_op": summarize(slopes), "intercept_J": summarize(inters),
            "decision": band_decision(s, T_D1, "PASS_NO_MATERIAL_LENGTH_EFFECT", "LENGTH_EFFECT")}
    ds = [v["decision"] for v in out["per_config"].values()]
    out["decision"] = ("PASS" if all(d == "PASS_NO_MATERIAL_LENGTH_EFFECT" for d in ds) else
                       "LENGTH_EFFECT" if any(d == "LENGTH_EFFECT" for d in ds) else "INCONCLUSIVE")
    return out


def d2(acc: pd.DataFrame) -> dict:
    out = {}
    for est, col in (("inst", "E_inst"), ("lag", "E_lag")):
        coefs = []
        for b, g in acc.groupby("block"):
            g = g[g[col].notna()]
            if len(g) < 4 or g.arm.eq("C").sum() == 0 or g.arm.ne("C").sum() == 0:
                continue
            y = np.log(g[col].astype(float) / g.E_counter.astype(float)).to_numpy()
            X = np.column_stack([np.ones(len(g)), 1.0 / g.T_region_s.astype(float), (g.arm == "C").astype(float)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pw = float((g.E_counter / g.T_region_s).mean())
            coefs.append((beta[0], beta[1], beta[2], beta[1] * pw))
        c = np.array(coefs) if coefs else np.zeros((0, 4))
        s_prot = summarize(c[:, 2]) if len(c) else {"n": 0, "ci": None}
        out[est] = {"intercept": summarize(c[:, 0]) if len(c) else None,
                    "per_inv_T": summarize(c[:, 1]) if len(c) else None,
                    "offset_J_per_region": summarize(c[:, 3]) if len(c) else None,
                    "protocol_LL128": s_prot,
                    "decision": band_decision(s_prot, T_D2, "PASS_PROTOCOL_INDEPENDENT", "PROTOCOL_DEPENDENT")}
    out["decision"] = {"PASS_PROTOCOL_INDEPENDENT": "PASS", "PROTOCOL_DEPENDENT": "PROTOCOL_DEPENDENT"}.get(
        out["inst"]["decision"], "INCONCLUSIVE")
    return out


def _paired(acc: pd.DataFrame, num: str, den: str, L: float, col: str, blocks=None) -> list[float]:
    vals = []
    for b, g in acc[acc.length_s.astype(float) == L].groupby("block"):
        if blocks is not None and int(b) not in blocks:
            continue
        a, c = g[g.arm == num], g[g.arm == den]
        if len(a) == 1 and len(c) == 1 and a[col].iloc[0] is not None and c[col].iloc[0] is not None:
            vals.append(math.log(a[col].iloc[0] / a.R.iloc[0] / (c[col].iloc[0] / c.R.iloc[0])))
    return vals


def d3(acc: pd.DataFrame, ctrl: dict) -> dict:
    good = {b for b, ok in ctrl["per_block"].items() if ok}
    out = {"per_length": {}}
    for L in LENGTHS:
        out["per_length"][L] = {e: summarize(_paired(acc, "B", "A", L, col, good))
                                for e, col in (("counter", "E_counter"), ("inst", "E_inst"), ("lag", "E_lag"))}
    if ctrl["status"] == "CONTROL_NOT_ESTABLISHED":
        out["decision"] = "INCONCLUSIVE_CONTROL_NOT_ESTABLISHED"
        return out
    ss = [out["per_length"][L]["counter"] for L in (10.0, 20.0, 40.0)]
    if any(s["n"] < MIN_BLOCKS or s["ci"] is None for s in ss):
        out["decision"] = "INCONCLUSIVE_INSUFFICIENT_BLOCKS"
    elif any(s["ci"][0] > 0 or s["ci"][1] < 0 for s in ss):
        out["decision"] = "SYSTEMATIC_OFFSET"
    elif all(-T_D3 <= s["ci"][0] and s["ci"][1] <= T_D3 for s in ss):
        out["decision"] = "PASS"
    else:
        out["decision"] = "INCONCLUSIVE"
    return out


def d4(acc: pd.DataFrame, r1: dict, r2: dict, r3: dict) -> dict:
    ca = {L: {e: summarize(_paired(acc, "C", "A", L, col))
              for e, col in (("counter", "E_counter"), ("inst", "E_inst"), ("lag", "E_lag"))} for L in LENGTHS}
    diff = []
    for b in sorted(acc.block.unique()):
        x25 = _paired(acc[acc.block == b], "C", "A", 2.5, "E_counter")
        x40 = _paired(acc[acc.block == b], "C", "A", 40.0, "E_counter")
        if x25 and x40:
            diff.append(x25[0] - x40[0])
    s40 = ca[40.0]["counter"]
    hw_ok = s40.get("ci") is not None and s40["n"] >= MIN_BLOCKS and s40["half_width"] <= T_D4
    ok = r1["decision"] == "PASS" and r2["decision"] == "PASS" and r3["decision"] == "PASS" and hw_ok
    return {"C_over_A_descriptive": ca, "C_over_A_2.5_minus_40_counter": summarize(diff),
            "half_width_40s_counter": s40.get("half_width"), "half_width_ok": bool(hw_ok),
            "decision": "SUFFICIENT" if ok else "NOT_SUFFICIENT",
            "note": "descriptive measurement-validation output; not a confirmatory test of the AllGather 2 MiB effect"}


def descriptive(df: pd.DataFrame, acc: pd.DataFrame, campaign: Path) -> dict:
    num = lambda c: pd.to_numeric(acc[c], errors="coerce")
    g = acc.assign(tail_frac=num("tail_J") / num("E_counter"),
                   timing_frac=(num("A1_J") + num("A2_J") + num("B1_J")) / num("E_counter"),
                   inst_over_counter=num("E_inst") / num("E_counter"), lag_over_counter=num("E_lag") / num("E_counter"),
                   period=num("period_ms_median"))
    tab = g.groupby(["arm", "length_s"]).agg(n=("block", "size"), tail_frac=("tail_frac", "median"),
                                             timing_frac=("timing_frac", "median"),
                                             inst_over_counter=("inst_over_counter", "median"),
                                             lag_over_counter=("lag_over_counter", "median"),
                                             period_ms=("period", "median")).reset_index()
    idle = []
    for bdir in sorted((campaign / "blocks").glob("block_*")):
        s = json.loads((bdir / "block_summary.json").read_text()) if (bdir / "block_summary.json").exists() else {}
        for k in ("idle_pre", "idle_post"):
            for pci, v in ((s.get(k) or {}).get("per_gpu") or {}).items():
                idle.append({"block": bdir.name, "when": k, "pci": pci, "counter_W": v.get("counter_W"),
                             "powerusage_W": v.get("sampled_mean_W"), "instant_W": v.get("sampled_inst_W")})
    return {"by_arm_length": tab.to_dict("records"), "flags": sorted({f for f in df.flags if f}),
            "idle_baselines": idle}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, default=REPO / "configs/v2.2/measval01_frozen.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = accept(load(a.campaign, json.loads(a.frozen.read_text())))
    df.to_csv(a.out / "processes.csv", index=False)
    acc = df[df.accepted]
    ctrl = control_status(acc)
    r1, r2, r3 = d1(acc), d2(acc), d3(acc, ctrl)
    r4 = d4(acc, r1, r2, r3)
    res = {"n_processes": int(len(df)), "n_accepted": int(len(acc)),
           "blocks": sorted(int(b) for b in df.block.unique()),
           "exclusions": df[~df.accepted][["block", "unit_id", "exclusion"]].to_dict("records"),
           "control": ctrl, "D1": r1, "D2": r2, "D3": r3, "D4": r4,
           "descriptive": descriptive(df, acc, a.campaign)}
    (a.out / "decisions.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"n_processes": res["n_processes"], "n_accepted": res["n_accepted"], "control": ctrl["status"],
                      "D1": r1["decision"], "D2": r2["decision"], "D3": r3["decision"], "D4": r4["decision"]}, indent=1))


if __name__ == "__main__":
    main()
