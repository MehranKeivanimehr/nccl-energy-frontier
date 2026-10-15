"""Block-level analysis for methodology v2 (exploratory and confirmatory).

Independent unit = block. Within a cell (placement x collective x size) the
block value of a configuration is ln(J/collective) (energy) and ln(t_gpu)
(latency). Paired comparisons use differences within the same block.
Nothing here pools pilot (v1) data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .attempts import final_attempt
from .common import COLL_FUNC, REPO, norm_pci
from .evidence import load_profiler_dir, verify
from .region import idle_adjusted

LAT_SLO = 1.05      # policy latency must be <= 1.05 x auto latency (pre-registered)
EFFECT = 0.05       # meaningful energy effect: >= 5 % lower raw J/collective


# ------------------------------------------------------------------ loading

def load_blocks(campaign_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (units, all_attempts, block_summaries)."""
    campaign_dir = Path(campaign_dir)
    rows, atts_all, summaries = [], [], []
    for bdir in sorted((campaign_dir / "blocks").glob("block_*")):
        summ = json.loads((bdir / "block_summary.json").read_text()) if (bdir / "block_summary.json").exists() else {}
        summ["dir"] = str(bdir)
        summaries.append(summ)
        idle = {}
        for k in ("idle_pre", "idle_post"):
            for pci_, v in ((summ.get(k) or {}).get("per_gpu") or {}).items():
                w = v.get("counter_W")
                if w is None:     # v2.2 sampled-energy campaigns: idle power from integrated sampled power
                    w = v.get("sampled_inst_W") if v.get("sampled_inst_W") is not None else v.get("sampled_mean_W")
                if w is not None:
                    idle.setdefault(norm_pci(pci_), []).append(w)
        idle_mean = {p: float(np.mean(v)) for p, v in idle.items()}
        atts = [json.loads(ln) for ln in (bdir / "attempts.jsonl").read_text().splitlines() if ln.strip()] \
            if (bdir / "attempts.jsonl").exists() else []
        dev = {}
        if (bdir / "deviations.jsonl").exists():
            for ln in (bdir / "deviations.jsonl").read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    dev[d["unit_id"]] = d
        by_unit: dict[str, list] = {}
        for a in atts:
            by_unit.setdefault(a["unit_id"], []).append(a)
            atts_all.append({"block": summ.get("block"), "unit_id": a["unit_id"], "attempt": a["attempt"],
                             "qc_status": a["qc_status"], "parent": a.get("parent_attempt"),
                             "qc_reasons": ";".join(a.get("qc_reasons") or [])})
        for uid, al in by_unit.items():
            a = final_attempt(al)
            m = a.get("region") or {}
            ev = reverify(a) or (a.get("evidence") or {})
            pl = a.get("placement") if isinstance(a.get("placement"), dict) else {}
            tel = a.get("telemetry") or {}
            qc_final = a["qc_status"]
            if uid in dev and dev[uid].get("action") == "EXCLUDE":
                qc_final = "EXCLUDED_DEVIATION"
            rec = {"block": summ.get("block"), "block_status": summ.get("status"), "unit_id": uid,
                   "qc_original": a["qc_status"], "deviation": (dev.get(uid) or {}).get("reason"),
                   "placement": uid.split("__")[0], "coll": uid.split("__")[1], "config": uid.split("__")[2].replace("-", "/"),
                   "size_bytes": int(uid.split("__")[3]), "qc_status": qc_final, "n_attempts": len(al),
                   "evidence": ev.get("status"), "actual": ev.get("actual_key"), "R": a.get("R"), "w": a.get("w"),
                   "exec_pos": a.get("exec_pos"), "planned_pos": a.get("planned_pos"),
                   "utc": (a.get("proc") or {}).get("utc_start"), "fingerprint": (pl.get("fingerprint") or {}).get("fingerprint"),
                   "J_per_coll": m.get("J_per_coll"), "E_raw_J": m.get("E_raw_J"), "t_gpu_s": m.get("t_gpu_s"),
                   "t_host_s": m.get("t_host_s"), "T_region_s": m.get("T_region_s"),
                   "region_flags": ";".join(m.get("flags") or []), "qc_reasons": ";".join(a.get("qc_reasons") or [])}
            if m.get("E_raw_J") is not None:
                adj = idle_adjusted(m, idle_mean)
                rec["J_idle_adj_per_coll"] = None if adj is None else adj / m["R"]
                per = m["per_gpu"]
                rec["tail_J"] = sum(p.get("tail_J") or 0 for p in per)
                rec["p_avg_W"] = sum(p["p_avg_W"] for p in per)
                inst = [tel.get(str(g), {}).get("energy_int_inst_J") for g in (pl.get("gpus") or [])]
                lag = [tel.get(str(g), {}).get("energy_int_lag_J") for g in (pl.get("gpus") or [])]
                rec["inst_over_counter"] = sum(inst) / m["E_raw_J"] if all(x is not None and np.isfinite(x) for x in inst) and inst else None
                rec["lag_over_counter"] = sum(lag) / m["E_raw_J"] if all(x is not None and np.isfinite(x) for x in lag) and lag else None
                temps = [tel.get(str(g), {}).get("temp_mean_c") for g in (pl.get("gpus") or [])]
                clks = [tel.get(str(g), {}).get("clk_gr_mean") for g in (pl.get("gpus") or [])]
                rec["temp_mean_c"] = float(np.mean([t for t in temps if t is not None])) if any(t is not None for t in temps) else None
                rec["temp_max_c"] = max((tel.get(str(g), {}).get("temp_max_c") or -1) for g in (pl.get("gpus") or []))
                rec["clk_gr_mean"] = float(np.mean([c for c in clks if c is not None])) if any(c is not None for c in clks) else None
            rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(atts_all), summaries


def reverify(a: dict) -> dict | None:
    """Recompute configuration evidence from the stored profiler files (single source of truth)."""
    m = a.get("region")
    if not m or not a.get("dir") or a.get("R") is None:
        return None
    files, _ = load_profiler_dir(REPO / a["dir"] / "prof")
    algo, proto = ("default", "default") if a["requested"] == "auto" else a["requested"].split("/")
    return verify(files, COLL_FUNC[a["coll"]], m["count"], len(a["placement"]["gpus"]), algo, proto,
                  min_expected=a["R"], exact_expected=a["R"] + a["w"] + 1)


def valid(units: pd.DataFrame, need_verified: bool = True) -> pd.DataFrame:
    v = units[(units.qc_status == "OK") & units.J_per_coll.notna() & units.t_gpu_s.notna()]
    if need_verified:
        v = v[v.evidence == "VERIFIED"]
    return v.assign(logE=np.log(v.J_per_coll), logT=np.log(v.t_gpu_s))


# ------------------------------------------------------------------ exploratory summaries

CELL = ["placement", "coll", "size_bytes"]


def cell_config_table(v: pd.DataFrame) -> pd.DataFrame:
    g = v.groupby(CELL + ["config"])
    t = g.agg(n_blocks=("block", "nunique"), J=("J_per_coll", "mean"), logE=("logE", "mean"), sdlogE=("logE", "std"),
              t_gpu=("t_gpu_s", "mean"), logT=("logT", "mean"), sdlogT=("logT", "std"),
              actual=("actual", lambda x: "|".join(sorted(set(x.dropna())))),
              J_idle_adj=("J_idle_adj_per_coll", "mean")).reset_index()
    t["edp"] = t.J * t.t_gpu
    return t


def collapse_effective(tab: pd.DataFrame) -> pd.DataFrame:
    """Merge requested configurations that executed the same (algo, proto, channels) in a cell.

    They are replicate measurements of one executed configuration; the representative label is
    'auto' when NCCL's own choice is among them, else the first forced label. Policies are chosen
    among these effective configurations so that noise between identical executions cannot create
    a spurious 'better' configuration.
    """
    rows = []
    for key, c in tab.groupby(CELL):
        for eff, g in c.groupby("actual"):
            members = sorted(g.config)
            w = g.n_blocks.to_numpy(float)
            rows.append({**dict(zip(CELL, key)), "actual": eff, "config": "auto" if "auto" in members else members[0],
                         "members": ";".join(members), "n_blocks": int(g.n_blocks.max()),
                         "logE": float(np.average(g.logE, weights=w)), "logT": float(np.average(g.logT, weights=w)),
                         "sdlogE": float(g.sdlogE.mean()), "sdlogT": float(g.sdlogT.mean()),
                         "J": float(np.exp(np.average(g.logE, weights=w))), "t_gpu": float(np.exp(np.average(g.logT, weights=w)))})
    return pd.DataFrame(rows)


def paired_vs_auto(v: pd.DataFrame) -> pd.DataFrame:
    """Per cell x config: block-paired differences of ln(J) and ln(t) against auto."""
    a = v[v.config == "auto"].set_index(CELL + ["block"])[["logE", "logT"]]
    out = []
    for key, grp in v[v.config != "auto"].groupby(CELL + ["config"]):
        gi = grp.set_index(CELL + ["block"])[["logE", "logT"]]
        j = gi.join(a, rsuffix="_auto", how="inner")
        if len(j) == 0:
            continue
        dE, dT = j.logE - j.logE_auto, j.logT - j.logT_auto
        out.append(dict(zip(CELL + ["config"], key), n_pairs=len(j), dE_mean=dE.mean(), dE_sd=dE.std(ddof=1),
                        dT_mean=dT.mean(), dT_sd=dT.std(ddof=1)))
    return pd.DataFrame(out)


def choose_policies(tab: pd.DataFrame, lat_slo: float = LAT_SLO) -> pd.DataFrame:
    """Per cell: auto, lat-opt, en-opt (raw), EDP-opt and Energy-SLO from block means."""
    out = []
    for key, c in tab.groupby(CELL):
        rec = dict(zip(CELL, key))
        if "auto" not in set(c.config):
            continue
        auto = c[c.config == "auto"].iloc[0]
        rec["lat_opt"] = c.loc[c.logT.idxmin(), "config"]
        rec["en_opt"] = c.loc[c.logE.idxmin(), "config"]
        rec["edp_opt"] = c.loc[(c.logE + c.logT).idxmin(), "config"]
        ok = c[np.exp(c.logT - auto.logT) <= lat_slo]
        rec["energy_slo"] = ok.loc[ok.logE.idxmin(), "config"] if len(ok) else "auto"
        rec["auto_actual"] = auto.actual
        for k in ("lat_opt", "en_opt", "edp_opt", "energy_slo"):
            r = c[c.config == rec[k]].iloc[0]
            rec[f"{k}_actual"] = r.actual
            rec[f"{k}_E_vs_auto"] = float(np.exp(r.logE - auto.logE))
            rec[f"{k}_T_vs_auto"] = float(np.exp(r.logT - auto.logT))
        out.append(rec)
    return pd.DataFrame(out)


def candidates(tab: pd.DataFrame, pol: pd.DataFrame, margin: float = 0.03) -> dict:
    """Confirmatory candidate configurations per cell: auto + policy picks + anything within
    `margin` of the best energy or best latency (exploratory block means)."""
    out = {}
    for key, c in tab.groupby(CELL):
        p = pol[(pol.placement == key[0]) & (pol.coll == key[1]) & (pol.size_bytes == key[2])]
        if p.empty:
            continue
        p = p.iloc[0]
        keep = {"auto", p.lat_opt, p.en_opt, p.edp_opt, p.energy_slo}
        keep |= set(c[np.exp(c.logE - c.logE.min()) <= 1 + margin].config)
        keep |= set(c[np.exp(c.logT - c.logT.min()) <= 1 + margin].config)
        out[f"{key[0]}|{key[1]}|{key[2]}"] = sorted(keep)
    return out


def required_blocks(sd: np.ndarray, half_width: float = math.log(1 + 0.025), n_max: int = 12,
                    frac: float = 0.8) -> dict:
    """Smallest n with predicted 95 % t-CI half-width <= half_width in >= frac of cells."""
    from scipy import stats
    sd = np.asarray([x for x in sd if np.isfinite(x)])
    for n in range(3, n_max + 1):
        hw = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        if np.mean(hw <= half_width) >= frac:
            return {"n": n, "frac_ok": float(np.mean(hw <= half_width)), "cells": len(sd)}
    hw = stats.t.ppf(0.975, n_max - 1) * sd / math.sqrt(n_max)
    return {"n": n_max, "frac_ok": float(np.mean(hw <= half_width)), "cells": len(sd), "cap_binds": True}


# ------------------------------------------------------------------ confirmatory statistics

def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def paired_test(d: np.ndarray, alternative: str = "less") -> dict:
    """t-based inference on block-paired log differences."""
    from scipy import stats
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return {"n": n}
    m, s = d.mean(), d.std(ddof=1)
    se = s / math.sqrt(n)
    tstat = m / se if se > 0 else np.inf * np.sign(m)
    p = stats.t.cdf(tstat, n - 1) if alternative == "less" else (1 - stats.t.cdf(tstat, n - 1) if alternative == "greater"
                                                                 else 2 * (1 - stats.t.cdf(abs(tstat), n - 1)))
    q = stats.t.ppf(0.975, n - 1)
    w = stats.wilcoxon(d).pvalue if n >= 5 and np.any(d != 0) else None
    hl = float(np.median([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)]))
    return {"n": n, "mean": m, "sd": s, "ci_lo": m - q * se, "ci_hi": m + q * se, "p": float(p),
            "wilcoxon_p_two_sided": w, "hodges_lehmann": hl, "per_block": d.tolist()}


def tost(d: np.ndarray, margin: float = math.log(1.03)) -> dict:
    """Two one-sided tests for equivalence within +/- margin (log scale)."""
    from scipy import stats
    d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        return {"n": n}
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return {"n": n, "p": 0.0 if abs(d.mean()) < margin else 1.0}
    p1 = 1 - stats.t.cdf((d.mean() + margin) / se, n - 1)
    p2 = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return {"n": n, "p": float(max(p1, p2))}


def block_bootstrap_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
