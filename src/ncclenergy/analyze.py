"""Raw run directory -> processed tables.

Outputs (in --out):
  selection.csv   per (collective, gpu set, requested algo/proto, size): probe status,
                  algorithm/protocol/channels NCCL actually selected, stability across
                  the start/end probes, rejection reason if any
  idle.csv        every idle measurement, per GPU
  launches.csv    every timed-launch attempt with status and QC flags
  regions.csv     one row per measured region (trial x condition x size x placement)
  exclusions.csv  regions not used in the statistics, with the reason
  summary.csv     per condition: trial means with bootstrap 95% intervals
  optima.csv      per (collective, gpu set, size, placement): latency-, energy-,
                  dynamic-energy- and EDP-optimal configuration, P(best), penalties
  pareto.csv      Pareto-front membership in (latency, energy) per cell
  qc.json         QC summary used by the smoke check and the README
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import mean_ci, pareto_front, prob_best, ratio_ci

KEEP_FLAGS_OK = {"wall_vs_reported_oop", "wall_vs_reported_ip", "few_samples_oop", "few_samples_ip",
                 "sampler_overruns"}
EXCLUDE_FLAGS = {"foreign_process_during", "no_counter_oop", "no_counter_ip"}
MIN_TRIALS = 3  # configurations with fewer valid trials are not ranked (smoke runs with T<3 use T)


def label(algo: str, proto: str) -> str:
    return f"{'auto' if algo == 'default' else algo}/{'auto' if proto == 'default' else proto}"


def load_records(run_dir: Path) -> list[dict]:
    lines = [ln for ln in (Path(run_dir) / "launches.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # record still being written by a running sweep
                break
            raise
    return out


def selection_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    probes = {}
    for r in records:
        if r.get("phase") in ("probe", "probe_end"):
            probes.setdefault(r["phase"], {})[(r["cond"]["coll"], r["cond"]["gpu_set"], r["cond"]["algo"],
                                                r["cond"]["proto"])] = r
    for key, r in sorted(probes.get("probe", {}).items()):
        end = probes.get("probe_end", {}).get(key)
        for size, p in r["per_size"].items():
            eff = "|".join(f"{s['algo']}/{s['proto']}[{s['ch_lo']}-{s['ch_hi']}]" for s in p.get("selected", []))
            eff_end = None
            if end is not None and size in end["per_size"]:
                pe = end["per_size"][size]
                eff_end = "|".join(f"{s['algo']}/{s['proto']}[{s['ch_lo']}-{s['ch_hi']}]" for s in pe.get("selected", []))
            sel = p.get("selected", [])
            rows.append({
                "coll": key[0], "gpu_set": key[1], "algo": key[2], "proto": key[3], "config": label(key[2], key[3]),
                "size_bytes": int(size), "probe_status": p["status"],
                "probe_end_status": None if end is None else end["per_size"].get(size, {}).get("status"),
                "selected": eff or None, "selected_end": eff_end,
                "selection_stable": None if end is None else (eff == eff_end),
                "sel_algo": sel[0]["algo"] if len(sel) == 1 else None,
                "sel_proto": sel[0]["proto"] if len(sel) == 1 else None,
                "forced_honoured": _forced_ok(key[2], key[3], sel),
                "oop_wrong": p.get("oop_wrong"), "ip_wrong": p.get("ip_wrong"),
                "reason": json.dumps(p.get("reason")) if p.get("reason") else None,
            })
    return pd.DataFrame(rows)


def _forced_ok(algo: str, proto: str, sel: list[dict]) -> bool | None:
    if not sel:
        return None
    ok = True
    if algo != "default":
        ok &= all(s["algo"].lower() == algo.lower() for s in sel)
    if proto != "default":
        ok &= all(s["proto"].lower() == proto.lower() for s in sel)
    return bool(ok)


def idle_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        if r.get("phase") != "idle":
            continue
        tag = r["launch_id"].removeprefix("idle-")
        trial = int(tag.split("-")[0][1:]) if tag.startswith("t") else None
        for g, v in (r.get("per_gpu") or {}).items():
            rows.append({"idle_id": r["launch_id"], "trial": trial, "status": r["status"], "gpu": int(g),
                         "t_wall_start": r["t_wall_start"], "duration_s": r.get("duration_s"),
                         "reached_p8": r.get("reached_p8"), "pstate_wait_s": r.get("pstate_wait_s"),
                         "power_counter_w": v.get("power_counter_w"),
                         "power_sampled_mean_w": v.get("power_sampled_mean_w"),
                         "temp_mean_c": v.get("temp_mean_c"), "pstates": str(v.get("pstates"))})
    return pd.DataFrame(rows)


def idle_lookup(idle: pd.DataFrame) -> dict:
    """Per (trial, gpu) idle power: median of that trial's valid idle windows."""
    ok = idle[(idle.status == "ok") & idle.power_counter_w.notna()] if len(idle) else idle
    out = {}
    if len(ok) == 0:
        return out
    overall = ok.groupby("gpu").power_counter_w.median().to_dict()
    for (t, g), grp in ok.groupby(["trial", "gpu"]):
        out[(t, g)] = float(grp.power_counter_w.median())
    out["overall"] = overall
    return out


def region_table(records: list[dict], idle_map: dict, sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eff = {}
    for _, s in sel.iterrows():
        eff[(s.coll, s.gpu_set, s.algo, s.proto, s.size_bytes)] = s.selected
    rows, launches = [], []
    for r in records:
        if r.get("phase") != "trial":
            continue
        c = r["cond"]
        base = {"launch_id": r["launch_id"], "trial": r["trial"], "attempt": r.get("attempt"), "final": r.get("final"),
                "coll": c["coll"], "gpu_set": c["gpu_set"], "n_gpus": len(c["gpus"]), "algo": c["algo"],
                "proto": c["proto"], "config": label(c["algo"], c["proto"]), "size_bytes": r["size_bytes"],
                "status": r["status"], "qc_flags": ";".join(r.get("flags") or [])}
        launches.append({**base, "iters": r.get("iters"), "warmup": r.get("warmup"), "rc": r.get("rc"),
                         "wall_s": r.get("wall_s"), "t_wall_start": r.get("t_wall_start"),
                         "foreign_pids_during": json.dumps(r.get("foreign_pids_during")),
                         "other_gpu_processes": json.dumps(r.get("other_gpu_processes")),
                         "load_avg_1m": (r.get("load_avg") or [None])[0]})
        if not r.get("final") or r["status"] not in ("ok", "ok_flagged"):
            continue
        n, w = r["iters"], r["warmup"]
        row = r["rows"][0]
        for g in r["regions"]:
            if g["size_bytes"] != r["size_bytes"] or not g.get("valid_boundaries"):
                continue
            pl = g["placement"]
            pg = g["per_gpu"]
            e_ctr = sum(v["energy_counter_j"] for v in pg.values()) if all(
                v["energy_counter_j"] is not None for v in pg.values()) else np.nan
            idle_w = 0.0
            idle_src = "trial"
            for gpu in c["gpus"]:
                v = idle_map.get((r["trial"], gpu))
                if v is None:
                    v = idle_map.get("overall", {}).get(gpu, np.nan)
                    idle_src = "overall"
                idle_w += v
            t_op = row[f"{pl}_time_us"] * 1e-6
            dur = g["duration_s"]
            ops = n + w
            e_op = e_ctr / ops
            e_dyn = (e_ctr - idle_w * dur) / ops
            size = r["size_bytes"]
            vals = list(pg.values())
            rows.append({
                **base, "placement": pl, "iters": n, "warmup": w, "ops_in_region": ops,
                "t_op_us": t_op * 1e6, "algbw_gbs": row[f"{pl}_algbw_gbs"], "busbw_gbs": row[f"{pl}_busbw_gbs"],
                "region_s": dur, "timed_s": n * t_op, "wall_per_op_us": dur / ops * 1e6,
                "energy_counter_j": e_ctr,
                "energy_integrated_j": sum(v["energy_integrated_j"] for v in vals),
                "energy_integrated_lag05_j": sum(v["energy_integrated_lag05_j"] for v in vals),
                "p_avg_w": e_ctr / dur, "idle_w": idle_w, "idle_source": idle_src,
                "e_op_j": e_op, "e_op_dyn_j": e_dyn, "e_op_rate_j": e_ctr / dur * t_op,
                "j_per_byte": e_op / size, "j_per_byte_dyn": e_dyn / size,
                "gb_per_j": size / e_op / 1e9, "gb_per_j_dyn": size / e_dyn / 1e9 if e_dyn > 0 else np.nan,
                "edp_js": e_op * t_op, "edp_dyn_js": e_dyn * t_op,
                "temp_mean_c": float(np.mean([v["temp_mean_c"] for v in vals])),
                "temp_max_c": float(np.max([v["temp_max_c"] for v in vals])),
                "clk_gr_mean_mhz": float(np.mean([v["clk_gr_mean_mhz"] for v in vals])),
                "clk_mem_mean_mhz": float(np.mean([v["clk_mem_mean_mhz"] for v in vals])),
                "util_gpu_mean": float(np.mean([v["util_gpu_mean"] for v in vals])),
                "util_mem_mean": float(np.mean([v["util_mem_mean"] for v in vals])),
                "pstates": str(sorted({p for v in vals for p in v["pstates"]})),
                "n_samples_min": int(min(v["n_samples"] for v in vals)),
                "t_wall_start": r["t_wall_start"],
                "selected": eff.get((c["coll"], c["gpu_set"], c["algo"], c["proto"], size)),
            })
    return pd.DataFrame(rows), pd.DataFrame(launches)


def split_exclusions(reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(reg) == 0:
        return reg, reg
    def reason(row):
        fl = set(filter(None, row["qc_flags"].split(";")))
        bad = fl & EXCLUDE_FLAGS
        if bad:
            return ";".join(sorted(bad))
        if not np.isfinite(row.energy_counter_j):
            return "no_counter"
        if row.timed_s < 2.0:
            return "timed_region_below_2s"
        return None
    reasons = reg.apply(reason, axis=1)
    return reg[reasons.isna()].copy(), reg[reasons.notna()].assign(exclusion_reason=reasons[reasons.notna()])


GROUP = ["coll", "gpu_set", "n_gpus", "algo", "proto", "config", "size_bytes", "placement"]
METRICS = ["t_op_us", "algbw_gbs", "busbw_gbs", "e_op_j", "e_op_dyn_j", "e_op_rate_j", "j_per_byte",
           "j_per_byte_dyn", "gb_per_j", "gb_per_j_dyn", "edp_js", "edp_dyn_js", "p_avg_w", "temp_mean_c",
           "clk_gr_mean_mhz"]


def summarize(reg: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for key, grp in reg.groupby(GROUP):
        rec = dict(zip(GROUP, key))
        rec["n_trials"] = int(grp.trial.nunique())
        rec["selected"] = grp.selected.iloc[0]
        for m in METRICS:
            mu, lo, hi = mean_ci(grp[m].to_numpy(), n_boot, rng)
            rec[m] = mu
            rec[f"{m}_lo"] = lo
            rec[f"{m}_hi"] = hi
            rec[f"{m}_cv"] = float(grp[m].std(ddof=1) / abs(grp[m].mean())) if len(grp) > 1 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


OBJECTIVES = {"lat": "t_op_us", "en": "e_op_j", "dyn": "e_op_dyn_j", "edp": "edp_js"}


def optima(reg: pd.DataFrame, summ: pd.DataFrame, n_boot: int, rng: np.random.Generator,
           min_trials: int = MIN_TRIALS) -> pd.DataFrame:
    out = []
    cell_keys = ["coll", "gpu_set", "n_gpus", "size_bytes", "placement"]
    for key, grp in reg.groupby(cell_keys):
        rec = dict(zip(cell_keys, key))
        counts = grp.groupby("config").trial.nunique()
        configs = sorted(counts[counts >= min_trials].index)
        rec["n_configs"] = len(configs)
        rec["configs_insufficient_trials"] = ";".join(sorted(counts[counts < min_trials].index)) or None
        if len(configs) < 2:
            out.append(rec)
            continue
        sel = grp.groupby("config").selected.first().to_dict()
        samples = {o: {c: grp[grp.config == c][m].to_numpy() for c in configs} for o, m in OBJECTIVES.items()}
        for o, m in OBJECTIVES.items():
            means = {c: float(np.mean(samples[o][c])) for c in configs}
            best = min(means, key=means.get)
            pb = prob_best(samples[o], n_boot, rng)
            rec[f"{o}_opt"] = best
            rec[f"{o}_opt_selected"] = sel.get(best)
            rec[f"{o}_opt_pbest"] = pb[best]
            rec[f"{o}_opt_mean"] = means[best]
        for o in ("en", "dyn", "edp"):
            m = OBJECTIVES[o]
            lat_c, o_c = rec["lat_opt"], rec[f"{o}_opt"]
            rec[f"disagree_{o}"] = lat_c != o_c
            rec[f"disagree_{o}_effective"] = (sel.get(lat_c) != sel.get(o_c)) if lat_c != o_c else False
            # cost (in objective o) of choosing the latency-optimal configuration
            pen, lo, hi = ratio_ci(samples[o][lat_c], samples[o][o_c], n_boot, rng)
            rec[f"{o}_penalty_of_lat_opt"] = pen
            rec[f"{o}_penalty_of_lat_opt_lo"] = lo
            rec[f"{o}_penalty_of_lat_opt_hi"] = hi
            # latency cost of choosing the o-optimal configuration
            pen, lo, hi = ratio_ci(samples["lat"][o_c], samples["lat"][lat_c], n_boot, rng)
            rec[f"lat_penalty_of_{o}_opt"] = pen
            rec[f"lat_penalty_of_{o}_opt_lo"] = lo
            rec[f"lat_penalty_of_{o}_opt_hi"] = hi
            # a disagreement is "resolved" when the energy penalty interval excludes 0
            rec[f"disagree_{o}_resolved"] = bool(rec[f"disagree_{o}_effective"] and lo is not None and
                                                 rec[f"{o}_penalty_of_lat_opt_lo"] > 0)
        out.append(rec)
    return pd.DataFrame(out)


def pareto_table(summ: pd.DataFrame, min_trials: int = MIN_TRIALS) -> pd.DataFrame:
    out = []
    keys = ["coll", "gpu_set", "n_gpus", "size_bytes", "placement"]
    for key, grp in summ[summ.n_trials >= min_trials].groupby(keys):
        for ycol in ("e_op_j", "e_op_dyn_j"):
            pts = [(r.config, r.t_op_us, getattr(r, ycol)) for r in grp.itertuples()]
            front = pareto_front(pts)
            for r in grp.itertuples():
                out.append({**dict(zip(keys, key)), "energy_metric": ycol, "config": r.config,
                            "selected": r.selected, "t_op_us": r.t_op_us, "energy_j": getattr(r, ycol),
                            "on_front": r.config in front})
    return pd.DataFrame(out)


def qc_summary(records, launches, reg, excl, idle, sel) -> dict:
    q = {}
    trial_final = launches[launches.final == True] if len(launches) else launches  # noqa: E712
    q["timed_launch_status_counts"] = trial_final.status.value_counts().to_dict() if len(trial_final) else {}
    q["timed_launch_attempts"] = int(len(launches))
    q["regions_used"] = int(len(reg))
    q["regions_excluded"] = int(len(excl))
    q["exclusion_reasons"] = excl.exclusion_reason.value_counts().to_dict() if len(excl) else {}
    if len(sel):
        q["probe_status_counts"] = sel.probe_status.value_counts().to_dict()
        q["forced_not_honoured"] = int((sel.forced_honoured == False).sum())  # noqa: E712
        q["selection_unstable"] = int((sel.selection_stable == False).sum())  # noqa: E712
        q["unsupported_conditions"] = sorted(
            {f"{r.coll}/{r.gpu_set}/{r.config}" for r in sel.itertuples() if r.probe_status == "unsupported"})
        q["incorrect_results"] = sorted(
            {f"{r.coll}/{r.gpu_set}/{r.config}/{r.size_bytes}" for r in sel.itertuples()
             if r.probe_status == "incorrect_results"})
    if len(reg):
        q["region_s"] = reg.region_s.describe().to_dict()
        q["timed_s_min"] = float(reg.timed_s.min())
        q["wall_vs_reported_ratio"] = (reg.wall_per_op_us / reg.t_op_us).describe().to_dict()
        r1 = reg.energy_integrated_j / reg.energy_counter_j
        r2 = reg.energy_integrated_lag05_j / reg.energy_counter_j
        q["integrated_over_counter"] = r1.describe().to_dict()
        q["integrated_lag05_over_counter"] = r2.describe().to_dict()
        q["dyn_fraction"] = (reg.e_op_dyn_j / reg.e_op_j).describe().to_dict()
        q["negative_dyn_energy_regions"] = int((reg.e_op_dyn_j <= 0).sum())
        q["temp_max_c"] = float(reg.temp_max_c.max())
        q["temp_mean_c_range"] = [float(reg.temp_mean_c.min()), float(reg.temp_mean_c.max())]
        q["clk_gr_mean_range"] = [float(reg.clk_gr_mean_mhz.min()), float(reg.clk_gr_mean_mhz.max())]
        q["pstates_seen"] = sorted(set(reg.pstates))
        q["n_samples_min"] = int(reg.n_samples_min.min())
        q["flag_counts"] = pd.Series([f for fl in launches["qc_flags"] for f in fl.split(";") if f]).value_counts().to_dict() \
            if len(launches) else {}
    if len(idle):
        ok = idle[idle.status == "ok"]
        q["idle_status_counts"] = idle.drop_duplicates("idle_id").status.value_counts().to_dict()
        q["idle_power_w_by_gpu"] = ok.groupby("gpu").power_counter_w.agg(["median", "min", "max", "count"]).to_dict("index")
        q["idle_reached_p8_all"] = bool(idle.reached_p8.all())
    samp = [r["sampler"] for r in records if r.get("phase") == "trial" and r.get("sampler")]
    if samp:
        dts = [v["median_dt_s"] for s in samp for v in s["per_gpu"].values() if v["median_dt_s"]]
        qms = [v["median_query_ms"] for s in samp for v in s["per_gpu"].values() if v["median_query_ms"]]
        q["sampler_median_dt_s"] = float(np.median(dts))
        q["sampler_max_dt_s"] = float(max(v["max_dt_s"] for s in samp for v in s["per_gpu"].values() if v["max_dt_s"]))
        q["sampler_median_query_ms_per_gpu"] = float(np.median(qms))
        q["sampler_overruns_total"] = int(sum(s["overruns"] for s in samp))
    return q


def analyze_run(run_dir: Path, out_dir: Path, n_boot: int = 10000, seed: int = 12345) -> None:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = load_records(run_dir)
    sel = selection_table(records)
    idle = idle_table(records)
    reg_all, launches = region_table(records, idle_lookup(idle), sel)
    reg, excl = split_exclusions(reg_all)
    min_trials = min(MIN_TRIALS, int(reg.trial.max())) if len(reg) else MIN_TRIALS
    summ = summarize(reg, n_boot, rng) if len(reg) else pd.DataFrame()
    opt = optima(reg, summ, n_boot, rng, min_trials) if len(reg) else pd.DataFrame()
    par = pareto_table(summ, min_trials) if len(summ) else pd.DataFrame()
    for name, df in (("selection", sel), ("idle", idle), ("launches", launches), ("regions", reg),
                     ("exclusions", excl), ("summary", summ), ("optima", opt), ("pareto", par)):
        df.to_csv(out_dir / f"{name}.csv", index=False)
    q = qc_summary(records, launches, reg, excl, idle, sel)
    q["n_boot"] = n_boot
    q["min_trials_for_ranking"] = min_trials
    q["seed"] = seed
    q["source_run"] = str(run_dir)
    (out_dir / "qc.json").write_text(json.dumps(q, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out_dir}: {len(reg)} regions, {len(excl)} excluded, {len(summ)} summary rows, {len(opt)} cells")
