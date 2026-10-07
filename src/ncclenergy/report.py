"""Generate results_summary.md / results_summary.json from processed tables.

Every number in the README's Results section is copied from this output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .plots import COLL_TITLE, fmt_bytes

SET_ORDER = {"2gpu_nvlink": 0, "4gpu": 1, "8gpu": 2}
COLL_ORDER = {"all_reduce": 0, "all_gather": 1, "reduce_scatter": 2}


def _pct(x, digits=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{digits}f} %"


def _ci(p, lo, hi, digits=1):
    return f"{_pct(p, digits)} [{_pct(lo, digits)}, {_pct(hi, digits)}]"


def _sorted_cells(df):
    return df.assign(_s=df.gpu_set.map(SET_ORDER), _c=df.coll.map(COLL_ORDER)).sort_values(
        ["_s", "_c", "size_bytes"]).drop(columns=["_s", "_c"])


def effective_mean_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Collapse requested configurations that ran the same algorithm/protocol/channels."""
    keys = ["coll", "gpu_set", "size_bytes", "placement", "selected"]
    return summ.groupby(keys, dropna=False).agg(t_op_us=("t_op_us", "mean"), e_op_j=("e_op_j", "mean"),
                                                e_op_dyn_j=("e_op_dyn_j", "mean"), p_avg_w=("p_avg_w", "mean"),
                                                n_req=("config", "count")).reset_index()


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build(processed: Path) -> dict:
    p = Path(processed)
    summ = pd.read_csv(p / "summary.csv")
    opt = pd.read_csv(p / "optima.csv")
    sel = pd.read_csv(p / "selection.csv")
    launches = pd.read_csv(p / "launches.csv")
    reg = pd.read_csv(p / "regions.csv")
    q = json.loads((p / "qc.json").read_text(encoding="utf-8"))
    out: dict = {"qc": q}

    # coverage
    final = launches[launches.final == True]  # noqa: E712
    out["coverage"] = {
        "final_units": int(len(final)),
        "status_counts": final.status.value_counts().to_dict(),
        "attempts": int(len(launches)),
        "retried_units": int(launches[launches.final == False].launch_id.nunique()),  # noqa: E712
        "regions_used": int(len(reg)),
        "trials": sorted(int(t) for t in reg.trial.unique()),
        "wall_clock_h": float((launches.t_wall_start.max() - launches.t_wall_start.min()) / 3600)
        if len(launches) else None,
        "gpu_sets": sorted(reg.gpu_set.unique()),
    }

    # NCCL's own choice (auto/auto) per size
    auto = sel[(sel.algo == "default") & (sel.proto == "default")]
    auto_rows = []
    for (coll, gs), g in auto.groupby(["coll", "gpu_set"]):
        g = g.sort_values("size_bytes")
        runs, prev, start = [], None, None
        for r in g.itertuples():
            if r.selected != prev:
                if prev is not None:
                    runs.append((start, last, prev))
                prev, start = r.selected, r.size_bytes
            last = r.size_bytes
        runs.append((start, last, prev))
        auto_rows.append({"coll": coll, "gpu_set": gs,
                          "ranges": [f"{fmt_bytes(a)}–{fmt_bytes(b)}: {s}" for a, b, s in runs]})
    out["auto_selection"] = auto_rows

    # disagreement summary
    o = opt[(opt.placement == "oop") & opt.lat_opt.notna()]
    rows = []
    for (coll, gs), g in o.groupby(["coll", "gpu_set"]):
        rec = {"coll": coll, "gpu_set": gs, "cells": int(len(g))}
        for k in ("en", "dyn", "edp"):
            d = g[g[f"disagree_{k}_effective"] == True]  # noqa: E712
            rec[f"{k}_disagree"] = int(len(d))
            rec[f"{k}_resolved"] = int((g[f"disagree_{k}_resolved"] == True).sum())  # noqa: E712
            rec[f"{k}_max_penalty"] = float(d[f"{k}_penalty_of_lat_opt"].max()) if len(d) else 0.0
        rows.append(rec)
    out["disagreement_by_group"] = rows
    tot = {k: int(sum(r[k] for r in rows)) for k in rows[0] if k not in ("coll", "gpu_set")} if rows else {}
    out["disagreement_total"] = tot

    # list of resolved disagreements (raw and dyn)
    det = []
    for k in ("en", "dyn"):
        d = o[o[f"disagree_{k}_resolved"] == True]  # noqa: E712
        for r in _sorted_cells(d).itertuples():
            det.append({
                "objective": k, "coll": r.coll, "gpu_set": r.gpu_set, "size_bytes": int(r.size_bytes),
                "lat_opt": r.lat_opt, "lat_opt_selected": r.lat_opt_selected, "lat_opt_pbest": r.lat_opt_pbest,
                "e_opt": getattr(r, f"{k}_opt"), "e_opt_selected": getattr(r, f"{k}_opt_selected"),
                "e_opt_pbest": getattr(r, f"{k}_opt_pbest"),
                "energy_penalty": getattr(r, f"{k}_penalty_of_lat_opt"),
                "energy_penalty_lo": getattr(r, f"{k}_penalty_of_lat_opt_lo"),
                "energy_penalty_hi": getattr(r, f"{k}_penalty_of_lat_opt_hi"),
                "latency_penalty": getattr(r, f"lat_penalty_of_{k}_opt"),
                "latency_penalty_lo": getattr(r, f"lat_penalty_of_{k}_opt_lo"),
                "latency_penalty_hi": getattr(r, f"lat_penalty_of_{k}_opt_hi"),
            })
    out["resolved_disagreements"] = det

    # rank correlation and spread of power vs time across distinct executed configurations
    eff = effective_mean_table(summ[summ.placement == "oop"])
    corr = []
    for (coll, gs, size), g in eff.groupby(["coll", "gpu_set", "size_bytes"]):
        if len(g) < 2:
            continue
        corr.append({"coll": coll, "gpu_set": gs, "size_bytes": int(size), "n_effective": int(len(g)),
                     "spearman_lat_energy": spearman(g.t_op_us, g.e_op_j),
                     "spearman_lat_dyn": spearman(g.t_op_us, g.e_op_dyn_j),
                     "time_ratio_max_min": float(g.t_op_us.max() / g.t_op_us.min()),
                     "power_ratio_max_min": float(g.p_avg_w.max() / g.p_avg_w.min())})
    corr = pd.DataFrame(corr)
    out["rank_correlation"] = {
        "cells": int(len(corr)),
        "cells_with_ge3_effective": int(corr.spearman_lat_energy.notna().sum()) if len(corr) else 0,
        "spearman_lat_energy_median": float(corr.spearman_lat_energy.median()) if len(corr) else None,
        "spearman_lat_energy_min": float(corr.spearman_lat_energy.min()) if len(corr) else None,
        "spearman_lat_dyn_median": float(corr.spearman_lat_dyn.median()) if len(corr) else None,
        "spearman_lat_dyn_min": float(corr.spearman_lat_dyn.min()) if len(corr) else None,
        "time_ratio_median": float(corr.time_ratio_max_min.median()) if len(corr) else None,
        "power_ratio_median": float(corr.power_ratio_max_min.median()) if len(corr) else None,
        "power_ratio_max": float(corr.power_ratio_max_min.max()) if len(corr) else None,
    }
    corr.to_csv(p / "rank_correlation.csv", index=False)

    # NCCL default vs optima
    a = summ[(summ.placement == "oop") & (summ.config == "auto/auto")].set_index(["coll", "gpu_set", "size_bytes"])
    best_e = summ[summ.placement == "oop"].groupby(["coll", "gpu_set", "size_bytes"]).e_op_j.min()
    best_t = summ[summ.placement == "oop"].groupby(["coll", "gpu_set", "size_bytes"]).t_op_us.min()
    pen_e = (a.e_op_j / best_e.reindex(a.index) - 1).dropna()
    pen_t = (a.t_op_us / best_t.reindex(a.index) - 1).dropna()
    out["auto_vs_best"] = {
        "cells": int(len(pen_e)),
        "energy_excess_median": float(pen_e.median()), "energy_excess_max": float(pen_e.max()),
        "energy_excess_max_cell": list(map(str, pen_e.idxmax())) if len(pen_e) else None,
        "latency_excess_median": float(pen_t.median()), "latency_excess_max": float(pen_t.max()),
        "latency_excess_max_cell": list(map(str, pen_t.idxmax())) if len(pen_t) else None,
        "cells_auto_within_1pct_energy": int((pen_e <= 0.01).sum()),
        "cells_auto_within_1pct_latency": int((pen_t <= 0.01).sum()),
    }

    # Compare the disagreement label between buffer placements.
    oi = opt[opt.lat_opt.notna()].pivot_table(index=["coll", "gpu_set", "size_bytes"], columns="placement",
                                              values="disagree_en_effective", aggfunc="first")
    if {"oop", "ip"} <= set(oi.columns):
        oi = oi.dropna()
        out["placement_agreement_en"] = {"cells": int(len(oi)),
                                         "same_disagreement_call": int((oi.oop == oi.ip).sum())}
    return out


def write_report(processed: Path) -> None:
    p = Path(processed)
    r = build(p)
    (p / "results_summary.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    L = ["# Results summary (generated)", "", f"Source: `{p.as_posix()}`", "",
         "Bootstrap intervals and optimum labels are descriptive, not confirmatory.", ""]
    c = r["coverage"]
    L += ["## Coverage", "",
          f"- final units: {c['final_units']} — status counts: {c['status_counts']}",
          f"- launch attempts: {c['attempts']} ({c['retried_units']} units needed a retry)",
          f"- regions used in statistics: {c['regions_used']} (excluded: {r['qc'].get('regions_excluded')}, "
          f"reasons {r['qc'].get('exclusion_reasons')})",
          f"- trials: {c['trials']}; GPU sets: {c['gpu_sets']}; wall clock of timed launches: "
          f"{c['wall_clock_h']:.2f} h", ""]
    L += ["## NCCL's own selection (auto/auto), from the probe", ""]
    for a in sorted(r["auto_selection"], key=lambda x: (SET_ORDER.get(x["gpu_set"], 9), COLL_ORDER[x["coll"]])):
        L.append(f"- {COLL_TITLE[a['coll']]} · {a['gpu_set']}: " + "; ".join(a["ranges"]))
    L += ["", "## Latency-optimal vs energy-optimal (out-of-place)", "",
          "| collective · GPUs | cells | disagree (raw E) | interval excludes 0 (raw E) | max penalty (raw E) | "
          "disagree (idle-sub. E) | interval excludes 0 (idle-sub. E) | max penalty (idle-sub. E) | disagree (EDP) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(r["disagreement_by_group"], key=lambda x: (SET_ORDER.get(x["gpu_set"], 9), COLL_ORDER[x["coll"]])):
        L.append(f"| {COLL_TITLE[d['coll']]} · {d['gpu_set']} | {d['cells']} | {d['en_disagree']} | "
                 f"{d['en_resolved']} | {_pct(d['en_max_penalty'])} | {d['dyn_disagree']} | {d['dyn_resolved']} | "
                 f"{_pct(d['dyn_max_penalty'])} | {d['edp_disagree']} |")
    t = r["disagreement_total"]
    if t:
        L.append(f"| **total** | {t['cells']} | {t['en_disagree']} | {t['en_resolved']} | | {t['dyn_disagree']} | "
                 f"{t['dyn_resolved']} | | {t['edp_disagree']} |")
    L += ["", "## Candidate disagreements whose descriptive interval excludes 0", "",
          "| objective | cell | latency-optimal (ran) | P(best) | energy-optimal (ran) | P(best) | "
          "energy penalty of latency-optimal [95% CI] | latency penalty of energy-optimal [95% CI] |",
          "|---|---|---|---|---|---|---|---|"]
    for d in r["resolved_disagreements"]:
        L.append(f"| {'raw' if d['objective'] == 'en' else 'idle-sub.'} | {COLL_TITLE[d['coll']]} · {d['gpu_set']} · "
                 f"{fmt_bytes(d['size_bytes'])}B | {d['lat_opt']} ({d['lat_opt_selected']}) | {d['lat_opt_pbest']:.2f} | "
                 f"{d['e_opt']} ({d['e_opt_selected']}) | {d['e_opt_pbest']:.2f} | "
                 f"{_ci(d['energy_penalty'], d['energy_penalty_lo'], d['energy_penalty_hi'])} | "
                 f"{_ci(d['latency_penalty'], d['latency_penalty_lo'], d['latency_penalty_hi'])} |")
    rc = r["rank_correlation"]
    L += ["", "## Latency vs energy across executed configurations within a cell", "",
          f"- cells with ≥2 distinct executed configurations: {rc['cells']} (≥3: {rc['cells_with_ge3_effective']})",
          f"- Spearman(latency, raw energy): median {rc['spearman_lat_energy_median']:.3f}, "
          f"min {rc['spearman_lat_energy_min']:.3f}",
          f"- Spearman(latency, idle-subtracted energy): median {rc['spearman_lat_dyn_median']:.3f}, "
          f"min {rc['spearman_lat_dyn_min']:.3f}",
          f"- max/min latency across executed configurations in a cell: median {rc['time_ratio_median']:.2f}×",
          f"- max/min average power across executed configurations in a cell: median "
          f"{rc['power_ratio_median']:.3f}×, max {rc['power_ratio_max']:.3f}×", ""]
    av = r["auto_vs_best"]
    L += ["## NCCL default (auto/auto) vs the best measured configuration", "",
          f"- cells: {av['cells']}",
          f"- energy above the energy-optimal configuration: median {_pct(av['energy_excess_median'])}, "
          f"max {_pct(av['energy_excess_max'])} at {av['energy_excess_max_cell']}; within 1 %: "
          f"{av['cells_auto_within_1pct_energy']} cells",
          f"- latency above the latency-optimal configuration: median {_pct(av['latency_excess_median'])}, "
          f"max {_pct(av['latency_excess_max'])} at {av['latency_excess_max_cell']}; within 1 %: "
          f"{av['cells_auto_within_1pct_latency']} cells", ""]
    if "placement_agreement_en" in r:
        pa = r["placement_agreement_en"]
        L += [f"- out-of-place vs in-place: same raw-energy disagreement call in {pa['same_disagreement_call']} of "
              f"{pa['cells']} cells", ""]
    (p / "results_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"report written to {p / 'results_summary.md'}")
