"""Validation checks for a smoke-test run (parsing + energy measurement).

Writes SMOKE_REPORT.md with PASS / FAIL / INFO lines; every number in the
report is computed from the run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .analyze import analyze_run, load_records
from .nccl_parse import parse_stdout
from .orchestrate import conditions, sizes_from_config


def smoke_check(run_dir: Path, out_dir: Path) -> bool:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    analyze_run(run_dir, out_dir, n_boot=2000)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    recs = load_records(run_dir)
    sel = pd.read_csv(out_dir / "selection.csv")
    reg = pd.read_csv(out_dir / "regions.csv")
    launches = pd.read_csv(out_dir / "launches.csv")
    q = json.loads((out_dir / "qc.json").read_text(encoding="utf-8"))
    lines, ok_all = [], True

    def check(name, passed, detail):
        nonlocal ok_all
        tag = "INFO" if passed is None else ("PASS" if passed else "FAIL")
        ok_all &= passed is not False
        lines.append(f"- **{tag}** {name}: {detail}")

    # 1 accounting: every planned unit has exactly one final record
    status = json.loads((run_dir / "gpu_set_status.json").read_text(encoding="utf-8"))
    active = {k for k, v in status.items() if v["status"] == "ok"}
    conds = [c for c in conditions(cfg) if c["gpu_set"] in active]
    planned = len(conds) * len(sizes_from_config(cfg)) * cfg["trials"]
    final = [r for r in recs if r.get("phase") == "trial" and r.get("final")]
    ids = [r["launch_id"] for r in final]
    check("accounting", len(set(ids)) == planned == len(ids),
          f"{planned} planned units, {len(ids)} final records, {len(set(ids))} unique; "
          f"status counts {pd.Series([r['status'] for r in final]).value_counts().to_dict()}")

    # 2 unsupported configurations are detected from NCCL's own error
    uns = sel[sel.probe_status == "unsupported"]
    tree_agrs = sel[(sel.algo == "Tree") & sel.coll.isin(["all_gather", "reduce_scatter"])]
    check("Tree for AllGather/ReduceScatter marked unsupported",
          len(tree_agrs) > 0 and (tree_agrs.probe_status == "unsupported").all()
          and tree_agrs.reason.fillna("").str.contains("no algorithm/protocol available").all(),
          f"{len(tree_agrs)} rows; unsupported conditions: {sorted(set(uns.coll + '/' + uns.gpu_set + '/' + uns.config))}")
    check("no other probe failures", set(sel.probe_status) <= {"ok", "unsupported"},
          f"probe status counts {sel.probe_status.value_counts().to_dict()}")

    # 3 forced algorithm/protocol actually used (TUNING lines)
    forced = sel[(sel.probe_status == "ok") & ((sel.algo != "default") | (sel.proto != "default"))]
    check("forced algo/proto honoured by NCCL", bool((forced.forced_honoured == True).all()),  # noqa: E712
          f"{int((forced.forced_honoured == True).sum())}/{len(forced)} forced (cond,size) rows match")  # noqa: E712
    auto = sel[(sel.probe_status == "ok") & (sel.algo == "default") & (sel.proto == "default")]
    check("auto selection recorded", None,
          "; ".join(f"{r.coll}/{r.gpu_set}/{r.size_bytes}B -> {r.selected}" for r in auto.itertuples()))

    # 4 correctness and 5 stability
    okp = sel[sel.probe_status == "ok"]
    check("data check (#wrong == 0) for every supported (cond,size)",
          bool(((okp.oop_wrong == 0) & (okp.ip_wrong == 0)).all()), f"{len(okp)} rows checked")
    if "selection_stable" in sel and sel.selection_stable.notna().any():
        check("selection identical in start and end probe", bool(sel.selection_stable.dropna().all()),
              f"{int(sel.selection_stable.dropna().sum())}/{int(sel.selection_stable.notna().sum())} stable")

    # 6 parser determinism: re-parse stdout from disk
    mism = 0
    n = 0
    for r in final:
        if r.get("dir") is None:
            continue
        n += 1
        rows = parse_stdout((run_dir / r["dir"] / "stdout.txt").read_text(encoding="utf-8"))["rows"]
        mism += rows != r["rows"]
    check("stdout re-parse identical to recorded rows", mism == 0, f"{n} launches re-parsed, {mism} mismatches")

    # 7 region timing
    check("all used regions have timed portion >= 2 s", bool(len(reg) and reg.timed_s.min() >= 2.0),
          f"timed_s min {reg.timed_s.min():.3f} s, region_s median {reg.region_s.median():.3f} s, "
          f"max {reg.region_s.max():.3f} s")
    ratio = reg.wall_per_op_us / reg.t_op_us
    for pl in ("oop", "ip"):
        rr = ratio[reg.placement == pl]
        check(f"region wall-clock per op vs nccl-tests time ({pl})", bool((rr - 1).abs().max() < 0.10),
              f"ratio median {rr.median():.4f}, min {rr.min():.4f}, max {rr.max():.4f}")

    # 8 energy measurement
    check("energy counter present and positive in every region",
          bool(len(reg) and (reg.energy_counter_j > 0).all()), f"{len(reg)} regions")
    check("idle-subtracted energy positive in every region", bool((reg.e_op_dyn_j > 0).all()),
          f"dynamic fraction of raw energy: median {(reg.e_op_dyn_j / reg.e_op_j).median():.3f}, "
          f"min {(reg.e_op_dyn_j / reg.e_op_j).min():.3f}")
    r_raw = reg.energy_integrated_j / reg.energy_counter_j
    r_lag = reg.energy_integrated_lag05_j / reg.energy_counter_j
    check("integrated PowerUsage vs energy counter (raw window)", None,
          f"ratio median {r_raw.median():.4f}, IQR [{r_raw.quantile(.25):.4f}, {r_raw.quantile(.75):.4f}], "
          f"min {r_raw.min():.4f}, max {r_raw.max():.4f}")
    check("integrated PowerUsage (window shifted +0.5 s) vs energy counter", bool(abs(r_lag.median() - 1) < 0.05),
          f"ratio median {r_lag.median():.4f}, IQR [{r_lag.quantile(.25):.4f}, {r_lag.quantile(.75):.4f}], "
          f"min {r_lag.min():.4f}, max {r_lag.max():.4f}")
    mono = []
    for key, g in reg[reg.placement == "oop"].groupby(["coll", "gpu_set", "config"]):
        m = g.groupby("size_bytes").e_op_j.mean().sort_index()
        mono.append(bool(np.all(np.diff(m.values) > 0)))
    check("energy/op increases with message size (every config)", all(mono), f"{sum(mono)}/{len(mono)} configs monotone")
    piv = reg.pivot_table(index=["coll", "gpu_set", "config", "size_bytes", "trial"], columns="placement",
                          values="e_op_j")
    d = (piv["oop"] / piv["ip"] - 1).dropna()
    check("out-of-place vs in-place energy/op", None,
          f"relative difference median {d.median():+.4f}, IQR [{d.quantile(.25):+.4f}, {d.quantile(.75):+.4f}]")
    cv = reg.groupby(["coll", "gpu_set", "config", "size_bytes", "placement"]).agg(
        e=("e_op_j", lambda x: x.std(ddof=1) / x.mean()), t=("t_op_us", lambda x: x.std(ddof=1) / x.mean()))
    check("trial-to-trial CV", None,
          f"energy/op CV median {cv.e.median():.4f} (max {cv.e.max():.4f}); latency CV median {cv.t.median():.4f} "
          f"(max {cv.t.max():.4f})")

    # 9 idle + sampler + interference
    check("idle measurements reached P8", bool(q.get("idle_reached_p8_all")),
          f"idle W by GPU {json.dumps(q.get('idle_power_w_by_gpu'), default=str)}")
    check("sampler cadence", bool(abs(q["sampler_median_dt_s"] - cfg["sampler"]["period_s"]) < 0.01),
          f"median dt {q['sampler_median_dt_s']:.4f} s (requested {cfg['sampler']['period_s']}), max dt "
          f"{q['sampler_max_dt_s']:.4f} s, NVML query {q['sampler_median_query_ms_per_gpu']:.2f} ms/GPU/sample, "
          f"overruns {q['sampler_overruns_total']}")
    fp = launches.foreign_pids_during.fillna("null")
    check("no foreign GPU processes during timed launches", bool((fp.isin(["null", "[]"])).all()),
          f"{int((~fp.isin(['null', '[]'])).sum())} launches with foreign PIDs")
    check("QC flags", None, json.dumps(q.get("flag_counts", {})))
    check("temperatures / clocks / P-states", None,
          f"temp mean range {q['temp_mean_c_range']}, max {q['temp_max_c']} C; graphics clock mean range "
          f"{q['clk_gr_mean_range']} MHz; P-states {q['pstates_seen']}")

    report = ["# Smoke-test report", "", f"Run directory: `{run_dir}`", "",
              f"Overall: **{'PASS' if ok_all else 'FAIL'}**", ""] + lines + [""]
    (out_dir / "SMOKE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return ok_all
