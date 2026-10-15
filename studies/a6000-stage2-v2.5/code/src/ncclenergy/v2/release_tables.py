"""Pilot-release tables, figures and consistency checks (CPU only, no NVML), generated from retained outputs.

    python -m ncclenergy.v2.release_tables --out logs/new_release_tables

Writes into --out:
  key_numbers.csv        every headline number quoted in the release documents, recomputed from the processed
                         outputs, with its evidence category, source file and the documents that must quote it
  gate3b_decisions.csv   pre-specified gate 3b quantities (D1-D4) as tables
  gate3b_posthoc.csv     post-hoc exploratory quantities (labeled)
  counts.json            accepted/excluded counts recomputed from the raw records and compared with processed outputs
  doc_check.json         every key number found (or not) verbatim in the documents that must quote it
  figures/*.png          gate 3b figures (pre-specified and post-hoc, labeled in the title)

Exit status 1 if a count disagrees or a document does not quote a key number.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO, read_jsonl

RA = REPO / "data/v2/reanalysis/2026-10-09_v2.2-audit"
MV = REPO / "data/v2.2/processed/v2.2-measval-01"
README, CLAIMS, G3B = "README.md", "docs/release/claims.md", "docs/v2/v22_audit/gate3b_measval_report.md"
CATS = {"PILOT_V1": "historical pilot (v1, driver 550, 3 trials)",
        "V2_CONFIRMATORY": "v2 confirmatory (pre-registered, 12 within-session blocks)",
        "V2_REANALYSIS": "v2.2 audit reanalysis of v2 data (Stage 2)",
        "V2_PHASE10": "v2 synthetic workload (Phase 10, 5 within-session blocks)",
        "V22_PRESPECIFIED": "v2.2 measurement validation, pre-specified (gate 3b)",
        "V22_POSTHOC": "v2.2 measurement validation, post-hoc exploratory"}


def m(x: float) -> str:
    """Typographic minus for negative numbers in documents."""
    return x.replace("-", "−") if isinstance(x, str) else x


def pct(lnx: float, d: int = 2, sign: bool = True) -> str:
    v = 100 * (math.exp(lnx) - 1)
    return m(f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def ci_pct(s: dict, d: int = 2) -> str:
    return f"{pct(s['mean'], d)} % [{pct(s['ci'][0], d)}, {pct(s['ci'][1], d)}]"


def key_numbers() -> list[dict]:
    rows = []

    def add(cid, cat, quantity, text, source, docs):
        rows.append({"id": cid, "category": cat, "quantity": quantity, "text": text, "source": source,
                     "must_appear_in": ";".join(docs)})

    # pilot v1
    ps = json.loads((REPO / "data/processed/full/results_summary.json").read_text())
    dt = ps["disagreement_total"]
    add("P1", "PILOT_V1", "cells where latency- and energy-optimal executed configurations coincide",
        f"{dt['cells'] - dt['en_disagree']} of {dt['cells']}", "data/processed/full/results_summary.json",
        [README, CLAIMS])
    pens = sorted(100 * r["energy_penalty"] for r in ps["resolved_disagreements"] if r["objective"] == "en")
    add("P2", "PILOT_V1", "raw energy penalty range of the latency-optimal choice in the disagreeing cells",
        f"{min(pens):.1f}–{max(pens):.1f} %", "data/processed/full/results_summary.json", [CLAIMS])

    # v2 confirmatory
    rq1 = pd.read_csv(REPO / "data/v2/processed/confirmatory/rq1.csv")
    th = pd.read_csv(RA / "rq1_threshold.csv")
    add("C1", "V2_CONFIRMATORY", "cells where the frozen pick uses less board energy than automatic",
        f"{int(rq1.saving_confirmed.sum())} of {len(rq1)}", "data/v2/processed/confirmatory/rq1.csv", [README, CLAIMS])
    add("C2a", "V2_CONFIRMATORY", "cells with >= 5 % saving, pre-registered point-estimate rule",
        f"{int(rq1.saving_meaningful.sum())} cells", "data/v2/processed/confirmatory/rq1.csv", [README, CLAIMS])
    add("C2b", "V2_REANALYSIS", "cells with >= 5 % saving supported by the threshold test (Holm)",
        f"{int(th['threshold_supported_holm_0.025'].sum())} cells", f"{RA.relative_to(REPO)}/rq1_threshold.csv",
        [README, CLAIMS])
    r = rq1[(rq1.placement == "P4_MIX") & (rq1.coll == "all_reduce") & (rq1.size_bytes == 524288)].iloc[0]
    add("C3", "V2_CONFIRMATORY", "largest effect: P4_MIX AllReduce 512 KiB energy ratio pick/automatic",
        f"{r.E_ratio:.3f} [{r.E_ratio_ci_lo:.3f}, {r.E_ratio_ci_hi:.3f}]", "data/v2/processed/confirmatory/rq1.csv",
        [README, CLAIMS])
    rq2 = pd.read_csv(REPO / "data/v2/processed/confirmatory/rq2.csv")
    ag = rq2[rq2.coll == "all_gather"].iloc[0]
    add("C6", "V2_CONFIRMATORY", "P2_NVL AllGather 2 MiB: LL128 lower latency, relative to auto",
        f"{100 * (1 - 1 / (1 + ag.latency_penalty_of_en_opt)):.1f} %",
        "data/v2/processed/confirmatory/rq2.csv", [README, CLAIMS])
    est = pd.read_csv(RA / "rq2_estimators.csv")
    for e, lab in (("counter", "counter"), ("integrated_instant", "instant"), ("lag_adjusted_powerusage", "lag")):
        x = est[(est.cell == "P2_NVL all_gather 2097152") & (est.estimator == e)].iloc[0]
        lo, hi = json.loads(x.ci95)
        add(f"C7_{lab}", "V2_REANALYSIS", f"P2_NVL AllGather 2 MiB LL128/auto energy ratio, {lab} estimator",
            f"{x.ratio_lat_over_en:.3f} [{lo:.3f}, {hi:.3f}]", f"{RA.relative_to(REPO)}/rq2_estimators.csv",
            [README, CLAIMS] if lab == "counter" else [CLAIMS])
    rev = rq1[(rq1.placement == "P2_PCI_SHM") & (rq1.coll == "all_reduce") & (rq1.size_bytes == 268435456)].iloc[0]
    add("C9", "V2_CONFIRMATORY", "P2_PCI_SHM AllReduce 256 MiB: energy of the pick vs automatic",
        f"+{100 * (rev.E_ratio - 1):.1f} %", "data/v2/processed/confirmatory/rq1.csv", [CLAIMS])
    ot = json.loads((RA / "order_temperature.json").read_text())
    add("C11", "V2_REANALYSIS", "paired energy vs paired temperature difference (association)",
        f"{100 * ot['coef']['per_degC']:.2f} %/°C", f"{RA.relative_to(REPO)}/order_temperature.json", [CLAIMS])
    vt = json.loads((REPO / "data/v2/raw/phase10/verify_tuner_r3/summary.json").read_text())
    add("C13", "V2_CONFIRMATORY", "tuner reproduces the measured executions: ruled sizes / all sizes",
        f"{vt['rule_matched']}/{vt['ruled_sizes']} rules and {vt['reference_matched']}/{vt['sizes_with_reference']} sizes",
        "data/v2/raw/phase10/verify_tuner_r3/summary.json", [CLAIMS])
    p10 = json.loads((RA / "phase10_v22/phase10_summary.json").read_text())
    for c in p10["contrasts"]:
        if c["contrast"] != "energy_slo/auto":
            continue
        key = {"ddp": "C14", "ddpb2048": "C15", "fsdp": "C16"}.get(c["family"])
        if key:
            add(key, "V2_PHASE10", f"Phase 10 {c['family']}: energy per step, tuner policy / automatic",
                f"{c['E_ratio']:.3f} [{c['E_ci'][0]:.3f}, {c['E_ci'][1]:.3f}]",
                f"{RA.relative_to(REPO)}/phase10_v22/phase10_summary.json", [README, CLAIMS])
            if key == "C14":
                add("C14_T", "V2_PHASE10", "Phase 10 ddp: step time, tuner policy / automatic", f"{c['T_ratio']:.3f}",
                    f"{RA.relative_to(REPO)}/phase10_v22/phase10_summary.json", [README, CLAIMS])

    # v2.2 gate 3b, pre-specified
    dec = json.loads((MV / "decisions.json").read_text())
    src = str((MV / "decisions.json").relative_to(REPO))
    add("G0", "V22_PRESPECIFIED", "processes accepted / run", f"{dec['n_accepted']} of {dec['n_processes']}", src,
        [README, CLAIMS, G3B])
    for arm in ("A", "B", "C"):
        s = dec["D1"]["per_config"][arm]["ln_Jop_2.5_over_40"]["counter"]
        add(f"G1_{arm}", "V22_PRESPECIFIED", f"D1 J/op 2.5 s / 40 s, counter, arm {arm}", ci_pct(s), src, [G3B])
    add("G2", "V22_PRESPECIFIED", "D2 LL128 term, instant estimator (deciding)",
        ci_pct(dec["D2"]["inst"]["protocol_LL128"]), src, [CLAIMS, G3B])
    s20 = dec["D3"]["per_length"]["20.0"]["counter"]
    add("G3", "V22_PRESPECIFIED", "D3 B/A at 20 s, counter", ci_pct(s20), src, [CLAIMS, G3B])
    add("G4", "V22_PRESPECIFIED", "D4 C/A 40 s counter CI half-width",
        f"{100 * dec['D4']['half_width_40s_counter']:.2f} %", src, [README, CLAIMS, G3B])
    for k in ("D1", "D2", "D3", "D4"):
        add(f"G{k}", "V22_PRESPECIFIED", f"{k} decision", dec[k]["decision"], src, [README, CLAIMS, G3B])
    # every interval in the gate 3b report's pre-specified tables
    for e in ("inst", "lag"):
        for term in ("intercept", "protocol_LL128"):
            add(f"G2_{e}_{term}", "V22_PRESPECIFIED", f"D2 {term}, {e}", ci_pct(dec["D2"][e][term]), src, [G3B])
        o = dec["D2"][e]["offset_J_per_region"]
        add(f"G2_{e}_offset", "V22_PRESPECIFIED", f"D2 1/T term as J per region, {e}",
            m(f"{o['mean']:.1f} J [{o['ci'][0]:.1f}, {o['ci'][1]:.1f}]"), src, [G3B])
    for L, v in dec["D3"]["per_length"].items():
        for e in ("counter", "inst", "lag"):
            add(f"G3_{L}_{e}", "V22_PRESPECIFIED", f"D3 B/A {L} s, {e}", ci_pct(v[e]), src, [G3B])
    add("G4_CA40", "V22_PRESPECIFIED", "D4 C/A 40 s counter", ci_pct(dec["D4"]["C_over_A_descriptive"]["40.0"]["counter"]),
        src, [G3B])
    add("G4_diff", "V22_PRESPECIFIED", "D4 C/A 2.5 s minus 40 s counter", ci_pct(dec["D4"]["C_over_A_2.5_minus_40_counter"]),
        src, [G3B])

    # v2.2 gate 3b, post-hoc
    ph = json.loads((MV / "posthoc_temperature.json").read_text())
    srcp = str((MV / "posthoc_temperature.json").relative_to(REPO))
    sw = ph["temp_slope_within_block"]
    add("H1", "V22_POSTHOC", "temperature slope of ln(J/op), within-block mean [95 % CI]",
        f"{100 * sw['mean']:.2f} [{100 * sw['ci'][0]:.2f}, {100 * sw['ci'][1]:.2f}]", srcp, [CLAIMS, G3B])
    f10 = ph["fit_length_ge_10s"]
    add("H2", "V22_POSTHOC", "residual SD of ln(J/op) without / with temperature, lengths >= 10 s",
        f"{100 * f10['resid_sd_ln_without_temp']:.2f} % to {100 * f10['resid_sd_ln_with_temp']:.2f} %", srcp,
        [CLAIMS, G3B])
    for k, v in ph["contrasts"].items():
        if k.startswith("temp_diff"):
            continue
        for L, vv in v.items():
            sa = vv["temp_adjusted"]
            add(f"H5_{k}_{L}", "V22_POSTHOC", f"temperature-adjusted {k} {L}",
                f"{pct(sa['mean'])} [{pct(sa['ci'][0])}, {pct(sa['ci'][1])}]", srcp,
                [G3B] if not (k.startswith("D4") and L != "40.0") else [])
    wg = ph["window_gap"]
    for e in ("inst", "lag"):
        g = wg[e]["gap_J_per_gpu_region"]
        add(f"H3_{e}", "V22_POSTHOC", f"matched-window energy gap counter - {e}, J per GPU per region",
            f"+{g['mean']:.1f} [{g['ci'][0]:.1f}, {g['ci'][1]:.1f}]", srcp, [CLAIMS, G3B])
        sp = wg[e]["steady_state_power_diff_pct"]
        add(f"H4_{e}", "V22_POSTHOC", f"steady-state power difference counter - {e} per GPU",
            m(f"{sp['mean']:+.3f} % [{sp['ci'][0]:+.3f}, {sp['ci'][1]:+.3f}]"), srcp, [G3B])
    return rows


def gate3b_tables(out: Path):
    dec = json.loads((MV / "decisions.json").read_text())
    rows = []

    def row(dname, quantity, s, thr, outcome):
        rows.append({"decision": dname, "quantity": quantity, "n_blocks": s.get("n"),
                     "estimate_pct": None if s.get("mean") is None else 100 * (math.exp(s["mean"]) - 1),
                     "ci_lo_pct": None if not s.get("ci") else 100 * (math.exp(s["ci"][0]) - 1),
                     "ci_hi_pct": None if not s.get("ci") else 100 * (math.exp(s["ci"][1]) - 1),
                     "threshold_pct": thr, "outcome": outcome})
    for arm, v in dec["D1"]["per_config"].items():
        for e in ("counter", "inst", "lag"):
            row("D1", f"J/op 2.5s/40s {e} arm {arm}", v["ln_Jop_2.5_over_40"][e], 0.3,
                v["decision"] if e == "counter" else "reported")
    for e in ("inst", "lag"):
        row("D2", f"LL128 term {e}", dec["D2"][e]["protocol_LL128"], 0.3,
            dec["D2"][e]["decision"] + (" (deciding)" if e == "inst" else " (secondary)"))
        row("D2", f"intercept {e}", dec["D2"][e]["intercept"], None, "reported")
    for L, v in dec["D3"]["per_length"].items():
        for e in ("counter", "inst", "lag"):
            row("D3", f"B/A {L}s {e}", v[e], 0.5, dec["D3"]["decision"] if e == "counter" and L != "2.5" else "reported")
    for L, v in dec["D4"]["C_over_A_descriptive"].items():
        row("D4", f"C/A {L}s counter (descriptive)", v["counter"], 0.4 if L == "40.0" else None,
            dec["D4"]["decision"] if L == "40.0" else "reported")
    pd.DataFrame(rows).to_csv(out / "gate3b_decisions.csv", index=False, float_format="%.4f")

    ph = json.loads((MV / "posthoc_temperature.json").read_text())
    prow = []
    for k, v in ph["contrasts"].items():
        if k.startswith("temp_diff"):
            continue
        for kk, vv in v.items():
            for adj in ("raw", "temp_adjusted"):
                s = vv[adj]
                prow.append({"label": "POST-HOC EXPLORATORY", "contrast": f"{k} {kk}", "version": adj, "n": s["n"],
                             "estimate_pct": 100 * (math.exp(s["mean"]) - 1),
                             "ci_lo_pct": 100 * (math.exp(s["ci"][0]) - 1) if s.get("ci") else None,
                             "ci_hi_pct": 100 * (math.exp(s["ci"][1]) - 1) if s.get("ci") else None})
    pd.DataFrame(prow).to_csv(out / "gate3b_posthoc.csv", index=False, float_format="%.4f")


def counts() -> dict:
    """Accepted/excluded counts from the raw records vs the processed outputs."""
    out = {}
    raw = [a for b in sorted((REPO / "data/v2.2/raw/v2.2-measval-01/blocks").glob("block_*"))
           for a in read_jsonl(b / "attempts.jsonl")[0]]
    dec = json.loads((MV / "decisions.json").read_text())
    proc = pd.read_csv(MV / "processes.csv")
    ok = sum(a["qc_status"] == "OK" for a in raw)
    out["gate3b"] = {"raw_attempts": len(raw), "raw_qc_ok": ok, "raw_not_ok": len(raw) - ok,
                     "processed_rows": len(proc), "processed_accepted": int(proc.accepted.sum()),
                     "decisions_n_processes": dec["n_processes"], "decisions_n_accepted": dec["n_accepted"],
                     "decisions_exclusions": len(dec["exclusions"]),
                     "raw_retries": sum(a["attempt"] != 0 for a in raw)}
    g = out["gate3b"]
    g["consistent"] = (g["raw_attempts"] == g["processed_rows"] == g["decisions_n_processes"] == 72
                       and g["raw_qc_ok"] == g["processed_accepted"] == g["decisions_n_accepted"] == 70
                       and g["raw_not_ok"] == g["decisions_exclusions"] == 2 and g["raw_retries"] == 0)
    conf = [a for b in sorted((REPO / "data/v2/raw/confirmatory/blocks").glob("block_*"))
            for a in read_jsonl(b / "attempts.jsonl")[0]]
    units = pd.read_csv(REPO / "data/v2/processed/confirmatory/units.csv")
    final = {}
    for a in conf:      # last attempt per (block, unit) is the unit's final status
        final[(a["block"], a["unit_id"])] = a["qc_status"]
    out["v2_confirmatory"] = {"raw_attempt_records": len(conf), "raw_units": len(final),
                              "raw_units_ok": sum(v == "OK" for v in final.values()), "processed_units": len(units)}
    c = out["v2_confirmatory"]
    c["consistent"] = c["raw_units"] == c["processed_units"] == 984 and c["raw_units_ok"] == 984

    # gate 3a: controller counter reads inside the STRICT region window [start read begins, post read ends]
    # (the gate 3a check used [start read, loop end]; the end and post reads follow the loop end)
    s2 = REPO / "data/v2.2/raw/v2.2-smoke-02"
    audit = [c for f in sorted((s2 / "nvml_audit").glob("audit_s2*.jsonl")) for c in read_jsonl(f)[0]]
    wins = []
    for a in read_jsonl(s2 / "blocks/block_01/attempts.jsonl")[0]:
        rec = json.loads((REPO / a["dir"] / "region.jsonl").read_text().splitlines()[0])
        wins += [(g["t_start_a_ns"], g["t_post_b_ns"]) for g in rec["gpus"]]
    ctr = [c for c in audit if c["fn"] == "nvmlDeviceGetTotalEnergyConsumption"]
    inside = [c for c in ctr if any(t0 <= c["t_ns"] <= t1 for t0, t1 in wins)]
    out["gate3a_counter_reads_strict_window"] = {"audit_calls": len(audit), "counter_calls": len(ctr),
                                                 "region_windows": len(wins), "counter_calls_inside": len(inside),
                                                 "consistent": bool(audit) and len(wins) == 6 and not inside}

    # Phase 10 (historical v2.1 meter, no flags recorded): post-hoc plausibility under the v2.2 meter rules
    from .aligned_meter import TAIL_P_MAX_W, TAIL_P_MIN_W
    runs = read_jsonl(REPO / "data/v2/raw/phase10/workload/runs.jsonl")[0]
    bad = []
    for r in runs:
        m = json.loads((REPO / r["dir"] / "result.json").read_text())["meter"]
        for g, v in (m.get("per_gpu") or {}).items():
            if not (TAIL_P_MIN_W <= (v.get("p_ctx_W") or -1) <= TAIL_P_MAX_W) or (v.get("tail_s") or -1) < 0 \
                    or not (v.get("energy_J") or 0) > 0:
                bad.append(f"b{r['block']}:{r['arm']}:GPU{g}")
    out["phase10_meter_plausibility_posthoc"] = {"runs": len(runs), "gpu_readings": sum(
        len(json.loads((REPO / r["dir"] / "result.json").read_text())["meter"].get("per_gpu") or {}) for r in runs),
        "implausible": bad, "note": "v2.1 meter recorded no QC flags; checked post hoc with the v2.2 tail rules",
        "consistent": not bad}
    return out


def temperature_partial_coordinates(d: pd.DataFrame) -> pd.DataFrame:
    """Residualize both axes against the arm×length effects used by the fit."""
    cells = d.groupby(["arm", "length_s"])
    return d.assign(
        resid=d.y - cells.y.transform("mean"),
        temp_resid_c=d.temp_c - cells.temp_c.transform("mean"),
    )


def figures(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    dec = json.loads((MV / "decisions.json").read_text())
    # pre-specified: D3 per-block B/A
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (L, v) in enumerate(dec["D3"]["per_length"].items()):
        s = v["counter"]
        pb = [100 * (math.exp(x) - 1) for x in s.get("per_block", [])]
        ax.scatter([i] * len(pb), pb, s=14, color="#4C72B0", zorder=3)
        if s.get("ci"):
            ax.errorbar(i + 0.15, 100 * (math.exp(s["mean"]) - 1),
                        yerr=[[100 * (math.exp(s["mean"]) - math.exp(s["ci"][0]))],
                              [100 * (math.exp(s["ci"][1]) - math.exp(s["mean"]))]], fmt="o", color="black", capsize=3)
    ax.axhspan(-0.5, 0.5, color="#DDDDDD", zorder=0, label="±0.5 % (D3 band)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(4), [f"{float(L):g} s" for L in dec["D3"]["per_length"]])
    ax.set_ylabel("B ÷ A energy per op, counter (%)")
    ax.set_title("Gate 3b D3 (pre-specified): identical-execution control\nper block (dots), mean and 95 % CI (black)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / "gate3b_D3_identical_execution_control.png", dpi=150)
    plt.close(fig)

    # post-hoc: temperature association
    from .measval_posthoc import load as ph_load
    d = ph_load(REPO / "data/v2.2/raw/v2.2-measval-01")
    d = temperature_partial_coordinates(d)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arm, col in zip(("A", "B", "C"), ("#4C72B0", "#DD8452", "#55A868")):
        g = d[d.arm == arm]
        ax.scatter(g.temp_resid_c, 100 * g.resid, s=14, color=col, label={"A": "auto", "B": "forced Ring/Simple",
                                                                     "C": "forced Ring/LL128"}[arm])
    ph = json.loads((MV / "posthoc_temperature.json").read_text())
    b = ph["fit_all_lengths"]["temp_slope_ln_per_c"]
    xs = np.linspace(d.temp_resid_c.min(), d.temp_resid_c.max(), 10)
    ax.plot(xs, 100 * b * xs, color="black", lw=1, label=f"partial slope {100 * b:.2f} log-%/°C")
    ax.set_xlabel("mean GPU temperature − arm×length mean (°C)")
    ax.set_ylabel("ln(J/op) − arm×length mean (%)")
    ax.set_title("Gate 3b POST-HOC (exploratory): association of energy per op with temperature\n"
                 "partial-regression coordinates; not causal or calibrated accuracy", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "gate3b_POSTHOC_temperature_association.png", dpi=150)
    plt.close(fig)

    # post-hoc: matched-window gap vs start phase
    rows = []
    for bdir in sorted((REPO / "data/v2.2/raw/v2.2-measval-01/blocks").glob("block_*")):
        for a in read_jsonl(bdir / "attempts.jsonl")[0]:
            if a.get("qc_status") != "OK":
                continue
            rec = json.loads((REPO / a["dir"] / "region.jsonl").read_text().splitlines()[0])
            P = rec["commit_period_ns"]
            per = {p["rank"]: p for p in a["region"]["per_gpu"]}
            for g in rec["gpus"]:
                p = per[g["rank"]]
                t0 = (g["t_start_a_ns"] + g["t_start_b_ns"]) / 2
                c_prev = g["phase_ns"] + math.floor((t0 - g["phase_ns"]) / P) * P
                tel = a["telemetry"][str(p["nvml_index"])]
                rows.append({"dstart_ms": (t0 - c_prev) / 1e6, "P": p["p_avg_W"],
                             "gap_lag": p["delta_J"] - tel["energy_int_lag_J"],
                             "gap_inst": p["delta_J"] - tel["energy_int_inst_J"]})
    w = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter(w.dstart_ms, w.gap_lag, s=10, color="#4C72B0", label="counter − lag PowerUsage")
    ax.scatter(w.dstart_ms, w.gap_inst, s=10, color="#C44E52", alpha=0.5, label="counter − POWER_INSTANT")
    xs = np.linspace(0, 100, 10)
    ax.plot(xs, w.P.mean() * xs / 1e3, color="black", lw=1, ls="--",
            label="prediction if the counter included the pre-start window")
    ax.set_xlabel("time from last counter update to start read (ms)")
    ax.set_ylabel("energy gap per GPU per region (J)")
    ax.set_title("Gate 3b POST-HOC (exploratory): counter vs sampled energy over matched windows\n"
                 "gap not explained by the start window", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "gate3b_POSTHOC_matched_window_gap.png", dpi=150)
    plt.close(fig)

    r = pd.read_csv(RA / "rq2_estimators.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (cell, group) in zip(axes, r.groupby("cell", sort=False)):
        for i, row in enumerate(group.itertuples()):
            lo, hi = json.loads(row.ci95)
            ax.errorbar(row.ratio_lat_over_en, i,
                        xerr=[[row.ratio_lat_over_en - lo], [hi - row.ratio_lat_over_en]],
                        fmt="o", color="#2166ac", capsize=3)
        ax.axvline(1, color="black", lw=.8)
        for bound in (1 / 1.03, 1.03):
            ax.axvline(bound, color="grey", ls=":", lw=.8)
        ax.set_yticks(range(len(group)), group.estimator, fontsize=8)
        ax.set_title(f"{cell}\npre-selected candidates, not confirmed optima", fontsize=8)
        ax.set_xlabel("energy ratio, 95% CI over 12 within-session blocks")
    fig.suptitle("RQ2 estimator sensitivity; shared board sensors, no independent calibration", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "rq2_estimator_sensitivity.png", dpi=150)
    plt.close(fig)


def doc_check(kn: pd.DataFrame) -> dict:
    res, missing = {}, []
    texts = {}
    for _, r in kn.iterrows():
        for doc in [x for x in r.must_appear_in.split(";") if x]:
            if doc not in texts:      # whitespace-normalized, so a number wrapped across lines still matches
                p = REPO / doc
                texts[doc] = re.sub(r"\s+", " ", p.read_text()) if p.exists() else None
            ok = texts[doc] is not None and r.text in texts[doc]
            res.setdefault(doc, {})[r.id] = ok
            if not ok:
                missing.append({"doc": doc, "id": r.id, "text": r.text})
    return {"documents": {k: {"n_required": len(v), "n_found": sum(v.values())} for k, v in res.items()},
            "missing": missing, "pass": not missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    kn = pd.DataFrame(key_numbers())
    kn.insert(2, "category_meaning", kn.category.map(CATS))
    kn.to_csv(out / "key_numbers.csv", index=False)
    gate3b_tables(out)
    cnt = counts()
    (out / "counts.json").write_text(json.dumps(cnt, indent=1))
    dc = doc_check(kn)
    (out / "doc_check.json").write_text(json.dumps(dc, indent=1, ensure_ascii=False))
    if not a.no_figures:
        figures(out)
        from . import plots_v2
        cf = out / "figures/v2_confirmatory"
        cf.mkdir(parents=True, exist_ok=True)
        plots_v2.rq1_blocks(pd.read_csv(REPO / "data/v2/processed/confirmatory/rq1.csv"), cf)
        plots_v2.counter_vs_integration(pd.read_csv(REPO / "data/v2/processed/confirmatory/units.csv"), cf)
        ef = out / "figures/v2_exploratory"
        ef.mkdir(parents=True, exist_ok=True)
        plots_v2.regret_heatmap(pd.read_csv(REPO / "data/v2/processed/exploratory/policies.csv"), ef,
                               title="Exploratory auto-energy regret; two within-session blocks")
        wf = out / "figures/historical_phase10"
        wf.mkdir(parents=True, exist_ok=True)
        plots_v2.workload_blocks(pd.read_csv(REPO / "data/v2/processed/phase10/workload_runs.csv"), wf)
    ok = all(v["consistent"] for v in cnt.values()) and dc["pass"]
    print(f"key numbers {len(kn)}; counts consistent: {[k for k, v in cnt.items() if v['consistent']]}; "
          f"documents: {dc['documents']}; missing {len(dc['missing'])}")
    for x in dc["missing"]:
        print("  MISSING", x)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
