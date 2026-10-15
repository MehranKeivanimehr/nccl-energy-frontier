"""v2.2 audit addendum 1: decomposition of what the historical counter-based region energies can and cannot
claim about uncertainty (CPU only; writes only into a new directory; frozen outputs hashed before/after).

    python -m ncclenergy.v2.reanalysis_addendum --out data/v2/reanalysis/2026-10-09_v2.2-audit-addendum1

Components, per GPU-region of the 984 accepted confirmatory units (worst case = half-widths added over the GPUs of
a unit, relative to the unit's region energy):
  A  read timestamps   A1 start read: the reset instant lies anywhere inside the start read call
                       A2 end read: idle-with-context power p_ctx uses the post window from the end read
  B  counter update    B1 commit phase: predicted commit time inside the probe bracket
                       B2 period model: drift of the predicted commit over N periods for a period error dP
                       B3 commit semantics (what a committed value covers in time): not identifiable here
  C  tail model        C1 magnitude of the subtracted idle tail; C2 one-sided sensitivity if the true tail power
                       were the region's mean power instead of p_ctx
  D  quantization      1 mJ per reading
  E  sensor accuracy   not characterized (no external meter; shared sensor for all estimators)
All of A-C rest on the interpretation, supported by the local Phase 4 experiments but not guaranteed by NVML, that a
counter read discards the energy accumulated since the last update and that updates occur on a fixed grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO
from .reanalysis import CONF, FROZEN_DIRS, tree_hash, unit_records

DP_LOAD_NS = 2_000            # Phase 4 load_counter_updates: p10-p90 99.998-100.002 ms -> 0.002 ms per period
MARGIN_NS = 8_000_000


def q(s) -> dict:
    s = pd.Series(s, dtype=float).dropna()
    return {"median": float(s.median()), "p95": float(s.quantile(.95)), "max": float(s.max()), "n": int(len(s))}


def components() -> pd.DataFrame:
    rows = []
    for (b, uid), a in unit_records(CONF).items():
        if a.get("qc_status") != "OK":
            continue
        rec = json.loads((REPO / a["dir"] / "region.jsonl").read_text().splitlines()[0])
        m = a["region"]
        P = rec["commit_period_ns"]
        for g, pm in zip(rec["gpus"], m["per_gpu"]):
            p_ctx, tail_s = pm["p_ctx_W"], pm["tail_s"]
            t_end_mid = (g["t_end_a_ns"] + g["t_end_b_ns"]) / 2
            post_s = (g["t_commit_ns"] + P - t_end_mid) * 1e-9
            tail_J = p_ctx * tail_s
            dur = pm["duration_s"]
            p_reg = pm["delta_J"] / dur
            n_per = (g["t_commit_ns"] - g["phase_ns"]) / P
            drift_ns = n_per * DP_LOAD_NS
            rows.append({
                "block": b, "unit_id": uid, "rank": g["rank"], "gpu_region_J": pm["delta_J"],
                "unit_region_J": m["E_raw_J"], "duration_s": dur, "p_ctx_W": p_ctx, "p_region_W": p_reg,
                "tail_s": tail_s, "tail_J": tail_J,
                "A1_start_read_J": p_ctx * (g["t_start_b_ns"] - g["t_start_a_ns"]) * 1e-9 / 2,
                "A2_end_read_J": tail_J * ((g["t_end_b_ns"] - g["t_end_a_ns"]) * 1e-9 / 2) / post_s,
                "B1_phase_bracket_J": p_ctx * g["bracket_ns"] * 1e-9 / 2,
                "B2_period_drift_J": p_ctx * drift_ns * 1e-9, "B2_drift_ms": drift_ns / 1e6,
                "B2_margin_left_ms": (g["t_end_a_ns"] - g["t_commit_ns"] - drift_ns) / 1e6,
                "C2_tail_model_bound_J": max(p_reg - p_ctx, 0.0) * tail_s,
                "D_quantization_J": 0.002 + 0.002 * tail_s / post_s,
                "periods_since_probe": n_per})
    return pd.DataFrame(rows)


def edge_period_evidence() -> dict:
    per = []
    for (b, uid), a in unit_records(CONF).items():
        if a.get("qc_status") != "OK":
            continue
        rec = json.loads((REPO / a["dir"] / "region.jsonl").read_text().splitlines()[0])
        P = rec["commit_period_ns"]
        for g in rec["gpus"]:
            mids = sorted((lo + hi) / 2 for lo, hi in g.get("edges") or [])
            for x, y in zip(mids, mids[1:]):
                k = round((y - x) / P)
                if k >= 1:
                    per.append((y - x) / k / 1e6)
    v4 = [json.loads(ln) for ln in (REPO / "data/v2/raw/validation/validation.jsonl").read_text().splitlines()]
    load = [{k: r[k] for k in ("median_update_ms", "p10_update_ms", "p90_update_ms", "n_reads", "n_changes")}
            for r in v4 if r.get("exp") == "load_counter_updates"]
    idle = [{k: r[k] for k in ("median_update_ms", "p10_update_ms", "p90_update_ms", "n_reads", "n_changes")}
            for r in v4 if r.get("exp") == "idle_counter"]
    return {"probe_edges_confirmatory": {**q(per), "unit": "ms per period, from consecutive probe-edge midpoints "
                                                          "(each edge bracketed to about 2 ms, so this resolves the "
                                                          "period only to about +-1 ms)"},
            "phase4_load_counter_updates": load, "phase4_idle_counter": idle,
            "interpretation": "Under load the update interval was 100.000 ms (p10-p90 99.998-100.002, 3 runs). At P8 "
                              "idle one run gave median 101.3 ms, p10-p90 93.5-101.5 ms; that run did not resolve the "
                              "idle period (read spacing not recorded at that precision). The Phase 4 report's "
                              "'100.00 ms idle and under load' overstated the idle evidence. The post-loop tail and the "
                              "post window are at idle-with-context power, where only the probe edges (bracket-limited) "
                              "are available."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else REPO / a.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    before = tree_hash(FROZEN_DIRS + [REPO / "data/v2/reanalysis/2026-10-09_v2.2-audit"])
    c = components()
    c.to_csv(out / "uncertainty_components.csv.gz", index=False, compression="gzip")
    keys = ["A1_start_read_J", "A2_end_read_J", "B1_phase_bracket_J", "B2_period_drift_J", "C2_tail_model_bound_J",
            "D_quantization_J", "tail_J"]
    u = c.groupby(["block", "unit_id"]).agg(E=("unit_region_J", "first"), **{k: (k, "sum") for k in keys})
    rel = {k: q(u[k] / u.E) for k in keys}
    res = {
        "n_units": int(len(u)), "n_gpu_regions": int(len(c)),
        "relative_to_unit_region_energy": {
            "A_read_timestamps": {"A1_start_read": rel["A1_start_read_J"], "A2_end_read_via_tail": rel["A2_end_read_J"],
                                  "A_total": q((u.A1_start_read_J + u.A2_end_read_J) / u.E)},
            "B_counter_update_and_phase": {"B1_phase_bracket": rel["B1_phase_bracket_J"],
                                           "B2_period_drift_at_dP_0.002ms": rel["B2_period_drift_J"],
                                           "B3_commit_semantics": "NOT_IDENTIFIABLE_FROM_THESE_DATA"},
            "C_tail_model": {"C1_correction_magnitude": rel["tail_J"],
                             "C2_one_sided_bound_if_tail_at_region_power": rel["C2_tail_model_bound_J"]},
            "D_quantization": rel["D_quantization_J"],
            "E_sensor_accuracy": "NOT_CHARACTERIZED (no external reference; NVML gives no accuracy figure for this "
                                 "board's energy counter that was verified here)",
            "previously_reported_0.10pct_bound": "= A1 + A2 + B1 worst case (max over units 0.10 %); it bounds only "
                                                 "boundary timing and phase, not total measurement accuracy"},
        "B2_drift": {"drift_ms": q(c.B2_drift_ms), "margin_left_ms_min": float(c.B2_margin_left_ms.min()),
                     "periods_since_probe": q(c.periods_since_probe),
                     "projection_40s_region": {"periods": 400 + 2, "drift_ms": (402 * DP_LOAD_NS) / 1e6,
                                               "note": "within the 8 ms end-read margin if dP <= 0.002 ms holds at "
                                                       "idle-with-context, which is not independently established"}},
        "period_evidence": edge_period_evidence(),
    }
    # the RQ2 AllGather 2 MiB cell: are the components of the two arms different enough to matter for ~1 %?
    s = c[c.unit_id.str.startswith("P2_NVL__all_gather__") & c.unit_id.str.contains("__2097152")].copy()
    s["arm"] = np.where(s.unit_id.str.contains("Ring-LL128"), "LL128", "auto")
    g = s.groupby(["block", "arm"]).agg(E=("gpu_region_J", "sum"), tailJ=("tail_J", "sum"),
                                        A=("A1_start_read_J", "sum"), A2=("A2_end_read_J", "sum"),
                                        B1=("B1_phase_bracket_J", "sum"), C2=("C2_tail_model_bound_J", "sum"))
    g = g.assign(tail_frac=g.tailJ / g.E, timing_frac=(g.A + g.A2 + g.B1) / g.E, c2_frac=g.C2 / g.E).reset_index()
    w = g.pivot(index="block", columns="arm")
    res["rq2_ag2m_components"] = {
        "tail_fraction": {arm: q(w["tail_frac"][arm]) for arm in ("auto", "LL128")},
        "timing_bound_fraction": {arm: q(w["timing_frac"][arm]) for arm in ("auto", "LL128")},
        "tail_model_bound_fraction": {arm: q(w["c2_frac"][arm]) for arm in ("auto", "LL128")},
        "paired_tail_fraction_difference_LL128_minus_auto": q(w["tail_frac"]["LL128"] - w["tail_frac"]["auto"]),
        "note": "Compare these per-arm fractions with the counter-based LL128/auto penalty (rq2_estimators.csv). "
                "Sensor accuracy is not included anywhere above."}
    after = tree_hash(FROZEN_DIRS + [REPO / "data/v2/reanalysis/2026-10-09_v2.2-audit"])
    res["frozen_and_first_reanalysis_unchanged"] = before == after
    (out / "uncertainty_components.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    if before != after:
        raise SystemExit("frozen outputs changed")


if __name__ == "__main__":
    main()
