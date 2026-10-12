"""Offline Energy-SLO recommender for NCCL configurations (transparent, table based).

Profiles come from verified, QC-passed block-level measurements. v2.2 guards:
* A query is answered only when its scope matches the profile's recorded scope
  exactly: GPU model, driver, CUDA driver version, NCCL version, dtype,
  reduction op, in-place flag, rank and node count, topology fingerprint and
  placement NCCL environment (profile_scope.py derives these from raw evidence).
  Missing, unknown, ambiguous or different metadata -> INSUFFICIENT_EVIDENCE,
  i.e. NCCL automatic. Missing legacy metadata is never filled in.
* Only measured sizes are answered. Interpolation between sizes is disabled by
  default; the experimental opt-in returns a separate status and never copies
  endpoint ratios or confidence bounds to the intermediate size.
Predicted savings are predictions from past blocks, not measurements.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

OBJECTIVES = ("min_energy_slo", "min_edp", "max_bytes_per_joule", "min_latency", "compare_auto")


@dataclass
class Query:
    collective: str
    message_size: int
    number_of_ranks: int
    number_of_nodes: int
    topology_fingerprint: str
    placement_env: str          # canonical "K=V;..." of placement-level NCCL env (e.g. NCCL_P2P_DISABLE=1)
    nccl_version: str | None    # e.g. "2.27.5+cuda12.9" as printed by NCCL
    cuda_driver_version: int | None   # NVML cuda driver version, e.g. 13000
    gpu_model: str | None = None
    driver_version: str | None = None
    dtype: str | None = None    # e.g. "float32"
    op: str | None = None       # reduction op for AllReduce/ReduceScatter, e.g. "sum"; ignored for AllGather
    in_place: int | None = None
    latency_budget: float = 1.05   # x NCCL-automatic latency (relative budget)
    objective: str = "min_energy_slo"


# (profile column, Query attribute) pairs that must match exactly; None on either side fails closed
SCOPE_MATCH = (("gpu_model", "gpu_model"), ("driver_version", "driver_version"),
               ("cuda_driver_version", "cuda_driver_version"), ("nccl_version", "nccl_version"),
               ("dtype", "dtype"), ("in_place", "in_place"))


def query_from_meta(coll: str, size: int, meta: dict, latency_budget: float = 1.05,
                    objective: str = "min_energy_slo") -> Query:
    """Query describing the profiled environment itself (used for LOBO and tuner export)."""
    return Query(coll, int(size), meta["nranks"], meta["nnodes"], meta["fingerprint"], meta["env"],
                 meta.get("nccl_version"), meta.get("cuda_driver_version"), meta.get("gpu_model"),
                 meta.get("driver_version"), meta.get("dtype"), (meta.get("ops_by_coll") or {}).get(coll),
                 meta.get("in_place"), latency_budget, objective)


def scope_problems(key_rows: pd.DataFrame, q: Query) -> list[str]:
    """Reasons why the profile rows cannot answer q (empty list = compatible)."""
    out = []
    for col, attr in SCOPE_MATCH:
        qv = getattr(q, attr)
        if col not in key_rows:
            out.append(f"PROFILE_METADATA_MISSING:{col}")
            continue
        vals = {v for v in key_rows[col].tolist() if v is not None and v == v}
        if len(key_rows[col].dropna()) != len(key_rows) or not vals:
            out.append(f"PROFILE_METADATA_MISSING:{col}")
        elif len(vals) > 1:
            out.append(f"PROFILE_METADATA_AMBIGUOUS:{col}")
        elif qv is None:
            out.append(f"QUERY_METADATA_MISSING:{attr}")
        elif str(qv) != str(next(iter(vals))):
            out.append(f"MISMATCH:{attr}")
    if q.collective != "all_gather":
        col = f"op_{q.collective}"
        vals = set(key_rows[col].dropna().tolist()) if col in key_rows else set()
        if len(vals) != 1 or (col in key_rows and len(key_rows[col].dropna()) != len(key_rows)):
            out.append(f"PROFILE_METADATA_MISSING:op[{q.collective}]")
        elif q.op is None:
            out.append("QUERY_METADATA_MISSING:op")
        elif q.op != next(iter(vals)):
            out.append("MISMATCH:op")
    return out


def build_profiles(units: pd.DataFrame, placement_meta: dict) -> pd.DataFrame:
    """Block-level profile rows: one per (placement key, coll, size, effective config, block).

    units: valid (QC OK + VERIFIED) unit table from analysis.load_blocks/valid.
    placement_meta: placement id -> profile_scope.campaign_scope() entry (fingerprint, env, nranks, nnodes and the
    recorded GPU/driver/CUDA/NCCL/dtype/op/in-place scope).
    """
    u = units.copy()
    for k in ("fingerprint", "env", "nranks", "nnodes") + tuple(c for c, _ in SCOPE_MATCH):
        u[k] = u.placement.map(lambda p, k=k: placement_meta[p].get(k))
    for coll in ("all_reduce", "reduce_scatter"):
        u[f"op_{coll}"] = u.placement.map(lambda p, c=coll: (placement_meta[p].get("ops_by_coll") or {}).get(c))
    return u


def _cell(prof: pd.DataFrame, q: Query, size: int) -> pd.DataFrame:
    return prof[(prof.fingerprint == q.topology_fingerprint) & (prof.env == q.placement_env) &
                (prof.nccl_version == q.nccl_version) & (prof.nranks == q.number_of_ranks) &
                (prof.nnodes == q.number_of_nodes) & (prof.coll == q.collective) & (prof.size_bytes == size)]


def _decide(cell: pd.DataFrame, q: Query, alpha: float = 0.05) -> dict:
    """Choose a configuration in one measured cell from block-paired evidence vs auto."""
    if cell.empty or "auto" not in set(cell.config):
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "no measured auto baseline"}
    auto_eff = cell[cell.config == "auto"].actual.iloc[0]
    a = cell[cell.config == "auto"].set_index("block")
    res = []
    for (cfg, eff), g in cell.groupby(["config", "actual"]):
        j = g.set_index("block")[["logE", "logT"]].join(a[["logE", "logT"]], rsuffix="_a", how="inner")
        n = len(j)
        if n < 2:
            continue
        dE, dT = (j.logE - j.logE_a).to_numpy(), (j.logT - j.logT_a).to_numpy()
        tq = stats.t.ppf(1 - alpha / 2, n - 1)
        seE, seT = dE.std(ddof=1) / math.sqrt(n), dT.std(ddof=1) / math.sqrt(n)
        res.append({"config": cfg, "actual": eff, "n_blocks": n, "E_ratio": math.exp(dE.mean()),
                    "E_ratio_hi": math.exp(dE.mean() + tq * seE), "E_ratio_lo": math.exp(dE.mean() - tq * seE),
                    "T_ratio": math.exp(dT.mean()), "T_ratio_hi": math.exp(dT.mean() + tq * seT),
                    "J": float(np.exp(g.logE.mean())), "t_gpu": float(np.exp(g.logT.mean())),
                    "J_sd_log": float(g.logE.std(ddof=1)), "t_sd_log": float(g.logT.std(ddof=1))})
    if not res:
        return {"status": "INSUFFICIENT_EVIDENCE", "reason": "fewer than 2 paired blocks"}
    r = pd.DataFrame(res)
    auto_row = r[r.actual == auto_eff].iloc[0] if (r.actual == auto_eff).any() else None
    obj = q.objective
    if obj == "min_latency":
        ok = r[r.T_ratio_hi < 1.0]
        pick = ok.loc[ok.T_ratio.idxmin()] if len(ok) else None
    elif obj in ("min_energy_slo", "compare_auto"):
        ok = r[(r.T_ratio_hi <= q.latency_budget) & (r.E_ratio_hi < 1.0)]   # SLO met and energy saving resolved
        pick = ok.loc[ok.E_ratio.idxmin()] if len(ok) else None
    elif obj == "min_edp":
        r = r.assign(edp_ratio=r.E_ratio * r.T_ratio, edp_hi=r.E_ratio_hi * r.T_ratio_hi)
        ok = r[r.edp_hi < 1.0]
        pick = ok.loc[ok.edp_ratio.idxmin()] if len(ok) else None
    elif obj == "max_bytes_per_joule":
        ok = r[r.E_ratio_hi < 1.0]                                          # bytes fixed -> min energy, no SLO
        pick = ok.loc[ok.E_ratio.idxmin()] if len(ok) else None
    else:
        raise ValueError(obj)
    base = {"auto_actual": auto_eff, "auto_J": None if auto_row is None else auto_row.J,
            "auto_t_gpu": None if auto_row is None else auto_row.t_gpu}
    if pick is None or pick.actual == auto_eff:
        return {**base, "status": "FALLBACK", "recommendation": "auto", "effective": auto_eff,
                "reason": "no configuration resolved better than NCCL automatic under the objective",
                "n_blocks": int(r.n_blocks.max())}
    return {**base, "status": "RECOMMEND", "recommendation": pick.config, "effective": pick.actual,
            "est_J": pick.J, "est_t_gpu": pick.t_gpu, "E_ratio_vs_auto": pick.E_ratio,
            "E_ratio_ci": [pick.E_ratio_lo, pick.E_ratio_hi], "T_ratio_vs_auto": pick.T_ratio,
            "T_ratio_hi": pick.T_ratio_hi, "n_blocks": int(pick.n_blocks), "reason": "block-paired evidence"}


def recommend(prof: pd.DataFrame, q: Query, allow_interpolation: bool = False) -> dict:
    out = {"query": q.__dict__, "fallback": "auto (NCCL_ALGO/NCCL_PROTO unset)"}
    if q.objective not in OBJECTIVES:
        return {**out, "status": "INVALID", "reason": f"unknown objective {q.objective}"}
    key = prof[(prof.fingerprint == q.topology_fingerprint) & (prof.env == q.placement_env) &
               (prof.nranks == q.number_of_ranks) & (prof.nnodes == q.number_of_nodes) &
               (prof.coll == q.collective)]
    if key.empty:
        return {**out, "status": "INSUFFICIENT_EVIDENCE", "reason": "no profile for this topology/env/ranks/collective"}
    probs = scope_problems(key, q)
    if probs:
        return {**out, "status": "INSUFFICIENT_EVIDENCE", "reason": "profile scope does not match the query",
                "scope_problems": probs}
    sizes = sorted(key.size_bytes.unique())
    s = q.message_size
    if s in sizes:
        return {**out, "size_basis": [s], **_decide(_cell(prof, q, s), q)}
    if s < sizes[0] or s > sizes[-1]:
        return {**out, "status": "INSUFFICIENT_EVIDENCE", "reason": f"size outside measured range {sizes[0]}-{sizes[-1]}"}
    lo = max(x for x in sizes if x < s)
    hi = min(x for x in sizes if x > s)
    if not allow_interpolation:
        return {**out, "status": "INSUFFICIENT_EVIDENCE", "size_basis": [lo, hi],
                "reason": "size not measured; interpolation between sizes is disabled (v2.2)"}
    # Experimental, unvalidated: report both endpoint decisions; no point estimate or interval for this size.
    dl, dh = _decide(_cell(prof, q, lo), q), _decide(_cell(prof, q, hi), q)
    ends = {str(lo): {k: dl.get(k) for k in ("status", "recommendation", "E_ratio_vs_auto", "E_ratio_ci")},
            str(hi): {k: dh.get(k) for k in ("status", "recommendation", "E_ratio_vs_auto", "E_ratio_ci")}}
    agree = (hi == 2 * lo and dl.get("status") == "RECOMMEND" and dh.get("status") == "RECOMMEND"
             and dl.get("recommendation") == dh.get("recommendation"))
    return {**out, "status": "EXPERIMENTAL_UNVALIDATED_INTERPOLATION", "size_basis": [lo, hi], "endpoints": ends,
            "candidate": dl.get("recommendation") if agree else None, "E_ratio_vs_auto": None, "E_ratio_ci": None,
            "reason": "unmeasured size; endpoint decisions only, no validated prediction for this size"}


def lobo_validate(v: pd.DataFrame, placement_meta: dict, latency_budget: float = 1.05) -> pd.DataFrame:
    """Leave-one-block-out: recommend from the other blocks, score on the held-out block.

    Regret = held-out J of the recommended config / held-out J of the best config that meets the
    SLO in that block, minus 1. Saving = 1 - held-out J(rec) / held-out J(auto) (measured, held out).
    """
    prof_all = build_profiles(v, placement_meta)
    rows = []
    blocks = sorted(prof_all.block.unique())
    for b in blocks:
        train, test = prof_all[prof_all.block != b], prof_all[prof_all.block == b]
        for (pl, coll, size), cell in test.groupby(["placement", "coll", "size_bytes"]):
            meta = placement_meta[pl]
            q = query_from_meta(coll, int(size), meta, latency_budget)
            r = recommend(train, q)
            rec = r.get("recommendation", "auto")
            byc = cell.groupby("config")[["J_per_coll", "t_gpu_s"]].mean()
            if "auto" not in byc.index:
                continue
            ja, ta = byc.loc["auto"]
            feas = byc[byc.t_gpu_s <= latency_budget * ta]
            jbest = feas.J_per_coll.min()
            jr, tr = byc.loc[rec] if rec in byc.index else byc.loc["auto"]
            rows.append({"heldout_block": b, "placement": pl, "coll": coll, "size_bytes": size, "status": r["status"],
                         "recommendation": rec, "regret": jr / jbest - 1, "measured_saving_vs_auto": 1 - jr / ja,
                         "predicted_E_ratio": r.get("E_ratio_vs_auto"), "measured_E_ratio": jr / ja,
                         "slo_violated": bool(tr > latency_budget * ta), "auto_regret": ja / jbest - 1})
    return pd.DataFrame(rows)


def tuner_conf(prof: pd.DataFrame, placement: str, placement_meta: dict, latency_budget: float = 1.05,
               objective: str = "min_energy_slo", experimental_ranges: bool = False) -> list[str]:
    """nccl_tuner.conf lines for the guarded plugin (native/nccl_tuner_guarded). v2.2: one rule per measured size
    (min_bytes == max_bytes), headed by "#@scope" lines the plugin verifies at startup; RECOMMEND cells only.
    experimental_ranges=True merges adjacent x2 sizes into ranges and marks the file experimental (the plugin
    honours ranges only with allow_ranges=1 in the file AND NCCLE_TUNER_ALLOW_RANGES=1)."""
    from .profile_scope import declared_scope_string
    meta = placement_meta[placement]
    lines = [f"# nccl-energy-frontier {objective} policy for placement {placement} (method v2.2)",
             f"# latency_budget={latency_budget}x auto; {'EXPERIMENTAL range rules' if experimental_ranges else 'exact measured sizes only'}",
             "# collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff",
             "#@scope method=v2.2",
             f"#@scope gpu_model={meta.get('gpu_model')}",
             f"#@scope driver_version={meta.get('driver_version')}",
             f"#@scope cuda_driver_version={meta.get('cuda_driver_version')}",
             f"#@scope nccl_version={meta.get('nccl_version')}",
             f"#@scope nccl_version_code={meta.get('nccl_version_code')}",
             f"#@scope nranks={meta.get('nranks')}",
             f"#@scope nnodes={meta.get('nnodes')}",
             f"#@scope pci_bus_ids={meta.get('pci_bus_ids')}",
             f"#@scope placement_env={meta.get('env') or 'none'}",
             f"#@scope fingerprint={meta.get('fingerprint')}",
             f"#@scope declared_scope={declared_scope_string(meta)}",
             f"#@scope allow_ranges={1 if experimental_ranges else 0}"]
    names = {"all_reduce": "allreduce", "all_gather": "allgather", "reduce_scatter": "reducescatter"}
    p = prof[prof.placement == placement]
    for coll in sorted(p.coll.unique()):
        sizes = sorted(p[p.coll == coll].size_bytes.unique())
        decisions = []
        for s in sizes:
            r = recommend(prof, query_from_meta(coll, int(s), meta, latency_budget, objective))
            decisions.append((s, r.get("recommendation") if r.get("status") == "RECOMMEND" else None))
        # NCCL passes the tuner nBytes = elementSize * ncclFuncMaxSendRecvCount(func, nRanks, count) (src/enqueue.cc),
        # i.e. the total buffer for AllGather/ReduceScatter (nRanks * per-rank count) and the buffer for AllReduce.
        # That equals the nccl-tests "size" in every case, so sizes are written unchanged.
        k = 0
        while k < len(decisions):
            s, rec = decisions[k]
            if rec is None:
                k += 1
                continue
            j = k
            if experimental_ranges:
                while j + 1 < len(decisions) and decisions[j + 1][1] == rec and decisions[j + 1][0] == 2 * decisions[j][0]:
                    j += 1
            algo, proto = rec.split("/")
            lines.append(f"{names[coll]},{int(decisions[k][0])},{int(decisions[j][0])},{algo.lower()},{proto.lower()},"
                         f"-1,{meta['nnodes']},{meta['nranks']},-1,-1")
            k = j + 1
    return lines
