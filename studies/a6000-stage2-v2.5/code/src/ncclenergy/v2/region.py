"""Region records written by the v2 nccl-tests patch -> energy/latency metrics.

All checks report; nothing is clamped or repaired.
"""

from __future__ import annotations

import json

from .common import NVML_ERROR_NO_PERMISSION, NVML_ERROR_NOT_SUPPORTED, NVML_SUCCESS, UNAVAILABLE, norm_pci

P_MIN_W, P_MAX_W = 2.0, 320.0   # plausible average board power per GPU during a region
TAIL_P_MIN_W, TAIL_P_MAX_W = 5.0, 250.0   # plausible idle-with-context power during the tail window
MAX_BRACKET_NS = 20_000_000               # commit-phase bracket must be <= 20 ms
SCHEMAS = ("nccle-region-v1", "nccle-region-v2")


def parse_region_text(text: str) -> tuple[list[dict], list[str]]:
    """Parse NCCLE_REGION_FILE content. Returns (records, flags)."""
    recs, flags = [], []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            flags.append("PARTIAL_REGION_LOG" if i == len(lines) - 1 else "CORRUPT_REGION_LINE")
            continue
        if r.get("schema") not in SCHEMAS:
            flags.append("UNKNOWN_REGION_SCHEMA")
            continue
        recs.append(r)
    return recs, flags


def _counter_flag(rc: int) -> str:
    if rc == NVML_ERROR_NO_PERMISSION:
        return UNAVAILABLE
    if rc == NVML_ERROR_NOT_SUPPORTED:
        return "COUNTER_NOT_SUPPORTED"
    return f"COUNTER_READ_ERROR_{rc}"


def region_metrics(rec: dict, expected_pci: list[str] | None = None, expected_R: int | None = None) -> dict:
    """Energy/latency for one region over the participating GPUs only.

    expected_pci: PCI bus ids of the GPUs that must participate (order free).
    expected_R:   frozen operation count; mismatch is a hard flag.
    """
    flags: list[str] = []
    gpus = rec.get("gpus") or []
    R = int(rec.get("R", 0))
    implied_R = int(rec.get("iters", 0)) * int(rec.get("agg_iters", 1)) * max(1, int(rec.get("graph_launches", 0) or 0))
    if R <= 0:
        flags.append("R_NONPOSITIVE")
    if R != implied_R:
        flags.append("R_INCONSISTENT")
    if expected_R is not None and R != expected_R:
        flags.append("R_NOT_FROZEN_VALUE")
    if not rec.get("nvml_loaded"):
        flags.append("NVML_NOT_LOADED")
    if int(rec.get("nranks", -1)) != len(gpus):
        flags.append("RANK_GPU_COUNT_MISMATCH")
    if expected_pci is not None:
        got = sorted(norm_pci(g["pci"]) for g in gpus)
        if got != sorted(norm_pci(p) for p in expected_pci):
            flags.append("GPU_SET_MISMATCH")

    per = []
    e_ok = True
    for g in gpus:
        f = []
        for key in ("handle_rc", "rc_start", "rc_end"):
            if g.get(key, -1) != NVML_SUCCESS:
                f.append(_counter_flag(g.get(key, -1)))
        delta_mj = None
        tail = {}
        if not f:
            delta_mj = int(g["e_end_mJ"]) - int(g["e_start_mJ"])
            if delta_mj == 0:
                f.append("COUNTER_NOT_UPDATING")
            elif delta_mj < 0:
                f.append("COUNTER_REVERSAL_OR_WRAP")
        t0 = (g["t_start_a_ns"] + g["t_start_b_ns"]) / 2
        v2 = rec.get("schema") == "nccle-region-v2"
        if v2:
            # collective region ends at the loop end (after sync); the end read follows the next counter commit
            t1 = g["t_loop_end_ns"]
            P = rec["commit_period_ns"]
            edges = g.get("edges") or []
            if g.get("n_edges", 0) == 0 or g.get("t_commit_ns", -1) < 0:
                f.append("COUNTER_PHASE_UNKNOWN")
            else:
                if g["bracket_ns"] > MAX_BRACKET_NS:
                    f.append("COUNTER_PHASE_IMPRECISE")
                for lo, hi in edges:                      # every observed commit must sit on the phase grid
                    k = round(((lo + hi) / 2 - g["phase_ns"]) / P)
                    if abs((lo + hi) / 2 - (g["phase_ns"] + k * P)) > (hi - lo) / 2 + g["bracket_ns"] / 2 + 2e6:
                        f.append("COUNTER_PHASE_INCONSISTENT")
                        break
                if g["t_end_a_ns"] < g["t_commit_ns"]:
                    f.append("COUNTER_END_READ_BEFORE_COMMIT")
                if g.get("rc_post", -1) != NVML_SUCCESS:
                    f.append(_counter_flag(g.get("rc_post", -1)))
                elif delta_mj is not None:
                    t_end_mid = (g["t_end_a_ns"] + g["t_end_b_ns"]) / 2
                    post_s = (g["t_commit_ns"] + P - t_end_mid) * 1e-9
                    p_ctx = (int(g["e_post_mJ"]) - int(g["e_end_mJ"])) / 1000.0 / post_s if post_s > 0 else None
                    tail_s = (g["t_commit_ns"] - g["t_loop_end_ns"]) * 1e-9
                    tail = {"tail_s": tail_s, "p_ctx_W": p_ctx, "aligned_delta_J": delta_mj / 1000.0}
                    if p_ctx is None or not (TAIL_P_MIN_W <= p_ctx <= TAIL_P_MAX_W) or tail_s < 0:
                        f.append("COUNTER_TAIL_IMPLAUSIBLE")
                    else:
                        tail["tail_J"] = p_ctx * tail_s
                        delta_mj = delta_mj - tail["tail_J"] * 1000.0
        else:
            t1 = (g["t_end_a_ns"] + g["t_end_b_ns"]) / 2
            f.append("SCHEMA_V1_UNALIGNED")
        dur = (t1 - t0) * 1e-9
        if not (g["t_start_a_ns"] <= g["t_start_b_ns"] < g["t_end_a_ns"] <= g["t_end_b_ns"]) or dur <= 0:
            f.append("TIME_REVERSAL")
        if g.get("event_rc", 0) != 0:
            f.append("EVENT_ERROR")
        p_avg = None
        if delta_mj is not None and dur > 0:
            p_avg = delta_mj / 1000.0 / dur
            if not (P_MIN_W <= p_avg <= P_MAX_W):
                f.append("COUNTER_IMPLAUSIBLE")
        if any(x.startswith(("COUNTER", UNAVAILABLE)) for x in f):
            e_ok = False
        per.append({"rank": g["rank"], "pci": norm_pci(g["pci"]), "nvml_index": g.get("nvml_index"),
                    "delta_J": None if delta_mj is None else delta_mj / 1000.0, "duration_s": dur, **tail,
                    "t_start_ns": t0, "t_end_ns": t1, "p_avg_W": p_avg, "event_ms": g.get("event_ms"),
                    "read_start_us": (g["t_start_b_ns"] - g["t_start_a_ns"]) / 1e3,
                    "read_end_us": (g["t_end_b_ns"] - g["t_end_a_ns"]) / 1e3, "flags": f})
        flags.extend(f)

    out = {"coll": rec.get("coll"), "size_bytes": rec.get("size_bytes"), "count": rec.get("count"),
           "in_place": rec.get("in_place"), "R": R, "warmup_iters": rec.get("warmup_iters"), "per_gpu": per,
           "flags": sorted(set(flags))}
    if per:
        out["T_region_s"] = (max(p["t_end_ns"] for p in per) - min(p["t_start_ns"] for p in per)) * 1e-9
        ev = [p["event_ms"] for p in per if p["event_ms"] is not None]
        out["t_gpu_s"] = (max(ev) / 1000.0 / R) if ev and R > 0 else None
        out["t_host_s"] = rec.get("host_sec_per_coll")
    if e_ok and per and R > 0:
        out["E_raw_J"] = sum(p["delta_J"] for p in per)
        out["J_per_coll"] = out["E_raw_J"] / R
    else:
        out["E_raw_J"] = None
        out["J_per_coll"] = None
    return out


def idle_adjusted(metrics: dict, p_idle_w: dict[str, float]) -> float | None:
    """E_raw - sum_g P_idle[g] * duration[g] (per-GPU durations). Negative values are returned as is."""
    if metrics.get("E_raw_J") is None:
        return None
    tot = 0.0
    idle = {norm_pci(k): v for k, v in p_idle_w.items()}
    for p in metrics["per_gpu"]:
        if norm_pci(p["pci"]) not in idle:
            return None
        tot += idle[norm_pci(p["pci"])] * p["duration_s"]
    return metrics["E_raw_J"] - tot
