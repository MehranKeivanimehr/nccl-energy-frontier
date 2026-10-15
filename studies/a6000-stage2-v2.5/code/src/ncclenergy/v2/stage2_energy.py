"""Stage 2 energy formulas and the v2.4 Stage 2 record (nccle-s2-v1). CPU only; never reads NVML.

Units are explicit in names: *_mJ raw counter values, *_ns CLOCK_MONOTONIC nanoseconds, *_s seconds, *_W watts, *_J
joules. Power-field timestamps (NVML nvmlFieldValue_t.timestamp, "CPU Timestamp of this value in microseconds since
1970", nvml.h R580 line 2786) belong to the REALTIME domain and are wrapped in RealtimeUs so they cannot be mixed with
monotonic times.

The v2.1 region record (schema nccle-region-v2) is unchanged and still parsed by region.py. The v2.4 patch writes a
separate record per region to NCCLE_S2_FILE; s2_metrics() joins the two by pid and checks every field it uses.
Nothing is clamped, repaired or inferred: missing or invalid inputs give None plus a flag.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .common import NVML_SUCCESS, norm_pci

NS_PER_S = 1_000_000_000
MJ_PER_J = 1000.0
S2_SCHEMA = "nccle-s2-v1"


class ClockDomainError(TypeError):
    pass


@dataclass(frozen=True)
class RealtimeUs:
    """A CLOCK_REALTIME-domain timestamp in microseconds since 1970 (NVML field timestamp). Arithmetic with monotonic
    times is refused; differences between two RealtimeUs values are allowed (same domain)."""
    us: int

    def __sub__(self, other):
        if not isinstance(other, RealtimeUs):
            raise ClockDomainError("cannot subtract a non-realtime value from a realtime field timestamp")
        return (self.us - other.us) * 1e-6          # seconds

    def __rsub__(self, other):
        raise ClockDomainError("cannot subtract a realtime field timestamp from a monotonic value")


class InvalidCounter(ValueError):
    pass


def counter_delta_J(c_start_mJ: int, c_end_mJ: int, rc_start: int = 0, rc_end: int = 0) -> float:
    """(C_end - C_start)/1000 J for one GPU; valid only if both NVML statuses are 0 and the counter did not decrease.
    Never clamps: a reversal or wrap raises InvalidCounter."""
    if rc_start != 0 or rc_end != 0:
        raise InvalidCounter(f"counter read status {rc_start}/{rc_end}")
    if not (isinstance(c_start_mJ, (int, np.integer)) and isinstance(c_end_mJ, (int, np.integer))):
        raise InvalidCounter("raw counter values must be integers in mJ")
    if c_end_mJ < c_start_mJ:
        raise InvalidCounter("counter decreased (reversal or wrap)")
    return (int(c_end_mJ) - int(c_start_mJ)) / MJ_PER_J


def e_counter_raw_J(gpus: list[dict]) -> float:
    """E_counter_raw = sum_g (C1_g - C0_g)/1000 J. Each gpu dict: c0_mJ, c1_mJ, rc0, rc1."""
    return sum(counter_delta_J(g["c0_mJ"], g["c1_mJ"], g.get("rc0", 0), g.get("rc1", 0)) for g in gpus)


def e_aligned_gpu_J(g: dict, period_ns: float) -> dict:
    """Historical v2.1 commit-aligned correction for one GPU, with an explicit period.

        E_g = (C1 - C0)/1000 - p_ctx * tail
        p_ctx = (Cpost - C1)/1000 / ((t_commit + P - t1_mid) * 1e-9)     [W]
        tail  = (t_commit - t_loop_end) * 1e-9                            [s]
        t1_mid = (t_end_a + t_end_b)/2

    t_commit is the update time the native code PREDICTED (phase + k * P_nom); it is not an observed update. Whether
    C1 really followed the update after the loop end is a separate question (stage2_phase.end_alignment_*).
    Raises InvalidCounter on a status error or a non-positive post window; reports tail < 0 as a flag."""
    if not (isinstance(period_ns, (int, float, np.floating)) and math.isfinite(period_ns) and period_ns > 0):
        raise InvalidCounter("period must be a positive finite number of ns")
    d_main = counter_delta_J(g["c0_mJ"], g["c1_mJ"], g.get("rc0", 0), g.get("rc1", 0))
    d_post = counter_delta_J(g["c1_mJ"], g["cpost_mJ"], g.get("rc1", 0), g.get("rcpost", 0))
    t1_mid = (g["t_end_a_ns"] + g["t_end_b_ns"]) / 2
    post_s = (g["t_commit_ns"] + period_ns - t1_mid) / NS_PER_S
    if post_s <= 0:
        raise InvalidCounter("post window is not positive")
    p_ctx = d_post / post_s
    tail_s = (g["t_commit_ns"] - g["t_loop_end_ns"]) / NS_PER_S
    flags = ["TAIL_NEGATIVE"] if tail_s < 0 else []
    return {"delta_J": d_main, "p_ctx_W": p_ctx, "tail_s": tail_s, "tail_J": p_ctx * tail_s,
            "energy_J": d_main - p_ctx * tail_s, "post_window_s": post_s, "period_ns": float(period_ns), "flags": flags}


def e_aligned_J(gpus: list[dict], period_ns_by_gpu: list[float]) -> float:
    if len(gpus) != len(period_ns_by_gpu):
        raise ValueError("one period per GPU is required")
    return sum(e_aligned_gpu_J(g, p)["energy_J"] for g, p in zip(gpus, period_ns_by_gpu))


def joules_per_op(energy_J: float, R: int) -> float:
    """Exact denominator: R is the frozen timed-operation count (integer, > 0)."""
    if not isinstance(R, (int, np.integer)) or isinstance(R, bool) or R <= 0:
        raise ValueError("R must be a positive integer operation count")
    if not math.isfinite(energy_J):
        raise ValueError("energy must be finite")
    return energy_J / int(R)


class CoverageError(ValueError):
    pass


def _window(t_s, p_W, t0_s: float, t1_s: float, max_gap_s: float):
    t = np.asarray(t_s, float)
    p = np.asarray(p_W, float)
    if t.ndim != 1 or t.shape != p.shape or len(t) < 2:
        raise CoverageError("need at least two samples")
    if not np.all(np.isfinite(t)):
        raise CoverageError("non-finite sample timestamp")
    if np.any(np.diff(t) <= 0):
        raise CoverageError("sample timestamps not strictly increasing")
    if not np.all(np.isfinite(p)):
        raise CoverageError("non-finite power value (unsupported field or read error)")
    if not (math.isfinite(t0_s) and math.isfinite(t1_s)) or t[0] > t0_s or t[-1] < t1_s or t1_s <= t0_s:
        raise CoverageError("samples do not cover the window on both sides")
    lo = int(np.searchsorted(t, t0_s, "right")) - 1     # last sample at or before t0
    hi = int(np.searchsorted(t, t1_s, "left"))          # first sample at or after t1
    if np.any(np.diff(t[lo:hi + 1]) > max_gap_s):
        raise CoverageError("sample gap inside the window exceeds the limit")
    return t, p, lo, hi


def sampled_energy_trapezoid_J(t_s, p_W, t0_s: float, t1_s: float, max_gap_s: float) -> float:
    """Trapezoid integral over [t0, t1] on the raw rows (repeated values kept); linear interpolation only at the two
    window edges, between the samples that bracket them. Rejects (CoverageError) instead of extrapolating."""
    t, p, lo, hi = _window(t_s, p_W, t0_s, t1_s, max_gap_s)
    inside = (t > t0_s) & (t < t1_s)
    tt = np.concatenate([[t0_s], t[inside], [t1_s]])
    pp = np.concatenate([[np.interp(t0_s, t, p)], p[inside], [np.interp(t1_s, t, p)]])
    return float(np.sum((pp[1:] + pp[:-1]) * np.diff(tt)) / 2.0)


def sampled_energy_hold_J(t_s, p_W, t0_s: float, t1_s: float, max_gap_s: float) -> float:
    """Previous-value hold over [t0, t1]: each reading holds until the next sample."""
    t, p, lo, hi = _window(t_s, p_W, t0_s, t1_s, max_gap_s)
    edges = np.concatenate([[t0_s], t[(t > t0_s) & (t < t1_s)], [t1_s]])
    vals = np.array([p[int(np.searchsorted(t, x, "right")) - 1] for x in edges[:-1]])
    return float(np.sum(vals * np.diff(edges)))


def log_ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        raise ValueError("log ratio needs positive finite energies")
    return math.log(a / b)


# ------------------------------------------------------------------ v2.4 Stage 2 record

def parse_s2_text(text: str) -> tuple[list[dict], list[str]]:
    """Parse NCCLE_S2_FILE content. Returns (records, flags); unknown schemas are flagged, never guessed."""
    recs, flags = [], []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            flags.append("S2_PARTIAL_LOG" if i == len(lines) - 1 else "S2_CORRUPT_LINE")
            continue
        if r.get("schema") != S2_SCHEMA:
            flags.append("S2_UNKNOWN_SCHEMA")
            continue
        recs.append(r)
    return recs, flags


def _mid(a, b):
    return (a + b) / 2


def brackets_from_reads(reads: list) -> list[tuple[int, int]]:
    """Update brackets from raw consecutive reads [(t_a, t_b, value, rc), ...] with the v2.1 rule: a value change
    between the read that started at t_a(prev) and the read that ended at t_b(cur) gives the bracket (t_a(prev),
    t_b(cur)). Stops at the first failed read."""
    out, prev = [], None
    for ta, tb, v, rc in reads:
        if rc != NVML_SUCCESS:
            break
        if prev is not None and v != prev[2]:
            out.append((int(prev[0]), int(tb)))
        prev = (ta, tb, v)
    return out


DOSE_READ_ALLOWANCE_NS = 10_000_000    # a request is eligible only if a read started at it can end before the work end
DOSE_MAX_LATENESS_NS = 20_000_000     # a read counts for its request only if it starts within 20 ms of it


def dose_summary(dose: dict, period_ms: int, t_c0_b_ns: int, t_work_end_ns: int) -> dict:
    """M4b delivery under ONE rule shared by the native poller, this parser and the fake benchmark.

    Requests: req_k = base + k P (k >= 1), base = end of the C0 read. The native poller attempts a read at each request
    until the controller's stop request, which follows the synchronized GPU-work end t_work_end.
      eligible   req_k + 10 ms <= t_work_end                        (denominator; never depends on what was read)
      delivered  eligible, a read for k with status 0, 0 <= t_a - req_k <= 20 ms and t_b <= t_work_end
      straddling a read with t_a < t_work_end < t_b                 (logged separately; never delivered)
      shutdown   a read with t_a >= t_work_end                      (logged separately; outside the region; never
                                                                     delivered; e.g. the read in flight at the stop)
      cancelled  requests before the stop request without a read    (skipped as late, or cancelled by the stop)
    delivered_frac = delivered / eligible; no eligible request -> None (flag). The 95 % acceptance threshold is
    unchanged; this rule only defines which requests it is measured over.

    Order (active poller only; a dose_period_ms = 0 launch never calls this): work_end <= stop_request <= join_end,
    all three present. A read in flight at the stop request (inflight_at_stop) is physically possible and is NOT a
    failure: it is classified as straddling or shutdown and excluded from delivery."""
    P = int(period_ms) * 1_000_000
    base, stop, joined = dose.get("base_ns", -1), dose.get("stop_req_ns", -1), dose.get("joined_ns", -1)
    flags = []
    if dose.get("start_failed"):
        flags.append("DOSE_START_FAILED")
    if dose.get("overflow"):
        flags.append("DOSE_OVERFLOW")
    order = {"work_end_ns": t_work_end_ns, "stop_request_ns": stop, "join_end_ns": joined}
    if base < 0 or stop < 0 or joined < 0 or t_work_end_ns is None or t_work_end_ns < 0:
        flags.append("DOSE_TIMES_MISSING")
        return {"eligible": None, "delivered": 0, "delivered_frac": None, "order": dict(order, ok=False),
                "flags": flags}
    if base != t_c0_b_ns:
        flags.append("DOSE_BASE_NOT_C0")
    if stop < t_work_end_ns:
        flags.append("DOSE_STOPPED_BEFORE_WORK_END")
    if joined < stop:
        flags.append("DOSE_JOINED_BEFORE_STOP")
    order["ok"] = t_work_end_ns <= stop <= joined
    eligible = max(0, (t_work_end_ns - DOSE_READ_ALLOWANCE_NS - base) // P)
    reads = [dict(zip(("k", "req", "ta", "tb", "v", "rc"), r)) for r in dose.get("reads", [])]
    if len(reads) != dose.get("n", -1):
        flags.append("DOSE_COUNT_MISMATCH")
    if any(r["req"] != base + r["k"] * P for r in reads):
        flags.append("DOSE_REQUEST_TIME_MISMATCH")
    by_k = {}
    for r in reads:
        by_k.setdefault(r["k"], []).append(r)
    if any(len(v) > 1 for v in by_k.values()):
        flags.append("DOSE_DUPLICATE_REQUEST")
    delivered = sum(1 for k in range(1, eligible + 1) for r in by_k.get(k, [])[:1]
                    if r["rc"] == NVML_SUCCESS and 0 <= r["ta"] - r["req"] <= DOSE_MAX_LATENESS_NS
                    and r["tb"] <= t_work_end_ns)
    straddling_reads = [r for r in reads if r["ta"] < t_work_end_ns < r["tb"]]
    shutdown_reads = [r for r in reads if r["ta"] >= t_work_end_ns]
    straddling, shutdown = len(straddling_reads), len(shutdown_reads)
    in_region = sum(1 for r in reads if r["ta"] >= base and r["tb"] <= t_work_end_ns)
    eligible_completed = sum(1 for k in range(1, eligible + 1) for r in by_k.get(k, [])[:1] if r["tb"] <= t_work_end_ns)
    requested_before_stop = max(0, (stop - base) // P)
    cancelled = max(0, requested_before_stop - len(by_k))
    if any(r["rc"] != NVML_SUCCESS for r in reads):
        flags.append("DOSE_STATUS_ERROR")
    vals = [r["v"] for r in reads if r["rc"] == NVML_SUCCESS]
    if any(b < a for a, b in zip(vals, vals[1:])):
        flags.append("DOSE_VALUE_DECREASE")
    if any(r["ta"] < base for r in reads):
        flags.append("DOSE_READ_BEFORE_BASE")
    if eligible == 0:
        flags.append("DOSE_NO_ELIGIBLE_REQUESTS")
    late = [r["ta"] - r["req"] for r in reads]
    raw = lambda rr: [[r["k"], r["req"], r["ta"], r["tb"], r["rc"]] for r in rr]  # noqa: E731
    return {"rule": "eligible: req + 10 ms <= work end; delivered: status 0, start within 20 ms, end <= work end",
            "eligible": int(eligible), "eligible_completed": int(eligible_completed), "delivered": int(delivered),
            "delivered_frac": (delivered / eligible) if eligible else None, "in_region_reads": in_region,
            "straddling": straddling, "shutdown_reads": shutdown, "cancelled_or_skipped": int(cancelled),
            "straddling_detail": raw(straddling_reads), "shutdown_detail": raw(shutdown_reads),
            "detail_columns": ["k", "req_ns", "t_a_ns", "t_b_ns", "status"],
            "skipped_native": dose.get("skipped"), "inflight_at_stop": dose.get("inflight_at_stop"),
            "inflight_at_stop_note": "logged, not a failure: such a read is straddling or shutdown and never delivered",
            "lateness_ms_median": float(np.median(late)) / 1e6 if late else None,
            "lateness_ms_max": max(late) / 1e6 if late else None,
            "stop_to_join_ms": (joined - stop) / 1e6, "work_end_to_stop_ms": (stop - t_work_end_ns) / 1e6,
            "order": order, "flags": flags}


def s2_metrics(region: dict, s2: dict) -> dict:
    """Stage 2 quantities for one launch from its v2.1 region record and its v2.4 Stage 2 record.

    Per GPU: E_prep_to_C0 (C_prep0 -> C0: warm-up, its synchronization, the probe, barriers and the M4c pause),
    warm-up duration, probe span and read count, requested vs actual pause, the counter-call inventory identity, M4b
    delivery and M4d raw post-probe brackets. Per launch: C_prep0 / C0 / C1 read skew between GPUs (C1 and Cpost are
    per-GPU by design, so their skew is reported, not checked)."""
    flags = []
    if s2.get("pid") != region.get("pid"):
        flags.append("S2_PID_MISMATCH")
    for k in ("R", "warmup_iters", "coll", "size_bytes"):
        if s2.get(k) != region.get(k):
            flags.append(f"S2_{k.upper()}_MISMATCH")
    rg = {int(g["rank"]): g for g in region.get("gpus") or []}
    sg = {int(g["rank"]): g for g in s2.get("gpus") or []}
    if sorted(rg) != sorted(sg) or not rg:
        flags.append("S2_GPU_SET_MISMATCH")
    per = []
    for rank in sorted(set(rg) & set(sg)):
        r, s = rg[rank], sg[rank]
        f = []
        if norm_pci(r["pci"]) != norm_pci(s["pci"]):
            f.append("S2_PCI_MISMATCH")
        e_prep = None
        try:
            e_prep = counter_delta_J(s["e_prep0_mJ"], r["e_start_mJ"], s["rc_prep0"], r["rc_start"])
        except InvalidCounter as e:
            f.append(f"E_PREP_TO_C0_UNAVAILABLE:{e}")
        order = [s["t_prep0_b_ns"], s["t_prep_start_ns"], s["t_prep_end_ns"], s["t_probe_start_ns"],
                 s["t_probe_end_ns"], s["t_pause_a_ns"], s["t_pause_b_ns"], r["t_start_a_ns"]]
        if any(x < 0 for x in order) or any(b < a for a, b in zip(order, order[1:])):
            f.append("S2_TIME_ORDER")
        t_work_end = s.get("t_work_end_ns")
        if t_work_end is None or t_work_end < 0:
            f.append("S2_NO_WORK_END")                  # v2.4 records lack it; v2.5 records it before the poller stop
            t_work_end = None
        elif not (r["t_start_b_ns"] <= t_work_end <= r["t_loop_end_ns"]):
            f.append("S2_WORK_END_ORDER")
        pause_act = s["t_pause_b_ns"] - s["t_pause_a_ns"]
        dose = s.get("dose") or {}
        pp = s.get("post_probe") or {}
        n_dose, n_pp = int(dose.get("n", 0)), int(pp.get("n", 0))
        expected = 1 + int(s["probe_reads"]) + 1 + n_dose + 1 + 1 + n_pp
        if int(s["n_counter_reads"]) != expected:
            f.append("COUNTER_CALL_INVENTORY_MISMATCH")
        d = None
        if int(region_dose := s2.get("dose_period_ms", 0)) > 0:
            d = dose_summary(dose, region_dose, r["t_start_b_ns"], t_work_end)
            d["poller"] = "active"
            f += d["flags"]
        else:
            # inactive poller (E0): no stop/join times exist and none are required; it must simply not have run
            d = {"poller": "inactive", "n_reads": n_dose, "order": "not applicable (no poller)"}
            if n_dose:
                f.append("DOSE_READS_WITHOUT_DOSE_REQUEST")
            if any(int(dose.get(k, -1)) >= 0 for k in ("base_ns", "stop_req_ns", "joined_ns")):
                f.append("DOSE_POLLER_TIMES_WITHOUT_DOSE_REQUEST")
        pp_out = None
        if int(s2.get("post_probe", 0)):
            reads = pp.get("reads", [])
            if len(reads) != n_pp:
                f.append("POST_PROBE_COUNT_MISMATCH")
            if pp.get("overflow"):
                f.append("POST_PROBE_OVERFLOW")
            if pp.get("start_ns", -1) < r.get("t_post_b_ns", 0):
                f.append("POST_PROBE_BEFORE_CPOST")
            pp_out = {"n": n_pp, "brackets": brackets_from_reads(reads), "span_s": (pp["end_ns"] - pp["start_ns"]) / 1e9,
                      "statuses_ok": all(x[3] == NVML_SUCCESS for x in reads)}
        elif n_pp:
            f.append("POST_PROBE_READS_WITHOUT_REQUEST")
        per.append({"rank": rank, "pci": norm_pci(r["pci"]), "E_prep_to_C0_J": e_prep,
                    "prep_duration_s": (s["t_prep_end_ns"] - s["t_prep_start_ns"]) / 1e9,
                    "probe_reads": int(s["probe_reads"]), "probe_span_s": (s["t_probe_end_ns"] - s["t_probe_start_ns"]) / 1e9,
                    "pause_req_ns": int(s2["pause_req_ns"]), "pause_act_ns": int(pause_act),
                    "pause_dev_ms": (pause_act - int(s2["pause_req_ns"])) / 1e6,
                    "n_counter_reads": int(s["n_counter_reads"]), "n_counter_reads_expected": expected,
                    "c_prep0_mid_ns": _mid(s["t_prep0_a_ns"], s["t_prep0_b_ns"]),
                    "c0_mid_ns": _mid(r["t_start_a_ns"], r["t_start_b_ns"]), "c1_mid_ns": _mid(r["t_end_a_ns"], r["t_end_b_ns"]),
                    "t_work_end_ns": t_work_end,
                    "join_after_work_end_ms": None if t_work_end is None else (r["t_loop_end_ns"] - t_work_end) / 1e6,
                    "dose": d, "post_probe": pp_out, "flags": f})
        flags += f

    def skew(key):
        v = [p[key] for p in per]
        return (max(v) - min(v)) / 1e6 if len(v) >= 2 else None
    out = {"per_gpu": per, "read_skew_ms": {"C_prep0": skew("c_prep0_mid_ns"), "C0": skew("c0_mid_ns"),
                                            "C1_by_design_per_gpu": skew("c1_mid_ns")},
           "mode": s2.get("mode"), "dose_period_ms": s2.get("dose_period_ms"), "post_probe": s2.get("post_probe"),
           "flags": sorted(set(flags))}
    e = [p["E_prep_to_C0_J"] for p in per]
    out["E_prep_to_C0_J"] = sum(e) if per and all(x is not None for x in e) else None
    pd = [p["prep_duration_s"] for p in per]
    out["prep_duration_s_max"] = max(pd) if pd else None
    out["prep_duration_s_min"] = min(pd) if pd else None
    return out


# ------------------------------------------------------------------ required telemetry channels (counter mode too)

V3_COLUMNS = ("t_mono", "gpu", "power_w", "temp_c", "clk_gr_mhz", "clk_mem_mhz", "util_gpu", "util_mem", "pstate",
              "clock_events", "enforced_limit_mw", "mem_used_mib", "query_ms", "power_inst_w", "t_power_a_ns",
              "t_power_b_ns", "t_inst_a_ns", "t_inst_b_ns", "inst_field_ts_us", "inst_latency_us", "inst_status",
              "t_realtime_ns", "t_mono_ns")
REQUIRED_CHANNELS = ("power_w", "temp_c", "clk_gr_mhz", "pstate", "clock_events", "power_inst_w")
OPTIONAL_CHANNELS = ("total_power_samples",)        # nvmlDeviceGetSamples: may be unsupported; recorded, never required


def required_channel_problems(rows: list, gpus: list[int], windows: dict, period_s: float) -> list[str]:
    """Every REQUIRED channel, every allocated GPU, over its window [t0, t1] (s, perf_counter clock): status 0 for
    POWER_INSTANT, finite values, integer P-state and clock-event fields, ordered per-channel query brackets, and
    coverage measured at each channel's own query time (bracket midpoint): first sample at or before t0, last at or
    after t1, no gap > 3 x period. rows: schema-v3 sampler rows (V3_COLUMNS order). Optional channels are not checked
    here; their status is recorded separately."""
    ix = {k: i for i, k in enumerate(V3_COLUMNS)}
    why = []
    for g in gpus:
        if g not in windows:
            why.append(f"NO_WINDOW_GPU{g}")
            continue
        t0, t1 = windows[g]
        rg = [r for r in rows if r[1] == g and len(r) == len(V3_COLUMNS)]
        if not rg:
            why.append(f"NO_V3_ROWS_GPU{g}")
            continue
        q_p = [((r[ix["t_power_a_ns"]] + r[ix["t_power_b_ns"]]) / 2e9, r) for r in rg]
        q_i = [((r[ix["t_inst_a_ns"]] + r[ix["t_inst_b_ns"]]) / 2e9, r) for r in rg]
        for name, q in (("power", q_p), ("inst", q_i)):
            ts = [t for t, _ in q]
            inside = [t for t in ts if t0 <= t <= t1]
            before = [t for t in ts if t <= t0]
            after = [t for t in ts if t >= t1]
            if not before or not after:
                why.append(f"{name.upper()}_WINDOW_NOT_COVERED_GPU{g}")
                continue
            seq = [before[-1]] + inside + [after[0]]
            if any(b - a > 3 * period_s for a, b in zip(seq, seq[1:])):
                why.append(f"{name.upper()}_SAMPLE_GAP_GPU{g}")
        win = [r for r in rg if t0 - 3 * period_s <= r[0] <= t1 + 3 * period_s]      # by the sampler loop time
        for r in win:
            if r[ix["inst_status"]] != 0:
                why.append(f"POWER_INSTANT_STATUS_GPU{g}")
                break
        for ch in ("power_w", "temp_c", "clk_gr_mhz", "power_inst_w"):
            if any(not isinstance(r[ix[ch]], (int, float)) or not math.isfinite(r[ix[ch]]) for r in win):
                why.append(f"NONFINITE_{ch.upper()}_GPU{g}")
        for ch in ("pstate", "clock_events"):
            if any(not isinstance(r[ix[ch]], (int, np.integer)) or isinstance(r[ix[ch]], bool) for r in win):
                why.append(f"INVALID_{ch.upper()}_GPU{g}")
        if any(r[ix["t_power_a_ns"]] > r[ix["t_power_b_ns"]] or r[ix["t_inst_a_ns"]] > r[ix["t_inst_b_ns"]] for r in rg):
            why.append(f"QUERY_BRACKET_REVERSED_GPU{g}")
        if any(r[ix["inst_field_ts_us"]] <= 0 for r in rg if r[ix["inst_status"]] == 0):
            why.append(f"FIELD_TIMESTAMP_MISSING_GPU{g}")
    return sorted(set(why))
