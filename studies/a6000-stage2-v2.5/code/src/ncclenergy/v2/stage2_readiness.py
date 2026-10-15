"""Thermal readiness, P8 baselines and pre-region comparability (CPU only; operates on recorded rows).

Kept apart:
  * safety/availability ceilings: W_max (bounded launch wait) and the 80 C stop. Never scientific criteria;
  * the held-out launch gate R1, whose threshold T_launch is derived from calibration only;
  * pre-region comparability R2 (state immediately before C0) and arm thermal balance R3 (per GPU, per contrast).

Decisions use the EXPLICITLY ALLOCATED GPU list, never a GPU set inferred from rows, and are made AT a decision time
t_now from rows at or before t_now only: a stable window must end at t_now (every GPU's newest row no older than
max_age_s). An earlier ready interval never authorizes a later launch, and an expired wait never does.

Row validity (every row in the window, every allocated GPU): finite temperature; pstate == 8; foreign is exactly
False (unknown process state is not False); clock_events present, and on a P8 row free of every bit except the idle
bits P8_TOLERATED_CLOCK_EVENTS (correction G, user-approved 2026-10-13); no sampler error in the window. The raw
bitmask is never altered; clock_event_bit_shares() reports how often each bit occurred.

Fail-closed numerics: NaN, inf, None, missing fields, missing GPUs, gaps and too few samples give a non-OK status with a
reason. Quantiles are nearest-rank order statistics (no interpolation); an unreachable or censored time is +inf.
"""

from __future__ import annotations

import math

import numpy as np

P8 = 8
# Correction G (2026-10-13): at P8 idle the RTX A6000 on driver 580 never reports clock events 0. It reports 0x1
# (GpuIdle: "Nothing is running on the GPU and the clocks are dropping to Idle state") or 0x4 (SwPowerCap; why it is
# set at ~25 W idle is undocumented): 0 of 14,904 gate 3b idle rows were 0. Both are tolerated on P8 rows of the IDLE
# gates only; every other bit fails. In-region QC and the pre-region check R2 are unchanged.
CE_GPU_IDLE, CE_SW_POWER_CAP = 0x1, 0x4
P8_TOLERATED_CLOCK_EVENTS = CE_GPU_IDLE | CE_SW_POWER_CAP
UNREACHABLE = math.inf          # sentinel: gate not satisfied within the recorded (censored) trace
MIN_CAL_LAUNCHES = 15           # of the 21 C1/C2 calibration launches
MIN_COOLDOWN_TRACES = 15
MAX_RAISE_C = 5


class NonFinite(ValueError):
    pass


def _finite(x, what: str) -> float:
    if x is None or isinstance(x, bool):
        raise NonFinite(f"{what}: missing ({x!r})")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise NonFinite(f"{what}: not a number ({x!r})") from None
    if not math.isfinite(v):
        raise NonFinite(f"{what}: non-finite ({x!r})")
    return v


def nearest_rank(values, q: float) -> float:
    """Nearest-rank quantile (smallest value with at least q of the sample at or below it). +inf sentinels are
    ordered last; NaN is refused; an empty sample is refused."""
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    v = [float(x) for x in values]
    if not v:
        raise ValueError("empty sample")
    if any(math.isnan(x) for x in v):
        raise NonFinite("NaN in quantile input")
    v.sort()
    return v[max(0, math.ceil(q * len(v)) - 1)]


def _row_problems(r: dict) -> list[str]:
    why = []
    T = r.get("temp_c")
    if T is None or isinstance(T, bool) or not isinstance(T, (int, float)) or not math.isfinite(T):
        why.append("TEMP_INVALID")
    if r.get("pstate") != P8:
        why.append("NOT_P8")
    if r.get("foreign") is not False:
        why.append("FOREIGN_OR_UNKNOWN_PROCESS")
    ce = r.get("clock_events")
    tolerated = P8_TOLERATED_CLOCK_EVENTS if r.get("pstate") == P8 else 0
    if ce is None or isinstance(ce, bool) or not isinstance(ce, (int, np.integer)) or int(ce) & ~tolerated:
        why.append("CLOCK_EVENTS_NOT_TOLERATED_OR_MISSING")
    if r.get("error"):
        why.append("SAMPLER_ERROR")
    return why


def clock_event_bit_shares(rows: list[dict], gpus: list[int]) -> dict:
    """Per allocated GPU: number of valid rows and the share of rows carrying each clock-event bit (raw bitmask), plus
    the share with the bitmask exactly 0. Evidence for correction G; never a decision input."""
    out = {}
    for g in gpus:
        ce = [int(r["clock_events"]) for r in rows if r.get("gpu") == g and not r.get("error")
              and isinstance(r.get("clock_events"), (int, np.integer)) and not isinstance(r.get("clock_events"), bool)]
        bits = sorted({b for v in ce for b in range(64) if v >> b & 1})
        out[str(g)] = {"n_rows": len(ce), "share_zero": (sum(1 for v in ce if v == 0) / len(ce)) if ce else None,
                       "share_by_bit": {hex(1 << b): sum(1 for v in ce if v >> b & 1) / len(ce) for b in bits}}
    return out


def gate_at(trace: list[dict], gpus: list[int], t_now: float, t_launch_c: float, stable_window_s: float = 30.0,
            stable_delta_c: float = 1.0, max_gap_s: float = 2.0, max_age_s: float = 2.0) -> dict:
    """R1 decision AT t_now. Uses rows with t_s <= t_now only. READY iff, for every allocated GPU: rows cover
    [t_now - W, t_now] (first row within max_gap_s of the window start, newest row at most max_age_s old, no gap >
    max_gap_s), every row is valid (_row_problems), max T <= t_launch_c and max - min T <= stable_delta_c. Rows of a
    GPU outside the allocation make the decision NOT_READY (the trace does not describe the allocation)."""
    reasons = []
    if not gpus or len(set(gpus)) != len(gpus):
        return {"status": "NOT_READY", "t_s": t_now, "reasons": ["ALLOCATION_INVALID"]}
    try:
        _finite(t_launch_c, "t_launch"), _finite(t_now, "t_now")
    except NonFinite as e:
        return {"status": "NOT_READY", "t_s": t_now, "reasons": [f"NONFINITE_PARAMETER: {e}"]}
    past = [r for r in trace if isinstance(r.get("t_s"), (int, float)) and math.isfinite(r["t_s"]) and r["t_s"] <= t_now]
    extra = sorted({r.get("gpu") for r in past} - set(gpus), key=str)
    if extra:
        reasons.append(f"ROWS_FOR_UNALLOCATED_GPUS_{extra}")
    if t_now < stable_window_s:
        reasons.append("WINDOW_NOT_ELAPSED")
    for g in gpus:
        rg = sorted((r for r in past if r.get("gpu") == g), key=lambda r: r["t_s"])
        if not rg:
            reasons.append(f"NO_ROWS_GPU{g}")
            continue
        if t_now - rg[-1]["t_s"] > max_age_s:
            reasons.append(f"STALE_GPU{g}")
        w = [r for r in rg if r["t_s"] >= t_now - stable_window_s]
        ts = [r["t_s"] for r in w]
        if not w or ts[0] > t_now - stable_window_s + max_gap_s or any(b - a > max_gap_s for a, b in zip(ts, ts[1:])):
            reasons.append(f"WINDOW_NOT_COVERED_GPU{g}")
            continue
        bad = sorted({p for r in w for p in _row_problems(r)})
        if bad:
            reasons += [f"{p}_GPU{g}" for p in bad]
            continue
        temps = [float(r["temp_c"]) for r in w]
        if max(temps) > t_launch_c:
            reasons.append(f"ABOVE_T_LAUNCH_GPU{g}")
        if max(temps) - min(temps) > stable_delta_c:
            reasons.append(f"UNSTABLE_GPU{g}")
    return {"status": "NOT_READY" if reasons else "READY", "t_s": t_now, "reasons": reasons}


def gate_over(trace: list[dict], gpus: list[int], t_launch_c: float, w_max_s: float, decision_period_s: float = 1.0,
              **kw) -> dict:
    """Offline replay of the live gate: decisions at t = W, W + dt, ... <= w_max_s, each from rows at or before t.
    The first READY decision time, else NOT_ACHIEVED (expiry is never readiness)."""
    W = kw.get("stable_window_s", 30.0)
    n = int(math.floor((w_max_s - W) / decision_period_s + 1e-9))
    for i in range(max(0, n) + 1):
        t = W + i * decision_period_s
        if t > w_max_s:
            break
        r = gate_at(trace, gpus, t, t_launch_c, **kw)
        if r["status"] == "READY":
            return {"status": "READY", "t_ready_s": t}
    return {"status": "NOT_ACHIEVED", "t_ready_s": None, "reason": "READINESS_NOT_ACHIEVED_WITHIN_W_MAX"}


def baseline_problems(rows: list[dict], gpus: list[int], t0: float, t1: float, max_gap_s: float) -> list[str]:
    """Validity of a baseline interval [t0, t1] (also used live, before each counter read): every allocated GPU has
    rows covering the interval (first within max_gap_s of t0, newest within max_gap_s of t1, no gap > max_gap_s) and
    every row is valid (P8, process state known and empty of foreign processes, clock events 0, finite temperature)
    with finite power."""
    why = []
    extra = sorted({r.get("gpu") for r in rows if t0 <= r.get("t_s", -math.inf) <= t1} - set(gpus), key=str)
    if extra:
        why.append(f"ROWS_FOR_UNALLOCATED_GPUS_{extra}")
    for g in gpus:
        w = sorted((r for r in rows if r.get("gpu") == g and t0 <= r["t_s"] <= t1), key=lambda r: r["t_s"])
        ts = [r["t_s"] for r in w]
        if not w or ts[0] > t0 + max_gap_s or ts[-1] < t1 - max_gap_s or any(b - a > max_gap_s for a, b in zip(ts, ts[1:])):
            why.append(f"BASELINE_SAMPLING_GAP_GPU{g}")
            continue
        bad = sorted({p for r in w for p in _row_problems(r)})
        why += [f"{p}_GPU{g}" for p in bad]
        if any(not isinstance(r.get("power_w"), (int, float)) or isinstance(r.get("power_w"), bool)
               or not math.isfinite(r["power_w"]) for r in w):
            why.append(f"NONFINITE_POWER_GPU{g}")
    return why


def p8_entry(rows: list[dict], gpus: list[int], wait_cap_s: float) -> float | None:
    """First time <= wait_cap_s at which every allocated GPU's newest row is valid (P8, known empty process state,
    clock events 0, finite temperature)."""
    for t in sorted({r["t_s"] for r in rows}):
        if t > wait_cap_s:
            return None
        ok = True
        for g in gpus:
            rg = [r for r in rows if r.get("gpu") == g and r["t_s"] <= t]
            if not rg or t - rg[-1]["t_s"] > 0.5 or _row_problems(rg[-1]):
                ok = False
                break
        if ok:
            return t
    return None


def p8_baseline(rows: list[dict], gpus: list[int], wait_cap_s: float = 60.0, settle_s: float = 10.0,
                measure_s: float = 30.0, max_gap_s: float = 0.2) -> dict:
    """Offline evaluation of a recorded baseline, fail-closed: P8 entry within the cap, then the whole interval
    [entry, entry + settle + measure] valid (baseline_problems). Mean power over the measure window per GPU."""
    if not gpus:
        return {"status": "FAILED", "reason": "ALLOCATION_INVALID"}
    entry = p8_entry(rows, gpus, wait_cap_s)
    if entry is None:
        return {"status": "FAILED", "reason": "P8_NOT_REACHED_WITHIN_CAP"}
    t_end = entry + settle_s + measure_s
    if max(r["t_s"] for r in rows) < t_end - max_gap_s:
        return {"status": "FAILED", "reason": "BASELINE_WINDOW_NOT_COVERED"}
    why = baseline_problems(rows, gpus, entry, t_end, max_gap_s)
    if why:
        return {"status": "FAILED", "reason": "INVALID_DURING_BASELINE: " + ",".join(why)}
    meas = [r for r in rows if entry + settle_s <= r["t_s"] <= t_end]
    return {"status": "OK", "entry_s": entry,
            "mean_power_W": {g: float(np.mean([r["power_w"] for r in meas if r["gpu"] == g])) for g in gpus}}


def time_to_ready(trace: list[dict], gpus: list[int], t_launch_c: float, **kw) -> float:
    """Time at which the FULL R1 gate (P8, process state, clock events, coverage, stability, threshold) is first
    satisfied in a recorded wait, replayed at 1 s decisions; +inf if never within the trace (censored at its end)."""
    if not trace:
        return UNREACHABLE
    end = max(r["t_s"] for r in trace)
    r = gate_over(trace, gpus, t_launch_c, end, **kw)
    return r["t_ready_s"] if r["status"] == "READY" else UNREACHABLE


def derive_heldout_rules(cal: list[dict], wait_traces: list[list[dict]], gpus: list[int],
                         wait_margin_s: float = 200.0, min_launches: int = MIN_CAL_LAUNCHES,
                         min_traces: int = MIN_COOLDOWN_TRACES, max_raise_c: int = MAX_RAISE_C, **gate_kw) -> dict:
    """Rules fixed before calibration, evaluated on calibration temperatures/clocks/states only (never energies).

    cal: one record per ACCEPTED C1/C2 calibration launch: {"gpus": {g: {t_launch_c, t_region_c, clk_mhz, pstate}}}.
    wait_traces: per launch, the recorded 1 Hz rows of its fixed wait ({t_s, gpu, temp_c, pstate, foreign,
    clock_events}). gpus: the allocated pair.

    T_launch := ceil(median_launch max_g t_launch_c); raised by 1 C (at most max_raise_c times) until the nearest-rank
    q90 over waits of time_to_ready (the full gate replayed on the trace) is <= wait_margin_s. The calibration waits
    are 120 s long, so a time is censored (+inf) beyond 120 s: in effect q90 must be OBSERVED within the 120 s waits.
    Bands per GPU: T_region in [q05 - 1, q95 + 1] C (nearest-rank); clock in [min - 15, max + 15] MHz; P-states seen.
    status OK is the only usable result; NOT_DERIVABLE / NOT_ACHIEVABLE carry no threshold (Freeze-H then needs an
    amendment)."""
    def fail(status, reason, **kw):
        return {"status": status, "reason": reason, "t_launch_c": None, "bands": None, **kw}
    if not gpus or len(set(gpus)) != len(gpus):
        return fail("NOT_DERIVABLE", "ALLOCATION_INVALID")
    if len(cal) < min_launches:
        return fail("NOT_DERIVABLE", f"INSUFFICIENT_CALIBRATION_LAUNCHES: {len(cal)} < {min_launches}")
    if len(wait_traces) < min_traces:
        return fail("NOT_DERIVABLE", f"INSUFFICIENT_WAIT_TRACES: {len(wait_traces)} < {min_traces}")
    if any(sorted(r["gpus"]) != sorted(gpus) for r in cal):
        return fail("NOT_DERIVABLE", "GPU_SET_DIFFERS_FROM_ALLOCATION")
    try:
        tl = [max(_finite(r["gpus"][g]["t_launch_c"], f"t_launch GPU{g}") for g in gpus) for r in cal]
        for i, tr in enumerate(wait_traces):
            for x in tr:
                _finite(x.get("t_s"), f"trace {i} t")
        tr_ = {g: [_finite(r["gpus"][g]["t_region_c"], f"t_region GPU{g}") for r in cal] for g in gpus}
        ck = {g: [_finite(r["gpus"][g]["clk_mhz"], f"clock GPU{g}") for r in cal] for g in gpus}
        ps = {g: sorted({int(_finite(r["gpus"][g]["pstate"], f"pstate GPU{g}")) for r in cal}) for g in gpus}
    except (NonFinite, KeyError) as e:
        return fail("NOT_DERIVABLE", f"NONFINITE_OR_MISSING_INPUT: {e}")
    t0 = math.ceil(float(np.median(tl)))
    tried, t_launch = [], None
    for r in range(max_raise_c + 1):
        thr = t0 + r
        q90 = nearest_rank([time_to_ready(tr, gpus, thr, **gate_kw) for tr in wait_traces], 0.9)
        tried.append({"t_launch_c": thr, "q90_time_to_full_gate_s": q90})
        if q90 <= wait_margin_s:
            t_launch = thr
            break
    if t_launch is None:
        return fail("NOT_ACHIEVABLE", f"q90 time to the full gate > {wait_margin_s} s (or censored) up to "
                                      f"T_launch = {t0 + max_raise_c} C", tried=tried)
    bands = {g: {"t_region_c": [nearest_rank(tr_[g], 0.05) - 1.0, nearest_rank(tr_[g], 0.95) + 1.0],
                 "clk_mhz": [min(ck[g]) - 15.0, max(ck[g]) + 15.0], "pstates": ps[g]} for g in gpus}
    return {"status": "OK", "t_launch_c": t_launch, "q90_time_to_full_gate_s": tried[-1]["q90_time_to_full_gate_s"],
            "raised_c": len(tried) - 1, "tried": tried, "bands": bands, "n_launches": len(cal),
            "n_traces": len(wait_traces)}


def pre_region_check(state: dict, bands: dict, gpus: list[int]) -> dict:
    """R2. state: {g: {t_region_c, clk_mhz, pstate, clock_events, inst_age_s}} for every ALLOCATED GPU. Every GPU must
    lie inside its band, with clock_events present and 0 and a POWER_INSTANT row at most 1 s old."""
    why = []
    for g in gpus:
        s, b = state.get(g), bands.get(g)
        if s is None or b is None:
            why.append(f"MISSING_STATE_OR_BAND_GPU{g}")
            continue
        try:
            tr, cl, pst = _finite(s.get("t_region_c"), "t_region"), _finite(s.get("clk_mhz"), "clock"), \
                int(_finite(s.get("pstate"), "pstate"))
        except NonFinite:
            why.append(f"NONFINITE_PRE_REGION_STATE_GPU{g}")
            continue
        if not (b["t_region_c"][0] <= tr <= b["t_region_c"][1]):
            why.append(f"T_REGION_OUT_OF_BAND_GPU{g}")
        if not (b["clk_mhz"][0] <= cl <= b["clk_mhz"][1]):
            why.append(f"CLOCK_OUT_OF_BAND_GPU{g}")
        if pst not in b["pstates"]:
            why.append(f"PSTATE_NOT_CALIBRATED_GPU{g}")
        ce = s.get("clock_events")
        if ce is None or isinstance(ce, bool) or ce != 0:
            why.append(f"CLOCK_EVENT_OR_MISSING_GPU{g}")
        age = s.get("inst_age_s")
        if not isinstance(age, (int, float)) or isinstance(age, bool) or not math.isfinite(age) or not 0 <= age <= 1.0:
            why.append(f"STALE_POWER_GPU{g}")
    extra = sorted(set(state) - set(gpus), key=str)
    if extra:
        why.append(f"STATE_FOR_UNALLOCATED_GPUS_{extra}")
    return {"ok": not why, "reasons": why}


def thermal_balance(units: list[dict], gpus: list[int], pairs=(("A", "B"), ("A", "C")), max_diff_c: float = 0.5) -> dict:
    """R3 per (session, length, contrast pair), per GPU: both arms of the pair must be present, every unit must have a
    finite pre-region temperature for every allocated GPU (t_region_gpu {g: C}), and |mean_arm,g - mean_A,g| <=
    max_diff_c for EVERY GPU g. A GPU-mean balance is not enough. A violation, a missing arm or a non-finite value
    labels that contrast 'thermal comparability not established'."""
    out = {}
    for s, L in sorted({(u["session"], u["length_s"]) for u in units}):
        sel = [u for u in units if u["session"] == s and u["length_s"] == L]
        for ref, arm in pairs:
            key = (s, L, f"{arm}-{ref}")
            a, b = [u for u in sel if u["arm"] == ref], [u for u in sel if u["arm"] == arm]
            if not a or not b:
                out[key] = {"ok": False, "reason": "ARM_MISSING"}
                continue
            try:
                diffs = {}
                for g in gpus:
                    ma = np.mean([_finite((u.get("t_region_gpu") or {}).get(g), f"T GPU{g}") for u in a])
                    mb = np.mean([_finite((u.get("t_region_gpu") or {}).get(g), f"T GPU{g}") for u in b])
                    diffs[g] = float(mb - ma)
            except NonFinite:
                out[key] = {"ok": False, "reason": "NONFINITE_OR_MISSING_T_REGION"}
                continue
            out[key] = {"ok": all(abs(d) <= max_diff_c for d in diffs.values()), "diff_c_by_gpu": diffs}
    return out
