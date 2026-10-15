"""Quality-control classification of one unit attempt (pure function).

qc_status answers "is this a valid energy/latency measurement?";
evidence_status (VERIFIED/UNVERIFIED) answers "do we know which configuration ran?".
"""

from __future__ import annotations
import math

INFRA = "INFRA_FAIL"           # retryable once, outcome-blind
OK = "OK"

# clock-event (throttle) reason bits that invalidate a region / stop a block
CLOCK_EVENT_BAD = {
    0x4: "SW_POWER_CAP",
    0x8: "HW_SLOWDOWN",
    0x20: "SW_THERMAL",
    0x40: "HW_THERMAL",
    0x80: "HW_POWER_BRAKE",
}


def bad_clock_events(mask: int) -> list[str]:
    return [name for bit, name in CLOCK_EVENT_BAD.items() if mask & bit]


QC_METHOD = "v2.3-finite-order-qc"  # Future-only; archived method IDs remain unchanged.


def classify(attempt: dict, T_min: float, energy_source: str = "counter") -> dict:
    """attempt keys used: exit_code, timed_out, region (metrics dict or None), wrong (int|None),
    foreign_pids (list), clock_events_or (int), temp_max_c, temp_stop_c, region_flags (list),
    data_check_required (default True), telemetry_problems (list; None = unknown), evidence_status.

    v2.2 is fail-closed: a missing correctness check, unknown or incomplete telemetry and missing or
    unverified execution evidence each prevent an OK classification (they are never silently accepted)."""
    reasons: list[str] = []
    if "exit_code" not in attempt:          # v2.2 release audit: never default a missing exit status to success
        return {"qc_status": INFRA, "reasons": ["EXIT_UNKNOWN"], "retryable": True}
    if attempt.get("timed_out"):
        return {"qc_status": INFRA, "reasons": ["TIMEOUT"], "retryable": True}
    if attempt.get("exit_code", 0) != 0:
        return {"qc_status": INFRA, "reasons": [f"EXIT_{attempt.get('exit_code')}"], "retryable": True}
    m = attempt.get("region")
    if not m:
        return {"qc_status": INFRA, "reasons": ["NO_REGION_RECORD"], "retryable": True}
    if attempt.get("region_parse_flags"):   # corrupt/unknown region lines: the record cannot be trusted
        return {"qc_status": "MEASUREMENT_INVALID",
                "reasons": sorted({f"REGION_PARSE:{f}" for f in attempt["region_parse_flags"]}), "retryable": False}
    if attempt.get("foreign_pids"):
        return {"qc_status": "FOREIGN_PROCESS", "reasons": ["FOREIGN_PROCESS_ON_ALLOCATED_GPU"], "retryable": False,
                "stop_block": True}
    prefixes = ("COUNTER", "UNAVAILABLE", "TIME_REVERSAL", "R_", "GPU_SET", "RANK_GPU", "NVML_NOT", "EVENT_ERROR",
                "SCHEMA")
    if energy_source == "sampled":       # declared fallback: counter flags are expected, everything else still hard
        prefixes = ("TIME_REVERSAL", "R_", "GPU_SET", "RANK_GPU", "EVENT_ERROR")
        se = attempt.get("sampled_energy_J")
        if se is None or not math.isfinite(se) or se <= 0:
            return {"qc_status": "MEASUREMENT_INVALID", "reasons": ["NO_SAMPLED_ENERGY"], "retryable": False}
    hard = [f for f in m.get("flags", []) if f.startswith(prefixes)]
    if hard:
        return {"qc_status": "MEASUREMENT_INVALID", "reasons": sorted(set(hard)), "retryable": False}
    if attempt.get("wrong") not in (0, None):
        return {"qc_status": "CORRECTNESS_FAIL", "reasons": [f"WRONG_{attempt['wrong']}"], "retryable": False}
    if attempt.get("wrong") is None:
        if attempt.get("data_check_required", True):
            return {"qc_status": "MEASUREMENT_INVALID", "reasons": ["NO_DATA_CHECK"], "retryable": False}
        reasons.append("NO_DATA_CHECK_NOT_REQUIRED")
    tp = attempt.get("telemetry_problems")
    if tp is None:
        return {"qc_status": "MEASUREMENT_INVALID", "reasons": ["TELEMETRY_UNKNOWN"], "retryable": False}
    if tp:
        return {"qc_status": "MEASUREMENT_INVALID", "reasons": ["TELEMETRY_INCOMPLETE"] + sorted(set(tp)),
                "retryable": False}
    ce = bad_clock_events(int(attempt.get("clock_events_or", 0)))
    # On this driver SW_POWER_CAP is reported intermittently at idle far below the limit (Phase 3);
    # it counts as throttling only when the region's power is within 10 % of the enforced limit.
    pf = attempt.get("power_frac_max")      # None = unknown limit or power: the flag is kept (fail closed)
    if "SW_POWER_CAP" in ce and pf is not None and pf < 0.9:
        ce.remove("SW_POWER_CAP")
        reasons.append("SW_POWER_CAP_FLAG_BELOW_LIMIT")
    if ce:
        return {"qc_status": "THROTTLED", "reasons": ce, "retryable": False, "stop_block": True}
    if attempt.get("temp_max_c") is not None and attempt.get("temp_stop_c") is not None \
            and attempt["temp_max_c"] >= attempt["temp_stop_c"]:
        return {"qc_status": "TOO_HOT", "reasons": [f"TEMP_{attempt['temp_max_c']}"], "retryable": False,
                "stop_block": True}
    ev = attempt.get("evidence_status")
    if ev != "VERIFIED":
        return {"qc_status": "EXECUTION_UNVERIFIED", "reasons": [f"EVIDENCE_{ev or 'MISSING'}"], "retryable": False}
    if m.get("T_region_s") is None or not math.isfinite(m["T_region_s"]) or m["T_region_s"] < T_min:
        return {"qc_status": "QC_SHORT", "reasons": [f"T_REGION_{m.get('T_region_s')}"], "retryable": False}
    return {"qc_status": OK, "reasons": reasons, "retryable": False}


def telemetry_problems(sampler_info: dict | None, per_gpu: dict | None, n_gpus: int, energy_source: str) -> list[str]:
    """Required-telemetry check for one region. sampler_info: {errors, truncated}; per_gpu: {gpu: window summary}."""
    if sampler_info is None or per_gpu is None:
        return ["NO_TELEMETRY"]
    out = []
    if sampler_info.get("errors"):
        out.append("SAMPLER_ERRORS")
    if sampler_info.get("truncated"):
        out.append("SAMPLER_TRUNCATED")
    if len(per_gpu) != n_gpus:
        out.append("TELEMETRY_GPU_MISSING")
    for g, w in per_gpu.items():
        gp = w.get("gaps") or {}
        if (w.get("n_samples") or 0) < 2:
            out.append(f"FEW_SAMPLES_GPU{g}")
        elif gp.get("n_gaps"):
            out.append(f"SAMPLE_GAP_GPU{g}")
        if gp.get("nonmonotonic") or gp.get("nonfinite"):
            out.append(f"SAMPLE_TIME_INVALID_GPU{g}")
        if w.get("temp_max_c") is None or w.get("clock_events_or") is None:
            out.append(f"NO_TEMP_OR_CLOCK_EVENTS_GPU{g}")
        elif not math.isfinite(w["temp_max_c"]) or not math.isfinite(w["clock_events_or"]):
            out.append(f"NONFINITE_TEMP_OR_CLOCK_EVENTS_GPU{g}")
        if energy_source == "sampled":
            e = w.get("energy_int_inst_J")
            if e is None or not math.isfinite(e) or e <= 0:
                lag = w.get("energy_int_lag_J")
                if lag is None or not math.isfinite(lag) or lag <= 0:
                    out.append(f"NO_SAMPLED_ENERGY_GPU{g}")
    return out
