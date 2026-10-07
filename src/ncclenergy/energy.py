"""Energy measurement primitives.

Primary energy source: NVML cumulative energy counter
(nvmlDeviceGetTotalEnergyConsumption, mJ since driver load), read only at
region boundaries. Secondary source: trapezoidal integration of sampled
nvmlDeviceGetPowerUsage. Both are recorded for every region.
"""

from __future__ import annotations

import time

import numpy as np
import pynvml as N

from .nvml_sampler import NvmlSampler, gpu_processes, handle


def counter_supported(h) -> bool:
    try:
        N.nvmlDeviceGetTotalEnergyConsumption(h)
        return True
    except N.NVMLError:
        return False


def read_counters_mj(handles) -> list[int | None]:
    out = []
    for h in handles:
        try:
            out.append(int(N.nvmlDeviceGetTotalEnergyConsumption(h)))
        except N.NVMLError:
            out.append(None)
    return out


def integrate_power_j(t: np.ndarray, p_w: np.ndarray, t0: float, t1: float) -> float:
    """Trapezoidal integral of power over [t0, t1] (J).

    Power is linearly interpolated at the boundaries. Returns NaN if the
    samples do not cover the interval.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p_w, dtype=float)
    if len(t) < 2 or t[0] > t0 or t[-1] < t1 or t1 <= t0:
        return float("nan")
    inside = (t > t0) & (t < t1)
    tt = np.concatenate([[t0], t[inside], [t1]])
    pp = np.concatenate([[np.interp(t0, t, p)], p[inside], [np.interp(t1, t, p)]])
    return float(np.trapezoid(pp, tt)) if hasattr(np, "trapezoid") else float(np.trapz(pp, tt))


def foreign_processes(gpus: list[int]) -> dict[int, list[dict]]:
    """Processes present on the given GPUs (any process counts as foreign
    because the orchestrator itself never creates a CUDA context)."""
    res = {}
    for g in gpus:
        procs = gpu_processes(handle(g))
        if procs:
            res[g] = procs
    return res


def wait_for_pstate(gpus: list[int], pstate: int, timeout_s: float, poll_s: float = 1.0) -> tuple[bool, float]:
    hs = [handle(g) for g in gpus]
    t0 = time.perf_counter()
    while True:
        states = [N.nvmlDeviceGetPerformanceState(h) for h in hs]
        if all(s == pstate for s in states):
            return True, time.perf_counter() - t0
        if time.perf_counter() - t0 > timeout_s:
            return False, time.perf_counter() - t0
        time.sleep(poll_s)


def measure_idle(gpus: list[int], duration_s: float, settle_timeout_s: float,
                 sampler_period_s: float, min_settle_s: float = 5.0) -> dict:
    """Measure idle power of each GPU with no process attached.

    Waits for every GPU to reach P8 (lowest idle P-state) and then an extra
    `min_settle_s`, then measures for `duration_s` with both the energy
    counter (2 reads) and the power sampler.
    """
    rec = {"kind": "idle_noctx", "gpus": gpus, "t_wall_start": time.time()}
    fp = foreign_processes(gpus)
    if fp:
        rec.update(status="foreign_process", foreign=fp)
        return rec
    ok, waited = wait_for_pstate(gpus, 8, settle_timeout_s)
    rec["pstate_wait_s"] = waited
    rec["reached_p8"] = ok
    time.sleep(min_settle_s)
    hs = [handle(g) for g in gpus]
    s = NvmlSampler(gpus=gpus, period_s=sampler_period_s).start()
    time.sleep(0.5)  # make sure the sampler covers t0
    t0 = time.perf_counter()
    c0 = read_counters_mj(hs)
    time.sleep(duration_s)
    c1 = read_counters_mj(hs)
    t1 = time.perf_counter()
    time.sleep(0.5)
    s.stop()
    fp_after = foreign_processes(gpus)
    rows = np.array([(r[0], r[1], r[2], r[5], r[8]) for r in s.rows])  # t, gpu, power, temp, pstate
    per = {}
    for i, g in enumerate(gpus):
        m = rows[:, 1] == g
        t, p = rows[m, 0], rows[m, 2]
        inwin = (t >= t0) & (t <= t1)
        e_ctr = None if (c0[i] is None or c1[i] is None) else (c1[i] - c0[i]) / 1000.0
        per[str(g)] = {
            "energy_counter_j": e_ctr,
            "power_counter_w": None if e_ctr is None else e_ctr / (t1 - t0),
            "energy_integrated_j": integrate_power_j(t, p, t0, t1),
            "power_sampled_mean_w": float(np.mean(p[inwin])) if inwin.any() else None,
            "temp_mean_c": float(np.mean(rows[m, 3][inwin])) if inwin.any() else None,
            "pstates": sorted({int(x) for x in rows[m, 4][inwin]}),
        }
    rec.update(status="ok" if not fp_after else "foreign_process_during",
               duration_s=t1 - t0, per_gpu=per, sampler=s.stats(),
               foreign_after=fp_after or None)
    return rec
