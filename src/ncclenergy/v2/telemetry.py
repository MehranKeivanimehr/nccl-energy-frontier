"""NVML diagnostic sampler (read-only) and region-level telemetry processing.

The sampler never calls nvmlDeviceGetTotalEnergyConsumption. On driver 580 the
local Phase 4 experiments support the interpretation that a read discards the
energy accumulated since the counter's last ~100 ms update (not an NVML
guarantee), and the counter is the primary measurement, read only at region
boundaries. NVML_FI_DEV_POWER_INSTANT
perturbed the counter on driver 550 but not on 580 (Phase 4 dose test), so it
is sampled here as the instantaneous-power cross-check.
"""

from __future__ import annotations

import csv
import gzip
import threading
import time
from dataclasses import dataclass, field

import numpy as np

FIELDS = ["t_mono", "gpu", "power_w", "temp_c", "clk_gr_mhz", "clk_mem_mhz", "util_gpu", "util_mem", "pstate",
          "clock_events", "enforced_limit_mw", "mem_used_mib", "query_ms", "power_inst_w"]


def _nvml():
    import pynvml
    return pynvml


@dataclass
class Sampler:
    gpus: list[int]                  # NVML indices
    period_s: float = 0.05
    proc_period_s: float = 1.0
    max_rows: int = 200_000          # hard cap on memory/disk per unit
    rows: list = field(default_factory=list)
    proc_events: list = field(default_factory=list)
    overruns: int = 0
    errors: list = field(default_factory=list)
    truncated: bool = False

    def __post_init__(self):
        N = _nvml()
        self._N = N
        self._h = [N.nvmlDeviceGetHandleByIndex(g) for g in self.gpus]
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="nccle-sampler", daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=10)

    def _sample(self, h):
        N = self._N
        t0 = time.perf_counter()
        p = N.nvmlDeviceGetPowerUsage(h) / 1000.0
        temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
        cg = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_GRAPHICS)
        cm = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM)
        u = N.nvmlDeviceGetUtilizationRates(h)
        ps = N.nvmlDeviceGetPerformanceState(h)
        try:
            ce = N.nvmlDeviceGetCurrentClocksEventReasons(h)
        except Exception:  # noqa: BLE001 - older bindings
            ce = N.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        lim = N.nvmlDeviceGetEnforcedPowerLimit(h)
        mem = N.nvmlDeviceGetMemoryInfo(h).used / 2**20
        fv = N.nvmlDeviceGetFieldValues(h, [N.NVML_FI_DEV_POWER_INSTANT])[0]
        pi = fv.value.uiVal / 1000.0 if fv.nvmlReturn == 0 else float("nan")
        return p, temp, cg, cm, u.gpu, u.memory, ps, int(ce), lim, mem, (time.perf_counter() - t0) * 1e3, pi

    def _run(self):
        N = self._N
        t_start = time.perf_counter()
        next_proc = t_start
        k = 0
        while not self._stop.is_set():
            now = time.perf_counter()
            for g, h in zip(self.gpus, self._h):
                if len(self.rows) >= self.max_rows:
                    self.truncated = True
                    break
                try:
                    self.rows.append((now, g) + self._sample(h))
                except Exception as e:  # noqa: BLE001
                    self.errors.append((now, g, repr(e)))
            if now >= next_proc:
                for g, h in zip(self.gpus, self._h):
                    # None = unknown (the compute-process query failed); [] = known to be empty. A failed query must
                    # never read as "no foreign process" (v2.2 release audit). The graphics query is optional.
                    pids, known = [], True
                    for fn, required in ((N.nvmlDeviceGetComputeRunningProcesses, True),
                                         (N.nvmlDeviceGetGraphicsRunningProcesses, False)):
                        try:
                            pids += [int(p.pid) for p in fn(h)]
                        except Exception:  # noqa: BLE001
                            known = known and not required
                    self.proc_events.append((now, g, pids if known else None))
                next_proc = now + self.proc_period_s
            k += 1
            delay = t_start + k * self.period_s - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                self.overruns += 1
                t_start, k = time.perf_counter(), 0

    def write(self, path) -> None:
        with gzip.open(path, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for r in self.rows:
                w.writerow([f"{r[0]:.6f}", r[1], f"{r[2]:.3f}", *r[3:12], f"{r[12]:.3f}", f"{r[13]:.3f}"])

    def array(self) -> np.ndarray:
        return np.array([r[:14] for r in self.rows], dtype=float) if self.rows else np.zeros((0, 14))


def integrate_trapezoid(t: np.ndarray, p: np.ndarray, t0: float, t1: float) -> float:
    """Trapezoidal integral of power over [t0, t1] with linear interpolation at both ends.
    Returns NaN if the samples do not cover the interval."""
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    if len(t) < 2 or len(p) != len(t) or not np.isfinite([t0, t1]).all() \
            or not np.isfinite(t).all() or not np.isfinite(p).all() or (np.diff(t) <= 0).any() \
            or t1 <= t0 or t[0] > t0 or t[-1] < t1:
        return float("nan")
    order = np.argsort(t)
    t, p = t[order], p[order]
    inside = (t > t0) & (t < t1)
    tt = np.concatenate([[t0], t[inside], [t1]])
    pp = np.concatenate([[np.interp(t0, t, p)], p[inside], [np.interp(t1, t, p)]])
    return float(np.sum((pp[1:] + pp[:-1]) * np.diff(tt)) / 2.0)


def gaps(t: np.ndarray, period: float, factor: float = 3.0) -> dict:
    t = np.asarray(t, float)
    nonmonotonic = int((np.diff(t) <= 0).sum())
    nonfinite = int((~np.isfinite(t)).sum())
    if len(t) < 2:
        return {"n": int(len(t)), "median_dt": None, "max_dt": None, "n_gaps": None,
                "nonmonotonic": nonmonotonic, "nonfinite": nonfinite}
    dt = np.diff(t)
    return {"n": int(len(t)), "median_dt": float(np.median(dt)), "max_dt": float(dt.max()),
            "n_gaps": int((dt > factor * period).sum()), "nonmonotonic": nonmonotonic,
            "nonfinite": nonfinite}


def window_summary(arr: np.ndarray, gpu: int, t0: float, t1: float, period: float, lag_s: float = 0.0) -> dict:
    """Per-GPU telemetry summary over [t0, t1] (seconds, perf_counter clock)."""
    m = arr[:, 1] == gpu if len(arr) else np.zeros(0, bool)
    a = arr[m]
    if len(a) == 0:
        return {"n_samples": 0}
    w = a[(a[:, 0] >= t0) & (a[:, 0] <= t1)]
    # gaps include the last sample before t0 and the first after t1, so a gap spanning a window edge is seen
    ts = np.sort(a[:, 0])
    lo, hi = max(int(np.searchsorted(ts, t0, "left")) - 1, 0), int(np.searchsorted(ts, t1, "right")) + 1
    ordered_check = gaps(a[:, 0], period)
    window_gaps = gaps(ts[lo:hi], period) if len(w) else None
    if window_gaps is not None:
        window_gaps["nonmonotonic"] = ordered_check["nonmonotonic"]
        window_gaps["nonfinite"] = ordered_check["nonfinite"]
    out = {"n_samples": int(len(w)), "gaps": window_gaps,
           "energy_int_J": integrate_trapezoid(a[:, 0], a[:, 2], t0, t1),
           "energy_int_lag_J": integrate_trapezoid(a[:, 0], a[:, 2], t0 + lag_s, t1 + lag_s) if lag_s else None,
           "energy_int_inst_J": integrate_trapezoid(a[:, 0], a[:, 13], t0, t1) if a.shape[1] > 13 else None}
    if len(w):
        out.update(temp_mean_c=float(w[:, 3].mean()), temp_max_c=float(w[:, 3].max()),
                   clk_gr_mean=float(w[:, 4].mean()), clk_gr_min=float(w[:, 4].min()), clk_gr_max=float(w[:, 4].max()),
                   clk_mem_mean=float(w[:, 5].mean()), util_gpu_mean=float(w[:, 6].mean()),
                   pstates=sorted({int(x) for x in w[:, 8]}),
                   clock_events_or=int(np.bitwise_or.reduce(w[:, 9].astype(np.int64))),
                   enforced_limit_mw=sorted({int(x) for x in w[:, 10]}),
                   power_mean_w=float(w[:, 2].mean()))
    return out
