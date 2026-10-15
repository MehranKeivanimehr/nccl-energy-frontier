"""Background NVML telemetry sampler.

Records, per GPU and per sample:
    t_mono          time.perf_counter() timestamp (s), same clock as region events
    power_w         nvmlDeviceGetPowerUsage (W)
    util_gpu_pct    nvmlDeviceGetUtilizationRates().gpu
    util_mem_pct    nvmlDeviceGetUtilizationRates().memory
    temp_c          nvmlDeviceGetTemperature(GPU)
    clk_gr_mhz      nvmlDeviceGetClockInfo(GRAPHICS)
    clk_mem_mhz     nvmlDeviceGetClockInfo(MEM)
    pstate          nvmlDeviceGetPerformanceState
    query_ms        wall time spent in the NVML calls for this GPU/sample

Deliberately NOT queried by the sampler:
    * nvmlDeviceGetTotalEnergyConsumption (the energy counter), and
    * nvmlDeviceGetFieldValues(NVML_FI_DEV_POWER_INSTANT).
On the driver/GPU combination used here (R550, GA102) reading either of these
was observed to perturb the energy counter, so the counter is read only at
measurement-region boundaries by the orchestrator (see docs/methodology.md).
Every API used here was checked not to perturb the counter on this machine.
"""

from __future__ import annotations

import gzip
import csv
import threading
import time
from dataclasses import dataclass, field

import pynvml as N

FIELDS = [
    "t_mono", "gpu", "power_w", "util_gpu_pct", "util_mem_pct", "temp_c",
    "clk_gr_mhz", "clk_mem_mhz", "pstate", "query_ms",
]

def nvml_init() -> None:
    N.nvmlInit()


def handle(gpu_index: int):
    return N.nvmlDeviceGetHandleByIndex(gpu_index)


def gpu_processes(h) -> list[dict]:
    """Compute + graphics processes currently using the GPU (NVML view)."""
    out = []
    for kind, fn in (("compute", N.nvmlDeviceGetComputeRunningProcesses),
                     ("graphics", N.nvmlDeviceGetGraphicsRunningProcesses)):
        try:
            for p in fn(h):
                out.append({"pid": int(p.pid), "kind": kind,
                            "used_mem_bytes": None if p.usedGpuMemory is None else int(p.usedGpuMemory)})
        except N.NVMLError as e:  # pragma: no cover - driver dependent
            out.append({"pid": None, "kind": kind, "error": str(e)})
    return out


def _sample_one(h):
    t0 = time.perf_counter()
    p = N.nvmlDeviceGetPowerUsage(h) / 1000.0
    u = N.nvmlDeviceGetUtilizationRates(h)
    temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
    cg = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_GRAPHICS)
    cm = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM)
    ps = N.nvmlDeviceGetPerformanceState(h)
    q = (time.perf_counter() - t0) * 1e3
    return p, u.gpu, u.memory, temp, cg, cm, ps, q


@dataclass
class NvmlSampler:
    """Fixed-period sampler running in a daemon thread.

    The schedule is absolute (t_k = t_start + k*period) so that slow NVML
    calls do not accumulate drift; if a sample overruns the period the next
    tick is taken immediately and `overruns` is incremented.
    """

    gpus: list[int]
    period_s: float = 0.05
    proc_period_s: float = 1.0
    rows: list = field(default_factory=list)
    proc_events: list = field(default_factory=list)
    overruns: int = 0
    errors: list = field(default_factory=list)

    def __post_init__(self):
        self._handles = [handle(g) for g in self.gpus]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nvml-sampler", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self):
        t_start = time.perf_counter()
        next_proc = t_start
        k = 0
        while not self._stop.is_set():
            now = time.perf_counter()
            for g, h in zip(self.gpus, self._handles):
                try:
                    vals = _sample_one(h)
                    self.rows.append((now, g) + vals)
                except N.NVMLError as e:
                    self.errors.append((now, g, str(e)))
            if now >= next_proc:
                for g, h in zip(self.gpus, self._handles):
                    self.proc_events.append({"t_mono": now, "gpu": g, "procs": gpu_processes(h)})
                next_proc = now + self.proc_period_s
            k += 1
            target = t_start + k * self.period_s
            delay = target - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                self.overruns += 1
                # re-anchor so a single long stall does not trigger a burst
                t_start = time.perf_counter()
                k = 0

    def write_csv_gz(self, path) -> None:
        with gzip.open(path, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for r in self.rows:
                w.writerow([f"{r[0]:.6f}", r[1], f"{r[2]:.3f}", r[3], r[4], r[5], r[6], r[7], r[8], f"{r[9]:.3f}"])

    def stats(self) -> dict:
        """Achieved sampling statistics (used for QC)."""
        by_gpu = {}
        for r in self.rows:
            by_gpu.setdefault(r[1], []).append(r)
        out = {"period_s_requested": self.period_s, "overruns": self.overruns,
               "n_errors": len(self.errors), "per_gpu": {}}
        for g, rs in by_gpu.items():
            ts = [r[0] for r in rs]
            dts = [b - a for a, b in zip(ts, ts[1:])]
            qs = sorted(r[9] for r in rs)
            out["per_gpu"][str(g)] = {
                "n": len(rs),
                "median_dt_s": sorted(dts)[len(dts) // 2] if dts else None,
                "max_dt_s": max(dts) if dts else None,
                "median_query_ms": qs[len(qs) // 2] if qs else None,
            }
        return out
