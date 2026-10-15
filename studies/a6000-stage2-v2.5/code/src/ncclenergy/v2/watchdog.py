"""Active safety watchdog for owned experiment processes (methodology v2.2).

A background thread evaluates checks about once per second while an owned
process runs. The first check that fails sets `tripped` to its reason; the
OwnedProcess loop (procctl.py) then cancels and cleans up the owned process
tree, and the caller stops the campaign. Nothing is retried.

The sampler-based check reads only what the telemetry Sampler already
collected (power, temperature, clock-event reasons, process lists). The
watchdog itself makes no NVML calls and never reads the energy counter.
"""

from __future__ import annotations

import shutil
import threading
import time
import math
from pathlib import Path

# NVML clock-event reason bits treated as health events during a run.
HW_SLOWDOWN, SW_THERMAL, HW_THERMAL, HW_POWER_BRAKE = 0x8, 0x20, 0x40, 0x80
DEFAULT_BAD_BITS = HW_SLOWDOWN | HW_THERMAL | HW_POWER_BRAKE | SW_THERMAL
# FIELDS index in telemetry.Sampler rows
_T, _GPU, _TEMP, _CE = 0, 1, 3, 9


class SamplerHealthCheck:
    """Contention / health / telemetry check over a running telemetry.Sampler.

    owned_pids: callable returning the set of pids we own (controller descendants).
    """

    def __init__(self, sampler, gpus, owned_pids, temp_stop_c: float, bad_bits: int = DEFAULT_BAD_BITS,
                 stale_s: float = 2.0, max_errors: int = 3, startup_s: float = 1.5, require_proc_list: bool = True):
        self.s, self.gpus, self.owned_pids = sampler, list(gpus), owned_pids
        self.temp_stop_c, self.bad_bits = temp_stop_c, bad_bits
        self.stale_s, self.max_errors, self.startup_s = stale_s, max_errors, startup_s
        self.require_proc_list = require_proc_list
        self.err0 = len(sampler.errors)
        self.t_begin = time.perf_counter()
        self._row_cursor = 0

    def __call__(self) -> str | None:
        now = time.perf_counter()
        if len(self.s.errors) - self.err0 >= self.max_errors:
            return f"TELEMETRY_ERRORS:{len(self.s.errors) - self.err0}"
        if self.s.truncated:
            return "TELEMETRY_TRUNCATED"
        latest = {}
        for r in reversed(self.s.rows[-8 * max(1, len(self.gpus)):]):
            if r[_GPU] not in latest:
                latest[r[_GPU]] = r
            if len(latest) == len(self.gpus):
                break
        if now - self.t_begin > self.startup_s:
            for g in self.gpus:
                r = latest.get(g)
                if r is None or now - r[_T] > self.stale_s:
                    return f"TELEMETRY_STALE:GPU{g}"
        # Inspect every newly recorded sample, not just the most recent row:
        # a transient event between watchdog polls still cancels the run.
        stop = len(self.s.rows)
        new_rows = self.s.rows[self._row_cursor:stop]
        self._row_cursor = stop
        for r in new_rows:
            g = r[_GPU]
            if g not in self.gpus:
                continue
            if r[_TEMP] is None or not math.isfinite(r[_TEMP]) or r[_CE] is None or not math.isfinite(r[_CE]):
                return f"TELEMETRY_INVALID:GPU{g}"
            if r[_TEMP] >= self.temp_stop_c:
                return f"TEMPERATURE:GPU{g}:{r[_TEMP]}C>={self.temp_stop_c}C"
            if int(r[_CE]) & self.bad_bits:
                return f"HEALTH_CLOCK_EVENT:GPU{g}:0x{int(r[_CE]) & self.bad_bits:x}"
        owned = self.owned_pids()
        seen = {}
        for t, g, pids in reversed(self.s.proc_events[-4 * max(1, len(self.gpus)):]):
            seen.setdefault(g, pids)
        for g, pids in seen.items():
            if pids is None:                    # the latest process-list query failed: contention undetectable
                return f"PROCESS_LIST_UNAVAILABLE:GPU{g}"
            foreign = sorted(p for p in pids if p not in owned)
            if foreign:
                return f"FOREIGN_PROCESS:GPU{g}:{foreign}"
        if self.require_proc_list and now - self.t_begin > self.startup_s + self.s.proc_period_s * 3 and not seen:
            return "TELEMETRY_NO_PROCESS_LIST"
        return None


class DiskFloorCheck:
    def __init__(self, path: Path, floor_gb: float):
        self.path, self.floor_gb = Path(path), floor_gb

    def __call__(self) -> str | None:
        free = shutil.disk_usage(self.path).free / 1e9
        return f"LOW_DISK:{free:.1f}GB<{self.floor_gb}GB" if free < self.floor_gb else None


class WallClockCheck:
    def __init__(self, limit_s: float):
        self.limit_s, self.t0 = limit_s, time.perf_counter()

    def __call__(self) -> str | None:
        return "WALL_TIME_EXCEEDED" if time.perf_counter() - self.t0 > self.limit_s else None


class Watchdog:
    """Runs checks every period_s in a daemon thread. A check that raises counts as a trip (fail closed)."""

    def __init__(self, checks, period_s: float = 1.0):
        self.checks, self.period_s = list(checks), period_s
        self.tripped: str | None = None
        self.trip_time: float | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="nccle-watchdog", daemon=True)

    def _run(self):
        while not self._stop.is_set() and self.tripped is None:
            for c in self.checks:
                try:
                    r = c()
                except Exception as e:  # noqa: BLE001 - a broken check must not silently pass
                    r = f"CHECK_FAILED:{type(c).__name__}:{e!r}"
                if r:
                    self.tripped, self.trip_time = r, time.perf_counter()
                    break
            self._stop.wait(self.period_s)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def check_now(self) -> str | None:
        """Synchronous evaluation (used before a launch)."""
        for c in self.checks:
            try:
                r = c()
            except Exception as e:  # noqa: BLE001
                r = f"CHECK_FAILED:{type(c).__name__}:{e!r}"
            if r:
                return r
        return None
