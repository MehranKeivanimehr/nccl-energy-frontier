"""Unprivileged preflight / health snapshot (NVML reads, nvidia-smi topo, host stats).

Never changes GPU state. Telemetry that the account cannot read is recorded as
UNAVAILABLE_DUE_TO_PERMISSIONS, never inferred as healthy.
"""

from __future__ import annotations

import os
import pwd
import shutil
import subprocess
import time

from .common import REPO, UNAVAILABLE, gpu_alias, utc_now
from .topology import parse_topo_matrix


def _N():
    import pynvml
    return pynvml


def _try(fn, *a):
    N = _N()
    try:
        v = fn(*a)
        return v.decode() if isinstance(v, bytes) else v
    except N.NVMLError as e:
        return UNAVAILABLE if "Permission" in str(e) or "No permission" in str(e) else f"NVMLError: {e}"


def process_owner(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/status") as f:
            for ln in f:
                if ln.startswith("Uid:"):
                    return pwd.getpwuid(int(ln.split()[1])).pw_name
    except (OSError, KeyError, ValueError):
        pass
    return UNAVAILABLE


THROTTLE_FIELDS = {"sw_power_cap": 74, "sync_boost": 76, "sw_thermal": 269, "hw_thermal": 270,
                   "hw_power_brake": 271}   # NVML_FI_DEV_CLOCKS_EVENT_REASON_* (ns)


def throttle_counters(h) -> dict:
    """Cumulative clock-event-reason durations (ns). Read per block only, never inside a region."""
    N = _N()
    try:
        vals = N.nvmlDeviceGetFieldValues(h, list(THROTTLE_FIELDS.values()))
        return {k: (int(v.value.ullVal) if v.nvmlReturn == 0 else f"rc{v.nvmlReturn}")
                for k, v in zip(THROTTLE_FIELDS, vals)}
    except N.NVMLError as e:
        return {"error": UNAVAILABLE if "ermission" in str(e) else str(e)}


def gpu_state(i: int, counters: bool = False) -> dict:
    N = _N()
    h = N.nvmlDeviceGetHandleByIndex(i)
    uuid = _try(N.nvmlDeviceGetUUID, h)
    procs = []
    for kind, fn in (("compute", N.nvmlDeviceGetComputeRunningProcesses),
                     ("graphics", N.nvmlDeviceGetGraphicsRunningProcesses)):
        try:
            for p in fn(h):
                procs.append({"pid": int(p.pid), "kind": kind, "owner": process_owner(int(p.pid)),
                              "used_mib": None if p.usedGpuMemory is None else p.usedGpuMemory / 2**20})
        except N.NVMLError as e:
            procs.append({"error": str(e)})
    try:
        ce = N.nvmlDeviceGetCurrentClocksEventReasons(h)
    except Exception:  # noqa: BLE001
        ce = _try(N.nvmlDeviceGetCurrentClocksThrottleReasons, h)
    mem = _try(lambda hh: N.nvmlDeviceGetMemoryInfo(hh, version=N.nvmlMemory_v2), h)  # 'used' excludes reserved
    return {
        "index": i, "alias": gpu_alias(uuid if isinstance(uuid, str) else ""), "uuid": uuid,
        "pci": _try(lambda hh: N.nvmlDeviceGetPciInfo(hh).busId, h),
        "name": _try(N.nvmlDeviceGetName, h),
        "pstate": _try(N.nvmlDeviceGetPerformanceState, h),
        "temp_c": _try(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU),
        "temp_slowdown_c": _try(N.nvmlDeviceGetTemperatureThreshold, h, N.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN),
        "temp_shutdown_c": _try(N.nvmlDeviceGetTemperatureThreshold, h, N.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN),
        "power_w": (lambda v: v / 1000 if isinstance(v, int) else v)(_try(N.nvmlDeviceGetPowerUsage, h)),
        "enforced_limit_mw": _try(N.nvmlDeviceGetEnforcedPowerLimit, h),
        "power_limit_mw": _try(N.nvmlDeviceGetPowerManagementLimit, h),
        "clk_gr_mhz": _try(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_GRAPHICS),
        "clk_mem_mhz": _try(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM),
        "max_clk_gr_mhz": _try(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_GRAPHICS),
        "max_clk_mem_mhz": _try(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_MEM),
        "app_clk_gr_mhz": _try(N.nvmlDeviceGetApplicationsClock, h, N.NVML_CLOCK_GRAPHICS),
        "util_gpu": _try(lambda hh: N.nvmlDeviceGetUtilizationRates(hh).gpu, h),
        "mem_used_mib": mem.used / 2**20 if hasattr(mem, "used") else mem,
        "mem_reserved_mib": mem.reserved / 2**20 if hasattr(mem, "reserved") else mem,
        "mem_free_mib": mem.free / 2**20 if hasattr(mem, "free") else mem,
        "throttle_counters_ns": throttle_counters(h) if counters else None,
        "clock_events": ce,
        "persistence": _try(N.nvmlDeviceGetPersistenceMode, h),
        "ecc_mode": _try(lambda hh: N.nvmlDeviceGetEccMode(hh)[0], h),
        "remapped_rows": _try(N.nvmlDeviceGetRemappedRows, h),
        "processes": procs,
    }


def host_state(path=REPO) -> dict:
    du = shutil.disk_usage(path)
    mem = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) // 1024
    except OSError:
        mem = UNAVAILABLE
    try:
        aff = sorted(os.sched_getaffinity(0))
    except OSError:
        aff = UNAVAILABLE
    return {"loadavg": os.getloadavg(), "disk_free_gb": du.free / 1e9, "disk_used_frac": du.used / du.total,
            "mem_mib": mem, "nproc": os.cpu_count(), "self_affinity_n": len(aff) if isinstance(aff, list) else aff,
            "self_nice": os.nice(0)}


def kernel_log_access() -> str:
    try:
        r = open("/proc/sys/kernel/dmesg_restrict").read().strip()
    except OSError:
        return UNAVAILABLE
    return UNAVAILABLE if r == "1" else "readable"


def xid_event_test(gpus: list[int]) -> dict:
    """Try to register for Xid critical-error events (read-only notification)."""
    N = _N()
    try:
        es = N.nvmlEventSetCreate()
    except N.NVMLError as e:
        return {"status": f"event set unavailable: {e}"}
    res = {}
    for g in gpus:
        h = N.nvmlDeviceGetHandleByIndex(g)
        try:
            N.nvmlDeviceRegisterEvents(h, N.nvmlEventTypeXidCriticalError, es)
            res[g] = "registered"
        except N.NVMLError as e:
            res[g] = UNAVAILABLE if "ermission" in str(e) else f"NVMLError: {e}"
    return {"status": "ok", "per_gpu": res, "event_set": es}


def poll_xid(es) -> list[dict]:
    """Non-blocking drain of pending Xid events."""
    N = _N()
    out = []
    while True:
        try:
            d = N.nvmlEventSetWait_v2(es, 0)
        except N.NVMLError:
            break
        out.append({"xid": int(d.eventData), "type": int(d.eventType), "t": time.time()})
    return out


def snapshot(alloc: list[int]) -> dict:
    N = _N()
    N.nvmlInit()
    n = N.nvmlDeviceGetCount()
    try:
        topo_txt = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60).stdout
        topo = parse_topo_matrix(topo_txt)
        topo_json = {"links": {f"{a}-{b}": v for (a, b), v in topo["links"].items()}, "numa": topo["numa"]}
    except Exception as e:  # noqa: BLE001
        topo_txt, topo_json = "", {"error": str(e)}
    nvl = {}
    for g in alloc:
        h = N.nvmlDeviceGetHandleByIndex(g)
        act = 0
        for link in range(N.NVML_NVLINK_MAX_LINKS):
            try:
                act += int(bool(N.nvmlDeviceGetNvLinkState(h, link)))
            except N.NVMLError:
                pass
        nvl[g] = act
    return {"utc": utc_now(), "driver": _try(N.nvmlSystemGetDriverVersion), "nvml": _try(N.nvmlSystemGetNVMLVersion),
            "cuda_driver": _try(N.nvmlSystemGetCudaDriverVersion_v2), "n_gpus": n, "allocation": alloc,
            "gpus": [gpu_state(i, counters=i in alloc) for i in range(n)], "topology": topo_json, "topo_text": topo_txt,
            "nvlink_active": nvl, "host": host_state(), "kernel_log": kernel_log_access(), "dcgm": dcgm_status()}


def dcgm_status() -> str:
    if shutil.which("dcgmi") is None:
        return "not installed"
    r = subprocess.run(["pgrep", "-x", "nv-hostengine"], capture_output=True)
    return "hostengine running" if r.returncode == 0 else "hostengine not running"


def light_health(alloc: list[int]) -> dict:
    """Cheap per-unit check of the allocated GPUs (no subprocess)."""
    N = _N()
    N.nvmlInit()
    return {"gpus": [gpu_state(g) for g in alloc], "host": host_state()}


def allocation_problems(snap: dict, alloc: list[int], mem_max_mib: float = 16.0) -> list[str]:
    probs = []
    by = {g["index"]: g for g in snap["gpus"]}
    for g in alloc:
        s = by.get(g)
        if s is None:
            probs.append(f"GPU{g}: not present")
            continue
        if s["processes"]:
            probs.append(f"GPU{g}: processes {[(p.get('pid'), p.get('owner')) for p in s['processes']]}")
        if isinstance(s["mem_used_mib"], (int, float)) and s["mem_used_mib"] > mem_max_mib:
            probs.append(f"GPU{g}: {s['mem_used_mib']:.0f} MiB used")
    return probs
