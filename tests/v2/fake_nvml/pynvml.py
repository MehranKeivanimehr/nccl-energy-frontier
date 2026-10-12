"""Test double for pynvml (CPU only; never touches a GPU). Put this directory first on sys.path.

Behaviour is set through FAKE (in-process) or environment variables (subprocesses):
  FAKE_NVML_COUNTER = "forbid"  -> nvmlDeviceGetTotalEnergyConsumption raises AssertionError (and is counted)
                      "commit"  -> synthetic counter: value advances by power x period at every commit edge
                      "error"   -> raises NVMLError (e.g. NOT_SUPPORTED, like Pascal boards)
  FAKE_NVML_POWER_W, FAKE_NVML_TEMP_C, FAKE_NVML_CLOCK_EVENTS, FAKE_NVML_PIDS ("4:123,5:456"),
  FAKE_NVML_PERIOD_MS, FAKE_NVML_INSTANT ("1" supported, "0" not supported),
  FAKE_NVML_PROCS_ERROR ("1": the compute-process query raises)
"""

import os
import time
from types import SimpleNamespace

NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_GRAPHICS, NVML_CLOCK_MEM = 0, 2
NVML_FI_DEV_POWER_INSTANT = 186
NVML_SUCCESS, NVML_ERROR_NOT_SUPPORTED = 0, 3

FAKE = {}
CALLS = {"counter": 0}


class NVMLError(Exception):
    def __init__(self, value):
        super().__init__(f"NVML error {value}")
        self.value = value


def _cfg(k, default):
    return FAKE.get(k, os.environ.get(k, default))


def nvmlInit():
    return None


nvmlInit_v2 = nvmlInit


def nvmlShutdown():
    return None


def nvmlDeviceGetHandleByIndex(i):
    return int(i)


def nvmlDeviceGetPowerUsage(h):
    return int(float(_cfg("FAKE_NVML_POWER_W", 100.0)) * 1000)


def nvmlDeviceGetTemperature(h, kind):
    return int(float(_cfg("FAKE_NVML_TEMP_C", 40)))


def nvmlDeviceGetClockInfo(h, kind):
    return 1900 if kind == NVML_CLOCK_GRAPHICS else 8000


def nvmlDeviceGetUtilizationRates(h):
    return SimpleNamespace(gpu=50, memory=10)


def nvmlDeviceGetPerformanceState(h):
    return 8


def nvmlDeviceGetCurrentClocksEventReasons(h):
    return int(_cfg("FAKE_NVML_CLOCK_EVENTS", 0))


def nvmlDeviceGetEnforcedPowerLimit(h):
    return 300000


def nvmlDeviceGetMemoryInfo(h):
    return SimpleNamespace(used=1 << 20, total=48 << 30, free=47 << 30)


def nvmlDeviceGetFieldValues(h, ids):
    ok = str(_cfg("FAKE_NVML_INSTANT", "1")) == "1"
    v = SimpleNamespace(uiVal=int(float(_cfg("FAKE_NVML_POWER_W", 100.0)) * 1000))
    return [SimpleNamespace(nvmlReturn=0 if ok else NVML_ERROR_NOT_SUPPORTED, value=v) for _ in ids]


def _procs(h):
    if str(_cfg("FAKE_NVML_PROCS_ERROR", "0")) == "1":        # test-only: the compute-process query fails
        raise NVMLError(999)
    spec = str(_cfg("FAKE_NVML_PIDS", ""))
    out = []
    for item in filter(None, spec.split(",")):
        g, pid = item.split(":")
        if int(g) == int(h):
            out.append(SimpleNamespace(pid=int(pid), usedGpuMemory=1 << 20))
    return out


nvmlDeviceGetComputeRunningProcesses = _procs


def nvmlDeviceGetGraphicsRunningProcesses(h):
    return []


def nvmlDeviceGetTotalEnergyConsumption(h):
    CALLS["counter"] += 1
    mode = str(_cfg("FAKE_NVML_COUNTER", "forbid"))
    if mode == "forbid":
        with open(os.environ.get("FAKE_NVML_COUNTER_LOG", os.devnull), "a") as f:
            f.write("COUNTER_CALLED\n")
        raise AssertionError("cumulative energy counter called in a mode that must not call it")
    if mode == "error":
        raise NVMLError(NVML_ERROR_NOT_SUPPORTED)
    P = float(_cfg("FAKE_NVML_PERIOD_MS", 100.0)) * 1e6
    phase = float(_cfg("FAKE_NVML_PHASE_NS", 37_000_000))
    k = int((time.monotonic_ns() - phase) // P)
    return int(k * (P / 1e9) * float(_cfg("FAKE_NVML_POWER_W", 100.0)) * 1000)   # mJ committed at the last edge
