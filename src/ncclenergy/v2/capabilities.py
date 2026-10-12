"""Hardware / counter capability detection for replication clusters (read-only by default).

    python -m ncclenergy.v2.capabilities --gpus 0,1 --out caps.json                    # allowed GPUs only
    python -m ncclenergy.v2.capabilities --gpus 0,1 --probe-counter 0,1 --out caps.json

Energy-counter and POWER_INSTANT reads can perturb the counter (driver dependent), so they are
made only on the GPUs listed in --gpus (your allocation). Other GPUs get passive NVML reads only.
--probe-counter additionally times the counter's commit period on the named idle GPUs.
Decides per GPU: COUNTER_ALIGNED (cumulative counter with a detectable commit period) or
SAMPLED_POWER_FALLBACK (no counter: energy from integrated sampled power, a different method
whose results must never be pooled with counter-based data).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
import pynvml as N

from .common import UNAVAILABLE, gpu_alias, redact, utc_now


def _try(fn, *a):
    try:
        v = fn(*a)
        return v.decode() if isinstance(v, bytes) else v
    except N.NVMLError as e:
        return UNAVAILABLE if "ermission" in str(e) else f"NVMLError: {e}"


def counter_supported(h) -> tuple[bool, str]:
    try:
        N.nvmlDeviceGetTotalEnergyConsumption(h)
        return True, "ok"
    except N.NVMLError as e:
        return False, str(e)


def instant_supported(h) -> bool:
    try:
        return N.nvmlDeviceGetFieldValues(h, [N.NVML_FI_DEV_POWER_INSTANT])[0].nvmlReturn == 0
    except N.NVMLError:
        return False


def commit_period(h, seconds: float = 1.5) -> dict:
    xs = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        a = time.monotonic_ns()
        v = N.nvmlDeviceGetTotalEnergyConsumption(h)
        xs.append((a, time.monotonic_ns(), v))
    ch = [(a, v) for (a, _, v), (_, _, v0) in zip(xs[1:], xs[:-1]) if v != v0]
    dt = np.diff([a for a, _ in ch]) / 1e6
    return {"reads": len(xs), "changes": len(ch), "median_period_ms": float(np.median(dt)) if len(dt) else None,
            "p10_ms": float(np.percentile(dt, 10)) if len(dt) else None,
            "p90_ms": float(np.percentile(dt, 90)) if len(dt) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices you are allowed to use")
    ap.add_argument("--probe-counter", default="", help="comma-separated idle GPU indices to time counter commits")
    a = ap.parse_args()
    N.nvmlInit()
    allowed = {int(x) for x in a.gpus.split(",") if x}
    probe = {int(x) for x in a.probe_counter.split(",") if x} & allowed
    gpus = []
    for i in range(N.nvmlDeviceGetCount()):
        h = N.nvmlDeviceGetHandleByIndex(i)
        uuid = _try(N.nvmlDeviceGetUUID, h)
        cc = _try(N.nvmlDeviceGetCudaComputeCapability, h)
        busy = bool(_try(N.nvmlDeviceGetComputeRunningProcesses, h) or _try(N.nvmlDeviceGetGraphicsRunningProcesses, h))
        if i in allowed and busy:   # never read counters on a GPU with any process (it may be measuring energy)
            sup, why = None, "not checked (GPU has a process)"
        else:
            sup, why = counter_supported(h) if i in allowed else (None, "not checked (outside --gpus)")
        g = {"index": i, "alias": gpu_alias(uuid if isinstance(uuid, str) else ""), "name": _try(N.nvmlDeviceGetName, h),
             "compute_capability": cc, "pci": _try(lambda hh: N.nvmlDeviceGetPciInfo(hh).busId, h),
             "energy_counter": sup, "energy_counter_detail": why, "power_usage": isinstance(_try(N.nvmlDeviceGetPowerUsage, h), int),
             "power_instant_field": instant_supported(h) if (i in allowed and not busy) else None, "enforced_limit_mw": _try(N.nvmlDeviceGetEnforcedPowerLimit, h),
             "nvlinks_active": sum(int(bool(_try(N.nvmlDeviceGetNvLinkState, h, k) == 1)) for k in range(N.NVML_NVLINK_MAX_LINKS)),
             "processes": len(_try(N.nvmlDeviceGetComputeRunningProcesses, h) or [])}
        if sup and i in probe:
            if g["processes"]:
                g["commit_probe"] = "skipped: GPU busy"
            else:
                g["commit_probe"] = commit_period(h)
        g["mode"] = None if sup is None else ("COUNTER_ALIGNED" if sup else "SAMPLED_POWER_FALLBACK")
        gpus.append(g)
    run = lambda *c: subprocess.run(list(c), capture_output=True, text=True, timeout=60).stdout
    caps = {"utc": utc_now(), "driver": _try(N.nvmlSystemGetDriverVersion), "nvml": _try(N.nvmlSystemGetNVMLVersion),
            "gpus": gpus, "topo_m": run("nvidia-smi", "topo", "-m"), "topo_p2p_r": run("nvidia-smi", "topo", "-p2p", "r"),
            "homogeneous_groups": {}}
    for g in gpus:
        caps["homogeneous_groups"].setdefault(f"{g['name']}|cc{g['compute_capability']}", []).append(g["index"])
    try:
        import torch
        caps["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "arch_list": torch.cuda.get_arch_list(),
                         "nccl": ".".join(map(str, torch.cuda.nccl.version()))}
        for g in gpus:
            cc = g["compute_capability"]
            if isinstance(cc, (tuple, list)):
                g["torch_supports_arch"] = f"sm_{cc[0]}{cc[1]}" in caps["torch"]["arch_list"]
    except Exception as e:  # noqa: BLE001
        caps["torch"] = {"error": str(e)}
    with open(a.out, "w") as f:
        f.write(redact(json.dumps(caps, indent=1, default=str)))
    for g in gpus:
        print(g["index"], g["name"], g["compute_capability"], g["mode"], "instant" if g["power_instant_field"] else "",
              g.get("commit_probe", ""))
    print("homogeneous groups:", caps["homogeneous_groups"])


if __name__ == "__main__":
    main()
