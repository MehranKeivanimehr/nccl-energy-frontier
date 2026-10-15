"""Audit every NVML call made by the controller (runner + Sampler + watchdog + owned-process control) while a
unit runs, with timestamps, and locate cumulative-counter calls relative to the measured region.

CPU only: pynvml is the test double in tests/v2/fake_nvml (put first on PYTHONPATH); the "benchmark" is
tests/v2/helpers/fake_bench.py, which writes a region record with real monotonic timestamps.

    python nvml_call_audit.py <out_dir_inside_repo>
Counter mode is audited (energy_source=counter): the controller-side idle baseline is expected to read the
counter (GPUs idle, outside any region); run_unit is expected to make zero counter calls.
"""
import collections
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pynvml
assert "fake_nvml" in pynvml.__file__, pynvml.__file__

CALLS = []                                    # (t_monotonic_ns, function, thread name)


def _wrap(name, fn):
    def w(*a, **k):
        CALLS.append((time.monotonic_ns(), name, threading.current_thread().name))
        return fn(*a, **k)
    return w


for _n in dir(pynvml):
    if _n.startswith("nvml") and callable(getattr(pynvml, _n)):
        setattr(pynvml, _n, _wrap(_n, getattr(pynvml, _n)))

from ncclenergy.v2 import runner  # noqa: E402  (imports after the wrapping: every lookup goes through pynvml.*)

HERE = Path(__file__).resolve().parent
tmp = Path(sys.argv[1])
cfg = {"energy_source": "counter", "sampler_period_s": 0.05, "power_lag_s": 0.75, "T_min_s": 2.0,
       "stop": {"temp_c": 80, "disk_floor_gb": 0.001, "mem_used_mib_max": 16},
       "timeouts": {"min_s": 60, "factor": 3}, "campaign": "audit", "mode": "TEST"}
ctx = SimpleNamespace(cfg=cfg, placements={"P": [4, 5]}, placement_env={},
                      by_index={4: {"pci": "00000000:81:00.0", "alias": "A6000-a", "uuid": "GPU-a"},
                                5: {"pci": "00000000:A1:00.0", "alias": "A6000-b", "uuid": "GPU-b"}})
ctx.check_allocation = lambda: []
ctx.uuids = lambda gpus: ["GPU-a", "GPU-b"]
ctx.numa = lambda gpus: None
ctx.watch = lambda gpus, sampler=None: {"gpus": list(gpus), "sampler": sampler, "temp_stop_c": 80,
                                        "disk_floor_gb": 0.001}
t_idle0 = time.monotonic_ns()
runner.idle_baseline(ctx, [4, 5], tmp / "idle", duration=1.0, settle=0.1, p8_timeout=2)
t_idle1 = time.monotonic_ns()
os.environ["FAKE_NVML_COUNTER"] = "forbid"          # from here on any counter call raises inside the controller
runner.build_cmd = lambda *a, **k: [sys.executable, str(HERE / "fake_bench.py"), "10", "2", "2.5", "counter"]
runner.build_env = lambda uuids, label, d, extra=None, **k: dict(os.environ)
uid = runner.unit_id("P", "all_reduce", "auto", 1048576)
bctx = {"dir": tmp / "block", "block": 1, "seed": "s", "exec_counter": 1,
        "frozen": {"entries": {uid: {"R": 10, "w": 2, "t_gpu_cal_s": 0.25}}},
        "placement_meta": {"P": {"id": "P", "gpus": [4, 5], "env": {}}}}
t_unit0 = time.monotonic_ns()
rec = runner.run_unit(ctx, bctx, {"placement": "P", "coll": "all_reduce", "config": "auto", "size": 1048576,
                                  "pos": 0}, 0, None)
t_unit1 = time.monotonic_ns()
region = json.loads((Path(runner.REPO) / rec["dir"] / "region.jsonl").read_text().splitlines()[0])
g0 = region["gpus"][0]
r0, r1 = (g0["t_start_a_ns"] + g0["t_start_b_ns"]) // 2, g0["t_loop_end_ns"]


def window(a, b):
    c = [x for x in CALLS if a <= x[0] <= b]
    return {"n_calls": len(c), "counter_calls": sum(1 for x in c if x[1] == "nvmlDeviceGetTotalEnergyConsumption"),
            "by_function": dict(collections.Counter(x[1] for x in c)),
            "by_thread": dict(collections.Counter(x[2] for x in c))}


print(json.dumps({"idle_baseline": window(t_idle0, t_idle1), "run_unit": window(t_unit0, t_unit1),
                  "measured_region": {**window(r0, r1), "duration_s": (r1 - r0) / 1e9},
                  "qc_status": rec["qc_status"], "cleanup_ok": rec["proc"]["cleanup"]["ok"],
                  "fake_counter_calls_total": pynvml.CALLS["counter"]}))
