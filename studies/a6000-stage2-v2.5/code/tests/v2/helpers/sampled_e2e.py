"""Sampled-energy fallback end to end through the real runner code (idle_baseline + run_unit), with the fake
pynvml whose cumulative-counter call raises. Prints a JSON summary. Run with PYTHONPATH=<fake_nvml>:<src>."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pynvml
assert "fake_nvml" in pynvml.__file__, pynvml.__file__     # never the real NVML

from ncclenergy.v2 import runner

HERE = Path(__file__).resolve().parent
tmp = Path(sys.argv[1])
cfg = {"energy_source": "sampled", "sampler_period_s": 0.05, "power_lag_s": 0.0, "T_min_s": 2.0,
       "stop": {"temp_c": 80, "disk_floor_gb": 0.001, "mem_used_mib_max": 16},
       "timeouts": {"min_s": 60, "factor": 3}, "campaign": "test", "mode": "TEST"}
ctx = SimpleNamespace(cfg=cfg, placements={"P": [4, 5]}, placement_env={},
                      by_index={4: {"pci": "00000000:81:00.0", "alias": "A6000-a", "uuid": "GPU-a"},
                                5: {"pci": "00000000:A1:00.0", "alias": "A6000-b", "uuid": "GPU-b"}})
ctx.check_allocation = lambda: []
ctx.uuids = lambda gpus: ["GPU-a", "GPU-b"]
ctx.numa = lambda gpus: None
ctx.watch = lambda gpus, sampler=None: {"gpus": list(gpus), "sampler": sampler, "temp_stop_c": 80,
                                        "disk_floor_gb": 0.001}
idle = runner.idle_baseline(ctx, [4, 5], tmp / "idle", duration=1.0, settle=0.1, p8_timeout=2)
runner.build_cmd = lambda *a, **k: [sys.executable, str(HERE / "fake_bench.py"), "10", "2", "2.5"]
runner.build_env = lambda uuids, label, d, extra=None, **k: dict(os.environ)
uid = runner.unit_id("P", "all_reduce", "auto", 1048576)
bctx = {"dir": tmp / "block", "block": 1, "seed": "s", "exec_counter": 1,
        "frozen": {"entries": {uid: {"R": 10, "w": 2, "t_gpu_cal_s": 0.25}}},
        "placement_meta": {"P": {"id": "P", "gpus": [4, 5], "env": {}}}}
rec = runner.run_unit(ctx, bctx, {"placement": "P", "coll": "all_reduce", "config": "auto", "size": 1048576,
                                  "pos": 0}, 0, None)
print(json.dumps({"idle": idle, "qc_status": rec["qc_status"], "qc_reasons": rec["qc_reasons"],
                  "energy_source": rec["energy_source"], "sampled_energy_J": rec["sampled_energy_J"],
                  "T_region_s": (rec.get("region") or {}).get("T_region_s"), "evidence": rec["evidence"]["status"],
                  "telemetry_problems": rec["telemetry_problems"], "cleanup_ok": rec["proc"]["cleanup"]["ok"],
                  "counter_calls": pynvml.CALLS["counter"]}, default=str))
