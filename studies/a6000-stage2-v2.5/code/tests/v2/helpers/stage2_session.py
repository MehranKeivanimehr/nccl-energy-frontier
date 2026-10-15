"""Run one Stage 2 session scenario with the real stage2_runner code (CPU only): fake NVML (tests/v2/fake_nvml first
on PYTHONPATH), the fake v2.4 bench, short test timing. Every NVML call is audited with its time and thread.

    python -W error stage2_session.py <scenario> <out_dir_inside_repo>
Prints one JSON line.
"""
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pynvml
assert "fake_nvml" in pynvml.__file__, pynvml.__file__

CALLS = []


def _wrap(name, fn):
    def w(*a, **k):
        CALLS.append((time.monotonic_ns(), name, threading.current_thread().name))
        return fn(*a, **k)
    return w


for _n in dir(pynvml):
    if _n.startswith("nvml") and callable(getattr(pynvml, _n)):
        setattr(pynvml, _n, _wrap(_n, getattr(pynvml, _n)))

from ncclenergy.v2 import stage2_runner as S  # noqa: E402
from ncclenergy.v2.stage2_budget import Deadline  # noqa: E402

scenario, out = sys.argv[1], Path(sys.argv[2]).resolve()
HERE = Path(__file__).resolve().parent
os.environ.setdefault("FAKE_NVML_COUNTER", "commit")
for k, v in {"heldout_hot": {"FAKE_NVML_TEMP_C": "50"}, "baseline_fail": {"FAKE_NVML_PSTATE": "2"},
             "baseline_foreign": {"FAKE_NVML_PIDS": "4:999999"}, "complete_not_accepted": {"FAKE_S2_BAD_PART": "C2r1"},
             "s0a_fail": {"FAKE_S2_PREP_S": "3.0"}, "inventory": {"FAKE_S2_EXTRA_READS": "1"},
             "idle_ce_gpuidle": {"FAKE_NVML_CLOCK_EVENTS": "1"}, "idle_ce_swpc": {"FAKE_NVML_CLOCK_EVENTS": "4"},
             "idle_ce_thermal": {"FAKE_NVML_CLOCK_EVENTS": "64"}}.get(scenario, {}).items():
    os.environ[k] = v

bindir = out / "bin"
bindir.mkdir(parents=True)
wrapper = bindir / "all_gather_perf"
wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} -W error {HERE / 'fake_bench_stage2.py'} \"$@\"\n")
wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

SNAP = {"driver": "580.178.04", "nvml": "13.580", "cuda_driver": 13000, "topology": {"links": {"4-5": "NODE"}, "numa": {}},
        "gpus": [{"index": 4, "uuid": "GPU-aaaa", "pci": "00000000:81:00.0", "alias": "A6000-a", "name": "RTX A6000"},
                 {"index": 5, "uuid": "GPU-bbbb", "pci": "00000000:A1:00.0", "alias": "A6000-b", "name": "RTX A6000"}]}
_LH = {"n": 0}


def _light_health(alloc):
    _LH["n"] += 1          # policy_change: the 3rd call (after the first unit) reports a different power limit
    lim = 250000 if scenario == "policy_change" and _LH["n"] >= 3 else 300000
    return {"gpus": [{"index": g, "processes": [], "mem_used_mib": 1.0, "enforced_limit_mw": lim,
                      "max_clk_gr_mhz": 2100, "app_clk_gr_mhz": 1800} for g in alloc], "host": {"disk_free_gb": 100.0}}


S.light_health = _light_health
S.xid_event_test = lambda alloc: {"status": "fake (no event set)", "event_set": None}     # the fake NVML has no events
held = scenario.startswith("heldout")
counts = {a: {"w": 95597, "R": {"2.5": 47799, "10": 191193, "20": 382385, "40": 764769}} for a in "ABC"}
cfg = {"implementation_id": S.IMPLEMENTATION_ID, "method_id": S.METHOD_ID, "output_root": S.OUTPUT_ROOT,
       "mode": "STAGE2_HELDOUT" if held else "STAGE2_CALIBRATION", "campaign": "cpu-test", "allocation": [4, 5],
       "placements": [{"id": "P", "gpus": [4, 5]}], "max_attempts": 1, "counts": counts, "power_lag_s": 0.75,
       "sampler_period_s": 0.05, "energy_source": "counter", "total_power_samples": not held, "session": 1,
       "scope_expected": {"gpu_name": "RTX A6000", "driver": "580.178.04", "link": "NODE", "uuids": ["GPU-aaaa", "GPU-bbbb"],
                          "numa": {"4": None, "5": None}, "power_limit_mw": [300000, 300000],
                          "max_clk_gr_mhz": [2100, 2100]}}
if scenario == "smoke":                                  # the frozen 3-launch smoke schedule (E0, D100, D500)
    import yaml
    cfg["mode"] = "STAGE2_SMOKE"
    cfg["campaign"] = "cpu-test-smoke"
if scenario == "scope_mismatch":
    cfg["scope_expected"] = dict(cfg["scope_expected"], driver="550.54.15")
if held:
    units = [{"launch_id": "s1-b1-p01", "arm": "A", "length_s": 2.5, "pause_ns": 31_000_000},
             {"launch_id": "s1-b1-p02", "arm": "C", "length_s": 2.5, "pause_ns": 7_000_000}]
    if scenario == "heldout_dose_refused":
        units[0]["dose_period_ms"] = 100
elif scenario == "smoke":
    units = yaml.safe_load((Path(S.REPO) / "configs/stage2/smoke01_DRAFT.yaml").read_text())["units"]
else:
    units = [{"launch_id": "S0a-01", "part": "S0a", "arm": "B", "length_s": 2.5, "dose": "E0", "dose_period_ms": 0,
              "pause_ns": 97_700_000, "post_probe": True},
             {"launch_id": "S0b-02", "part": "S0b", "arm": "B", "length_s": 2.5, "dose": "D100", "dose_period_ms": 100,
              "pause_ns": 26_000_000, "post_probe": True},
             {"launch_id": "C2r1-03", "part": "C2r1", "arm": "B", "length_s": 2.5, "dose": "D500", "dose_period_ms": 500,
              "pause_ns": 54_600_000, "post_probe": True}]
timing = S.Timing(cal_fixed_wait_s=0.6, ho_w_max_s=4.0, stable_window_s=1.0, p8_wait_cap_s=2.0, baseline_settle_s=0.2,
                  baseline_measure_s=0.5, readiness_period_s=0.2)
ctx = S.Stage2Ctx(cfg, "cfg-sha", snapshot_fn=lambda alloc: SNAP, bin_dir=bindir, out_dir=out / "data")
if held:
    cfg["readiness_rules"] = {"t_launch_c": 45, "bands": {"4": {"t_region_c": [38, 45], "clk_mhz": [1880, 1950],
                                                                 "pstates": [8]},
                                                           "5": {"t_region_c": [38, 45], "clk_mhz": [1880, 1950],
                                                                 "pstates": [8]}}}
deadline = Deadline({"budget": 300.0, "budget_start": 150.0}.get(scenario, 10_000.0))
unit_windows = []
_orig = S.run_stage2_unit


_orig_run_process = S.R.run_process


def interrupting_run_process(*a, **k):
    raise KeyboardInterrupt("test: interrupted while the unit runs")


def timed_unit(*a, **k):
    t0 = time.monotonic_ns()
    if scenario == "foreign_in_unit":
        os.environ["FAKE_NVML_PIDS"] = "4:999999"          # another user's process appears during the unit
    if scenario == "interrupt_unit":
        S.R.run_process = interrupting_run_process
    try:
        return _orig(*a, **k)
    finally:
        unit_windows.append((t0, time.monotonic_ns()))


S.run_stage2_unit = timed_unit
if scenario == "baseline_fail_after_first_read":     # another user's process appears right after the first read
    _orig_counter_read = S._counter_read

    def counter_read_then_foreign(handles):
        out = _orig_counter_read(handles)
        os.environ["FAKE_NVML_PIDS"] = "4:999999"
        return out
    S._counter_read = counter_read_then_foreign
sdir = ctx.out / ("smoke" if scenario == "smoke" else ("calibration" if not held else "session_1"))
try:
    summary = S.run_session(ctx, units, deadline, timing)
except SystemExit as e:                       # schedule refused before anything ran
    summary = {"execution_status": "REFUSED", "stop_reason": str(e)}
except KeyboardInterrupt:
    summary = json.loads((sdir / "session_summary.json").read_text())
rerun = None
if scenario == "calibration":
    try:
        S.run_session(ctx, units, deadline, timing)
    except SystemExit as e:
        rerun = str(e)
recs = [json.loads(ln) for ln in (sdir / "attempts.jsonl").read_text().splitlines()] if (sdir / "attempts.jsonl").exists() else []
counter_in_units = sum(1 for t, n, _ in CALLS if n == "nvmlDeviceGetTotalEnergyConsumption"
                       and any(a <= t <= b for a, b in unit_windows))
part = sdir / "units" / units[0]["launch_id"]


def _evidence(name, key="counter_read_evidence"):
    f = sdir / name / "baseline.json"
    return json.loads(f.read_text()).get(key) if f.exists() else None
print(json.dumps({
    "status": summary["execution_status"], "stop_reason": summary.get("stop_reason"), "incomplete": summary.get("incomplete"),
    "n_accepted": summary.get("n_accepted"), "not_accepted": summary.get("not_accepted"),
    "partial_unit": {"unit_json": (part / "unit.json").exists(), "telemetry": (part / "telemetry.csv.gz").exists()},
    "baseline_start": summary.get("baseline_start"), "baseline_end": summary.get("baseline_end"),
    "postflight": summary.get("postflight"), "rerun_refused": rerun,
    "units": [{"id": r["unit_id"], "qc": r["qc_status"], "accepted": r.get("accepted"),
               "why": r.get("not_accepted_reasons"), "readiness": (r.get("readiness") or {}).get("status"),
               "s2_flags": (r.get("s2") or {}).get("flags"), "nccle_env": sorted(k for k in (r.get("nccl_env") or {})
                                                                                if k.startswith("NCCLE_")),
               "dose": [(p.get("dose") or {}).get("delivered_frac") for p in (r.get("s2") or {}).get("per_gpu", [])],
               "dose_detail": [{k: (p.get("dose") or {}).get(k) for k in ("eligible", "delivered", "straddling",
                                "shutdown_reads", "inflight_at_stop")} for p in (r.get("s2") or {}).get("per_gpu", [])],
               "required_channels": r.get("required_channel_problems"), "unsafe": r.get("unsafe"),
               "pp_brackets": [len((p.get("post_probe") or {}).get("brackets") or []) for p in (r.get("s2") or {}).get("per_gpu", [])],
               "inventory": [(p["n_counter_reads"], p["n_counter_reads_expected"]) for p in (r.get("s2") or {}).get("per_gpu", [])],
               "pre_region_ok": (r.get("pre_region_check") or {}).get("ok"),
               "tps_file": (Path(S.REPO) / r["dir"] / "total_power_samples.csv.gz").exists() if r.get("dir") else None}
              for r in recs],
    "unit_dirs": sorted(p.name for p in (sdir / "units").iterdir()) if (sdir / "units").exists() else [],
    "counter_calls_total": sum(1 for _, n, _ in CALLS if n == "nvmlDeviceGetTotalEnergyConsumption"),
    "counter_calls_in_units": counter_in_units, "n_unit_windows": len(unit_windows),
    "baseline_evidence": {"start": _evidence("baseline_start"), "end": _evidence("baseline_end")},
    "baseline_bit_shares": {"start": _evidence("baseline_start", "clock_event_bit_shares"),
                            "end": _evidence("baseline_end", "clock_event_bit_shares")},
    "unit_qc_reasons": [r.get("qc_reasons") for r in recs],
    "smoke_checks": [r.get("smoke_checks") for r in recs], "loader": summary.get("loader"),
    "deadline": summary.get("deadline"), "smoke_verdict": summary.get("smoke_verdict")}))
