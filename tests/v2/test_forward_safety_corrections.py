"""Malformed-input and disabled-launch regressions; no GPU access."""
import math
from pathlib import Path

import numpy as np

from ncclenergy.v2.aligned_meter import aligned_energy
from ncclenergy.v2.telemetry import gaps
from ncclenergy.v2.runner import calibration_usable, build_env
from ncclenergy.v2.phase10 import require_validated_gpu_protocol, prospective_acceptance


def _raw():
    return {"c0_mJ": 1000, "rc0": 0, "br0": (0, 1000), "c1_mJ": 500000,
            "rc1": 0, "br1": (5008000000, 5008000000), "cpost_mJ": 509000,
            "rcpost": 0, "brpost": (5108000000, 5108000000), "phase_ns": 0,
            "bracket_ns": 2000000, "edges": [], "period_ns": 100000000,
            "period_est_ns": None, "commit_ns": 5000000000, "t_end_ns": 4970000000}


def test_meter_rejects_reversed_and_nonfinite_records():
    assert aligned_energy(_raw())["energy_J"] > 0
    for change in ({"br0": (10000000000, 10000000001)},
                   {"br1": (5008000001, 5008000000)},
                   {"c1_mJ": math.inf}, {"period_ns": 0}, {"t_end_ns": math.nan}):
        result = aligned_energy({**_raw(), **change})
        assert result["energy_J"] is None and result["flags"]


def test_original_telemetry_order_is_checked():
    assert gaps(np.array([1., .5, 2.]), .05)["nonmonotonic"] == 1
    assert gaps(np.array([1., 1., 2.]), .05)["nonmonotonic"] == 1
    assert gaps(np.array([1., math.inf]), .05)["nonfinite"] == 1


def test_failed_calibration_cannot_seed_schedule():
    good = {"rc": 0, "flags": [], "parse_flags": [], "t_gpu_s": .001}
    assert calibration_usable(good)
    for change in ({"rc": 1}, {"flags": ["BAD"]}, {"parse_flags": ["TRUNCATED"]},
                   {"t_gpu_s": math.inf}, {"t_gpu_s": -1}):
        assert not calibration_usable({**good, **change})


def test_sampled_mode_is_conveyed_to_native():
    env = build_env(["GPU-test"], "auto", Path("/tmp/fake-nccle-unit"), energy_source="sampled")
    assert env["NCCLE_ENERGY_SOURCE"] == "sampled"


def test_environment_cannot_override_allocation_or_measurement():
    for key in ("CUDA_VISIBLE_DEVICES", "NCCL_ALGO", "NCCLE_REGION_FILE",
                "NCCLE_ENERGY_SOURCE", "NCCL_PROFILER_PLUGIN", "LD_PRELOAD"):
        try:
            build_env(["GPU-test"], "auto", Path("/tmp/fake"), extra={key: "wrong"})
        except ValueError:
            pass
        else:
            raise AssertionError(key)
    env = build_env(["GPU-test"], "auto", Path("/tmp/fake"), extra={"NCCL_P2P_DISABLE": "1"})
    assert env["NCCL_P2P_DISABLE"] == "1"


def test_duplicate_or_empty_allocations_rejected_before_nvml():
    from ncclenergy.v2.runner import Ctx
    for alloc, placements in (([], []), ([4, 4], []), ([4], [{"id": "P", "gpus": [4, 4]}]),
                              ([4], [{"id": "P", "gpus": [4]}, {"id": "P", "gpus": [4]}])):
        try:
            Ctx({"allocation": alloc, "placements": placements}, "test")
        except SystemExit:
            pass
        else:
            raise AssertionError("invalid allocation accepted")


def test_historical_and_unvalidated_native_campaigns_disabled_before_nvml():
    from ncclenergy.v2.runner import _run_gpu_command
    for cfg in ({}, {"implementation_id": "release-corrections-2026-10-11", "energy_source": "sampled"}):
        try:
            _run_gpu_command(None, cfg, "test")
        except SystemExit:
            pass
        else:
            raise AssertionError("unvalidated collection was enabled")


def test_unvalidated_gpu_workload_is_disabled():
    try:
        require_validated_gpu_protocol()
    except SystemExit as e:
        assert "disabled" in str(e)
    else:
        raise AssertionError("Unvalidated GPU workload was allowed")


def test_all_historical_workloads_fail_strict_numerical_acceptance():
    import pandas as pd
    from ncclenergy.v2.common import REPO, read_jsonl
    runs = pd.DataFrame(read_jsonl(REPO / "data/v2/raw/phase10/workload/runs.jsonl")[0])
    mask, decisions = prospective_acceptance(runs, {})
    assert len(mask) == 50 and not any(mask)
    assert all("NUMERICAL_SANITY_NOT_PASSED" in d["reasons"] for d in decisions.values())


def test_watchdog_sees_transient_health_between_poll_ticks():
    from types import SimpleNamespace
    import time
    from ncclenergy.v2.watchdog import SamplerHealthCheck
    t = time.perf_counter()
    rows = [(t, 4, 100., 40., 1900, 7601, 50, 10, 0, 0x40, 300000, 0., 0., 100.),
            (t + .001, 4, 100., 40., 1900, 7601, 50, 10, 0, 0, 300000, 0., 0., 100.)]
    s = SimpleNamespace(rows=rows, proc_events=[(t, 4, [])], errors=[], truncated=False, proc_period_s=1.)
    c = SamplerHealthCheck(s, [4], lambda: set(), 80)
    assert c().startswith("HEALTH_CLOCK_EVENT")
